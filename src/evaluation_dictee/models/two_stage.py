"""Scorer en deux étapes (APPROCHE 1) : étape 1 HTR (lecture image), étape 2 codage textuel.

Découple lecture et jugement ; même interface `Scorer` que `VLMScorer` pour une comparaison juste.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, cast

# Client instrumenté Langfuse (comme VLMScorer) : sans lui, les appels des deux étapes
# n'apparaissent pas comme « générations » sous la trace de la copie.
from langfuse.openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from evaluation_dictee.config import ModelConfig, PromptConfig
from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.data.loaders import Copy, load_image
from evaluation_dictee.evaluation.ensemble import vote_majoritaire
from evaluation_dictee.models.base import CODE_NON_PARSE, CopyPrediction, ItemPrediction, Scorer
from evaluation_dictee.models.vlm import (
    _image_to_data_url,
    _items_json_schema,
    extract_reasoning,
    log_if_truncated,
    thinking_kwargs,
)
from evaluation_dictee.pipeline.alignement_local import Alignement, aligner, tokeniser
from evaluation_dictee.pipeline.alignment import best_realignment, needs_realignment
from evaluation_dictee.pipeline.prompts import (
    PROMPT_TRANSCRIPTION_PASSE1,
    attach_image,
    build_text_coding_aligned_prompt,
    build_text_coding_prompt,
    build_transcription_passe1_prompt,
    build_transcription_prompt,
    fetch_prompt,
)
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)

# Schéma de la réponse attendue à l'étape 1 (transcription seule).
_TRANSCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"transcription": {"type": "string"}},
    "required": ["transcription"],
}

# Passe 1 de l'option `transcription_verification` (transcription aveugle + incertains),
# suivie d'un alignement local sur les items (`pipeline/alignement_local.py`).
_PASSE1_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "incertains": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mot": {"type": "string"},
                    "lettres_vues": {"type": "string"},
                    "variantes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["mot", "lettres_vues", "variantes"],
            },
        },
        "transcription": {"type": "string"},
    },
    # `incertains` AVANT `transcription` : l'hésitation est décrite avant d'écrire le texte.
    "required": ["incertains", "transcription"],
}


def _parse_json_object(content: str) -> dict[str, Any] | None:
    """Parse un objet JSON de réponse modèle (balises ``` tolérées) ; None si illisible.

    Args:
        content: Contenu textuel brut renvoyé par le modèle.

    Returns:
        L'objet JSON décodé, ou None si le contenu n'est pas un objet JSON valide.
    """
    cleaned = content.strip().removeprefix("```json").removeprefix("```")
    cleaned = cleaned.removesuffix("```").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Repli : premier « { » au dernier « } » (texte parasite autour du JSON).
        debut, fin = content.find("{"), content.rfind("}")
        if debut < 0 or fin <= debut:
            return None
        try:
            data = json.loads(content[debut : fin + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


@dataclass
class _ResultatPasse:
    """Sortie parsée d'une passe de transcription."""

    transcription: str
    n_incertains: int | None = None


def _request_kwargs(
    model_config: ModelConfig, schema: dict[str, Any], schema_name: str
) -> dict[str, Any]:
    """Options d'appel communes aux deux étapes : décodage contraint et mode thinking.

    Ces options étaient appliquées à l'end-to-end (`VLMScorer`) mais pas ici, d'où les
    nombreux « JSON non extractible » à l'étape 2 : sans `response_format`, le modèle
    produit du JSON libre (souvent tronqué ou précédé d'un bloc de raisonnement).

    Args:
        model_config: Configuration du modèle de l'étape concernée.
        schema: Schéma JSON de la réponse attendue.
        schema_name: Nom du schéma transmis à l'API.

    Returns:
        Les kwargs à passer à `chat.completions.create` (éventuellement vides).
    """
    kwargs: dict[str, Any] = {"extra_body": thinking_kwargs(model_config)}
    if model_config.structured_output:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema},
        }
    return kwargs


def _extract_items_from_content(content: str) -> list[dict]:
    """Extrait la liste d'items d'une réponse modèle, du parsing strict au plus permissif.

    Args:
        content: Contenu textuel brut renvoyé par le modèle.

    Returns:
        Liste des items (dictionnaires bruts), vide si rien n'est extractible.
    """
    if not content:
        return []

    # Tentative 1 : JSON complet
    cleaned = content.strip().removeprefix("```json").removeprefix("```")
    cleaned = cleaned.removesuffix("```").strip()
    try:
        data = json.loads(cleaned)
        items = data.get("items")
        if isinstance(items, list) and items:
            return items
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        pass

    # Tentative 2 : trouver le premier objet racine avec une clé "items"
    m = re.search(r'\{[^{}]*"items"\s*:\s*\[', content)
    if m:
        start = m.start()
        # Aller jusqu'au } équilibré correspondant (approximation)
        depth = 0
        end = None
        for i in range(start, len(content)):
            c = content[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end:
            try:
                data = json.loads(content[start:end])
                items = data.get("items")
                if isinstance(items, list) and items:
                    return items
            except json.JSONDecodeError:
                pass

    # Tentative 3 : récupérer les sous-objets d'item isolés (utile si la liste est tronquée).
    items = []
    for match in re.finditer(
        r'\{\s*"item_id"\s*:\s*"[^"]+"[^{}]*"code"\s*:\s*"[^"]+"[^{}]*\}',
        content,
    ):
        try:
            items.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    return items


class TwoStageScorer(Scorer):
    """Évalue une copie en deux étapes : transcription HTR puis codage du texte."""

    def __init__(
        self,
        model_config: ModelConfig,
        prompt_config: PromptConfig,
        base_url: str,
        api_key: str,
        grid_items: list[GridItem],
        scheme: str = "simplifiee",
        model_config_stage2: ModelConfig | None = None,
        lecteurs_supplementaires: list[ModelConfig] | None = None,
    ) -> None:
        """Initialise le client, l'index des items et les configs des deux étapes.

        Args:
            model_config: Configuration du modèle de l'étape 1 (transcription HTR).
            prompt_config: Configuration du prompt (lecture de l'état final, etc.).
            base_url: URL de base de l'API compatible OpenAI.
            api_key: Clé d'API.
            grid_items: Items de la grille de codage.
            scheme: Grille de codage utilisée (par défaut « simplifiee »).
            model_config_stage2: Configuration du modèle de l'étape 2 (codage textuel) ;
                reprend `model_config` si None.
            lecteurs_supplementaires: Modèles qui relisent aussi la copie ; le code
                retenu par item est alors le code majoritaire entre lecteurs.
        """
        self.model_config = model_config
        self.model_config_stage2 = model_config_stage2 or model_config
        self.prompt_config = prompt_config
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.grid_items = grid_items
        self.scheme = scheme
        self._items_by_id = {it.item_id: it for it in grid_items}
        self.lecteurs_supplementaires = list(lecteurs_supplementaires or [])

    # ── Étape 1 : transcription HTR ──────────────────────────────────────────
    def _transcribe(self, copy: Copy) -> tuple[str, int]:
        """Lit l'image et renvoie (transcription, nb_tentatives) ; vide si échec total.

        Args:
            copy: Copie dont l'image est à transcrire.

        Returns:
            Couple (transcription, nombre de tentatives) ; transcription vide si aucun
            essai n'a produit de texte exploitable.
        """
        image = load_image(copy.image_path)
        messages = cast(
            "list[ChatCompletionMessageParam]",
            attach_image(
                build_transcription_prompt(read_final_state=self.prompt_config.read_final_state),
                _image_to_data_url(image),
            ),
        )
        request_kwargs = _request_kwargs(
            self.model_config, _TRANSCRIPTION_SCHEMA, "transcription_dictee"
        )
        for attempt in range(self.model_config.max_retries + 1):
            temp = self.model_config.temperature + (0.3 if attempt > 0 else 0.0)
            response = self.client.chat.completions.create(
                model=self.model_config.name,
                temperature=temp,
                max_tokens=self.model_config.max_tokens,
                messages=messages,
                **request_kwargs,
            )
            log_if_truncated(response, copy.copy_id, self.model_config.max_tokens)
            content = response.choices[0].message.content or "{}"
            transcription = self._parse_transcription(content)
            if transcription.strip():
                return transcription, attempt + 1
        return "", self.model_config.max_retries + 1

    @staticmethod
    def _parse_transcription(content: str) -> str:
        """Extrait le champ 'transcription' du JSON renvoyé à l'étape 1.

        Args:
            content: Contenu textuel brut renvoyé par le modèle.

        Returns:
            Transcription extraite ; repli sur le texte brut si le JSON est illisible.
        """
        try:
            cleaned = content.strip().removeprefix("```json").removeprefix("```")
            cleaned = cleaned.removesuffix("```").strip()
            data = json.loads(cleaned)
            return str(data.get("transcription", "")).strip()
        except (json.JSONDecodeError, KeyError, TypeError):
            # Repli : réponse en texte brut prise telle quelle.
            return content.strip()

    # ── Étape 1 en deux passes (option transcription_verification) ──────────
    def _appel_image(
        self,
        copy: Copy,
        messages_texte: list[Any],
        image_url: str | None,
        schema: dict[str, Any],
        schema_name: str,
        parse: Any,
        label: str,
        prompt_name: str,
        model_config: ModelConfig | None = None,
    ) -> tuple[_ResultatPasse | None, int]:
        """Appelle le modèle (avec ou sans image), avec retry tant que `parse` échoue.

        Args:
            copy: Copie concernée (pour les logs).
            messages_texte: Messages du prompt, sans image.
            image_url: Image de la copie en data URL ; None pour un appel texte seul.
            schema: Schéma JSON attendu (décodage contraint si activé).
            schema_name: Nom du schéma transmis à l'API.
            parse: Fonction contenu → `_ResultatPasse | None`.
            label: Nom de la passe, pour les logs.
            prompt_name: Nom du prompt Langfuse, pour lier la génération à sa version.
            model_config: Modèle à appeler ; celui de l'étape 1 si None.

        Returns:
            Couple (résultat parsé ou None, nombre de tentatives).
        """
        cfg = model_config or self.model_config
        avec_image = attach_image(messages_texte, image_url) if image_url else messages_texte
        messages = cast("list[ChatCompletionMessageParam]", avec_image)
        request_kwargs = _request_kwargs(cfg, schema, schema_name)
        # Lie la génération à la version du prompt dans Langfuse (rien si repli local).
        prompt_ref = fetch_prompt(prompt_name)
        if prompt_ref is not None:
            request_kwargs["langfuse_prompt"] = prompt_ref
        n_essais = cfg.max_retries + 1
        for attempt in range(n_essais):
            temp = cfg.temperature + (0.3 if attempt > 0 else 0.0)
            response = self.client.chat.completions.create(
                model=cfg.name,
                temperature=temp,
                max_tokens=cfg.max_tokens,
                messages=messages,
                **request_kwargs,
            )
            # Une génération tronquée n'est jamais acceptée : le repli « texte brut » de la
            # passe 1 prenait sinon 16 000 tokens de JSON coupé (boucle du modèle) pour une
            # transcription, et faussait les 83 items de la copie (13/11/2026).
            tronquee = log_if_truncated(response, copy.copy_id, cfg.max_tokens)
            resultat = None if tronquee else parse(response.choices[0].message.content or "")
            if resultat is not None and resultat.transcription.strip():
                return resultat, attempt + 1
            logger.warning(
                "%s : réponse inexploitable à l'essai %d/%d pour %s.",
                label,
                attempt + 1,
                n_essais,
                copy.copy_id,
            )
        return None, n_essais

    @staticmethod
    def _parse_passe1(content: str) -> _ResultatPasse | None:
        """Parse la réponse de la passe 1 (transcription à l'aveugle + incertains).

        Args:
            content: Contenu textuel brut renvoyé par le modèle.

        Returns:
            Le résultat parsé ; repli sur le texte brut si le JSON est illisible.
        """
        data = _parse_json_object(content)
        if data is None:
            # Même repli que l'étape 1 historique : texte brut pris tel quel.
            return _ResultatPasse(transcription=content.strip()) if content.strip() else None
        incertains = data.get("incertains")
        return _ResultatPasse(
            transcription=str(data.get("transcription", "")).strip(),
            n_incertains=len(incertains) if isinstance(incertains, list) else None,
        )

    def _transcribe_et_aligner(
        self,
        copy: Copy,
        lecteur: ModelConfig | None = None,
        image_url: str | None = None,
    ) -> tuple[str, int, _ResultatPasse | None, Alignement | None]:
        """Passe 1 (lecture à l'aveugle avec l'image), puis alignement LOCAL sur les items.

        Args:
            copy: Copie dont l'image est à transcrire.
            lecteur: Modèle qui lit la copie ; celui de l'étape 1 si None.
            image_url: Image déjà encodée (évite de la recharger pour chaque lecteur).

        Returns:
            (transcription, nb de tentatives, résultat passe 1, alignement) ;
            transcription vide et alignement None si la passe 1 a échoué.
        """
        if image_url is None:
            image_url = _image_to_data_url(load_image(copy.image_path))
        cfg = lecteur or self.model_config
        passe1, n1 = self._appel_image(
            copy,
            build_transcription_passe1_prompt(read_final_state=self.prompt_config.read_final_state),
            image_url,
            _PASSE1_SCHEMA,
            "transcription_passe1",
            self._parse_passe1,
            f"Passe 1 (transcription aveugle, {cfg.name})",
            PROMPT_TRANSCRIPTION_PASSE1,
            model_config=cfg,
        )
        if passe1 is None:
            return "", n1, None, None
        items = [self._items_by_id[i] for i in copy.item_ids if i in self._items_by_id]
        mots = tokeniser(passe1.transcription)
        if not mots or not items:
            return passe1.transcription, n1, passe1, None
        return passe1.transcription, n1, passe1, aligner(mots, items)

    # ── Étape 2 : codage du texte transcrit ──────────────────────────────────
    def _code_text(
        self,
        copy: Copy,
        reference_text: str,
        transcription: str,
        ecrits: dict[str, str] | None = None,
    ) -> CopyPrediction:
        """Code chaque item à partir de la transcription (sans image).

        Le retry évite qu'un échec ponctuel de génération JSON code toute la copie en '?'.

        Args:
            copy: Copie évaluée (fournit la liste d'items à coder).
            reference_text: Texte de référence de la dictée.
            transcription: Transcription produite à l'étape 1.
            ecrits: alignement local (mot écrit par item) ; si fourni, le
                modèle code des paires déjà alignées au lieu d'aligner lui-même.

        Returns:
            Prédiction de la copie ; items codés « ? » si aucun essai n'est exploitable.
        """
        items_a_coder = [self._items_by_id[i] for i in copy.item_ids if i in self._items_by_id]
        if ecrits is not None:
            prompt = build_text_coding_aligned_prompt(items_a_coder, ecrits, scheme=self.scheme)
        else:
            prompt = build_text_coding_prompt(
                reference_text=reference_text,
                transcription=transcription,
                items=items_a_coder,
                scheme=self.scheme,
            )
        messages = cast("list[ChatCompletionMessageParam]", prompt)
        # chain_of_thought=False : le prompt de l'étape 2 ne demande pas de champ
        # `comparaison`, inutile de le rendre obligatoire dans le schéma.
        request_kwargs = _request_kwargs(
            self.model_config_stage2, _items_json_schema(chain_of_thought=False), "codage_dictee"
        )
        for attempt in range(self.model_config_stage2.max_retries + 1):
            temp = self.model_config_stage2.temperature + (0.3 if attempt > 0 else 0.0)
            response = self.client.chat.completions.create(
                model=self.model_config_stage2.name,
                temperature=temp,
                max_tokens=self.model_config_stage2.max_tokens,
                messages=messages,
                **request_kwargs,
            )
            log_if_truncated(response, copy.copy_id, self.model_config_stage2.max_tokens)
            message = response.choices[0].message
            content = message.content or ""
            if _extract_items_from_content(content):
                prediction = self._parse_coding(copy, content, ecrits)
                # Raisonnement de l'étape 2 : c'est elle qui décide des codes.
                prediction.reasoning = extract_reasoning(message)
                return prediction
            logger.warning(
                "Étape 2 : JSON non extractible à l'essai %d/%d pour %s. Nouvel essai.",
                attempt + 1,
                self.model_config_stage2.max_retries + 1,
                copy.copy_id,
            )
        # Échec total : on parse quand même pour laisser _parse_coding logger et remonter des '?'.
        return self._parse_coding(copy, content, ecrits)

    def _parse_coding(
        self, copy: Copy, content: str, ecrits: dict[str, str] | None = None
    ) -> CopyPrediction:
        """Parse le JSON de codage de l'étape 2, avec ré-alignement de sécurité (cf. VLMScorer).

        Args:
            copy: Copie évaluée (fournit la liste d'items attendus).
            content: Contenu textuel brut renvoyé par le modèle à l'étape 2.
            ecrits: alignement local ; s'il est fourni, pas de ré-alignement
                (déjà fait) et la transcription de chaque item est celle de l'alignement.

        Returns:
            Prédiction de la copie ; items codés « ? » pour ceux absents de la réponse.
        """
        raw_items = _extract_items_from_content(content)
        if not raw_items:
            logger.warning(
                "Étape 2 (codage textuel) : aucun item extractible du JSON pour la "
                "copie %s. Réponse tronquée ou format cassé (longueur brute : %d). "
                "Tous les items seront codés '?'. Extrait : %r",
                copy.copy_id,
                len(content),
                content[:200],
            )

        codes_seq = [str(it.get("code", CODE_NON_PARSE)).strip() for it in raw_items]
        trans_seq = [it.get("transcription") for it in raw_items]
        conf_seq = [it.get("confidence") for it in raw_items]

        expected_words = [
            self._items_by_id[i].attendu for i in copy.item_ids if i in self._items_by_id
        ]

        if ecrits is None and codes_seq and needs_realignment(expected_words, trans_seq):
            aligned = best_realignment(expected_words, codes_seq, trans_seq, conf_seq)
            items = [
                ItemPrediction(
                    item_id=item_id,
                    code=a.code,
                    confidence=a.confidence,
                    transcription=a.transcription,
                )
                for item_id, a in zip(copy.item_ids, aligned, strict=False)
            ]
            return CopyPrediction(copy_id=copy.copy_id, items=items)

        by_id = {it.get("item_id"): it for it in raw_items}
        items = []
        for item_id in copy.item_ids:
            entry = by_id.get(item_id)
            if entry is None:
                items.append(ItemPrediction(item_id=item_id, code=CODE_NON_PARSE, confidence=0.0))
            else:
                items.append(
                    ItemPrediction(
                        item_id=item_id,
                        code=str(entry.get("code", CODE_NON_PARSE)).strip(),
                        confidence=entry.get("confidence"),
                        transcription=(
                            ecrits.get(item_id)
                            if ecrits is not None
                            else entry.get("transcription")
                        ),
                        comparaison=entry.get("comparaison"),
                    )
                )
        return CopyPrediction(copy_id=copy.copy_id, items=items)

    # ── Interface Scorer ─────────────────────────────────────────────────────
    def score_copy(self, copy: Copy, reference_text: str | None) -> CopyPrediction:
        """Évalue une copie : transcription puis codage (non transcrite si l'étape 1 échoue).

        Avec des lecteurs supplémentaires, chaque lecteur lit et fait coder la copie,
        puis on vote item par item (`_voter`).

        Args:
            copy: Copie à évaluer.
            reference_text: Texte de référence de la dictée, ou None.

        Returns:
            Prédiction de la copie ; `transcribed=False` si aucune lecture n'a produit
            de transcription. La transcription brute est conservée dans
            `raw_transcription` (celle du lecteur principal).
        """
        if not self.lecteurs_supplementaires:
            return self._score_un_lecteur(copy, reference_text, self.model_config)
        image_url = _image_to_data_url(load_image(copy.image_path))
        lecteurs = [self.model_config, *self.lecteurs_supplementaires]
        predictions = [
            self._score_un_lecteur(copy, reference_text, lecteur, image_url) for lecteur in lecteurs
        ]
        return self._voter(copy, predictions)

    def _voter(self, copy: Copy, predictions: list[CopyPrediction]) -> CopyPrediction:
        """Combine les prédictions de plusieurs lecteurs par vote majoritaire, item par item.

        Confiance d'un item = part des lecteurs qui donnent le code retenu (1 si tous
        concordent) : c'est le score de renvoi humain. Le lecteur principal (premier)
        l'emporte en cas d'égalité, et fournit les champs de niveau copie.

        Args:
            copy: Copie évaluée.
            predictions: une prédiction par lecteur, lecteur principal en premier.

        Returns:
            La prédiction combinée.
        """
        par_lecteur = [
            {it.item_id: it for it in p.items} if p.transcribed else {} for p in predictions
        ]
        items: list[ItemPrediction] = []
        for item_id in copy.item_ids:
            preds = [d.get(item_id) for d in par_lecteur]
            codes: list[str | None] = [
                p.code if p is not None and p.code != CODE_NON_PARSE else None for p in preds
            ]
            code, confiance, idx = vote_majoritaire(codes)
            source = preds[idx] if idx is not None else None
            items.append(
                ItemPrediction(
                    item_id=item_id,
                    code=code if code is not None else CODE_NON_PARSE,
                    confidence=confiance,
                    transcription=source.transcription if source is not None else None,
                    codes_lecteurs=[c if c is not None else CODE_NON_PARSE for c in codes],
                )
            )
        principal = predictions[0]
        combinee = CopyPrediction(
            copy_id=copy.copy_id,
            items=items,
            transcribed=any(p.transcribed for p in predictions),
            n_attempts=sum(p.n_attempts for p in predictions),
        )
        for champ in (
            "raw_transcription",
            "transcription_passe1",
            "n_incertains_passe1",
            "n_absents_alignement",
            "n_mots_en_trop_alignement",
            "reasoning",
        ):
            setattr(combinee, champ, getattr(principal, champ))
        combinee.transcriptions_lecteurs = [p.raw_transcription for p in predictions]
        return combinee

    def _score_un_lecteur(
        self,
        copy: Copy,
        reference_text: str | None,
        lecteur: ModelConfig,
        image_url: str | None = None,
    ) -> CopyPrediction:
        """Lecture par un modèle, puis codage : la chaîne complète pour UN lecteur.

        Args:
            copy: Copie à évaluer.
            reference_text: Texte de référence de la dictée, ou None.
            lecteur: Modèle qui lit la copie (passe 1).
            image_url: Image déjà encodée, partagée entre lecteurs.

        Returns:
            Prédiction de la copie pour ce lecteur.
        """
        passe1: _ResultatPasse | None = None
        alignement: Alignement | None = None
        if self.prompt_config.transcription_verification:
            transcription, n_attempts, passe1, alignement = self._transcribe_et_aligner(
                copy, lecteur, image_url
            )
        else:
            transcription, n_attempts = self._transcribe(copy)

        if not transcription.strip():
            items_vides = [
                ItemPrediction(item_id=i, code=CODE_NON_PARSE, confidence=0.0)
                for i in copy.item_ids
            ]
            return CopyPrediction(
                copy_id=copy.copy_id,
                items=items_vides,
                transcribed=False,
                n_attempts=n_attempts,
            )

        ecrits = alignement.ecrits() if alignement is not None else None
        prediction = self._code_text(copy, reference_text or "", transcription, ecrits)
        prediction.n_attempts = n_attempts
        prediction.raw_transcription = transcription
        if passe1 is not None:
            prediction.transcription_passe1 = passe1.transcription
            prediction.n_incertains_passe1 = passe1.n_incertains
        if alignement is not None:
            prediction.n_absents_alignement = alignement.n_absents()
            prediction.n_mots_en_trop_alignement = len(alignement.mots_en_trop)
        return prediction
