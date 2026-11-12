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
from evaluation_dictee.models.base import CODE_NON_PARSE, CopyPrediction, ItemPrediction, Scorer
from evaluation_dictee.models.vlm import (
    _image_to_data_url,
    _items_json_schema,
    extract_reasoning,
    log_if_truncated,
    thinking_kwargs,
)
from evaluation_dictee.pipeline.alignment import best_realignment, needs_realignment
from evaluation_dictee.pipeline.prompts import (
    PROMPT_TRANSCRIPTION_PASSE1,
    PROMPT_TRANSCRIPTION_PASSE2,
    attach_image,
    build_text_coding_prompt,
    build_transcription_passe1_prompt,
    build_transcription_passe2_prompt,
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

# Transcription en deux passes (option `transcription_verification`).
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

_PASSE2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "reference": {"type": "string"},
                    "passe1": {"type": "string"},
                    "lettres_vues": {"type": "string"},
                    "ecrit": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["conforme", "faute de l'élève", "mot oublié", "incertain"],
                    },
                    "correction_passe1": {"type": "boolean"},
                },
                "required": [
                    "reference",
                    "passe1",
                    "lettres_vues",
                    "ecrit",
                    "verdict",
                    "correction_passe1",
                ],
            },
        },
        "mots_ajoutes": {"type": "array", "items": {"type": "string"}},
        "transcription": {"type": "string"},
        "nb_incertains": {"type": "integer"},
    },
    "required": ["verifications", "mots_ajoutes", "transcription", "nb_incertains"],
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
    n_corrections: int | None = None


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
        """
        self.model_config = model_config
        self.model_config_stage2 = model_config_stage2 or model_config
        self.prompt_config = prompt_config
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.grid_items = grid_items
        self.scheme = scheme
        self._items_by_id = {it.item_id: it for it in grid_items}

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
        image_url: str,
        schema: dict[str, Any],
        schema_name: str,
        parse: Any,
        label: str,
        prompt_name: str,
    ) -> tuple[_ResultatPasse | None, int]:
        """Appelle le modèle d'étape 1 avec l'image, avec retry tant que `parse` échoue.

        Args:
            copy: Copie concernée (pour les logs).
            messages_texte: Messages du prompt, sans image.
            image_url: Image de la copie en data URL.
            schema: Schéma JSON attendu (décodage contraint si activé).
            schema_name: Nom du schéma transmis à l'API.
            parse: Fonction contenu → `_ResultatPasse | None`.
            label: Nom de la passe, pour les logs.
            prompt_name: Nom du prompt Langfuse, pour lier la génération à sa version.

        Returns:
            Couple (résultat parsé ou None, nombre de tentatives).
        """
        messages = cast("list[ChatCompletionMessageParam]", attach_image(messages_texte, image_url))
        request_kwargs = _request_kwargs(self.model_config, schema, schema_name)
        # Lie la génération à la version du prompt dans Langfuse (rien si repli local).
        prompt_ref = fetch_prompt(prompt_name)
        if prompt_ref is not None:
            request_kwargs["langfuse_prompt"] = prompt_ref
        n_essais = self.model_config.max_retries + 1
        for attempt in range(n_essais):
            temp = self.model_config.temperature + (0.3 if attempt > 0 else 0.0)
            response = self.client.chat.completions.create(
                model=self.model_config.name,
                temperature=temp,
                max_tokens=self.model_config.max_tokens,
                messages=messages,
                **request_kwargs,
            )
            log_if_truncated(response, copy.copy_id, self.model_config.max_tokens)
            resultat = parse(response.choices[0].message.content or "")
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

    @staticmethod
    def _parse_passe2(content: str) -> _ResultatPasse | None:
        """Parse la réponse de la passe 2 (vérification mot à mot avec la référence).

        Pas de repli texte brut : un texte libre en passe 2 n'est pas une transcription
        vérifiée, on préfère alors garder la passe 1.

        Args:
            content: Contenu textuel brut renvoyé par le modèle.

        Returns:
            Le résultat parsé, ou None si le JSON est illisible.
        """
        data = _parse_json_object(content)
        if data is None:
            return None
        verifications = data.get("verifications")
        n_corrections = None
        if isinstance(verifications, list):
            n_corrections = sum(
                1 for v in verifications if isinstance(v, dict) and v.get("correction_passe1")
            )
        nb_incertains = data.get("nb_incertains")
        return _ResultatPasse(
            transcription=str(data.get("transcription", "")).strip(),
            n_incertains=nb_incertains if isinstance(nb_incertains, int) else None,
            n_corrections=n_corrections,
        )

    def _transcribe_deux_passes(
        self, copy: Copy, reference_text: str
    ) -> tuple[str, int, _ResultatPasse | None, _ResultatPasse | None]:
        """Transcription en deux passes : aveugle, puis vérifiée avec la référence.

        Si la passe 2 échoue, on garde la passe 1 (la copie reste codable).

        Args:
            copy: Copie dont l'image est à transcrire.
            reference_text: Texte de référence de la dictée.

        Returns:
            (transcription retenue, nb total de tentatives, résultat passe 1, résultat
            passe 2) ; transcription vide si la passe 1 a échoué.
        """
        image_url = _image_to_data_url(load_image(copy.image_path))
        read_final = self.prompt_config.read_final_state
        passe1, n1 = self._appel_image(
            copy,
            build_transcription_passe1_prompt(read_final_state=read_final),
            image_url,
            _PASSE1_SCHEMA,
            "transcription_passe1",
            self._parse_passe1,
            "Passe 1 (transcription aveugle)",
            PROMPT_TRANSCRIPTION_PASSE1,
        )
        if passe1 is None:
            return "", n1, None, None
        if not reference_text.strip():
            logger.warning("Pas de texte de référence pour %s : passe 2 sautée.", copy.copy_id)
            return passe1.transcription, n1, passe1, None
        passe2, n2 = self._appel_image(
            copy,
            build_transcription_passe2_prompt(
                reference_text=reference_text,
                transcription_passe1=passe1.transcription,
                read_final_state=read_final,
            ),
            image_url,
            _PASSE2_SCHEMA,
            "transcription_passe2",
            self._parse_passe2,
            "Passe 2 (vérification)",
            PROMPT_TRANSCRIPTION_PASSE2,
        )
        if passe2 is None:
            logger.warning(
                "Passe 2 en échec pour %s : on garde la transcription de la passe 1.",
                copy.copy_id,
            )
            return passe1.transcription, n1 + n2, passe1, None
        return passe2.transcription, n1 + n2, passe1, passe2

    # ── Étape 2 : codage du texte transcrit ──────────────────────────────────
    def _code_text(self, copy: Copy, reference_text: str, transcription: str) -> CopyPrediction:
        """Code chaque item à partir de la transcription (sans image).

        Le retry évite qu'un échec ponctuel de génération JSON code toute la copie en '?'.

        Args:
            copy: Copie évaluée (fournit la liste d'items à coder).
            reference_text: Texte de référence de la dictée.
            transcription: Transcription produite à l'étape 1.

        Returns:
            Prédiction de la copie ; items codés « ? » si aucun essai n'est exploitable.
        """
        items_a_coder = [self._items_by_id[i] for i in copy.item_ids if i in self._items_by_id]
        messages = cast(
            "list[ChatCompletionMessageParam]",
            build_text_coding_prompt(
                reference_text=reference_text,
                transcription=transcription,
                items=items_a_coder,
                scheme=self.scheme,
            ),
        )
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
                prediction = self._parse_coding(copy, content)
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
        return self._parse_coding(copy, content)

    def _parse_coding(self, copy: Copy, content: str) -> CopyPrediction:
        """Parse le JSON de codage de l'étape 2, avec ré-alignement de sécurité (cf. VLMScorer).

        Args:
            copy: Copie évaluée (fournit la liste d'items attendus).
            content: Contenu textuel brut renvoyé par le modèle à l'étape 2.

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

        if codes_seq and needs_realignment(expected_words, trans_seq):
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
                        transcription=entry.get("transcription"),
                        comparaison=entry.get("comparaison"),
                    )
                )
        return CopyPrediction(copy_id=copy.copy_id, items=items)

    # ── Interface Scorer ─────────────────────────────────────────────────────
    def score_copy(self, copy: Copy, reference_text: str | None) -> CopyPrediction:
        """Évalue une copie : transcription puis codage (non transcrite si l'étape 1 échoue).

        Args:
            copy: Copie à évaluer.
            reference_text: Texte de référence de la dictée, ou None.

        Returns:
            Prédiction de la copie ; `transcribed=False` si l'étape 1 (HTR) n'a produit
            aucune transcription. La transcription brute est conservée dans
            `raw_transcription`.
        """
        passe1: _ResultatPasse | None = None
        passe2: _ResultatPasse | None = None
        if self.prompt_config.transcription_verification:
            transcription, n_attempts, passe1, passe2 = self._transcribe_deux_passes(
                copy, reference_text or ""
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

        prediction = self._code_text(copy, reference_text or "", transcription)
        prediction.n_attempts = n_attempts
        prediction.raw_transcription = transcription
        if passe1 is not None:
            prediction.transcription_passe1 = passe1.transcription
            prediction.n_incertains_passe1 = passe1.n_incertains
        if passe2 is not None:
            prediction.n_corrections_passe2 = passe2.n_corrections
            prediction.n_incertains_passe2 = passe2.n_incertains
        return prediction
