"""Chargement et validation des configurations d'expérience (YAML validé par Pydantic)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    """Secrets lus depuis les variables d'environnement (ou le fichier .env)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_s3_endpoint: str = "minio.lab.sspcloud.fr"
    # Répertoire S3 où sont déposées les prédictions finies (JAMAIS dans Git : les
    # transcriptions sont des données d'élèves mineurs). Sert à relancer notebooks
    # et site Quarto sans réexécuter le pipeline. Surchargé par S3_PREDICTIONS_PREFIX.
    s3_predictions_prefix: str = "s3://projet-production-ecrits-depp/predictions"

    llm_base_url: str = "https://llm.lab.sspcloud.fr/api/v1"
    llm_api_key: str = "dummy"


class ImageConfig(BaseModel):
    """Préparation de l'image envoyée au lecteur (passe 1). Défaut : page telle quelle.

    Voir `data/pretraitement.py` pour les mesures qui motivent chaque option.
    """

    # Recadre sur la zone d'écriture (intérieur du cadre pré-imprimé).
    recadrage: bool = False
    # Découpe en bandes horizontales (coupes au creux des interlignes), envoyées comme
    # images successives d'un même appel. Utile aux modèles à budget d'image fixe (gemma).
    n_bandes: int = Field(default=1, ge=1, le=8)
    # Facteur d'agrandissement (Lanczos). Utile aux modèles en pleine résolution (qwen).
    agrandissement: float = Field(default=1.0, ge=1.0, le=3.0)

    def slug(self) -> str:
        """Suffixe de nom de run ("" pour les options par défaut)."""
        parts = []
        if self.recadrage:
            parts.append("crop")
        if self.n_bandes > 1:
            parts.append(f"b{self.n_bandes}")
        if self.agrandissement != 1.0:
            parts.append(f"x{self.agrandissement:g}".replace(".", "p"))
        return "-".join(parts)


class ModelConfig(BaseModel):
    """Paramètres du modèle à interroger."""

    name: str  # nom du modèle servi par vLLM (ex: "Qwen/Qwen2.5-VL-7B-Instruct")
    # "vlm" = multimodal image+texte ; "llm" = texte seul ; "htr" = HTR classique
    kind: Literal["vlm", "llm", "htr"] = "vlm"
    temperature: float = 0.0
    max_tokens: int = 2048
    request_logprobs: bool = True  # log-probs pour estimer la confiance par item
    # 0 = pas de retry. Copie sans transcription après tous les essais = exclue des métriques.
    max_retries: int = 2
    # Désactive le bloc <think> des modèles thinking (Qwen3, DeepSeek-R1, QwQ) qui casse
    # le parsing JSON et multiplie la latence. Sans effet sur gemma4, Qwen2.5-VL...
    disable_thinking: bool = True
    # Force une sortie JSON conforme au schéma (décodage contraint vLLM), supprimant les
    # copies non parsables. Mettre à False si l'endpoint ne supporte pas json_schema.
    structured_output: bool = True
    # Préparation de l'image (lecteurs de la passe 1 seulement ; ignoré en étape 2).
    image: ImageConfig = Field(default_factory=ImageConfig)


class DataConfig(BaseModel):
    """Localisation et périmètre des données."""

    corpus: Literal["dictee", "production_ecrite"] = "dictee"
    images_path: str  # chemin local OU préfixe S3 (s3://.../dictee_2015/)
    # CSV des codes experts, faisant office de gold standard (annotateur unique). Local ou S3.
    labels_path: str
    # Grille JSON (versionnée dans configs/) : mot attendu + fautes connues par item.
    grid_path: str = "configs/grille_dictee_2015.json"
    limit: int | None = None  # limiter le nombre de copies (tests rapides)
    # Seuil de densité d'encre en dessous duquel une copie est jugée VIERGE (l'élève
    # n'a rien écrit : seul le pré-imprimé marque ~2%). Ces copies sont codées "0"
    # (absent) sur tous les items, sans appel modèle, identiquement pour toutes les
    # méthodes — évite que l'end-to-end hallucine la référence. 0 pour désactiver.
    blank_ink_threshold: float = 0.025


class GridConfig(BaseModel):
    """Schéma de codage cible."""

    # "simplifiee" = 1/erreur/0 (cible principale) ; "complete" = 1/3/4/5/9/0
    scheme: Literal["simplifiee", "complete"] = "simplifiee"


class PromptConfig(BaseModel):
    """Stratégie de prompting."""

    method: Literal["A", "B", "C", "D"] = "C"
    n_few_shot: int = 0  # nombre d'exemples annotés dans le prompt
    enforce_faithful: bool = True  # consigne anti-sur-correction
    read_final_state: bool = True  # règle des ratures : lire l'état final
    # Force un champ "comparaison" avant le code (verbalise la différence lue-attendue).
    chain_of_thought: bool = False
    # Demande au modèle de COMPTER les items effectivement écrits sur la copie (mots +
    # ponctuation) et de confronter ce total au nombre d'items attendus, AVANT de coder.
    # Vise le biais principal mesuré : le modèle lit le texte de référence et code
    # "présent" des mots que l'élève n'a jamais écrits.
    count_items: bool = False
    # Fournit au modèle, item par item, les fautes RÉELLEMENT observées par les
    # correcteurs sur cet item (champs `ex_*` de la grille). 55 items sur 83 en ont.
    # Vise la cause dominante mesurée : 90 % des fautes manquées sont des items que
    # le modèle transcrit à l'identique du mot attendu, sans voir la différence.
    show_error_examples: bool = False
    # Oblige le nombre d'items codés "0" à coïncider avec le comptage déclaré.
    # Mesuré : le modèle annonce un décompte puis en code un autre sur 9 copies
    # sur 50 (qwen3-6) et 10 sur 35 (qwen3-8) — le comptage reste décoratif.
    # Sans effet si `count_items` est faux (il n'y a alors rien à faire coïncider).
    enforce_count: bool = False
    # Vérification de voisinage N-1 / N / N+1 appliquée à CHAQUE item, et non
    # conditionnée à un manque déclaré : conditionner serait inopérant sur un modèle
    # qui annonce 83 items sur 49 copies sur 50 alors que 27 en manquent.
    check_neighbours: bool = False
    # Expérimentation 1 : ne montre PAS le texte de référence en phrase continue —
    # seulement la liste d'items isolés (déjà affichée par ailleurs). Vise la tension
    # structurelle identifiée le 18/09/2026 : une phrase familière invite à la lecture
    # fluide par reconnaissance plutôt qu'à l'examen lettre à lettre de l'image, ce qui
    # va dans le sens du biais dominant mesuré (sous-détection des fautes).
    reference_items_only: bool = False
    # Expérimentation 3 : pour un item ponctuation, vérifie D'ABORD qu'un signe existe
    # avant d'en juger la nature. Vise le mécanisme distinct de la ponctuation (omission
    # plutôt que faute), documenté au rapport de bras section 7 (mots vs ponctuation).
    check_punctuation_presence: bool = False
    # Expérimentation 4 : présente le mot ATTENDU en contraste des fautes connues,
    # plutôt que les fautes seules. Sans effet si show_error_examples est faux.
    contrastive_examples: bool = False
    # two_stage uniquement : l'étape 1 (transcription) se fait en DEUX passes.
    # Passe 1 : transcription à l'aveugle (sans référence) avec liste des mots incertains.
    # Passe 2 : le modèle revoit l'image AVEC la référence et la passe 1, et vérifie mot
    # à mot. Vise les deux erreurs de la passe 1 : faute de l'élève corrigée par erreur
    # (sur-correction) et erreur de lecture prise pour une faute. La transcription de la
    # passe 2 alimente l'étape de codage. Sans effet en end_to_end.
    transcription_verification: bool = False


class ExperimentConfig(BaseModel):
    """Configuration complète d'une expérience (un fichier YAML = un run)."""

    name: str = Field(..., description="Nom unique du run, utilisé comme nom de trace Langfuse")
    seed: int = 42
    # Nombre de copies évaluées EN PARALLÈLE (le endpoint vLLM batche les requêtes
    # concurrentes). 1 = séquentiel (ancien comportement). Monter tant que le serveur
    # suit (débit borné par le GPU) ; redescendre en cas d'erreurs/timeout.
    concurrency: int = Field(default=8, ge=1)
    # "end_to_end" : un VLM lit l'image ET code en une passe.
    # "two_stage" : étape 1 HTR (transcription) puis étape 2 codage (isole lecture/jugement).
    approach: Literal["end_to_end", "two_stage"] = "end_to_end"
    model: ModelConfig
    # Modèle de l'étape 2 (codage textuel) en two_stage ; si absent, on réutilise `model`.
    model_stage2: ModelConfig | None = None
    # Lecteurs SUPPLÉMENTAIRES (two_stage + transcription_verification) : chacun relit
    # la copie (passe 1), son texte est aligné puis codé par `model_stage2`, et le code
    # retenu par item est le code majoritaire (`model` ne tranche que sans majorité :
    # trois codes différents, ou un lecteur en échec et les deux autres en désaccord). La
    # confiance d'un item = part des lecteurs qui donnent ce code : la discordance
    # entre lecteurs sert de score de renvoi humain. Mesuré le 13/11/2026 sur 50 copies
    # (qwen3-8-27b, gemma4-26b-moe, qwen3-6-35b-moe) : kappa 0,728 contre 0,687 pour
    # le meilleur lecteur seul. Coût : un appel de lecture et un de codage par lecteur.
    extra_readers: list[ModelConfig] = Field(default_factory=list)
    data: DataConfig
    grid: GridConfig = GridConfig()
    prompt: PromptConfig = PromptConfig()

    @model_validator(mode="after")
    def _lecteurs_supplementaires_valides(self) -> ExperimentConfig:
        """Le vote entre lecteurs suppose l'alignement local (two_stage + vérification)."""
        if self.extra_readers and not (
            self.approach == "two_stage" and self.prompt.transcription_verification
        ):
            raise ValueError(
                "`extra_readers` exige approach: two_stage et "
                "prompt.transcription_verification: true (alignement par item)."
            )
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    """Charge et valide une configuration d'expérience depuis un fichier YAML.

    Args:
        path: Chemin du fichier YAML de configuration.

    Returns:
        La configuration d'expérience validée.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig.model_validate(raw)


def _slugify_model_name(name: str) -> str:
    """Convertit un nom de modèle en fragment de nom de fichier sûr, sans perte.

    Ne remplace QUE les caractères réellement invalides dans un nom de fichier
    (`/`, espaces, `:`…) : `.` et `-` sont conservés tels quels et NE SONT PAS
    interchangés, pour que deux noms de modèle différents (ex. un nom fautif
    avec un point vs. le nom réel avec un tiret) ne produisent jamais le même
    suffixe — et donc n'écrasent jamais le même fichier de sortie.
    """
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", name).strip("-.")


def run_output_name(config: ExperimentConfig) -> str:
    """Nom de base des fichiers de sortie d'un run : `name` + modèle(s), sans doublon.

    Source unique de vérité pour nommer `<...>_predictions.jsonl` et
    `<...>_failed_copies.txt`. Le(s) modèle(s) figurent TOUJOURS dans le nom, que
    le run soit lancé via le YAML seul ou via les surcharges `--model-name` /
    `--model-stage2-name` : deux modèles n'écrasent donc jamais le même fichier.

    Idempotent : si `config.name` porte déjà le suffixe de modèle (cas d'un
    lancement CLI, où `override_model_names` a déjà renommé le run), il n'est pas
    ajouté une seconde fois. C'est ce qui garantit qu'un même run écrit dans le
    même fichier — et retrouve donc son checkpoint de reprise — quel que soit son
    mode de lancement.

    Args:
        config: Configuration de l'expérience.

    Returns:
        Le préfixe des fichiers de sortie du run.
    """

    def _avec_image(m: ModelConfig) -> str:
        slug_image = m.image.slug()
        base = _slugify_model_name(m.name)
        return f"{base}-{slug_image}" if slug_image else base

    parts = [_avec_image(config.model)]
    if config.model_stage2 is not None:
        slug_stage2 = _slugify_model_name(config.model_stage2.name)
        if slug_stage2 not in parts:
            parts.append(slug_stage2)
    # Lecteurs supplémentaires : un vote change les prédictions, il ne doit jamais
    # partager le checkpoint du run à lecteur unique.
    for lecteur in config.extra_readers:
        slug = _avec_image(lecteur)
        if slug not in parts:
            parts.append(slug)
    suffix = "_".join(parts)

    if config.name == suffix or config.name.endswith(f"_{suffix}"):
        return config.name
    return f"{config.name}_{suffix}"


def override_model_names(
    config: ExperimentConfig,
    model_name: str | None = None,
    model_stage2_name: str | None = None,
) -> ExperimentConfig:
    """Surcharge le(s) nom(s) de modèle d'une config, sans dupliquer le YAML.

    Les noms doivent correspondre EXACTEMENT à ceux servis sur llm.lab. Dès qu'un
    nom est surchargé, le `name` du run (donc le fichier de sortie
    `data/processed/<name>_predictions.jsonl`) est suffixé par le(s) modèle(s)
    utilisé(s), pour ne jamais écraser le checkpoint d'un autre modèle.

    Args:
        config: Configuration chargée depuis le YAML.
        model_name: Nom de modèle pour l'étape 1 (unique étape en end_to_end,
            transcription en two_stage). None = ne pas surcharger.
        model_stage2_name: Nom de modèle pour l'étape 2 (codage textuel,
            two_stage uniquement). None = ne pas surcharger.

    Returns:
        Une nouvelle config avec les noms de modèle et le `name` mis à jour.

    Raises:
        ValueError: Si `model_stage2_name` est fourni alors que la config n'a
            pas de bloc `model_stage2` (approche `end_to_end`).
    """
    if model_name is None and model_stage2_name is None:
        return config

    current_stage2 = config.model_stage2
    if model_stage2_name is not None and current_stage2 is None:
        raise ValueError(
            "--model-stage2-name n'a de sens qu'en approche two_stage "
            "(la config chargée n'a pas de bloc `model_stage2`)."
        )

    updates: dict[str, object] = {}
    suffix_parts = []

    if model_name is not None:
        updates["model"] = config.model.model_copy(update={"name": model_name})
        suffix_parts.append(_slugify_model_name(model_name))

    if model_stage2_name is not None and current_stage2 is not None:
        updates["model_stage2"] = current_stage2.model_copy(update={"name": model_stage2_name})
        stage2_slug = _slugify_model_name(model_stage2_name)
        if stage2_slug not in suffix_parts:
            suffix_parts.append(stage2_slug)

    updates["name"] = f"{config.name}_{'_'.join(suffix_parts)}"
    return config.model_copy(update=updates)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration pour le pipeline HTR (transcription seule, corpus Scoledit)
# ─────────────────────────────────────────────────────────────────────────────


class HTRDataConfig(BaseModel):
    """Localisation des données Scoledit pour l'évaluation HTR."""

    scans_path: str  # local ou s3://.../scans/CE1/
    annotations_path: str  # transcriptions JSON de référence, local ou s3://.../annotation/CE1/
    limit: int | None = None  # limiter le nombre d'échantillons (tests rapides)


class HTRExperimentConfig(BaseModel):
    """Configuration d'une expérience HTR : données et métriques (CER/WER) spécifiques."""

    name: str = Field(..., description="Nom unique du run.")
    seed: int = 42
    # Nombre d'échantillons transcrits EN PARALLÈLE (cf. `concurrency` du scoring).
    concurrency: int = Field(default=8, ge=1)
    model: ModelConfig
    data: HTRDataConfig
    read_final_state: bool = True  # en cas de rature, lire l'état final


def load_htr_config(path: str | Path) -> HTRExperimentConfig:
    """Charge et valide une configuration HTR depuis un fichier YAML.

    Args:
        path: Chemin du fichier YAML de configuration HTR.

    Returns:
        La configuration d'expérience HTR validée.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return HTRExperimentConfig.model_validate(raw)
