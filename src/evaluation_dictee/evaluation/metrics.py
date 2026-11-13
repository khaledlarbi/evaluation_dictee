"""Métriques de scoring : accord brut, kappa de Cohen, matrice de confusion.

Voir `.claude/rules/methodologie.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sklearn.metrics import cohen_kappa_score, confusion_matrix

from evaluation_dictee.data import reference


@dataclass
class ScoringMetrics:
    """Métriques d'accord entre prédictions et codes experts."""

    n_items: int
    raw_agreement: float
    cohen_kappa: float
    labels: list[str]
    confusion: list[list[int]]


def compute_scoring_metrics(y_true: list[str], y_pred: list[str]) -> ScoringMetrics:
    """Calcule les métriques d'accord entre codes experts et codes prédits.

    Args:
        y_true: codes de l'annotateur expert.
        y_pred: codes prédits par le modèle, alignés item par item sur y_true.

    Returns:
        Les métriques agrégées (effectif, accord brut, kappa de Cohen, labels
        et matrice de confusion).

    Raises:
        ValueError: si les deux listes n'ont pas la même longueur ou sont vides.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true et y_pred doivent avoir la même longueur.")
    if not y_true:
        raise ValueError("Listes vides : aucune métrique calculable.")

    n = len(y_true)
    raw = sum(1 for a, b in zip(y_true, y_pred, strict=True) if a == b) / n
    kappa = float(cohen_kappa_score(y_true, y_pred))

    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return ScoringMetrics(
        n_items=n,
        raw_agreement=raw,
        cohen_kappa=kappa,
        labels=labels,
        confusion=matrix,
    )


def compute_diagnostic_metrics(
    y_true: list[str],
    y_pred: list[str],
    item_ids: list[str],
    item_types: dict[str, str],
    transcriptions: list[str | None] | None = None,
    attendus: dict[str, str] | None = None,
) -> dict[str, float]:
    """Métriques de diagnostic d'un run, au-delà de l'accord et du kappa.

    Ces quatre chiffres sont ceux qui ont permis de comprendre POURQUOI un bras
    d'expérience fonctionne ou non, là où l'accord et le kappa disent seulement s'il
    fonctionne. Les enregistrer à chaque run évite d'avoir à rejouer l'analyse.

    - `taux_sous_detection` : part des fautes réelles que le modèle code « correct ».
      C'est l'erreur dominante, autour de 45 % sur tous les modèles testés.
    - `biais_taux_faute_pts` : écart, en points, entre le taux de faute attribué aux
      élèves par le modèle et celui mesuré par l'expert. Négatif = sous-détection.
    - `kappa_mots` et `kappa_ponctuation` : mot et ponctuation sont deux tâches
      distinctes (l'un se rate par faute, l'autre par omission) qu'un kappa global
      confond ; l'écart entre les deux atteint 0,23 sur certains modèles.

    Args:
        y_true: codes experts, déjà filtrés des items inévaluables.
        y_pred: codes prédits, alignés sur `y_true`.
        item_ids: identifiant d'item de chaque décision.
        item_types: nature de chaque item ("mot" ou "ponctuation"), par identifiant.
        transcriptions: transcription retenue par item (alignée sur `y_true`) ; si
            fournie avec `attendus`, ajoute la décomposition lecture/codage.
        attendus: mot attendu par identifiant d'item.

    Returns:
        Les métriques de diagnostic. Une métrique indéfinie (strate vide ou kappa non
        calculable) est omise plutôt que renvoyée à zéro, qui se lirait comme une
        mesure.
    """
    if not y_true:
        return {}

    fautes = [i for i, code in enumerate(y_true) if code == reference.SIMPLE_ERREUR]
    out: dict[str, float] = {}
    if fautes:
        manquees = sum(1 for i in fautes if y_pred[i] == reference.SIMPLE_CORRECT)
        out["taux_sous_detection"] = manquees / len(fautes)

    taux_vrai = len(fautes) / len(y_true)
    taux_pred = sum(1 for code in y_pred if code == reference.SIMPLE_ERREUR) / len(y_pred)
    out["biais_taux_faute_pts"] = (taux_pred - taux_vrai) * 100

    for nature, suffixe in [("mot", "mots"), ("ponctuation", "ponctuation")]:
        idx = [i for i, item in enumerate(item_ids) if item_types.get(item) == nature]
        if not idx:
            continue
        kappa = compute_scoring_metrics(
            [y_true[i] for i in idx], [y_pred[i] for i in idx]
        ).cohen_kappa
        if kappa is not None and not math.isnan(kappa):
            out[f"kappa_{suffixe}"] = kappa
    if transcriptions is not None and attendus is not None:
        out.update(decompose_lecture_codage(y_true, y_pred, item_ids, transcriptions, attendus))
    return out


def code_coherent(ecrit: str, attendu: str) -> str:
    """Code simplifié qu'implique la transcription seule : identique, vide ou différente.

    Sert de RÉFÉRENCE de diagnostic, pas de codage : elle dit ce qu'un codeur
    parfaitement cohérent avec la lecture aurait rendu.
    """
    if not ecrit.strip():
        return reference.SIMPLE_ABSENT
    return reference.SIMPLE_CORRECT if ecrit == attendu else reference.SIMPLE_ERREUR


def decompose_lecture_codage(
    y_true: list[str],
    y_pred: list[str],
    item_ids: list[str],
    transcriptions: list[str | None],
    attendus: dict[str, str],
) -> dict[str, float]:
    """Sépare les désaccords modèle/expert dus au CODAGE de ceux dus à la LECTURE.

    Pour chaque item, on compare le code prédit au code qu'implique la transcription
    (`code_coherent`) :
    - différents → erreur de CODAGE (le modèle code autre chose que ce qu'il a lu) ;
    - identiques mais en désaccord avec l'expert → erreur de LECTURE (la transcription
      ne reflète pas la copie, ou la grille tolère une forme que la lecture distingue).

    Pertinent seulement quand la transcription par item vient d'un alignement explicite
    (option `transcription_verification`) ; les items sans transcription sont ignorés.

    Args:
        y_true: codes experts (grille simplifiée).
        y_pred: codes prédits, alignés sur `y_true`.
        item_ids: identifiant d'item de chaque décision.
        transcriptions: transcription retenue par item, None si inconnue.
        attendus: mot attendu par identifiant d'item.

    Returns:
        `part_desaccords_codage` et `part_desaccords_lecture` (sommant à 1, sur les
        désaccords), `taux_incoherence_codage` (sur tous les items) et les effectifs.
        Vide si aucun item n'a de transcription.
    """
    n_items = n_desaccords = n_codage = n_lecture = n_incoherents = 0
    for vrai, pred, item_id, ecrit in zip(y_true, y_pred, item_ids, transcriptions, strict=True):
        if ecrit is None or item_id not in attendus:
            continue
        n_items += 1
        coherent = code_coherent(ecrit, attendus[item_id])
        incoherent = pred != coherent
        n_incoherents += incoherent
        if pred != vrai:
            n_desaccords += 1
            if incoherent:
                n_codage += 1
            else:
                n_lecture += 1
    if n_items == 0:
        return {}
    out: dict[str, float] = {
        "taux_incoherence_codage": n_incoherents / n_items,
        "n_desaccords": float(n_desaccords),
    }
    if n_desaccords:
        out["part_desaccords_codage"] = n_codage / n_desaccords
        out["part_desaccords_lecture"] = n_lecture / n_desaccords
    return out
