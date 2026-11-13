"""Tests des métriques de scoring."""

import pytest

from evaluation_dictee.evaluation.metrics import compute_scoring_metrics


def test_accord_parfait() -> None:
    y = ["1", "9", "0", "1"]
    metrics = compute_scoring_metrics(y, y)
    assert metrics.raw_agreement == 1.0
    assert metrics.cohen_kappa == 1.0
    assert metrics.n_items == 4


def test_accord_partiel() -> None:
    y_true = ["1", "1", "9", "0"]
    y_pred = ["1", "9", "9", "0"]
    metrics = compute_scoring_metrics(y_true, y_pred)
    assert metrics.raw_agreement == 0.75


def test_longueurs_differentes_leve_erreur() -> None:
    with pytest.raises(ValueError):
        compute_scoring_metrics(["1"], ["1", "9"])


def test_listes_vides_levent_erreur() -> None:
    with pytest.raises(ValueError):
        compute_scoring_metrics([], [])


def test_decompose_lecture_codage() -> None:
    from evaluation_dictee.evaluation.metrics import decompose_lecture_codage

    attendus = {"a": "soir", "b": "Le", "c": "tombait", "d": "."}
    # a : lu « soire », codé 1 (le modèle ignore ce qu'il a lu) → CODAGE
    # b : lu « Le », codé 1, l'expert dit 9 → LECTURE
    # c : lu « tombait », codé 1, expert 1 → accord
    # d : lu "", codé 9, expert 0 → CODAGE (vide implique 0)
    out = decompose_lecture_codage(
        y_true=["9", "9", "1", "0"],
        y_pred=["1", "1", "1", "9"],
        item_ids=["a", "b", "c", "d"],
        transcriptions=["soire", "Le", "tombait", ""],
        attendus=attendus,
    )
    assert out["n_desaccords"] == 3
    assert out["part_desaccords_codage"] == 2 / 3
    assert out["part_desaccords_lecture"] == 1 / 3
    assert out["taux_incoherence_codage"] == 2 / 4
    assert decompose_lecture_codage(["1"], ["1"], ["a"], [None], attendus) == {}
