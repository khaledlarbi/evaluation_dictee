"""Tests de la combinaison de deux runs."""

import json

from evaluation_dictee.evaluation.ensemble import charger_codes, combiner, decider


def test_decider() -> None:
    assert decider("9", "9", "run_b") == "9"
    assert decider("1", "9", "run_a") == "1"
    assert decider("1", "9", "run_b") == "9"
    assert decider("1", "9", "erreur_prioritaire") == "9"
    assert decider("0", "1", "erreur_prioritaire") == "0"
    assert decider("9", "1", "correct_prioritaire") == "1"


def test_charger_et_combiner(tmp_path) -> None:
    def ecrire(nom: str, lignes: list[dict]) -> object:
        p = tmp_path / nom
        p.write_text("\n".join(json.dumps(x) for x in lignes), encoding="utf-8")
        return p

    a = ecrire(
        "a.jsonl",
        [
            {"copy_id": "c", "item_id": "i1", "y_true": "1", "y_pred": "1"},
            {"copy_id": "c", "item_id": "i2", "y_true": "9", "y_pred": "1"},
            {"copy_id": "c", "item_id": "i3", "y_true": "1", "y_pred": "1", "blank": True},
        ],
    )
    b = ecrire(
        "b.jsonl",
        [
            {"copy_id": "c", "item_id": "i1", "y_true": "1", "y_pred": "1"},
            {"copy_id": "c", "item_id": "i2", "y_true": "9", "y_pred": "9"},
        ],
    )
    c = combiner(charger_codes(a), charger_codes(b))
    assert c.cles == [("c", "i1"), ("c", "i2")]  # i3 exclu (copie vierge)
    assert c.concordants == [True, False]
    assert c.predire("erreur_prioritaire") == ["1", "9"]
