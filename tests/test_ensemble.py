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


def test_arbitrage_par_troisieme_lecture() -> None:
    from evaluation_dictee.evaluation.ensemble import Combinaison

    c = Combinaison(
        cles=[("c", f"i{k}") for k in range(4)],
        y_true=["1", "9", "9", "0"],
        pred_a=["1", "1", "9", "0"],
        pred_b=["1", "9", "1", "9"],
        pred_c=["9", "9", "9", "1"],
    )
    # i0 : A = B → C ignoré ; i1, i2 : C départage ; i3 : tous différents → indécis.
    assert c.arbitrer() == ["1", "9", "9", None]


def test_vote_majoritaire() -> None:
    from evaluation_dictee.evaluation.ensemble import vote_majoritaire

    assert vote_majoritaire(["1", "1", "1"]) == ("1", 1.0, 0)
    assert vote_majoritaire(["1", "9", "9"]) == ("9", 2 / 3, 1)
    # Trois codes différents : le lecteur principal l'emporte, confiance 1/3.
    assert vote_majoritaire(["0", "9", "1"]) == ("0", 1 / 3, 0)
    # Un lecteur en échec ne vote pas mais compte dans la confiance.
    assert vote_majoritaire([None, "9", "9"]) == ("9", 2 / 3, 1)
    assert vote_majoritaire([None, None]) == (None, 0.0, None)
