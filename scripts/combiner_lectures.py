"""Évalue la combinaison de deux runs item par item (aucun appel modèle).

N'affiche QUE des compteurs agrégés : aucune transcription ni donnée d'élève.

    uv run scripts/combiner_lectures.py \
        --run-a data/processed/<run A>_predictions.jsonl \
        --run-b data/processed/<run B>_predictions.jsonl \
        --grid configs/grille_dictee_2015.json \
        [--run-c data/processed/<run C>_predictions.jsonl]   # arbitre des désaccords
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation_dictee.data.grid import load_grid
from evaluation_dictee.evaluation.ensemble import (
    POLITIQUES,
    Combinaison,
    charger_codes,
    combiner,
)
from evaluation_dictee.evaluation.metrics import (
    compute_diagnostic_metrics,
    compute_scoring_metrics,
)


def _ligne(nom: str, y_true: list[str], y_pred: list[str], ids: list[str], types: dict) -> str:
    m = compute_scoring_metrics(y_true, y_pred)
    d = compute_diagnostic_metrics(y_true, y_pred, ids, types)
    return (
        f"{nom:<22} accord {m.raw_agreement * 100:5.1f} % | kappa {m.cohen_kappa:.3f} | "
        f"mots {d.get('kappa_mots', float('nan')):.3f} | "
        f"ponct {d.get('kappa_ponctuation', float('nan')):.3f} | "
        f"non détectées {d.get('taux_sous_detection', float('nan')) * 100:4.1f} % | "
        f"biais {d.get('biais_taux_faute_pts', float('nan')):+.1f} pt"
    )


def main() -> None:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--run-c", type=Path, help="Troisième lecture, arbitre des désaccords.")
    parser.add_argument("--grid", default="configs/grille_dictee_2015.json")
    args = parser.parse_args()

    types = {it.item_id: it.type for it in load_grid(args.grid).items}
    c = combiner(
        charger_codes(args.run_a),
        charger_codes(args.run_b),
        charger_codes(args.run_c) if args.run_c else None,
    )
    ids = [item_id for _, item_id in c.cles]
    n = len(c.cles)
    n_copies = len({copy_id for copy_id, _ in c.cles})
    print(f"Items communs : {n} sur {n_copies} copies\n")

    print("── Chaque run seul, puis combinaisons (désaccord tranché par la politique)")
    print(_ligne("A seul", c.y_true, c.pred_a, ids, types))
    print(_ligne("B seul", c.y_true, c.pred_b, ids, types))
    if c.pred_c is not None:
        print(_ligne("C seul", c.y_true, c.pred_c, ids, types))
    for pol in POLITIQUES[2:]:
        print(_ligne(pol, c.y_true, c.predire(pol), ids, types))

    conc = c.concordants
    idx_ok = [i for i in range(n) if conc[i]]
    idx_ko = [i for i in range(n) if not conc[i]]
    juste = [c.y_true[i] == c.pred_a[i] for i in range(n)]  # sur idx_ok, A == B

    print("\n── Concordance des deux runs")
    print(f"Items concordants : {len(idx_ok)} ({len(idx_ok) / n * 100:.1f} %)")
    if idx_ok:
        acc_ok = sum(juste[i] for i in idx_ok) / len(idx_ok)
        print(f"  accord avec l'expert quand A = B  : {acc_ok * 100:.1f} %")
    if idx_ko:
        for nom, pred in [("A", c.pred_a), ("B", c.pred_b)]:
            acc = sum(c.y_true[i] == pred[i] for i in idx_ko) / len(idx_ko)
            print(f"  accord avec l'expert quand A ≠ B ({nom}) : {acc * 100:.1f} %")

    # Renvoi humain : les items discordants vont à l'expert (supposé juste), le reste
    # garde le code concordant.
    if idx_ok:
        erreurs = sum(not juste[i] for i in idx_ok)
        print("\n── Si les items discordants sont renvoyés à un humain")
        print(
            f"Taux de renvoi : {len(idx_ko) / n * 100:.1f} % des items | erreur "
            f"résiduelle : {erreurs / n * 100:.1f} % des items "
            f"(accord {100 - erreurs / n * 100:.1f} %)"
        )

    if c.pred_c is not None:
        _arbitrage(c, ids, types)


def _arbitrage(c: Combinaison, ids: list[str], types: dict) -> None:
    """Résultats du vote à trois (C départage les désaccords entre A et B)."""
    n = len(c.cles)
    assert c.pred_c is not None
    votes = c.arbitrer()
    ko = [i for i in range(n) if c.pred_a[i] != c.pred_b[i]]
    tranches = [i for i in ko if votes[i] is not None]
    indecis = [i for i in ko if votes[i] is None]

    print("\n── Arbitrage par la troisième lecture (C), sur les items où A ≠ B")
    print(f"Items discordants A ≠ B : {len(ko)} ({len(ko) / n * 100:.1f} %)")
    if ko:
        print(
            f"  tranchés par C : {len(tranches)} ({len(tranches) / len(ko) * 100:.1f} % des "
            f"discordants) | restés indécis (A, B, C tous différents) : {len(indecis)}"
        )
    if tranches:
        acc = sum(c.y_true[i] == votes[i] for i in tranches) / len(tranches)
        cote_a = sum(votes[i] == c.pred_a[i] for i in tranches) / len(tranches)
        print(
            f"  accord avec l'expert sur les items tranchés : {acc * 100:.1f} % "
            f"(C rejoint A dans {cote_a * 100:.0f} % des cas)"
        )

    # Code automatique pour tous : vote majoritaire, A si indécis.
    auto = [v if v is not None else c.pred_a[i] for i, v in enumerate(votes)]
    print(_ligne("vote à 3 (A si indécis)", c.y_true, auto, ids, types))

    # Renvoi humain des seuls items indécis (l'humain est supposé juste).
    erreurs = sum(votes[i] is not None and votes[i] != c.y_true[i] for i in range(n))
    print("\n── Si seuls les items indécis sont renvoyés à un humain")
    print(
        f"Taux de renvoi : {len(indecis) / n * 100:.1f} % des items | erreur résiduelle : "
        f"{erreurs / n * 100:.1f} % des items (accord {100 - erreurs / n * 100:.1f} %)"
    )


if __name__ == "__main__":
    main()
