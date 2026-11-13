"""Combinaison de deux runs (deux lectures d'une même copie) item par item.

Deux modèles qui se trompent en sens inverse (l'un rate des fautes, l'autre en
invente) se complètent : là où ils concordent, le code est fiable ; là où ils
divergent, l'item est incertain et peut être renvoyé à un humain. L'alignement local
des transcriptions sur les items rend la comparaison item à item directe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from evaluation_dictee.data import reference

# Politiques de décision en cas de désaccord entre les deux runs.
POLITIQUES = ("run_a", "run_b", "erreur_prioritaire", "correct_prioritaire")


def charger_codes(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """Codes (expert, prédit) par (copy_id, item_id) d'un JSONL de prédictions.

    Même filtrage que les métriques du run : dernière occurrence d'un doublon, items
    exclus (copie vierge, code expert illisible) écartés.

    Args:
        path: JSONL de prédictions d'un run.

    Returns:
        Pour chaque (copy_id, item_id) évaluable : (code expert, code prédit).
    """
    par_cle: dict[tuple[str, str], dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            par_cle[(rec["copy_id"], rec["item_id"])] = rec
    out = {}
    for cle, rec in par_cle.items():
        if rec.get("exclusion") is not None or rec.get("blank"):
            continue
        if not reference.est_evaluable(str(rec["y_true"])):
            continue
        out[cle] = (str(rec["y_true"]), str(rec["y_pred"]))
    return out


def decider(code_a: str, code_b: str, politique: str) -> str:
    """Code retenu pour un item selon la politique (identique si les deux concordent).

    Args:
        code_a: code du run A.
        code_b: code du run B.
        politique: une valeur de `POLITIQUES`.

    Returns:
        Le code combiné.
    """
    if code_a == code_b or politique == "run_a":
        return code_a
    if politique == "run_b":
        return code_b
    correct = reference.SIMPLE_CORRECT
    if politique == "erreur_prioritaire":
        # L'un dit « correct », l'autre non → on retient le non-correct.
        return code_b if code_a == correct else code_a
    if politique == "correct_prioritaire":
        return correct if correct in (code_a, code_b) else code_a
    raise ValueError(f"Politique inconnue : {politique}")


@dataclass
class Combinaison:
    """Items communs aux deux runs, alignés : codes expert, A, B."""

    cles: list[tuple[str, str]]
    y_true: list[str]
    pred_a: list[str]
    pred_b: list[str]
    # Troisième lecture optionnelle, consultée seulement là où A et B divergent.
    pred_c: list[str] | None = None

    @property
    def concordants(self) -> list[bool]:
        """Vrai pour chaque item où les deux runs rendent le même code."""
        return [a == b for a, b in zip(self.pred_a, self.pred_b, strict=True)]

    def predire(self, politique: str) -> list[str]:
        """Codes combinés de tous les items selon la politique."""
        return [decider(a, b, politique) for a, b in zip(self.pred_a, self.pred_b, strict=True)]

    def arbitrer(self) -> list[str | None]:
        """Vote à trois : code majoritaire, ou None si A, B et C sont tous différents.

        Là où A et B concordent, C n'est pas consulté (il ne peut que confirmer ou
        être mis en minorité). Là où ils divergent, C départage s'il rejoint l'un
        des deux ; sinon l'item reste indécidé (à renvoyer à un humain).

        Returns:
            Le code retenu par item, None pour les items indécidés.

        Raises:
            ValueError: si aucune troisième lecture n'est disponible.
        """
        if self.pred_c is None:
            raise ValueError("Arbitrage impossible sans troisième lecture (pred_c).")
        out: list[str | None] = []
        for a, b, c in zip(self.pred_a, self.pred_b, self.pred_c, strict=True):
            out.append(a if a == b or c == a else b if c == b else None)
        return out


def combiner(
    codes_a: dict[tuple[str, str], tuple[str, str]],
    codes_b: dict[tuple[str, str], tuple[str, str]],
    codes_c: dict[tuple[str, str], tuple[str, str]] | None = None,
) -> Combinaison:
    """Aligne deux (ou trois) runs sur leurs items communs (même copie, même item).

    Args:
        codes_a: sortie de `charger_codes` pour le run A.
        codes_b: sortie de `charger_codes` pour le run B.
        codes_c: troisième lecture optionnelle, qui sert d'arbitre.

    Returns:
        Les items communs à tous les runs fournis, triés, avec les codes expert et
        les prédictions.
    """
    communs = set(codes_a) & set(codes_b)
    if codes_c is not None:
        communs &= set(codes_c)
    cles = sorted(communs)
    return Combinaison(
        cles=cles,
        y_true=[codes_a[c][0] for c in cles],
        pred_a=[codes_a[c][1] for c in cles],
        pred_b=[codes_b[c][1] for c in cles],
        pred_c=[codes_c[c][1] for c in cles] if codes_c is not None else None,
    )
