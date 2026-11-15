"""Chargement de la grille de codage de la dictée (JSON versionné dans configs/).

Le texte de référence est désormais dérivé de la grille.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class GridItem:
    """Un item de la grille : mot attendu, type et fautes connues."""

    item_id: str
    attendu: str
    type: str  # "mot" ou "ponctuation"
    ex_lexicale: list[str]
    ex_grammaticale: list[str]
    ex_les_deux: list[str]
    # Formes acceptées comme correctes en plus de l'attendue (ex. « À » pour « A »).
    # Remplacées par l'attendue avant le codage : décision de grille, pas du modèle.
    formes_acceptees: list[str] = field(default_factory=list)


@dataclass
class DictationGrid:
    """Grille complète de la dictée."""

    reference_text: str
    items: list[GridItem]


def load_grid(path: str) -> DictationGrid:
    """Charge la grille de codage depuis un fichier JSON.

    Args:
        path: Chemin du fichier JSON de la grille.

    Returns:
        La grille désérialisée : texte de référence et liste des items.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    items = [
        GridItem(
            item_id=it["id"],
            attendu=it["attendu"],
            type=it["type"],
            ex_lexicale=it.get("ex_lexicale", []),
            ex_grammaticale=it.get("ex_grammaticale", []),
            ex_les_deux=it.get("ex_les_deux", []),
            formes_acceptees=it.get("formes_acceptees", []),
        )
        for it in raw["items"]
    ]
    return DictationGrid(reference_text=raw["reference_text"], items=items)


def appliquer_formes_acceptees(
    ecrits: dict[str, str], items_by_id: dict[str, GridItem]
) -> dict[str, str]:
    """Remplace par la forme attendue tout mot écrit listé dans `formes_acceptees`.

    Args:
        ecrits: mot écrit par item (alignement local).
        items_by_id: items de la grille, par identifiant.

    Returns:
        Une copie de `ecrits` où les formes acceptées sont ramenées à l'attendue.
    """
    out = dict(ecrits)
    for item_id, ecrit in ecrits.items():
        it = items_by_id.get(item_id)
        if it is not None and ecrit.replace("’", "'") in it.formes_acceptees:
            out[item_id] = it.attendu
    return out
