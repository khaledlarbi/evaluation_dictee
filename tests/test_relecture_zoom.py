"""Tests de la relecture en zoom (fonctions pures, aucune donnée élève)."""

from __future__ import annotations

from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.pipeline.alignement_local import aligner, tokeniser
from evaluation_dictee.pipeline.relecture_zoom import (
    HAUTEUR_MIN,
    ItemARelire,
    boite_bande,
    contexte,
    ligne_item,
    n_lignes,
    prompt_relecture,
)


def _items(mots: list[str]) -> list[GridItem]:
    return [
        GridItem(f"i{k}", m, "ponctuation" if m in ".," else "mot", [], [], [])
        for k, m in enumerate(mots, 1)
    ]


ITEMS = _items(["Le", "chat", "dort", ".", "Il", "rêve", "de", "souris", "."])


def test_ligne_item_et_absent() -> None:
    al = aligner(tokeniser("Le chat dort.\\nIl rève\\nsouris."), ITEMS)
    assert n_lignes(al) == 3
    assert ligne_item(al, ITEMS, "i2") == 1
    assert ligne_item(al, ITEMS, "i6") == 2
    # « de » absent : ligne de l'item lu le plus proche, avant d'abord.
    assert ligne_item(al, ITEMS, "i7") == 2


def test_boite_bande_centree_et_bornee() -> None:
    g, h, d, b = boite_bande(1185, 1540, ligne=5, nb_lignes=10)
    assert (g, d) == (0, 1185)
    assert h < 0.45 * 1540 < b  # le centre de la ligne 5 est à 0,45 de la hauteur
    assert b - h >= HAUTEUR_MIN
    assert boite_bande(1185, 1540, ligne=1, nb_lignes=10)[1] == 0
    assert boite_bande(1185, 1540, ligne=10, nb_lignes=10)[3] == 1540


def test_contexte_et_prompt() -> None:
    assert contexte(ITEMS, "i6") == ("dort . Il", "de souris .")
    a = ItemARelire(ITEMS[5], ["rêve", "rève"], ligne=2, n_lignes=3)
    p = prompt_relecture(ITEMS, a)
    assert "« dort . Il ___ de souris . »" in p and "« rève »" in p
    # Le mot attendu n'apparaît que parmi les pistes des lecteurs, pas dans le passage.
    assert p.count("rêve") == 1
    ponct = prompt_relecture(ITEMS, ItemARelire(ITEMS[3], ["."], 1, 3))
    assert "ponctuation" in ponct


def test_lignes_ecriture_et_bande_sur_bloc() -> None:
    import numpy as np
    from PIL import Image

    from evaluation_dictee.pipeline.relecture_zoom import lignes_ecriture

    a = np.full((1540, 1185), 255, dtype=np.uint8)
    for y in range(100, 900, 100):  # 8 lignes écrites dans le haut seulement
        a[y : y + 40, 50:1100:3] = 0
    a[1300:1302, :] = 0  # trait réglé plein : ignoré
    blocs = lignes_ecriture(Image.fromarray(a, mode="L"))
    assert len(blocs) == 8
    _, h, _, b = boite_bande(1185, 1540, ligne=8, nb_lignes=8, blocs=blocs)
    assert h < 820 < b < 1100  # la dernière ligne, pas le bas du cadre
