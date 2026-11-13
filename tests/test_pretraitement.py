"""Tests de la préparation d'image (pages synthétiques, aucune donnée élève)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from evaluation_dictee.config import ExperimentConfig, ImageConfig, run_output_name
from evaluation_dictee.data.pretraitement import points_de_coupe, preparer, zone_ecriture

H, W = 2044, 1594
CADRE = (112, 257, 1300, 1800)  # gauche, haut, droite, bas (traits du cadre)


def _page(lignes: list[tuple[int, int]] | None = None, cadre: bool = True) -> Image.Image:
    """Page blanche, cadre pré-imprimé et « lignes d'écriture » (blocs d'encre partiels)."""
    a = np.full((H, W), 255, dtype=np.uint8)
    a[40:60, 300:900:4] = 0  # en-tête pré-imprimé (texte : pas de suite continue)
    if cadre:
        g, h, d, b = CADRE
        a[h : h + 2, g : d + 1] = 0
        a[b : b + 2, g : d + 1] = 0
        a[h : b + 2, g : g + 2] = 0
        a[h : b + 2, d : d + 2] = 0
    for y0, y1 in lignes or []:
        a[y0:y1, 150:1200:3] = 0  # encre clairsemée : pas un trait plein
    return Image.fromarray(a, mode="L")


def test_zone_ecriture_interieur_du_cadre() -> None:
    g, h, d, b = zone_ecriture(_page())
    assert CADRE[0] < g < CADRE[0] + 10 and CADRE[1] < h < CADRE[1] + 10
    assert CADRE[2] - 10 < d <= CADRE[2] and CADRE[3] - 10 < b <= CADRE[3]


def test_zone_ecriture_trait_incline() -> None:
    a = np.asarray(_page()).copy()
    g, h, d, _ = CADRE
    a[h : h + 2, g : d + 1] = 255
    for x in range(g, d + 1):  # trait du haut incliné de 2 px sur la largeur
        y = h + (2 * (x - g)) // (d - g)
        a[y, x] = 0
    assert zone_ecriture(Image.fromarray(a, mode="L"))[1] < h + 10


def test_zone_ecriture_sans_cadre_garde_la_page() -> None:
    assert zone_ecriture(_page(cadre=False)) == (0, 0, W, H)


def test_coupes_dans_les_interlignes() -> None:
    lignes = [(y, y + 40) for y in range(300, 1760, 70)]  # interlignes de 30 px
    page = _page(lignes)
    coupes = points_de_coupe(page, 4)
    assert len(coupes) == 3 and coupes == sorted(coupes)
    for c in coupes:
        assert not any(y0 <= c < y1 for y0, y1 in lignes), c


def test_preparer_defaut_renvoie_la_page() -> None:
    page = _page()
    assert preparer(page, ImageConfig()) == [page]


def test_preparer_bandes_sans_perte_ni_chevauchement() -> None:
    page = _page([(y, y + 40) for y in range(300, 1760, 70)])
    opts = ImageConfig(recadrage=True, n_bandes=3)
    bandes = preparer(page, opts)
    g, h, d, b = zone_ecriture(page)
    assert len(bandes) == 3
    assert sum(x.height for x in bandes) == b - h
    assert all(x.width == d - g for x in bandes)


def test_preparer_agrandissement() -> None:
    bandes = preparer(_page(), ImageConfig(agrandissement=2.0))
    assert bandes[0].size == (2 * W, 2 * H)


def test_slug_et_nom_de_run() -> None:
    assert ImageConfig().slug() == ""
    assert ImageConfig(recadrage=True, n_bandes=4, agrandissement=1.5).slug() == "crop-b4-x1p5"
    cfg = ExperimentConfig.model_validate(
        {
            "name": "essai",
            "approach": "two_stage",
            "model": {"name": "qwen3-8-27b", "image": {"recadrage": True}},
            "data": {"images_path": "x", "labels_path": "y"},
        }
    )
    assert run_output_name(cfg) == "essai_qwen3-8-27b-crop"


def test_coupes_decalees() -> None:
    lignes = [(y, y + 40) for y in range(300, 1760, 70)]
    page = _page(lignes)
    normales = points_de_coupe(page, 4)
    decalees = points_de_coupe(page, 4, decalage=0.5)
    assert len(decalees) == 4 and decalees == sorted(decalees)
    # Chaque coupe normale tombe entre deux coupes décalées.
    for c, (a, b) in zip(normales, zip(decalees, decalees[1:], strict=False), strict=True):
        assert a < c < b
    for c in decalees:
        assert not any(y0 <= c < y1 for y0, y1 in lignes), c
    assert ImageConfig(recadrage=True, n_bandes=4, decalage_bandes=0.5).slug() == "crop-b4-d0p5"


def test_epaississement() -> None:
    a = np.full((50, 50), 255, dtype=np.uint8)
    a[25, 25] = 0  # un point d'un pixel
    (sortie,) = preparer(Image.fromarray(a, mode="L"), ImageConfig(epaississement=1))
    encre = np.asarray(sortie) < 128
    assert encre.sum() == 9 and encre[24:27, 24:27].all()
    assert ImageConfig(recadrage=True, n_bandes=4, epaississement=1).slug() == "crop-b4-e1"
