"""Formes acceptées de la grille (« À » pour « A ») : remplacées avant le codage."""

from __future__ import annotations

from evaluation_dictee.data.grid import GridItem, appliquer_formes_acceptees, load_grid


def _item(item_id: str, attendu: str, acceptees: list[str] | None = None) -> GridItem:
    return GridItem(item_id, attendu, "mot", [], [], [], acceptees or [])


def test_forme_acceptee_ramenee_a_l_attendue() -> None:
    items = {"i73": _item("i73", "A", ["À"]), "i74": _item("i74", "ce")}
    ecrits = {"i73": "À", "i74": "se"}
    assert appliquer_formes_acceptees(ecrits, items) == {"i73": "A", "i74": "se"}
    assert ecrits["i73"] == "À"  # l'entrée n'est pas modifiée


def test_autres_formes_inchangees() -> None:
    items = {"i73": _item("i73", "A", ["À"])}
    for lu in ("à", "a", "Et", ""):
        assert appliquer_formes_acceptees({"i73": lu}, items) == {"i73": lu}


def test_grille_par_defaut_sans_forme_acceptee() -> None:
    g = load_grid("configs/grille_dictee_2015.json")
    assert all(not it.formes_acceptees for it in g.items)


def test_variante_a_accentue() -> None:
    base = load_grid("configs/grille_dictee_2015.json")
    var = load_grid("configs/grille_dictee_2015_A_accentue.json")
    diff = [(a, b) for a, b in zip(base.items, var.items, strict=True) if a != b]
    assert len(diff) == 1
    assert diff[0][1].attendu == "A" and diff[0][1].formes_acceptees == ["À"]
