"""Tests de la transcription épelée (reconstruction, questions fermées, réponses)."""

from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.pipeline.transcription_epelee import (
    formater_mots_lus,
    lignes_depuis_reponse,
    lignes_vers_texte,
    mots_lus,
    valider_alignement,
)


def _items(mots: list[str]) -> list[GridItem]:
    return [
        GridItem(f"i{k}", m, "ponctuation" if m in {".", ","} else "mot", [], [], [])
        for k, m in enumerate(mots)
    ]


def test_reconstruction_ne_corrige_pas() -> None:
    lignes = lignes_depuis_reponse(
        {"lignes": [["L e s", "e n f a n t", "p a r t i t", "a", "l ' e c o l e", "."]]}
    )
    assert lignes is not None
    assert lignes_vers_texte(lignes, final=True, garder_marques=False) == (
        "Les enfant partit a l'ecole."
    )


def test_ratures_resolues_vers_etat_final() -> None:
    lignes = lignes_depuis_reponse(
        {"lignes": [["~~M a m a n~~", "p e r d {a→u} [?]", "[ajout: i l s]", "[illisible]"]]}
    )
    assert lignes is not None
    assert lignes_vers_texte(lignes, final=True, garder_marques=False) == "perdu ils [illisible]"
    assert lignes_vers_texte(lignes, final=False, garder_marques=True) == (
        "~~Maman~~ perd{a→u}[?] [ajout: ils] [illisible]"
    )


def test_mots_lus_decoupe_elisions_et_resout_ratures() -> None:
    lignes = [["Ils", "n'étaient", "~~pa~~", "pas"], ["perd{a→u}[?]", "."]]
    lus = mots_lus(lignes)
    assert [(m.numero, m.ligne, m.texte) for m in lus] == [
        (1, 1, "Ils"),
        (2, 1, "n'"),
        (3, 1, "étaient"),
        (4, 1, "pas"),
        (5, 2, "perdu"),
        (6, 2, "."),
    ]
    assert "[ligne 2 de la copie]" in formater_mots_lus(lus)


def _data(items: list[GridItem], lus: list[list[int]]) -> dict:
    return {
        "alignement": [{"item_id": it.item_id, "lus": n} for it, n in zip(items, lus, strict=True)]
    }


def test_alignement_trou_fusion_coupure() -> None:
    items = _items(["nous", "les", "avons", "retrouvés", "."])
    lus = mots_lus([["nousles", "re", "trouvé", "."]])
    al = valider_alignement(_data(items, [[1], [1], [], [2, 3], [4]]), items, lus)
    assert al is not None
    assert [al.ecrit(it.item_id, lus) for it in items] == [
        "nousles",
        "nousles",
        "",
        "re trouvé",
        ".",
    ]
    assert al.n_absents() == 1 and al.n_reparations == 0 and al.mots_en_trop == []


def test_alignement_croisement_ramene_a_absent() -> None:
    items = _items(["Le", "soir", "tombait"])
    lus = mots_lus([["Le", "soir", "tombait", "vite"]])
    # « tombait » pointe en arrière (croisement) : ramené à absent, la suite n'en
    # pâtit pas ; « vite » n'est associé à rien.
    al = valider_alignement(_data(items, [[1], [2], [1]]), items, lus)
    assert al is not None
    assert al.lus_par_item == {"i0": [1], "i1": [2], "i2": []}
    assert al.n_reparations == 1
    assert al.mots_en_trop == [3, 4]


def test_alignement_incomplet_rejete() -> None:
    items = _items(["a", "b", "c", "d", "e"])
    lus = mots_lus([["a", "b"]])
    assert valider_alignement(_data(items[:2], [[1], [2]]), items, lus) is None
