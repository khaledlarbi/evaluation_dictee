"""Tests de l'alignement local (Needleman-Wunsch) des mots lus sur les items."""

from evaluation_dictee.data.grid import GridItem, load_grid
from evaluation_dictee.pipeline.alignement_local import aligner, tokeniser


def _items(mots: list[str]) -> list[GridItem]:
    return [
        GridItem(
            f"i{k}", m, "ponctuation" if not any(c.isalnum() for c in m) else "mot", [], [], []
        )
        for k, m in enumerate(mots)
    ]


def _ecrits(texte: str, items: list[GridItem]) -> list[str]:
    al = aligner(tokeniser(texte), items)
    return [al.ecrit(it.item_id) for it in items]


def test_tokeniser_elisions_ponctuation_lignes_marques() -> None:
    toks = tokeniser("Ils n'étaient pas[?] perdus,\\n- dit-il.")
    assert [t.texte for t in toks] == [
        "Ils", "n'", "étaient", "pas", "perdus", ",", "-", "dit-il", ".",
    ]  # fmt: skip
    assert [t.ligne for t in toks][-3:] == [2, 2, 2]


def test_reference_alignee_sur_elle_meme() -> None:
    grille = load_grid("configs/grille_dictee_2015.json")
    al = aligner(tokeniser(grille.reference_text), grille.items)
    assert all(al.ecrit(it.item_id) == it.attendu for it in grille.items)
    assert al.mots_en_trop == []


def test_mot_oublie_est_un_trou_isole() -> None:
    items = _items(["Papa", "et", "maman", ",", "inquiets", ","])
    assert _ecrits("Papa maman , inquiets ,", items) == ["Papa", "", "maman", ",", "inquiets", ","]


def test_fautes_restent_appariees() -> None:
    items = _items(["Le", "soir", "tombait", "."])
    assert _ecrits("le soire tombé", items) == ["le", "soire", "tombé", ""]


def test_fusion_coupure_et_elision_collee() -> None:
    items = _items(["nous", "les", "avons", "re", "Ils", "n'", "étaient", "retrouvés"])
    ecrits = _ecrits("nousles avons re Ils nétaient re trouvés", items)
    assert ecrits == [
        "nousles", "nousles", "avons", "re", "Ils", "nétaient", "nétaient", "re trouvés",
    ]  # fmt: skip


def test_elision_oubliee_nest_pas_une_fusion() -> None:
    items = _items(["Ils", "n'", "étaient"])
    assert _ecrits("ils étaient", items) == ["ils", "", "étaient"]


def test_ponctuation_jamais_appariee_a_un_mot() -> None:
    items = _items(["inquiets", ",", "se"])
    assert _ecrits("inquiets se", items) == ["inquiets", "", "se"]
    assert _ecrits("inquiets . se", items) == ["inquiets", ".", "se"]


def test_mots_ajoutes_et_copie_inachevee() -> None:
    items = _items(["Le", "soir", "tombait", "."])
    al = aligner(tokeniser("Le grand soir"), items)
    assert [al.ecrit(it.item_id) for it in items] == ["Le", "soir", "", ""]
    assert [al.mots[n - 1].texte for n in al.mots_en_trop] == ["grand"]
    assert al.n_absents() == 2
