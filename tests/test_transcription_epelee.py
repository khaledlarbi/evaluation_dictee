"""Tests de la transcription épelée (reconstruction, questions fermées, réponses)."""

from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.pipeline.transcription_epelee import (
    appliquer_reponses,
    construire_questions,
    formater_questions,
    lignes_depuis_reponse,
    lignes_vers_texte,
    variantes_courantes,
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


def test_variantes_courantes() -> None:
    assert set(variantes_courantes("perdus")) >= {"perdu", "perdue", "perdues"}
    assert set(variantes_courantes("arriver")) >= {"arrivé", "arrivez", "arrivés"}
    assert "a" in variantes_courantes("à")


def test_questions_contiennent_lecture_et_attendu_sans_ponctuation() -> None:
    items = _items(["Ils", "n'", "étaient", "perdus", "."])
    lignes = [["Ils", "n'etaient", "perdu", "."]]
    questions = construire_questions(lignes, items, graine="c1")
    lus = [q.lu for q in questions]
    assert lus == ["Ils", "n'etaient", "perdu"]  # pas de question sur « . »
    q_perdu = questions[2]
    assert {"perdu", "perdus"} <= set(q_perdu.options)
    assert "n'étaient" in questions[1].options
    texte = formater_questions(questions, lignes)
    assert "ligne 1, position 3 (entre « n'etaient » et « . »)" in texte


def test_melange_reproductible() -> None:
    items = _items(["perdus"])
    a = construire_questions([["perdu"]], items, graine="x")
    b = construire_questions([["perdu"]], items, graine="x")
    assert a[0].options == b[0].options


def test_appliquer_reponses_lettre_et_autre() -> None:
    items = _items(["Le", "soir"])
    lignes = [["Le", "soir"]]
    qs = construire_questions(lignes, items, graine="c")
    q_soir = next(q for q in qs if q.lu == "soir")
    reponses = [{"id": q_soir.id, "choix": "AUTRE", "epele": "s o i e r", "certain": False}]
    res = appliquer_reponses(lignes, qs, reponses)
    assert res.lignes == [["Le", "soier"]]
    assert res.n_corrections == 1 and res.n_incertains == 1
