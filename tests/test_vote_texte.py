"""Tests du vote entre lectures pour produire une transcription (textes inventés)."""

from __future__ import annotations

from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.transcription.vote_texte import voter_transcription


def _items(phrase: str) -> list[GridItem]:
    mots = phrase.replace(".", " .").replace(",", " ,").split()
    return [
        GridItem(f"i{k}", m, "ponctuation" if m in ".," else "mot", [], [], [])
        for k, m in enumerate(mots)
    ]


ITEMS = _items("Le chat dort sur le tapis.")


def test_unanime_texte_inchange() -> None:
    t = "Le chat dort\nsur le tapis."
    res = voter_transcription([t, t, t], ITEMS)
    assert res.texte == t
    assert res.compte()["unanimes"] == len(res.unites)


def test_majorite_corrige_la_base_en_gardant_les_lignes() -> None:
    base = "Le cha dort\nsur le tapis."
    autre = "Le chat dort sur le tapis."
    res = voter_transcription([base, autre, autre], ITEMS, base=0)
    assert res.texte == "Le chat dort\nsur le tapis."


def test_sans_majorite_garde_la_base() -> None:
    res = voter_transcription(
        ["Le chas dort sur le tapis.", "Le chat dort sur le tapis.", "Le chal dort sur le tapis."],
        ITEMS,
        base=0,
    )
    assert res.texte.startswith("Le chas ")
    assert res.compte()["sans_majorite"] == 1


def test_mot_oublie_par_la_base_est_insere() -> None:
    autre = "Le chat dort sur le tapis."
    res = voter_transcription(["Le chat sur le tapis.", autre, autre], ITEMS, base=0)
    assert res.texte == autre


def test_mot_en_trop_de_la_base_retire_si_majorite_absente() -> None:
    autre = "Le chat dort sur tapis."
    res = voter_transcription(["Le chat dort sur le tapis.", autre, autre], ITEMS, base=0)
    assert res.texte == autre


def test_ponctuation_inseree_collee() -> None:
    autre = "Le chat dort sur le tapis."
    res = voter_transcription(["Le chat dort sur le tapis", autre, autre], ITEMS, base=0)
    assert res.texte == autre


def test_lecteur_de_base_en_echec_repli() -> None:
    t = "Le chat dort sur le tapis."
    res = voter_transcription([None, t, t], ITEMS, base=0)
    assert res.base == 1
    assert res.texte == t


def test_aucun_lecteur() -> None:
    assert voter_transcription([None, "", None], ITEMS).texte == ""
