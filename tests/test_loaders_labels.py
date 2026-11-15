"""Lecture des CSV de codes experts (formats 2015 et 2021) et tirage d'échantillon."""

from __future__ import annotations

from pathlib import Path

from evaluation_dictee.data import loaders
from evaluation_dictee.data.loaders import load_dataset, load_labels


def test_format_large_2015(tmp_path: Path) -> None:
    p = tmp_path / "l.csv"
    p.write_text('"";"dictee";"I1";"I2"\n"1";"a.png";" 1";" 4"\n', encoding="utf-8")
    assert load_labels(str(p)) == {"a.png": {"I1": "1", "I2": "4"}}


def test_format_long_2021(tmp_path: Path) -> None:
    p = tmp_path / "l.csv"
    p.write_text(
        "dictee;num_item;corrige\nb.png;I2;0\nb.png;I1;1\na.png;I1;3\na.png;I2;1\n",
        encoding="utf-8",
    )
    lab = load_labels(str(p))
    assert lab == {"a.png": {"I1": "3", "I2": "1"}, "b.png": {"I1": "1", "I2": "0"}}
    assert list(lab["b.png"]) == ["I1", "I2"]  # items rangés par identifiant


def _corpus(tmp_path: Path, n: int, monkeypatch) -> str:
    p = tmp_path / "l.csv"
    lignes = ["dictee;num_item;corrige"] + [f"c{i:03d}.png;I1;1" for i in range(n)]
    p.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    monkeypatch.setattr(loaders, "_exists", lambda _p: True)
    return str(p)


def test_limit_sans_graine_prend_les_premieres(tmp_path: Path, monkeypatch) -> None:
    csv_path = _corpus(tmp_path, 50, monkeypatch)
    ids = [c.copy_id for c in load_dataset("imgs", csv_path, limit=5)]
    assert ids == [f"c{i:03d}.png" for i in range(5)]


def test_echantillon_aleatoire_reproductible_et_emboite(tmp_path: Path, monkeypatch) -> None:
    csv_path = _corpus(tmp_path, 200, monkeypatch)
    a = [c.copy_id for c in load_dataset("imgs", csv_path, limit=20, sample_seed=42)]
    b = [c.copy_id for c in load_dataset("imgs", csv_path, limit=20, sample_seed=42)]
    grand = [c.copy_id for c in load_dataset("imgs", csv_path, limit=40, sample_seed=42)]
    assert a == b == sorted(a)
    assert a != [f"c{i:03d}.png" for i in range(20)]
    # Agrandir l'échantillon garde les copies déjà faites (reprise du checkpoint).
    assert set(a) <= set(grand)
