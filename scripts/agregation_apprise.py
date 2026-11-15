"""Agrégation apprise des trois lecteurs, comparée au vote majoritaire.

Pour chaque item de la dictée, un modèle donne la probabilité que l'item soit
correct (1), fautif (9) ou absent (0) à partir des codes des trois lecteurs et
d'autres variables :

- la copie (taux de non-unanimité, taux de 9 et de 0 vus par chaque lecteur) ;
- les items voisins (nombre de lecteurs qui y voient une faute ou une absence) ;
- la forme lue par chaque lecteur (option `--formes`) : classe de la lecture
  (identique, faute connue de la grille, casse, accent, apostrophe, vide, autre),
  distance à l'attendu, accord des lecteurs sur la forme, et formes les plus
  fréquentes de l'item (« m' » pour « n' »).

Deux modes :

- **validation croisée** (défaut) : par copie, sur un seul corpus ;
- **transposition** (`--cible`) : apprendre sur le corpus source et/ou sur quelques
  copies corrigées du corpus cible, évaluer sur le reste de la cible, et comparer
  l'évolution du taux de faute (réelle et estimée).

N'imprime que des agrégats : aucune transcription ni code d'élève n'est écrit.

Usage :
    python scripts/agregation_apprise.py --predictions <JSONL vote 2015> [--formes]
    python scripts/agregation_apprise.py --predictions <JSONL 2015> \
        --cible <JSONL vote 2021> --formes
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import pickle
import unicodedata
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold

from evaluation_dictee.data.grid import GridItem, load_grid
from evaluation_dictee.pipeline.alignement_local import aligner, tokeniser

CODES = ["1", "9", "0"]
LECTEURS = ["q27", "gem", "q35"]  # ordre de codes_lecteurs : model puis extra_readers
CLASSES = ["identique", "connue", "casse", "accent", "apostrophe", "vide", "autre", "na"]
N_FORMES_FREQ = 6  # formes fréquentes retenues par item (hors forme attendue)
MIN_FORMES_FREQ = 15  # occurrences minimales dans l'apprentissage


# ─── Chargement ──────────────────────────────────────────────────────────────
def _sans_accent(s: str) -> str:
    d = unicodedata.normalize("NFD", s)
    return "".join(c for c in d if unicodedata.category(c) != "Mn")


def classe_lecture(lu: str | None, it: GridItem) -> str:
    """Classe de la forme lue, comparée à la forme attendue et aux fautes de la grille."""
    if lu is None:
        return "na"
    lu = lu.replace("’", "'").strip()
    att = it.attendu
    if not lu:
        return "vide"
    if lu == att:
        return "identique"
    if lu in (*it.ex_lexicale, *it.ex_grammaticale, *it.ex_les_deux):
        return "connue"
    if lu.lower() == att.lower():
        return "casse"
    if _sans_accent(lu).lower() == _sans_accent(att).lower():
        return "accent"
    if lu.replace("'", "") == att.replace("'", ""):
        return "apostrophe"
    return "autre"


def distance(lu: str | None, att: str) -> float:
    """1 − ressemblance (difflib) entre la forme lue et l'attendue ; 1 si vide."""
    if not lu:
        return 1.0
    return 1 - difflib.SequenceMatcher(None, lu.lower(), att.lower()).ratio()


def _aligner(tache: tuple) -> tuple[tuple[str, int], dict[str, str]]:
    cid, j, t, its = tache
    return (cid, j), aligner(tokeniser(t), its).ecrits()


def charger(path: str, grid_path: str, formes: bool) -> pd.DataFrame:
    """Une ligne par item évaluable : codes des lecteurs, vote, code expert, formes lues."""
    grid = load_grid(grid_path)
    par_id = {it.item_id: it for it in grid.items}
    pos = {it.item_id: i for i, it in enumerate(grid.items)}
    par_cle: dict = {}
    trans: dict[str, list] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            par_cle[(r["copy_id"], r["item_id"])] = r
            if formes:
                trans[r["copy_id"]] = r.get("transcriptions_lecteurs") or [None] * 3

    # Formes lues : chaque transcription est réalignée sur les items de la copie
    # (≈ 0,16 s par alignement : en parallèle, puis mis en cache à côté du JSONL,
    # sur le SSP Cloud, dans data/ qui n'est jamais versionné).
    lues: dict[tuple[str, int], dict[str, str]] = {}
    if formes:
        cache = Path(path + ".formes_lues.pkl")
        st = os.stat(path)
        signature = (st.st_size, st.st_mtime_ns)
        if cache.exists():
            with open(cache, "rb") as fc:
                sig, lues = pickle.load(fc)
            if sig != signature:
                lues = {}
        if not lues:
            items_copie: dict[str, list[GridItem]] = {}
            for cid, iid in par_cle:
                if iid in par_id:
                    items_copie.setdefault(cid, []).append(par_id[iid])
            taches = []
            for cid, its in items_copie.items():
                its.sort(key=lambda it: pos[it.item_id])
                for j, t in enumerate(trans.get(cid, [])[:3]):
                    if t:
                        taches.append((cid, j, t, its))
            with Pool(processes=min(16, os.cpu_count() or 1)) as pool:
                for cle, ecrits in pool.imap_unordered(_aligner, taches, chunksize=20):
                    lues[cle] = ecrits
            with open(cache, "wb") as fc:
                pickle.dump((signature, lues), fc)

    rows = []
    for (cid, iid), r in par_cle.items():
        if r.get("exclusion") is not None or r.get("blank"):
            continue
        y = str(r["y_true"]).strip()
        if y not in CODES or iid not in pos:
            continue
        cl = r.get("codes_lecteurs") or [None, None, None]
        cl = [c if c in CODES else "NA" for c in cl]
        row = dict(
            copy=cid,
            item=iid,
            pos=pos[iid],
            type=par_id[iid].type,
            y=y,
            vote=str(r["y_pred"]),
            conf=r.get("confidence") or 0.0,
            **{f"c_{lec}": c for lec, c in zip(LECTEURS, cl, strict=True)},
        )
        if formes:
            for j, lec in enumerate(LECTEURS):
                ecrits = lues.get((cid, j))
                lu = None if ecrits is None or cl[j] == "NA" else ecrits.get(iid, "")
                row[f"f_{lec}"] = lu
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["copy", "pos"]).reset_index(drop=True)


def ajouter_variables(df: pd.DataFrame, formes: bool) -> pd.DataFrame:
    """Variables de la copie, des voisins et des formes (sans le code expert)."""
    c = df[[f"c_{lec}" for lec in LECTEURS]]
    df["unanime"] = (c.nunique(axis=1) == 1).astype(int)
    df["n9"] = (c == "9").sum(axis=1)
    df["n0"] = (c == "0").sum(axis=1)
    g = df.groupby("copy")
    df["copie_non_unan"] = 1 - g["unanime"].transform("mean")
    for lec in LECTEURS:
        df[f"copie_9_{lec}"] = g[f"c_{lec}"].transform(lambda s: (s == "9").mean())
        df[f"copie_0_{lec}"] = g[f"c_{lec}"].transform(lambda s: (s == "0").mean())
    for d, nom in ((1, "prec"), (-1, "suiv")):
        ok = (df["copy"].shift(d) == df["copy"]) & (df["pos"].shift(d) == df["pos"] - d)
        for k in ("n9", "n0"):
            df[f"{k}_{nom}"] = np.where(ok, df[k].shift(d), -1)
    if formes:
        for lec in LECTEURS:
            df[f"d_{lec}"] = [
                distance(lu, a) for lu, a in zip(df[f"f_{lec}"], df["attendu"], strict=True)
            ]
        f = df[[f"f_{lec}" for lec in LECTEURS]].fillna("<na>")
        df["accord_formes"] = f.apply(lambda r: max(Counter(r).values()), axis=1)
    return df


VARS_COPIE = ["copie_non_unan"] + [f"copie_{k}_{lec}" for k in "90" for lec in LECTEURS]
VARS_VOISINS = [f"{k}_{n}" for k in ("n9", "n0") for n in ("prec", "suiv")]


def matrice(df: pd.DataFrame, contexte: bool, formes: bool) -> np.ndarray:
    """Variables communes à tous les items (codes, contexte, classes de lecture)."""
    cols = [(df[f"c_{lec}"] == v).to_numpy(float) for lec in LECTEURS for v in CODES]
    if contexte:
        v = df[VARS_VOISINS].to_numpy(float)
        cols += list(df[VARS_COPIE].to_numpy(float).T) + list((v > 0).T) + list((v < 0).T)
    if formes:
        for lec in LECTEURS:
            cols += [(df[f"cl_{lec}"] == k).to_numpy(float) for k in CLASSES]
            cols.append(df[f"d_{lec}"].to_numpy(float))
        cols.append(df["accord_formes"].to_numpy(float))
    return np.column_stack(cols).astype(float)


def formes_frequentes(tr: pd.DataFrame) -> dict[str, list[str]]:
    """Par item, les formes lues les plus fréquentes autres que l'attendue (apprentissage)."""
    out: dict[str, list[str]] = {}
    for item, d in tr.groupby("item"):
        att = d["attendu"].iloc[0]
        cnt: Counter = Counter()
        for lec in LECTEURS:
            cnt.update(x for x in d[f"f_{lec}"] if x is not None and x != att)
        out[item] = [f for f, n in cnt.most_common(N_FORMES_FREQ) if n >= MIN_FORMES_FREQ]
    return out


def matrice_formes_item(d: pd.DataFrame, liste: list[str]) -> np.ndarray:
    """Pour un item : nombre de lecteurs ayant lu chacune de ses formes fréquentes."""
    if not liste:
        return np.zeros((len(d), 0))
    f = d[[f"f_{lec}" for lec in LECTEURS]].to_numpy(object)
    return np.column_stack([(f == x).sum(1) for x in liste]).astype(float)


# ─── Modèles ─────────────────────────────────────────────────────────────────
def proba_pleine(clf, X: np.ndarray) -> np.ndarray:
    """Probabilités sur les 3 codes, même si une classe manque à l'apprentissage."""
    p_full = np.zeros((len(X), 3))
    p = clf.predict_proba(X)
    for j, cl in enumerate(clf.classes_):
        p_full[:, CODES.index(cl)] = p[:, j]
    return p_full


def m_table(tr: pd.DataFrame, te: pd.DataFrame, alpha: float = 2.0) -> np.ndarray:
    """Par item : fréquence du code expert pour chaque combinaison des trois codes,
    lissée vers la même combinaison tous items du même type confondus."""
    cle = [f"c_{lec}" for lec in LECTEURS]
    y1 = pd.get_dummies(tr["y"]).reindex(columns=CODES, fill_value=0).astype(float)
    t = pd.concat([tr[["item", "type", *cle]], y1], axis=1)
    glob = t.groupby(["type", *cle])[CODES].sum()
    loc = t.groupby(["item", *cle])[CODES].sum()
    prior_item = t.groupby("item")[CODES].sum()
    prior_item = prior_item.div(prior_item.sum(axis=1), axis=0)
    out = np.zeros((len(te), 3))
    for i, r in enumerate(te[["item", "type", *cle]].itertuples(index=False)):
        prior = prior_item.loc[r.item].to_numpy() if r.item in prior_item.index else np.ones(3) / 3
        kg = (r.type, *r[2:])
        if kg in glob.index:
            n = glob.loc[kg].to_numpy()
            prior = (n + alpha * prior) / (n.sum() + alpha)
        kl = (r.item, *r[2:])
        n = loc.loc[kl].to_numpy() if kl in loc.index else np.zeros(3)
        out[i] = (n + alpha * prior) / (n.sum() + alpha)
    return out


def m_logit_item(
    tr: pd.DataFrame, te: pd.DataFrame, contexte: bool, formes: bool, c_reg: float = 1.0
) -> np.ndarray:
    """Une régression logistique multinomiale par item (régularisation L2)."""
    out = np.zeros((len(te), 3))
    x_tr, x_te = matrice(tr, contexte, formes), matrice(te, contexte, formes)
    freq = formes_frequentes(tr) if formes else {}
    ytr_all = tr["y"].to_numpy()
    g_tr = tr.groupby("item").indices
    for item, idx_te in te.groupby("item").indices.items():
        idx_tr = g_tr.get(item, np.array([], dtype=int))
        ytr = ytr_all[idx_tr]
        if len(set(ytr)) < 2:
            out[idx_te, CODES.index(ytr[0]) if len(ytr) else 0] = 1.0
            continue
        a, b = x_tr[idx_tr], x_te[idx_te]
        if formes:
            liste = freq.get(item, [])
            a = np.column_stack([a, matrice_formes_item(tr.iloc[idx_tr], liste)])
            b = np.column_stack([b, matrice_formes_item(te.iloc[idx_te], liste)])
        mu, sd = a.mean(0), a.std(0) + 1e-9
        clf = LogisticRegression(C=c_reg, max_iter=3000)
        clf.fit((a - mu) / sd, ytr)
        out[idx_te] = proba_pleine(clf, (b - mu) / sd)
    return out


def m_boosting(tr: pd.DataFrame, te: pd.DataFrame, formes: bool) -> np.ndarray:
    """Un seul modèle commun (gradient boosting), l'item en variable catégorielle."""

    def x(d: pd.DataFrame) -> np.ndarray:
        return np.column_stack([d["pos"].to_numpy(float), matrice(d, True, formes)])

    clf = HistGradientBoostingClassifier(
        categorical_features=[0],
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=0,
    )
    clf.fit(x(tr), tr["y"].to_numpy())
    return proba_pleine(clf, x(te))


def modeles(formes: bool) -> dict:
    m = {
        "table (3 codes × item)": lambda tr, te: m_table(tr, te),
        "logit/item, + copie + voisins": lambda tr, te: m_logit_item(tr, te, True, False),
        "boosting, + copie + voisins": lambda tr, te: m_boosting(tr, te, False),
    }
    if formes:
        m["logit/item, + formes lues"] = lambda tr, te: m_logit_item(tr, te, True, True)
        m["boosting, + formes lues"] = lambda tr, te: m_boosting(tr, te, True)
    return m


# ─── Décision et indicateurs ─────────────────────────────────────────────────
def decision(p: np.ndarray, sans_biais: bool) -> np.ndarray:
    """Code retenu. `sans_biais` : on pondère P(9) pour que la part de 9 prédits égale
    la part attendue Σ P(9) / n (estimée sans code expert si P est calibrée)."""
    if not sans_biais:
        return np.array(CODES)[p.argmax(1)]
    cible = p[:, 1].mean()
    lo, hi = -4.0, 4.0
    for _ in range(40):
        w = (lo + hi) / 2
        q = p.copy()
        q[:, 1] *= np.exp(w)
        part = (q.argmax(1) == 1).mean()
        lo, hi = (w, hi) if part < cible else (lo, w)
    q = p.copy()
    q[:, 1] *= np.exp((lo + hi) / 2)
    return np.array(CODES)[q.argmax(1)]


def indicateurs(df: pd.DataFrame, pred: np.ndarray) -> dict:
    y = df["y"].to_numpy()
    mot = (df["type"] == "mot").to_numpy()
    return {
        "accord": (pred == y).mean(),
        "kappa": cohen_kappa_score(y, pred),
        "kappa_mots": cohen_kappa_score(y[mot], pred[mot]),
        "kappa_ponct": cohen_kappa_score(y[~mot], pred[~mot]),
        "biais_pt": ((pred == "9").mean() - (y == "9").mean()) * 100,
        "fautes_ratees": (pred[mot & (y == "9")] == "1").mean(),
        "fautes_inventees": (pred[mot & (y == "1")] == "9").mean(),
        "ponct_non_vue": (pred[~mot & (y == "1")] == "0").mean(),
    }


def residuelle(df: pd.DataFrame, pred: np.ndarray, incertitude: np.ndarray, taux: float) -> float:
    """Erreur résiduelle si l'humain corrige la part `taux` des items les plus incertains."""
    y = df["y"].to_numpy()
    n = int(round(taux * len(y)))
    faux = pred != y
    faux[np.argsort(-incertitude, kind="stable")[:n]] = False
    return faux.mean()


def ece(df: pd.DataFrame, p: np.ndarray, nbins: int = 10) -> float:
    y = df["y"].to_numpy()
    conf = p.max(1)
    juste = np.array(CODES)[p.argmax(1)] == y
    bins = np.minimum((conf * nbins).astype(int), nbins - 1)
    return sum(
        (bins == b).mean() * abs(juste[bins == b].mean() - conf[bins == b].mean())
        for b in range(nbins)
        if (bins == b).any()
    )


def preparer(path: str, grid_path: str, formes: bool) -> pd.DataFrame:
    df = charger(path, grid_path, formes)
    grid = load_grid(grid_path)
    att = {it.item_id: it.attendu for it in grid.items}
    df["attendu"] = df["item"].map(att)
    if formes:
        par_id = {it.item_id: it for it in grid.items}
        for lec in LECTEURS:
            df[f"cl_{lec}"] = [
                classe_lecture(lu, par_id[i])
                for lu, i in zip(df[f"f_{lec}"], df["item"], strict=True)
            ]
    return ajouter_variables(df, formes)


# ─── Mode 1 : validation croisée sur un corpus ───────────────────────────────
ITEMS_DIAG = [17, 22, 33, 34, 35, 59, 67, 73, 74]


def validation_croisee(df: pd.DataFrame, formes: bool, n_folds: int) -> None:
    mods = modeles(formes)
    oof = {k: np.zeros((len(df), 3)) for k in mods}
    for f, (itr, ite) in enumerate(GroupKFold(n_splits=n_folds).split(df, groups=df["copy"])):
        tr, te = df.iloc[itr].reset_index(drop=True), df.iloc[ite].reset_index(drop=True)
        for k, fn in mods.items():
            oof[k][ite] = fn(tr, te)
        print(f"  pli {f + 1}/{n_folds} fait", flush=True)

    vote = df["vote"].to_numpy()
    taux_vote = 1 - df["unanime"].mean()
    lignes = [{"approche": "vote majoritaire", **indicateurs(df, vote)}]
    renvois = [
        {
            "approche": "vote majoritaire",
            f"{taux_vote:.1%}": residuelle(df, vote, 1 - df["conf"].to_numpy(), taux_vote),
        }
    ]
    for k, p in oof.items():
        for sb in (False, True):
            pred = decision(p, sb)
            nom = k + (" [sans biais]" if sb else "")
            lignes.append({"approche": nom, **indicateurs(df, pred), "ECE": ece(df, p)})
            if not sb:
                rv = {"approche": k}
                for t in (0.05, 0.10, 0.15, taux_vote, 0.25):
                    rv[f"{t:.1%}"] = residuelle(df, pred, 1 - p.max(1), t)
                renvois.append(rv)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n== Indicateurs (validation croisée par copie, prédictions hors pli) ==")
    print(pd.DataFrame(lignes).set_index("approche").round(4).to_string())
    print("\n== Erreur résiduelle selon le taux de renvoi ==")
    print(pd.DataFrame(renvois).set_index("approche").round(4).to_string())
    cible = renvois[0][f"{taux_vote:.1%}"]
    print(f"\nRenvoi nécessaire pour l'erreur résiduelle du vote ({cible:.2%}) :")
    for k, p in oof.items():
        pred = decision(p, False)
        t = next(
            (
                t
                for t in np.arange(0, 0.4, 0.0025)
                if residuelle(df, pred, 1 - p.max(1), t) <= cible
            ),
            None,
        )
        print(f"  {k:40s} {t:.2%}" if t is not None else f"  {k:40s} > 40 %")

    print("\n== Taux de faute (part des items codés 9) ==")
    print(f"  expert {(df['y'] == '9').mean():.2%} | vote {(vote == '9').mean():.2%}")
    for k, p in oof.items():
        print(
            f"  {k:40s} Σ P(9)/n {p[:, 1].mean():.2%} | argmax {(decision(p, False) == '9').mean():.2%}"
        )

    print("\n== Accord sur les items du diagnostic ==")
    d = df.assign(vote_ok=vote == df["y"])
    for k, p in oof.items():
        d[k] = decision(p, False) == df["y"]
    tab = d.groupby(["pos", "attendu"])[["vote_ok", *oof]].mean().reset_index()
    tab["pos"] += 1
    tab = tab[tab["pos"].isin(ITEMS_DIAG)]
    print(tab.round(3).to_string(index=False))


# ─── Mode 2 : transposition vers un autre corpus ─────────────────────────────
def transposition(
    src: pd.DataFrame, cib: pd.DataFrame, formes: bool, n_train: list[int], n_rep: int
) -> None:
    commun = set(src["copy"]) & set(cib["copy"])
    if commun:
        print(f"  {len(commun)} copies présentes dans les deux corpus : retirées de la source")
        src = src[~src["copy"].isin(commun)].reset_index(drop=True)
    print(f"source : {src['copy'].nunique()} copies, cible : {cib['copy'].nunique()} copies")
    nom_m = "logit/item, + formes lues" if formes else "logit/item, + copie + voisins"
    fn = modeles(formes)[nom_m]
    # Taux de faute source : expert et estimation hors pli (même modèle).
    oof = np.zeros((len(src), 3))
    for itr, ite in GroupKFold(n_splits=5).split(src, groups=src["copy"]):
        oof[ite] = fn(src.iloc[itr].reset_index(drop=True), src.iloc[ite].reset_index(drop=True))
    t_src = {
        "expert": (src["y"] == "9").mean(),
        "vote": (src["vote"] == "9").mean(),
        "modèle": oof[:, 1].mean(),
    }

    rng = np.random.default_rng(0)
    copies = cib["copy"].unique()
    res = []
    for n in n_train:
        for rep in range(n_rep):
            sel = set(rng.choice(copies, size=n, replace=False))
            tr_c = cib[cib["copy"].isin(sel)].reset_index(drop=True)
            te = cib[~cib["copy"].isin(sel)].reset_index(drop=True)
            vote = te["vote"].to_numpy()
            taux_vote = 1 - te["unanime"].mean()
            r_vote = residuelle(te, vote, 1 - te["conf"].to_numpy(), taux_vote)
            ligne = {
                "n": n,
                "rep": rep,
                "vote_kappa": cohen_kappa_score(te["y"], vote),
                "vote_biais": indicateurs(te, vote)["biais_pt"],
                "vote_resid": r_vote,
                "t_expert": (te["y"] == "9").mean(),
                "t_vote": (vote == "9").mean(),
            }
            for nom, tr in (
                ("source", src),
                ("cible", tr_c),
                ("source+cible", pd.concat([src, tr_c], ignore_index=True)),
            ):
                p = fn(tr, te)
                pred = decision(p, False)
                ligne[f"{nom}_kappa"] = cohen_kappa_score(te["y"], pred)
                ligne[f"{nom}_biais"] = indicateurs(te, decision(p, True))["biais_pt"]
                ligne[f"{nom}_resid"] = residuelle(te, pred, 1 - p.max(1), taux_vote)
                ligne[f"{nom}_t"] = p[:, 1].mean()
            res.append(ligne)
            print(f"  n={n} rép. {rep + 1}/{n_rep} faite", flush=True)
    r = pd.DataFrame(res)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(
        f"\n== Transposition ({nom_m}), cible hors copies d'apprentissage, moyenne des répétitions =="
    )
    cols = [c for c in r.columns if c.endswith(("kappa", "biais", "resid"))]
    print(r.groupby("n")[cols].mean().round(4).T.to_string())
    print("\n== Taux de faute : source → cible (copies de test) ==")
    m = r.groupby("n").mean(numeric_only=True)
    for n, row in m.iterrows():
        d_exp = row["t_expert"] - t_src["expert"]
        print(
            f"  n={n}: expert {t_src['expert']:.2%} → {row['t_expert']:.2%} (Δ {d_exp * 100:+.2f} pt)"
        )
        print(
            f"        vote   {t_src['vote']:.2%} → {row['t_vote']:.2%} "
            f"(Δ {(row['t_vote'] - t_src['vote']) * 100:+.2f} pt)"
        )
        for nom in ("source", "cible", "source+cible"):
            if f"{nom}_t" in row and not np.isnan(row[f"{nom}_t"]):
                print(
                    f"        modèle {nom:13s} {t_src['modèle']:.2%} → {row[f'{nom}_t']:.2%} "
                    f"(Δ {(row[f'{nom}_t'] - t_src['modèle']) * 100:+.2f} pt)"
                )


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="JSONL du vote (corpus source)")
    ap.add_argument("--cible", help="JSONL du vote sur un autre corpus : mode transposition")
    ap.add_argument("--grid", default="configs/grille_dictee_2015.json")
    ap.add_argument("--formes", action="store_true", help="ajouter les variables de forme lue")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-train", default="100,200", help="copies cible pour apprendre")
    ap.add_argument("--repetitions", type=int, default=5)
    args = ap.parse_args()

    src = preparer(args.predictions, args.grid, args.formes)
    print(f"{src['copy'].nunique()} copies, {len(src)} items évaluables (source)")
    print("Répartition expert :", dict(Counter(src["y"])))
    if args.formes:
        for lec in LECTEURS:
            print(f"  classes lues {lec} :", dict(Counter(src[f"cl_{lec}"]).most_common()))
    if args.cible:
        cib = preparer(args.cible, args.grid, args.formes)
        transposition(
            src, cib, args.formes, [int(x) for x in args.n_train.split(",")], args.repetitions
        )
    else:
        validation_croisee(src, args.formes, args.folds)


if __name__ == "__main__":
    main()
