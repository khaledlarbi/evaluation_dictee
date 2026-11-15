"""Agrégation apprise des trois lecteurs, comparée au vote majoritaire.

Pour chaque item de la dictée, un modèle donne la probabilité que l'item soit
correct (1), fautif (9) ou absent (0) à partir des codes des trois lecteurs, et
d'autres variables. On compare au vote à la majorité, par validation croisée par
copie (une copie entière dans le même pli).

N'imprime que des agrégats : aucune transcription ni code d'élève n'est écrit.

Usage :
    python scripts/agregation_apprise.py --predictions <JSONL du vote> [--grid ...]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold

from evaluation_dictee.data.grid import load_grid

CODES = ["1", "9", "0"]
LECTEURS = ["q27", "gem", "q35"]  # ordre de codes_lecteurs : model puis extra_readers
VAL_CODES = ["1", "9", "0", "NA"]


# ─── Chargement ──────────────────────────────────────────────────────────────
def charger(path: str, grid_path: str) -> pd.DataFrame:
    """Une ligne par item évaluable : codes des lecteurs, vote, code expert."""
    grid = load_grid(grid_path)
    pos = {it.item_id: i for i, it in enumerate(grid.items)}
    typ = {it.item_id: it.type for it in grid.items}
    par_cle: dict = {}
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
    rows = []
    for (cid, iid), r in par_cle.items():
        if r.get("exclusion") is not None or r.get("blank"):
            continue
        y = str(r["y_true"])
        if y not in CODES or iid not in pos:
            continue
        cl = r.get("codes_lecteurs") or [None, None, None]
        cl = [c if c in CODES else "NA" for c in cl]
        rows.append(
            dict(
                copy=cid,
                item=iid,
                pos=pos[iid],
                type=typ[iid],
                y=y,
                vote=str(r["y_pred"]),
                conf=r.get("confidence") or 0.0,
                **{f"c_{l}": c for l, c in zip(LECTEURS, cl)},
            )
        )
    df = pd.DataFrame(rows).sort_values(["copy", "pos"]).reset_index(drop=True)
    return df


def ajouter_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Variables de la copie et des items voisins (calculées sans le code expert)."""
    c = df[[f"c_{l}" for l in LECTEURS]]
    df["unanime"] = (c.nunique(axis=1) == 1).astype(int)
    df["n9"] = (c == "9").sum(axis=1)
    df["n0"] = (c == "0").sum(axis=1)
    g = df.groupby("copy")
    # Copie : taux de non-unanimité, taux de fautes et d'absences vus par chaque lecteur.
    df["copie_non_unan"] = 1 - g["unanime"].transform("mean")
    for l in LECTEURS:
        df[f"copie_9_{l}"] = g[f"c_{l}"].transform(lambda s: (s == "9").mean())
        df[f"copie_0_{l}"] = g[f"c_{l}"].transform(lambda s: (s == "0").mean())
    # Voisins : nombre de lecteurs codant 9 / 0 sur l'item précédent et le suivant.
    for d, nom in ((1, "prec"), (-1, "suiv")):
        meme = df["copy"].shift(d) == df["copy"]
        voisin = df["pos"].shift(d) == df["pos"] - d
        ok = meme & voisin
        for k in ("n9", "n0"):
            df[f"{k}_{nom}"] = np.where(ok, df[k].shift(d), -1)
    return df


VARS_COPIE = ["copie_non_unan"] + [f"copie_{k}_{l}" for k in "90" for l in LECTEURS]
VARS_VOISINS = [f"{k}_{n}" for k in ("n9", "n0") for n in ("prec", "suiv")]


def onehot_codes(df: pd.DataFrame) -> np.ndarray:
    cols = [(df[f"c_{l}"] == v).to_numpy(float) for l in LECTEURS for v in VAL_CODES[:-1]]
    return np.column_stack(cols)


def matrice(df: pd.DataFrame, avec_contexte: bool) -> np.ndarray:
    X = onehot_codes(df)
    if avec_contexte:
        voisins = df[VARS_VOISINS].to_numpy(float)
        X = np.column_stack(
            [
                X,
                df[VARS_COPIE].to_numpy(float),
                (voisins > 0).astype(float),
                (voisins < 0).astype(float),
            ]
        )
    return X


# ─── Modèles ─────────────────────────────────────────────────────────────────
def proba_pleine(clf, X) -> np.ndarray:
    """Probabilités sur les 3 codes, même si une classe manque à l'apprentissage."""
    P = np.zeros((len(X), 3))
    p = clf.predict_proba(X)
    for j, cl in enumerate(clf.classes_):
        P[:, CODES.index(cl)] = p[:, j]
    return P


def m_table(tr: pd.DataFrame, te: pd.DataFrame, alpha: float = 2.0) -> np.ndarray:
    """Par item : fréquence du code expert pour chaque combinaison des trois codes,
    lissée vers la même combinaison tous items confondus (puis vers l'item seul)."""
    cle = [f"c_{l}" for l in LECTEURS]
    y1 = pd.get_dummies(tr["y"]).reindex(columns=CODES, fill_value=0).astype(float)
    t = pd.concat([tr[["item", "type"] + cle], y1], axis=1)
    glob = t.groupby(["type"] + cle)[CODES].sum()
    loc = t.groupby(["item"] + cle)[CODES].sum()
    item_prior = t.groupby("item")[CODES].sum()
    item_prior = item_prior.div(item_prior.sum(axis=1), axis=0)
    out = np.zeros((len(te), 3))
    for i, r in enumerate(te[["item", "type"] + cle].itertuples(index=False)):
        prior = item_prior.loc[r.item].to_numpy() if r.item in item_prior.index else np.ones(3) / 3
        kg = (r.type, *r[2:])
        if kg in glob.index:
            n = glob.loc[kg].to_numpy()
            prior = (n + alpha * prior) / (n.sum() + alpha)
        kl = (r.item, *r[2:])
        n = loc.loc[kl].to_numpy() if kl in loc.index else np.zeros(3)
        out[i] = (n + alpha * prior) / (n.sum() + alpha)
    return out


def m_logit_item(tr, te, avec_contexte: bool, C: float = 1.0) -> np.ndarray:
    """Une régression logistique multinomiale par item (régularisation L2)."""
    out = np.zeros((len(te), 3))
    Xtr_all, Xte_all = matrice(tr, avec_contexte), matrice(te, avec_contexte)
    mu, sd = Xtr_all.mean(0), Xtr_all.std(0) + 1e-9
    Xtr_all, Xte_all = (Xtr_all - mu) / sd, (Xte_all - mu) / sd
    for item, idx_te in te.groupby("item").indices.items():
        m = (tr["item"] == item).to_numpy()
        ytr = tr["y"].to_numpy()[m]
        if len(set(ytr)) < 2:
            out[idx_te, CODES.index(ytr[0]) if len(ytr) else 0] = 1.0
            continue
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(Xtr_all[m], ytr)
        out[idx_te] = proba_pleine(clf, Xte_all[idx_te])
    return out


def m_boosting(tr, te) -> np.ndarray:
    """Un seul modèle commun (gradient boosting), l'item en variable catégorielle."""

    def X(d):
        return np.column_stack([d["pos"].to_numpy(float), matrice(d, True)])

    clf = HistGradientBoostingClassifier(
        categorical_features=[0],
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=0,
    )
    clf.fit(X(tr), tr["y"].to_numpy())
    return proba_pleine(clf, X(te))


# ─── Indicateurs ─────────────────────────────────────────────────────────────
def indicateurs(df: pd.DataFrame, pred: np.ndarray) -> dict:
    y = df["y"].to_numpy()
    mot = (df["type"] == "mot").to_numpy()
    res = {
        "accord": (pred == y).mean(),
        "kappa": cohen_kappa_score(y, pred),
        "kappa_mots": cohen_kappa_score(y[mot], pred[mot]),
        "kappa_ponct": cohen_kappa_score(y[~mot], pred[~mot]),
        "biais_pt": ((pred == "9").mean() - (y == "9").mean()) * 100,
        "fautes_ratees": (pred[mot & (y == "9")] == "1").mean(),
        "fautes_inventees": (pred[mot & (y == "1")] == "9").mean(),
        "ponct_non_vue": (pred[~mot & (y == "1")] == "0").mean(),
    }
    return res


def residuelle(df, pred, score_incertitude, taux) -> float:
    """Erreur résiduelle si on renvoie à l'humain la part `taux` des items les plus
    incertains (l'humain rend le code expert)."""
    y = df["y"].to_numpy()
    n = int(round(taux * len(y)))
    ordre = np.argsort(-score_incertitude, kind="stable")
    faux = pred != y
    faux[ordre[:n]] = False
    return faux.mean()


def ece(df, P, nbins=10) -> float:
    y = df["y"].to_numpy()
    conf = P.max(1)
    juste = np.array(CODES)[P.argmax(1)] == y
    bins = np.minimum((conf * nbins).astype(int), nbins - 1)
    e = 0.0
    for b in range(nbins):
        m = bins == b
        if m.any():
            e += m.mean() * abs(juste[m].mean() - conf[m].mean())
    return e


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--grid", default="configs/grille_dictee_2015.json")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    df = ajouter_variables(charger(args.predictions, args.grid))
    copies = df["copy"].unique()
    print(f"{len(copies)} copies, {len(df)} items évaluables")
    print("Répartition expert :", dict(Counter(df["y"])))

    modeles = {
        "table (3 codes × item)": lambda tr, te: m_table(tr, te),
        "logit/item, 3 codes": lambda tr, te: m_logit_item(tr, te, False),
        "logit/item, + copie + voisins": lambda tr, te: m_logit_item(tr, te, True),
        "boosting commun, + copie + voisins": m_boosting,
    }
    oof = {k: np.zeros((len(df), 3)) for k in modeles}
    gkf = GroupKFold(n_splits=args.folds)
    for f, (itr, ite) in enumerate(gkf.split(df, groups=df["copy"])):
        tr, te = df.iloc[itr].reset_index(drop=True), df.iloc[ite].reset_index(drop=True)
        for k, fn in modeles.items():
            oof[k][ite] = fn(tr, te)
        print(f"  pli {f + 1}/{args.folds} fait", flush=True)

    # Vote : incertitude = 1 − confiance (3/3, 2/3, 1/3).
    vote = df["vote"].to_numpy()
    taux_vote = 1 - df["unanime"].mean()
    taux = [0.05, 0.10, 0.15, round(taux_vote, 3), 0.25, 0.30]

    lignes, renvois = [], []
    lignes.append({"approche": "vote majoritaire", **indicateurs(df, vote)})
    rv = {"approche": "vote majoritaire"}
    rv[f"{taux_vote:.1%}"] = residuelle(df, vote, 1 - df["conf"].to_numpy(), taux_vote)
    renvois.append(rv)
    for k, P in oof.items():
        pred = np.array(CODES)[P.argmax(1)]
        lignes.append({"approche": k, **indicateurs(df, pred), "ECE": ece(df, P)})
        rv = {"approche": k}
        for t in taux:
            rv[f"{t:.1%}"] = residuelle(df, pred, 1 - P.max(1), t)
        renvois.append(rv)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n== Indicateurs (validation croisée par copie, prédictions hors pli) ==")
    print(pd.DataFrame(lignes).set_index("approche").round(4).to_string())
    print("\n== Erreur résiduelle selon le taux de renvoi ==")
    print(pd.DataFrame(renvois).set_index("approche").round(4).to_string())

    # Taux de renvoi nécessaire pour atteindre l'erreur résiduelle du vote.
    cible = renvois[0][f"{taux_vote:.1%}"]
    print(f"\nRenvoi nécessaire pour l'erreur résiduelle du vote ({cible:.2%}) :")
    for k, P in oof.items():
        pred = np.array(CODES)[P.argmax(1)]
        for t in np.arange(0.0, 0.40, 0.005):
            if residuelle(df, pred, 1 - P.max(1), t) <= cible:
                print(f"  {k:38s} {t:.1%}")
                break

    # Gains par item du meilleur modèle simple (logit/item + contexte).
    best = "logit/item, + copie + voisins"
    pred = np.array(CODES)[oof[best].argmax(1)]
    grid = load_grid(args.grid)
    att = {it.item_id: it.attendu for it in grid.items}
    d = df.assign(ok_vote=(vote == df["y"]), ok_mod=(pred == df["y"]))
    gi = d.groupby(["pos", "item"])[["ok_vote", "ok_mod"]].mean()
    gi["gain_pt"] = (gi["ok_mod"] - gi["ok_vote"]) * 100
    gi = gi.reset_index()
    gi["attendu"] = gi["item"].map(att)
    gi["pos"] += 1
    cols = ["pos", "attendu", "ok_vote", "ok_mod", "gain_pt"]
    print(f"\n== Items où « {best} » change le plus l'accord ==")
    print(gi.sort_values("gain_pt", ascending=False)[cols].head(10).round(3).to_string(index=False))
    print(gi.sort_values("gain_pt")[cols].head(5).round(3).to_string(index=False))

    # Courbe d'apprentissage : combien de copies corrigées faut-il ?
    print("\n== Courbe d'apprentissage (pli 1 en test, n copies tirées du reste) ==")
    rng = np.random.default_rng(0)
    itr, ite = next(gkf.split(df, groups=df["copy"]))
    te = df.iloc[ite].reset_index(drop=True)
    cop_tr = df.iloc[itr]["copy"].unique()
    k_vote = cohen_kappa_score(te["y"], te["vote"])
    print(f"  vote sur ce pli : kappa {k_vote:.4f}")
    for n in (50, 100, 200, 500, 1000, len(cop_tr)):
        sel = set(rng.choice(cop_tr, size=min(n, len(cop_tr)), replace=False))
        tr = df[df["copy"].isin(sel)].reset_index(drop=True)
        out = []
        for k in ("table (3 codes × item)", "logit/item, + copie + voisins"):
            pr = np.array(CODES)[modeles[k](tr, te).argmax(1)]
            out.append(f"{k}: kappa {cohen_kappa_score(te['y'], pr):.4f}")
        print(f"  n={n:5d}  " + " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
