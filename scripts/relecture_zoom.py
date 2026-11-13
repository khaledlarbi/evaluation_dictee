"""Relecture en zoom des items discordants entre trois lectures déjà faites.

Pour chaque copie, les items où les trois runs ne donnent pas le même code sont
relus sur une bande de quelques lignes autour du mot, par un lecteur dédié, puis
codés par le modèle de l'étape 2. Compare ensuite plusieurs façons d'utiliser cette
relecture face au vote à trois.

N'affiche QUE des compteurs agrégés. Les relectures (données d'élèves) sont écrites
dans data/processed/, jamais affichées.

    uv run scripts/relecture_zoom.py --config configs/scoring/dictee_two_stage_vote.yaml \
        --run-a <principal>.jsonl --run-b <...>.jsonl --run-c <...>.jsonl \
        --lecteur gemma4-26b-moe --limit 30 [--agrandissement 2]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from evaluation_dictee.config import Secrets, load_config
from evaluation_dictee.data.grid import GridItem, load_grid
from evaluation_dictee.data.loaders import Copy, load_dataset, load_image
from evaluation_dictee.data.pretraitement import zone_ecriture
from evaluation_dictee.evaluation.ensemble import charger_codes, vote_majoritaire
from evaluation_dictee.evaluation.metrics import (
    compute_diagnostic_metrics,
    compute_scoring_metrics,
)
from evaluation_dictee.models.two_stage import (
    TwoStageScorer,
    _parse_json_object,
    _request_kwargs,
)
from evaluation_dictee.models.vlm import _image_to_data_url, log_if_truncated
from evaluation_dictee.pipeline.alignement_local import aligner, tokeniser
from evaluation_dictee.pipeline.relecture_zoom import (
    SCHEMA_RELECTURE,
    ItemARelire,
    extraire_bande,
    ligne_item,
    n_lignes,
    prompt_relecture,
)


def _charger_bruts(path: Path) -> dict[tuple[str, str], dict]:
    """Enregistrements bruts par (copy_id, item_id), dernière occurrence."""
    out: dict[tuple[str, str], dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out[(rec["copy_id"], rec["item_id"])] = rec
    return out


def _ligne(nom: str, y_true: list[str], y_pred: list[str], ids: list[str], types: dict) -> str:
    m = compute_scoring_metrics(y_true, y_pred)
    d = compute_diagnostic_metrics(y_true, y_pred, ids, types)
    return (
        f"{nom:<34} accord {m.raw_agreement * 100:5.1f} % | kappa {m.cohen_kappa:.3f} | "
        f"ponct {d.get('kappa_ponctuation', float('nan')):.3f} | "
        f"non détectées {d.get('taux_sous_detection', float('nan')) * 100:4.1f} % | "
        f"biais {d.get('biais_taux_faute_pts', float('nan')):+.1f} pt"
    )


class Relecteur:
    """Relit les items discordants d'une copie et les fait coder."""

    def __init__(self, scorer: TwoStageScorer, lecteur: str, agrandissement: float) -> None:
        """Lecteur de zoom (modèle) et scorer qui fournit client et codage."""
        self.scorer = scorer
        self.cfg = scorer.model_config.model_copy(update={"name": lecteur})
        self.agrandissement = agrandissement

    def _lire(self, copy_id: str, zone: Any, items: list[GridItem], a: ItemARelire) -> str | None:
        bande = extraire_bande(zone, a, self.agrandissement)
        contenu = [
            {"type": "text", "text": prompt_relecture(items, a)},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(bande)}},
        ]
        kwargs = _request_kwargs(self.cfg, SCHEMA_RELECTURE, "relecture_item")
        for essai in range(self.cfg.max_retries + 1):
            r = self.scorer.client.chat.completions.create(
                model=self.cfg.name,
                temperature=self.cfg.temperature + (0.3 if essai else 0.0),
                max_tokens=1024,
                messages=[{"role": "user", "content": contenu}],  # type: ignore[list-item]
                **kwargs,
            )
            if log_if_truncated(r, copy_id, 1024):
                continue
            data = _parse_json_object(r.choices[0].message.content or "")
            if data is not None and "ecrit" in data:
                if not data.get("passage_trouve", True):
                    return None
                return str(data["ecrit"]).strip()
        return None

    def relire(
        self, copy: Copy, items: list[GridItem], a_relire: list[ItemARelire]
    ) -> dict[str, tuple[str, str]]:
        """Lecture zoomée puis code de chaque item relu (absent si passage non trouvé)."""
        image = load_image(copy.image_path)
        zone = image.crop(zone_ecriture(image))
        lus = {a.item.item_id: self._lire(copy.copy_id, zone, items, a) for a in a_relire}
        ecrits = {i: e for i, e in lus.items() if e is not None}
        if not ecrits:
            return {}
        sous_copie = replace(copy, item_ids=list(ecrits))
        pred = self.scorer._code_text(sous_copie, "", "", ecrits)
        return {it.item_id: (ecrits[it.item_id], it.code) for it in pred.items}


def main() -> None:
    """Point d'entrée CLI."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--run-a", type=Path, required=True, help="Lecteur principal (localisation).")
    p.add_argument("--run-b", type=Path, required=True)
    p.add_argument("--run-c", type=Path, required=True)
    p.add_argument("--lecteur", required=True, help="Modèle qui relit en zoom.")
    p.add_argument("--agrandissement", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--concurrence", type=int, default=8)
    args = p.parse_args()

    config = load_config(args.config)
    grille = load_grid(config.data.grid_path)
    par_id = {it.item_id: it for it in grille.items}
    types = {it.item_id: it.type for it in grille.items}
    secrets = Secrets()
    scorer = TwoStageScorer(
        model_config=config.model,
        prompt_config=config.prompt,
        base_url=secrets.llm_base_url,
        api_key=secrets.llm_api_key,
        grid_items=grille.items,
        scheme=config.grid.scheme,
        model_config_stage2=config.model_stage2,
    )
    relecteur = Relecteur(scorer, args.lecteur, args.agrandissement)

    copies = load_dataset(config.data.images_path, config.data.labels_path, limit=args.limit)
    runs = [charger_codes(r) for r in (args.run_a, args.run_b, args.run_c)]
    bruts = [_charger_bruts(r) for r in (args.run_a, args.run_b, args.run_c)]

    travaux = []
    for copy in copies:
        items = [par_id[i] for i in copy.item_ids if i in par_id]
        cles = [(copy.copy_id, it.item_id) for it in items]
        if not all(all(c in r for c in cles) for r in runs):
            continue  # copie écartée ou absente d'un run
        transcription = next(
            (bruts[0][c].get("transcription_passe1") for c in cles if c in bruts[0]), ""
        )
        mots = tokeniser(transcription or "")
        if not mots:
            continue
        al = aligner(mots, items)
        a_relire = []
        for it, cle in zip(items, cles, strict=True):
            if len({r[cle][1] for r in runs}) > 1:
                lectures = list(
                    dict.fromkeys(str(b[cle].get("transcription") or "") for b in bruts)
                )
                a_relire.append(
                    ItemARelire(it, lectures, ligne_item(al, items, it.item_id), n_lignes(al))
                )
        travaux.append((copy, items, cles, a_relire))

    with ThreadPoolExecutor(args.concurrence) as ex:
        resultats = list(ex.map(lambda t: relecteur.relire(t[0], t[1], t[3]), travaux))

    slug = args.lecteur.replace("/", "_") + (
        f"-x{args.agrandissement:g}" if args.agrandissement != 1 else ""
    )
    sortie = Path("data/processed") / f"relecture_zoom_{slug}.jsonl"
    y_true, ids, vote, zoom_tranche, quatre, a_pred = [], [], [], [], [], []
    n_relus = n_trouves = 0
    renvoi_vote = renvoi_zoom = err_vote = err_zoom = 0
    with open(sortie, "w", encoding="utf-8") as f:
        for (_copy, _items, cles, a_relire), res in zip(travaux, resultats, strict=True):
            n_relus += len(a_relire)
            n_trouves += len(res)
            for cle in cles:
                codes = [r[cle][1] for r in runs]
                vt, conf, _ = vote_majoritaire(codes)
                vt = vt or codes[0]
                z = res.get(cle[1])
                zc = z[1] if z is not None else None
                y = runs[0][cle][0]
                y_true.append(y)
                ids.append(cle[1])
                a_pred.append(codes[0])
                vote.append(vt)
                zoom_tranche.append(zc if zc is not None else vt)
                # 4e voix : en cas d'égalité 2-2 ou de dispersion, la relecture tranche.
                c4, _, _ = vote_majoritaire([zc, *codes] if zc is not None else codes)
                quatre.append(c4 or vt)
                # Renvoi humain (l'humain est supposé juste ; erreurs = items non renvoyés).
                # Vote : tout item non unanime est renvoyé.
                if conf < 1:
                    renvoi_vote += 1
                elif vt != y:
                    err_vote += 1
                # Relecture : un non-unanime n'est renvoyé que si la relecture manque ou
                # contredit la majorité ; sinon le code majoritaire, confirmé, est gardé.
                if conf < 1 and (zc is None or zc != vt):
                    renvoi_zoom += 1
                elif vt != y:
                    err_zoom += 1
                if z is not None:
                    f.write(
                        json.dumps(
                            {
                                "copy_id": cle[0],
                                "item_id": cle[1],
                                "ecrit_zoom": z[0],
                                "code_zoom": z[1],
                                "codes_lecteurs": codes,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    n = len(y_true)
    print(
        f"Copies : {len(travaux)} | items : {n} | relus (non unanimes) : {n_relus} "
        f"({n_relus / n * 100:.1f} %) | passage trouvé : {n_trouves} "
        f"({n_trouves / max(1, n_relus) * 100:.0f} %)\n"
    )
    print(_ligne("A seul", y_true, a_pred, ids, types))
    print(_ligne("vote à 3", y_true, vote, ids, types))
    print(_ligne("relecture tranche (non unanimes)", y_true, zoom_tranche, ids, types))
    print(_ligne("relecture = 4e voix", y_true, quatre, ids, types))
    print(
        f"\nRenvoi humain, vote à 3 (non unanimes renvoyés) : {renvoi_vote / n * 100:.1f} % "
        f"des items | erreur résiduelle {err_vote / n * 100:.1f} %"
    )
    print(
        f"Renvoi humain, relecture (non unanimes où la relecture contredit le vote, ou "
        f"passage non trouvé) : {renvoi_zoom / n * 100:.1f} % | erreur résiduelle "
        f"{err_zoom / n * 100:.1f} %"
    )
    print(f"\nRelectures écrites dans {sortie} (non affichées).")


if __name__ == "__main__":
    main()
