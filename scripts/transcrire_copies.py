"""Transcrit des copies avec la lecture du pipeline et range chaque copie dans un dossier.

Réutilise la passe 1 de `TwoStageScorer` (transcription à l'aveugle, état final des
ratures, découpage en bandes selon la config) sans l'étape de codage. Avec `--vote`,
les trois lecteurs de `dictee_two_stage_vote.yaml` lisent chaque copie et leurs
lectures sont combinées mot par mot (`transcription/vote_texte.py`).

    <sortie>/[lot_XX/]<copy_id>/<copy_id>.png            ← image en vrai PNG
    <sortie>/[lot_XX/]<copy_id>/<copy_id>.txt            ← transcription (lignes conservées)
    <sortie>/[lot_XX/]<copy_id>/<copy_id>_lecteurs.json  ← avec --vote : lectures et vote

Relancer la même commande reprend : les copies dont le .txt existe sont sautées.

Exemple (20 lots de 20 copies, vote à trois lecteurs) :
    uv run scripts/transcrire_copies.py --vote --limit 400 --taille-lot 20 --out <dossier>
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import fsspec
from PIL import Image

from evaluation_dictee.config import ModelConfig, Secrets, load_config
from evaluation_dictee.data.grid import GridItem, load_grid
from evaluation_dictee.data.loaders import Copy, load_dataset, load_image
from evaluation_dictee.models.factory import build_scorer
from evaluation_dictee.models.two_stage import TwoStageScorer
from evaluation_dictee.transcription.vote_texte import voter_transcription
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)

CONFIG_UN_LECTEUR = "configs/scoring/dictee_two_stage_gemma_bandes.yaml"
CONFIG_VOTE = "configs/scoring/dictee_two_stage_vote.yaml"
BASE_VOTE = "gemma4-26b-moe"  # meilleur lecteur seul : son texte sert de trame


def _sauver_png(copy: Copy, destination: Path) -> None:
    """Copie l'image source (TIFF 1 bit déguisé en .png) en vrai PNG."""
    with fsspec.open(copy.image_path, "rb") as f:
        Image.open(f).convert("L").save(destination, format="PNG")


def _lire(scorer: TwoStageScorer, copy: Copy, lecteur: ModelConfig, image: Image.Image) -> str:
    """Passe 1 d'un lecteur ; chaîne vide si échec (exception API comprise)."""
    try:
        transcription, _, _, _ = scorer._transcribe_et_aligner(copy, lecteur, image)
    except Exception as exc:  # noqa: BLE001 — un lecteur en panne ne bloque pas le vote
        logger.warning("%s : échec de %s (%s).", copy.copy_id, lecteur.name, exc)
        return ""
    return transcription


def _traiter(
    scorer: TwoStageScorer,
    lecteurs: list[ModelConfig],
    base: int,
    items: list[GridItem],
    copy: Copy,
    dossier: Path,
) -> bool:
    """Lit une copie, écrit son dossier ; True si la transcription est non vide."""
    nom = PurePosixPath(copy.copy_id).stem
    dossier.mkdir(parents=True, exist_ok=True)
    _sauver_png(copy, dossier / f"{nom}.png")
    image = load_image(copy.image_path)
    lectures = [_lire(scorer, copy, lecteur, image) for lecteur in lecteurs]

    if len(lecteurs) == 1:
        texte = lectures[0]
    else:
        items_copie = [it for it in items if it.item_id in set(copy.item_ids)]
        res = voter_transcription(lectures, items_copie, base=base)
        texte = res.texte
        detail = {
            "copy_id": copy.copy_id,
            "lecteurs": [lecteur.name for lecteur in lecteurs],
            "lecteur_base": lecteurs[res.base].name,
            "transcriptions_lecteurs": lectures,
            "compte": res.compte(),
            "mots_non_unanimes": [
                {
                    "items": u.item_ids,
                    "attendu": u.attendu,
                    "lectures": u.lectures,
                    "retenu": u.retenu,
                    "n_accord": u.n_accord,
                }
                for u in res.unites
                if not u.unanime
            ],
        }
        (dossier / f"{nom}_lecteurs.json").write_text(
            json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    # Le .txt en dernier : sa présence marque la copie comme faite (reprise).
    (dossier / f"{nom}.txt").write_text(texte.rstrip() + "\n", encoding="utf-8")
    logger.info("%s : %d caractères.", copy.copy_id, len(texte))
    return bool(texte.strip())


def main() -> None:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="Config two_stage (défaut selon --vote).")
    parser.add_argument("--vote", action="store_true", help="Vote entre les trois lecteurs.")
    parser.add_argument("--base", default=BASE_VOTE, help="Lecteur servant de trame au vote.")
    parser.add_argument("--limit", type=int, default=10, help="Nombre de copies.")
    parser.add_argument("--ids", nargs="*", help="Identifiants précis (sinon les premières).")
    parser.add_argument("--taille-lot", type=int, default=0, help="Copies par lot (0 = aucun).")
    parser.add_argument("--concurrency", type=int, help="Copies en parallèle (défaut : config).")
    parser.add_argument("--out", required=True, help="Dossier local de sortie (hors data/).")
    args = parser.parse_args()

    config = load_config(args.config or (CONFIG_VOTE if args.vote else CONFIG_UN_LECTEUR))
    if config.approach != "two_stage":
        parser.error("La config doit utiliser approach: two_stage.")

    copies = load_dataset(
        config.data.images_path,
        config.data.labels_path,
        limit=None if args.ids else args.limit,
    )
    if args.ids:
        voulus = set(args.ids)
        copies = [c for c in copies if c.copy_id in voulus]
    if not copies:
        parser.error("Aucune copie trouvée.")

    secrets = Secrets()
    grid = load_grid(config.data.grid_path)
    scorer = build_scorer(
        config=config,
        grid_items=grid.items,
        base_url=secrets.llm_base_url,
        api_key=secrets.llm_api_key,
    )
    assert isinstance(scorer, TwoStageScorer)

    lecteurs = [config.model]
    if args.vote:
        lecteurs += list(config.extra_readers or [])
        if len(lecteurs) < 2:
            parser.error("--vote demande une config avec extra_readers.")
    noms = [lecteur.name for lecteur in lecteurs]
    base = noms.index(args.base) if args.base in noms else 0
    logger.info("Lecteurs : %s (trame : %s).", ", ".join(noms), noms[base])

    sortie = Path(args.out)
    a_faire: list[tuple[Copy, Path]] = []
    for k, copy in enumerate(copies):
        nom = PurePosixPath(copy.copy_id).stem
        parent = sortie / f"lot_{k // args.taille_lot + 1:02d}" if args.taille_lot else sortie
        dossier = parent / nom
        if (dossier / f"{nom}.txt").exists():
            continue
        a_faire.append((copy, dossier))
    logger.info("%d copie(s) à traiter (%d déjà faites).", len(a_faire), len(copies) - len(a_faire))

    echecs: list[str] = []
    with ThreadPoolExecutor(max_workers=args.concurrency or config.concurrency) as pool:
        futures = {
            pool.submit(_traiter, scorer, lecteurs, base, grid.items, c, d): c for c, d in a_faire
        }
        for fut in as_completed(futures):
            copy = futures[fut]
            try:
                if not fut.result():
                    echecs.append(copy.copy_id)
            except Exception:
                logger.exception("%s : échec.", copy.copy_id)
                echecs.append(copy.copy_id)

    logger.info("Terminé : %d copie(s) dans %s.", len(copies), sortie)
    if echecs:
        logger.warning(
            "%d copie(s) sans transcription : %s", len(echecs), ", ".join(sorted(echecs))
        )


if __name__ == "__main__":
    main()
