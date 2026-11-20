"""Transcrit des copies avec la lecture du pipeline et range chaque copie dans un dossier.

Réutilise la passe 1 de `TwoStageScorer` (transcription à l'aveugle, état final des
ratures, découpage en bandes selon la config) sans l'étape de codage. Pour chaque copie :

    <sortie>/<copy_id>/<copy_id>.png   ← image convertie en vrai PNG
    <sortie>/<copy_id>/<copy_id>.txt   ← transcription, retours à la ligne conservés

Exemple :
    uv run scripts/transcrire_copies.py --limit 10 --out /tmp/transcriptions --dest s3://projet-production-ecrits-depp/transcriptions/test10
"""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath

import fsspec
from PIL import Image

from evaluation_dictee.config import Secrets, load_config
from evaluation_dictee.data.grid import load_grid
from evaluation_dictee.data.loaders import Copy, load_dataset, load_image
from evaluation_dictee.models.factory import build_scorer
from evaluation_dictee.models.two_stage import TwoStageScorer
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)

CONFIG_DEFAUT = "configs/scoring/dictee_two_stage_gemma_bandes.yaml"


def _sauver_png(copy: Copy, destination: Path) -> None:
    """Copie l'image source (TIFF 1 bit déguisé en .png) en vrai PNG."""
    with fsspec.open(copy.image_path, "rb") as f:
        Image.open(f).convert("L").save(destination, format="PNG")


def _transcrire(scorer: TwoStageScorer, copy: Copy) -> str:
    """Lit la copie avec la passe 1 du pipeline ; chaîne vide en cas d'échec."""
    transcription, n_essais, _, _ = scorer._transcribe_et_aligner(
        copy, image=load_image(copy.image_path)
    )
    logger.info("%s : %d caractères (%d essai(s)).", copy.copy_id, len(transcription), n_essais)
    return transcription


def main() -> None:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=CONFIG_DEFAUT, help="Config two_stage du lecteur.")
    parser.add_argument("--limit", type=int, default=10, help="Nombre de copies.")
    parser.add_argument("--ids", nargs="*", help="Identifiants précis (sinon les premières).")
    parser.add_argument("--out", required=True, help="Dossier local de sortie (hors data/).")
    parser.add_argument("--dest", help="Préfixe S3 où envoyer les dossiers (optionnel).")
    args = parser.parse_args()

    config = load_config(args.config)
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
    scorer = build_scorer(
        config=config,
        grid_items=load_grid(config.data.grid_path).items,
        base_url=secrets.llm_base_url,
        api_key=secrets.llm_api_key,
    )
    assert isinstance(scorer, TwoStageScorer)

    sortie = Path(args.out)
    echecs: list[str] = []
    for copy in copies:
        nom = PurePosixPath(copy.copy_id).stem
        dossier = sortie / nom
        dossier.mkdir(parents=True, exist_ok=True)
        _sauver_png(copy, dossier / f"{nom}.png")
        transcription = _transcrire(scorer, copy)
        if not transcription:
            echecs.append(copy.copy_id)
        (dossier / f"{nom}.txt").write_text(transcription.rstrip() + "\n", encoding="utf-8")

    logger.info("%d copie(s) écrite(s) dans %s.", len(copies), sortie)
    if echecs:
        logger.warning("Transcription vide pour : %s", ", ".join(echecs))

    if args.dest:
        fs, _, (racine,) = fsspec.get_fs_token_paths(args.dest.rstrip("/") + "/")
        fs.put(str(sortie) + "/", racine, recursive=True)
        logger.info("Dossiers envoyés vers %s.", args.dest)


if __name__ == "__main__":
    main()
