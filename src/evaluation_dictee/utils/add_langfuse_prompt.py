"""Pousse les templates chat de `pipeline/prompts.py` vers Langfuse.

Nouvelle version promue en production à chaque appel.
La logique conditionnelle (schéma, flags) reste dans le code : un seul prompt
versionné par fonction, pas une version par combinaison de flags.

Usage :
    uv run add-langfuse-prompt                                  # tous les prompts
    uv run add-langfuse-prompt "Transcription pass 1" "..."      # seulement ceux-là
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from langfuse import get_client

from evaluation_dictee.pipeline.prompts import PROMPT_TEMPLATES
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)


def push_prompts(names: Sequence[str] | None = None) -> None:
    """Crée/met à jour les templates dans Langfuse et les promeut en production.

    Args:
        names: Noms des prompts à pousser ; tous si None ou vide (par défaut, lus sur
            la ligne de commande). Évite de recréer une version des prompts inchangés.
    """
    if names is None:
        names = sys.argv[1:]
    inconnus = [n for n in names if n not in PROMPT_TEMPLATES]
    if inconnus:
        raise SystemExit(f"Prompts inconnus : {inconnus}. Connus : {list(PROMPT_TEMPLATES)}")
    langfuse = get_client()

    for name, messages in PROMPT_TEMPLATES.items():
        if names and name not in names:
            continue
        prompt = langfuse.create_prompt(
            name=name,
            type="chat",
            prompt=list(messages),
            labels=["production"],
        )
        logger.info(
            "Prompt « %s » poussé (version %s), labels : %s",
            prompt.name,
            prompt.version,
            prompt.labels,
        )

    # Flush explicite : Langfuse envoie en asynchrone, sinon les créations sont perdues.
    langfuse.flush()


if __name__ == "__main__":
    push_prompts()
