"""Relecture en zoom des items où les lecteurs divergent.

Pourquoi : qwen voit déjà la page en pleine résolution et rate pourtant ~21 % des
fautes (il lit le mot attendu). Plutôt que de relire toute la copie, on ne relit que
les items discordants, sur une bande de quelques lignes autour du mot, avec une
question ciblée : « recopie ce que l'élève a écrit entre tel et tel mot ».

Localisation : la passe 1 respecte les retours à la ligne de la copie (règle 6). Les
lignes d'écriture sont repérées sur l'image (blocs du profil d'encre, traits réglés
retirés). Si leur nombre égale celui des lignes transcrites, la ligne de l'item donne
directement son bloc ; sinon, elle est rapportée proportionnellement à l'étendue de
l'écriture. La bande couvre une marge d'une ligne et demie de part et d'autre.

Mesuré le 13/11/2026 (30 copies) avec une répartition uniforme sur toute la hauteur
du cadre : passage trouvé pour 100 % des items du premier quart de la dictée, 6 % du
dernier. L'écriture n'occupe pas le cadre uniformément.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.pipeline.alignement_local import Alignement

# Demi-hauteur de la bande, en lignes, au-delà de la ligne de l'item.
MARGE_LIGNES = 1.5
# Hauteur minimale de la bande (px de la zone d'écriture) : ~3 interlignes.
HAUTEUR_MIN = 200
# Mots de contexte de part et d'autre de l'item, tirés de la dictée.
N_CONTEXTE = 3

SCHEMA_RELECTURE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "passage_trouve": {"type": "boolean"},
        "ecrit": {"type": "string"},
    },
    "required": ["passage_trouve", "ecrit"],
    "additionalProperties": False,
}


@dataclass
class ItemARelire:
    """Un item discordant, avec ce que les lecteurs y ont lu et sa ligne."""

    item: GridItem
    lectures: list[str]  # lectures distinctes des lecteurs ("" = rien lu)
    ligne: int  # 1-indexée, dans la transcription du lecteur principal
    n_lignes: int  # nombre de lignes de cette transcription


def ligne_item(alignement: Alignement, items: list[GridItem], item_id: str) -> int:
    """Ligne (1-indexée) de l'item dans la transcription alignée.

    Un item sans mot lu prend la ligne de l'item lu le plus proche (avant d'abord).

    Args:
        alignement: alignement de la transcription du lecteur principal.
        items: items de la copie, dans l'ordre.
        item_id: item recherché.

    Returns:
        La ligne ; 1 si aucun item n'a de mot lu.
    """
    par_num = {m.numero: m.ligne for m in alignement.mots}
    ids = [it.item_id for it in items]
    j = ids.index(item_id)
    for k in sorted(range(len(ids)), key=lambda k: (abs(k - j), k > j)):
        nums = alignement.lus_par_item.get(ids[k], [])
        if nums:
            return par_num[nums[0]]
    return 1


def n_lignes(alignement: Alignement) -> int:
    """Nombre de lignes de la transcription (au moins 1)."""
    return max((m.ligne for m in alignement.mots), default=1)


def lignes_ecriture(zone: Image.Image) -> list[tuple[int, int]]:
    """Blocs de rangées encrées (lignes d'écriture) de la zone, de haut en bas.

    Les rangées presque pleines (traits réglés, soulignements) sont retirées avant le
    profil ; les blocs séparés de moins de 12 px sont fusionnés, ceux de moins de 15 px
    de haut (accents, points isolés) écartés.

    Args:
        zone: zone d'écriture (recadrée).

    Returns:
        Les blocs (haut, bas), bas exclu.
    """
    a = np.asarray(zone.convert("L")) < 128
    profil = a.mean(1)
    profil = np.where(profil > 0.4, 0.0, profil)
    lisse = np.convolve(profil, np.ones(7) / 7, mode="same")
    on = lisse > 0.01
    blocs: list[list[int]] = []
    y = 0
    while y < len(on):
        if on[y]:
            debut = y
            while y < len(on) and on[y]:
                y += 1
            if blocs and debut - blocs[-1][1] < 12:
                blocs[-1][1] = y
            else:
                blocs.append([debut, y])
        y += 1
    return [(h, b) for h, b in blocs if b - h >= 15]


def boite_bande(
    largeur: int,
    hauteur: int,
    ligne: int,
    nb_lignes: int,
    blocs: list[tuple[int, int]] | None = None,
) -> tuple[int, int, int, int]:
    """Bande horizontale (boîte PIL) autour de la ligne `ligne` sur `nb_lignes`.

    Args:
        largeur: largeur de la zone d'écriture.
        hauteur: hauteur de la zone d'écriture.
        ligne: ligne de l'item (1-indexée).
        nb_lignes: nombre de lignes transcrites.
        blocs: lignes d'écriture repérées sur l'image (`lignes_ecriture`) ; sans
            elles, répartition uniforme sur toute la hauteur.

    Returns:
        (0, haut, largeur, bas).
    """
    ligne = min(max(ligne, 1), nb_lignes)
    if blocs and len(blocs) == nb_lignes:
        h, b = blocs[ligne - 1]
        centre = (h + b) / 2
        pas = (blocs[-1][1] - blocs[0][0]) / nb_lignes
    elif blocs:
        haut, bas = blocs[0][0], blocs[-1][1]
        pas = (bas - haut) / max(1, nb_lignes)
        centre = haut + (ligne - 0.5) * pas
    else:
        pas = hauteur / max(1, nb_lignes)
        centre = (ligne - 0.5) * pas
    demi = max((0.5 + MARGE_LIGNES) * pas, HAUTEUR_MIN / 2)
    haut = max(0, round(centre - demi))
    bas = min(hauteur, round(centre + demi))
    return (0, haut, largeur, bas)


def extraire_bande(
    zone: Image.Image, item: ItemARelire, agrandissement: float = 1.0
) -> Image.Image:
    """Bande de la zone d'écriture autour de l'item, agrandie si demandé."""
    blocs = lignes_ecriture(zone)
    bande = zone.crop(boite_bande(zone.width, zone.height, item.ligne, item.n_lignes, blocs))
    if agrandissement != 1.0:
        taille = (round(bande.width * agrandissement), round(bande.height * agrandissement))
        bande = bande.resize(taille, Image.Resampling.LANCZOS)
    return bande


def contexte(items: list[GridItem], item_id: str) -> tuple[str, str]:
    """Mots de la dictée juste avant et juste après l'item."""
    ids = [it.item_id for it in items]
    j = ids.index(item_id)
    avant = " ".join(it.attendu for it in items[max(0, j - N_CONTEXTE) : j])
    apres = " ".join(it.attendu for it in items[j + 1 : j + 1 + N_CONTEXTE])
    return avant, apres


def prompt_relecture(items: list[GridItem], a_relire: ItemARelire) -> str:
    """Question ciblée sur un item : recopier ce que l'élève a écrit à sa place.

    Le contexte vient de la dictée (mots voisins, pas l'item lui-même) pour repérer
    le passage ; les lectures des lecteurs sont données comme pistes, pas comme
    réponses, pour ne pas pousser vers le mot attendu.

    Args:
        items: items de la copie, dans l'ordre.
        a_relire: l'item à relire.

    Returns:
        Le texte du message user.
    """
    avant, apres = contexte(items, a_relire.item.item_id)
    trou = f"« {avant} ___ {apres} »".replace("«  ", "« ").replace("  »", " »")
    pistes = ", ".join(f"« {lu} »" if lu else "(rien)" for lu in a_relire.lectures)
    if a_relire.item.type == "ponctuation":
        consigne = (
            "À la place de ___, la dictée attend un signe de ponctuation. Recopie le signe "
            "que l'élève a réellement tracé à cet endroit (point, virgule, deux-points, "
            "point d'exclamation…), ou une chaîne vide s'il n'en a tracé aucun."
        )
    else:
        consigne = (
            "Recopie exactement, lettre par lettre, le mot que l'élève a écrit à la place "
            "de ___ : SON orthographe, avec ses fautes, ses accents, ses majuscules et ses "
            "apostrophes, sans rien corriger. Les enfants dessinent parfois un petit rond "
            "au lieu du point sur le i. Si l'élève n'a rien écrit à cet endroit, réponds "
            "une chaîne vide."
        )
    return (
        "Cette image est un extrait de quelques lignes de la dictée manuscrite d'un "
        "élève de CM2. Dans la dictée, le passage est : "
        f"{trou}.\n\n"
        "Repère ce passage dans l'image (l'élève a pu faire des fautes sur les mots "
        f"voisins aussi). {consigne} En cas de rature, lis l'état final, c'est-à-dire "
        "la version corrigée par l'élève.\n\n"
        f"Des lecteurs automatiques ont lu à cet endroit : {pistes}. Ils peuvent se "
        "tromper tous : ne te fie qu'à l'image.\n\n"
        "Réponds en JSON : passage_trouve (false si tu ne trouves pas le passage dans "
        "l'image) et ecrit (ce que l'élève a écrit)."
    )
