"""Préparation de l'image envoyée au lecteur (passe 1) : recadrage, bandes, agrandissement.

Mesuré le 13/11/2026 (nombre de tokens d'image sur des images synthétiques) :
- qwen3-8-27b et qwen3-6-35b-moe voient l'image en pleine résolution (un token par
  carré de 32 × 32 pixels, sans plafond). Découper ne leur donne aucun détail en
  plus ; agrandir, si : une lettre manuscrite de 20 à 30 px tient sinon dans moins
  d'un token.
- gemma4-26b-moe réduit toute image à environ 256 tokens, quelle que soit sa
  taille (budget non réglable par requête sur llm.lab). Chaque bande envoyée comme
  image séparée reçoit son propre budget : 4 bandes = 4 fois plus de détail.

Mise en page des copies : un cadre pré-imprimé (traits pleins en haut, en bas, à
gauche et à droite) entoure la zone d'écriture ; au-dessus et en dessous, seulement
du pré-imprimé identique d'une copie à l'autre.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from evaluation_dictee.config import ImageConfig

# Un trait du cadre est une longue suite CONTINUE d'encre (au moins 30 % de la page),
# ce que n'est jamais du texte, même dense (l'en-tête pré-imprimé dépasse 40 % d'encre
# sur 3 rangées cumulées). La suite est cherchée sur l'union de 5 rangées voisines :
# un trait légèrement incliné ne tient sur aucune rangée isolée (4 copies sur 50 avec
# un simple seuil de densité à 0,5, mesuré le 13/11/2026).
# Vertical : 20 % de la hauteur suffit (aucune lettre ne fait 400 px de haut), et un
# trait gauche interrompu ne dépassait pas 434 px sur une copie.
_PART_TRAIT = 0.3
_PART_TRAIT_VERTICAL = 0.2
_VOISINAGE_TRAIT = 5
_RETRAIT_CADRE = 4
_SEUIL_ENCRE = 128


def _encre(image: Image.Image) -> np.ndarray:
    """Masque booléen des pixels d'encre."""
    return np.asarray(image.convert("L")) < _SEUIL_ENCRE


def _plus_longue_suite(a: np.ndarray) -> np.ndarray:
    """Longueur de la plus longue suite d'encre de chaque rangée (union de voisines)."""
    r = _VOISINAGE_TRAIT // 2
    union = a.copy()
    for k in range(1, r + 1):
        union[k:] |= a[:-k]
        union[:-k] |= a[k:]
    out = np.zeros(a.shape[0], dtype=int)
    bord = np.zeros((a.shape[0], 1), dtype=np.int8)
    d = np.diff(np.hstack([bord, union.astype(np.int8), bord]), axis=1)
    lignes_deb, cols_deb = np.nonzero(d == 1)
    _, cols_fin = np.nonzero(d == -1)  # même ordre (ligne, colonne) que les débuts
    np.maximum.at(out, lignes_deb, cols_fin - cols_deb)
    return out


def zone_ecriture(image: Image.Image) -> tuple[int, int, int, int]:
    """Boîte (gauche, haut, droite, bas) de la zone d'écriture, à l'intérieur du cadre.

    Le cadre est repéré par ses traits continus : premier trait horizontal du tiers
    supérieur, dernier du tiers inférieur, premier trait vertical du tiers gauche,
    dernier de la moitié droite. Si un trait manque ou si la boîte est aberrante
    (moins de la moitié de la page), toute la page est gardée.

    Args:
        image: page scannée.

    Returns:
        La boîte au format PIL (`Image.crop`).
    """
    a = _encre(image)
    h, w = a.shape
    page = (0, 0, w, h)
    rangees = np.where(_plus_longue_suite(a) >= _PART_TRAIT * w)[0]
    colonnes = np.where(_plus_longue_suite(a.T) >= _PART_TRAIT_VERTICAL * h)[0]
    # Les 2 % de bord sont ignorés : un bord noir de scan n'est pas le cadre.
    by, bx = h // 50, w // 50
    hauts = rangees[(rangees >= by) & (rangees < h // 3)]
    bas = rangees[(rangees > 2 * h // 3) & (rangees < h - by)]
    gauches = colonnes[(colonnes >= bx) & (colonnes < w // 3)]
    droites = colonnes[(colonnes > w // 2) & (colonnes < w - bx)]
    if not (len(hauts) and len(bas) and len(gauches) and len(droites)):
        return page
    boite = (
        int(gauches[0]) + _RETRAIT_CADRE,
        int(hauts[0]) + _RETRAIT_CADRE,
        int(droites[-1]) - _RETRAIT_CADRE + 1,
        int(bas[-1]) - _RETRAIT_CADRE + 1,
    )
    if boite[2] - boite[0] < w // 2 or boite[3] - boite[1] < h // 2:
        return page
    return boite


def points_de_coupe(image: Image.Image, n_bandes: int, decalage: float = 0.0) -> list[int]:
    """Rangées où couper l'image en `n_bandes` bandes, au creux des interlignes.

    Pour chaque coupe, on cherche autour de la position régulière ((k + décalage) ×
    hauteur / n) la rangée la moins encrée du profil horizontal lissé ; à encre égale,
    la plus proche de la position régulière.

    Args:
        image: image à découper (déjà recadrée si besoin).
        n_bandes: nombre de bandes voulu (sans décalage).
        decalage: décalage des coupes, en fraction de hauteur de bande. Non nul, il
            ajoute une coupe : avec 0,5, les coupes tombent au milieu des bandes non
            décalées (n − 1 bandes pleines et deux demi-bandes aux extrémités).

    Returns:
        Les rangées de coupe, strictement croissantes (n − 1 sans décalage, n avec).
    """
    if n_bandes <= 1 or image.height < 2 * n_bandes:
        return []
    a = _encre(image)
    h = a.shape[0]
    k = max(3, h // 200)
    profil = np.convolve(a.mean(1), np.ones(k) / k, mode="same")
    fenetre = max(1, h // (3 * n_bandes))
    coupes: list[int] = []
    positions = [i + decalage for i in range(n_bandes) if 0 < i + decalage < n_bandes]
    for pos in positions:
        cible = round(pos * h / n_bandes)
        lo = max(coupes[-1] + 1 if coupes else 1, cible - fenetre)
        hi = min(h - 1, cible + fenetre)
        if lo >= hi:
            coupes.append(max(lo, cible))
            continue
        seg = profil[lo:hi]
        candidats = np.where(seg <= seg.min() + 1e-12)[0] + lo
        coupes.append(int(candidats[np.argmin(np.abs(candidats - cible))]))
    return coupes


def epaissir(image: Image.Image, rayon: int) -> Image.Image:
    """Épaissit le trait de `rayon` pixels de chaque côté (encre sombre sur fond clair).

    Appliqué après le découpage : les coupes restent calculées sur le trait d'origine.

    Args:
        image: image en niveaux de gris.
        rayon: épaississement, en pixels.

    Returns:
        L'image au trait épaissi.
    """
    return image.convert("L").filter(ImageFilter.MinFilter(2 * rayon + 1))


def preparer(image: Image.Image, options: ImageConfig) -> list[Image.Image]:
    """Images à envoyer au lecteur, dans l'ordre de lecture (de haut en bas).

    Recadrage sur le cadre, puis découpage en bandes sans chevauchement, puis
    épaississement du trait et agrandissement de chaque bande. Options par défaut :
    la page telle quelle.

    Args:
        image: page scannée.
        options: options de préparation du lecteur.

    Returns:
        Une image (page entière) ou une image par bande.
    """
    if options == ImageConfig():
        return [image]
    if options.recadrage:
        image = image.crop(zone_ecriture(image))
    coupes = points_de_coupe(image, options.n_bandes, options.decalage_bandes)
    bornes = [0, *coupes, image.height]
    paires = zip(bornes, bornes[1:], strict=False)
    bandes = [image.crop((0, y0, image.width, y1)) for y0, y1 in paires]
    if options.epaississement:
        bandes = [epaissir(b, options.epaississement) for b in bandes]
    if options.agrandissement != 1.0:
        f = options.agrandissement
        bandes = [
            b.resize((round(b.width * f), round(b.height * f)), Image.Resampling.LANCZOS)
            for b in bandes
        ]
    return bandes
