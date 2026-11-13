"""Transcription épelée en deux passes : reconstruction, puis alignement sur la dictée.

Passe 1 : le modèle épelle chaque mot (« p e r d u »), ligne par ligne. Le texte est
reconstruit ICI, jamais par le modèle : recoller les lettres lui-même lui redonne
l'occasion de corriger l'élève.
Passe 2 : le modèle (texte seul) aligne les mots lus, numérotés, sur les items de la
dictée. Il ne rend que des NUMÉROS, jamais de mots : il ne peut donc pas corriger
l'élève en recopiant. Le code vérifie l'alignement (ordre croissant, pas de
croisement) et ramène à « absent » tout item incohérent, pour qu'un mot oublié reste
un trou isolé au lieu de décaler tous les items suivants.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from evaluation_dictee.data.grid import GridItem

MARQUE_INCERTAIN = "[?]"

_PONCTUATION = {".", ",", "?", "!", ";", ":", "-", "—", "–", "«", "»", '"', "(", ")", "…"}
# Signes collés au mot précédent (pas d'espace avant) ; les autres gardent l'espace
# français (« ? », « ! », « ; », « : »).
_COLLES_A_GAUCHE = {".", ",", "…", ")"}

# ─────────────────────────────────────────────────────────────────────────────
# Passe 1 : reconstruction du texte depuis les éléments épelés
# ─────────────────────────────────────────────────────────────────────────────


def recoller(element: str) -> str:
    """Recolle un élément épelé (« p e r d {a→e} » → « perd{a→e} »).

    Les espaces internes sont supprimés, sauf celui qui suit « [ajout: » (lisibilité).

    Args:
        element: élément tel que rendu par le modèle en passe 1.

    Returns:
        L'élément sans les espaces d'épellation.
    """
    element = element.strip()
    m = re.fullmatch(r"\[ajout:\s*(.*)\]", element)
    if m:
        return "[ajout: " + "".join(m.group(1).split()) + "]"
    return "".join(element.split())


def est_ponctuation(element: str) -> bool:
    """Vrai si l'élément (recollé) est un signe de ponctuation seul."""
    return element in _PONCTUATION


def est_annotation(element: str) -> bool:
    """Vrai si l'élément est une rature, un ajout ou un passage illisible."""
    return (
        element.startswith("~~")
        or element.startswith("[ajout:")
        or element == "[illisible]"
        or "{" in element
    )


def sans_marque(element: str) -> str:
    """Retire la marque d'hésitation « [?] » d'un élément recollé."""
    return element.replace(MARQUE_INCERTAIN, "").strip()


def etat_final(element: str) -> str | None:
    """Version finale (corrigée par l'élève) d'un élément annoté ; None s'il disparaît.

    Mot barré → disparaît ; ajout → le mot ajouté ; « perd{a→e} » → « perde ».

    Args:
        element: élément recollé.

    Returns:
        Le texte final de l'élément, ou None si l'élève l'a barré.
    """
    if element.startswith("~~") and element.endswith("~~"):
        return None
    m = re.fullmatch(r"\[ajout:\s*(.*)\]", element)
    if m:
        return m.group(1)
    return re.sub(r"\{[^{}]*?→([^{}]*)\}", r"\1", element)


def joindre_ligne(elements: list[str]) -> str:
    """Joint les éléments d'une ligne en texte, avec les conventions de ponctuation.

    Args:
        elements: éléments recollés d'une ligne.

    Returns:
        La ligne en texte continu.
    """
    texte = ""
    for el in elements:
        if not texte:
            texte = el
        elif el in _COLLES_A_GAUCHE:
            texte += el
        else:
            texte += " " + el
    return texte


def lignes_vers_texte(lignes: list[list[str]], final: bool, garder_marques: bool) -> str:
    """Reconstruit le texte à partir des lignes d'éléments recollés.

    Args:
        lignes: éléments recollés, ligne par ligne.
        final: si True, résout ratures/ajouts/réécritures vers l'état final (DEPP).
        garder_marques: si True, conserve les « [?] » (diagnostic) ; sinon les retire.

    Returns:
        Le texte, une ligne de la copie par ligne de texte.
    """
    sortie = []
    for ligne in lignes:
        elements = []
        for el in ligne:
            if not garder_marques:
                el = sans_marque(el)
            if final:
                resolu = etat_final(el)
                if resolu is None:
                    continue
                el = resolu
            if el:
                elements.append(el)
        sortie.append(joindre_ligne(elements))
    return "\n".join(sortie)


def lignes_depuis_reponse(data: dict[str, Any]) -> list[list[str]] | None:
    """Extrait et recolle les lignes de la réponse JSON de la passe 1 ; None si absentes.

    Args:
        data: objet JSON décodé de la réponse.

    Returns:
        Les éléments recollés, ligne par ligne, ou None si « lignes » est inexploitable.
    """
    lignes = data.get("lignes")
    if not isinstance(lignes, list):
        return None
    out = []
    for ligne in lignes:
        if isinstance(ligne, list):
            elements = [recoller(str(el)) for el in ligne if str(el).strip()]
        else:  # ligne rendue en chaîne : on la découpe telle quelle
            elements = [recoller(el) for el in str(ligne).split("  ") if el.strip()]
        out.append(elements)
    return out if any(out) else None


# ─────────────────────────────────────────────────────────────────────────────
# Passe 2 : alignement des mots lus sur les items de la dictée
# ─────────────────────────────────────────────────────────────────────────────


def _decouper_elision(mot: str) -> list[str]:
    """« n'étaient » → [« n' », « étaient »] (les élisions sont des items à part)."""
    parts = re.findall(r"[^'’]+['’]|[^'’]+$", mot)
    return parts if "".join(parts) == mot else [mot]


@dataclass
class MotLu:
    """Un mot (ou signe) lu en passe 1, numéroté pour l'alignement."""

    numero: int  # 1-indexé, dans l'ordre de lecture
    ligne: int  # ligne de la copie, 1-indexée
    texte: str


def mots_lus(lignes: list[list[str]], final: bool = True) -> list[MotLu]:
    """Liste à plat et numérotée des mots lus, élisions découpées comme dans la grille.

    Args:
        lignes: éléments recollés de la passe 1.
        final: si True, ratures résolues vers l'état final (mot barré retiré, etc.).

    Returns:
        Les mots lus, numérotés à partir de 1.
    """
    out: list[MotLu] = []
    for li, ligne in enumerate(lignes, 1):
        for el in ligne:
            el = sans_marque(el)
            if final:
                resolu = etat_final(el)
                if resolu is None:
                    continue
                el = resolu
            for mot in el.split():
                for part in _decouper_elision(mot):
                    out.append(MotLu(len(out) + 1, li, part))
    return out


def formater_mots_lus(mots: list[MotLu]) -> str:
    """Bloc « mots lus » du prompt : un mot numéroté par ligne, regroupés par ligne."""
    blocs: list[str] = []
    ligne_courante = 0
    for m in mots:
        if m.ligne != ligne_courante:
            ligne_courante = m.ligne
            blocs.append(f"[ligne {m.ligne} de la copie]")
        blocs.append(f"  {m.numero:>3}. {m.texte}")
    return "\n".join(blocs)


def formater_items_attendus(items: list[GridItem]) -> str:
    """Bloc « items attendus » du prompt : identifiant et mot attendu, dans l'ordre."""
    return "\n".join(
        f"  {k:>3}. {it.item_id} → « {it.attendu} »"
        + (" (ponctuation)" if it.type == "ponctuation" else "")
        for k, it in enumerate(items, 1)
    )


@dataclass
class Alignement:
    """Alignement validé : pour chaque item, les numéros des mots lus associés."""

    lus_par_item: dict[str, list[int]]
    mots_en_trop: list[int]
    # Items dont l'alignement rendu était incohérent (numéro inconnu, croisement) et
    # a été ramené à « absent » ; 0 si le modèle a respecté toutes les contraintes.
    n_reparations: int = 0

    def ecrit(self, item_id: str, mots: list[MotLu]) -> str:
        """Ce que l'élève a écrit pour cet item ("" si absent)."""
        par_num = {m.numero: m.texte for m in mots}
        return " ".join(par_num[n] for n in self.lus_par_item.get(item_id, []) if n in par_num)

    def n_absents(self) -> int:
        """Nombre d'items sans aucun mot lu associé."""
        return sum(1 for nums in self.lus_par_item.values() if not nums)


def valider_alignement(
    data: dict[str, Any], items: list[GridItem], mots: list[MotLu], seuil: float = 0.8
) -> Alignement | None:
    """Vérifie et répare l'alignement rendu par le modèle ; None s'il est inexploitable.

    Contraintes imposées : numéros existants, ordre croissant le long de la dictée (pas
    de croisement), un même numéro partagé seulement par des items CONSÉCUTIFS (mot
    fusionné). Un item qui viole une contrainte est ramené à « absent » plutôt que de
    laisser une erreur se propager aux suivants.

    Args:
        data: objet JSON décodé de la réponse de la passe 2.
        items: items de la grille, dans l'ordre.
        mots: mots lus numérotés.
        seuil: part minimale des items présents dans la réponse pour l'accepter.

    Returns:
        L'alignement validé, ou None si trop d'items manquent (nouvel essai).
    """
    brut = data.get("alignement")
    if not isinstance(brut, list):
        return None
    par_id: dict[str, list[int]] = {}
    for entree in brut:
        if not isinstance(entree, dict) or "item_id" not in entree:
            continue
        lus = entree.get("lus") or []
        if not isinstance(lus, list):
            lus = [lus]
        nums: list[int] = []
        for n in lus:
            try:
                nums.append(int(n))
            except (TypeError, ValueError):
                continue
        par_id[str(entree["item_id"])] = nums
    if len(par_id) < seuil * len(items):
        return None

    valides = {m.numero for m in mots}
    lus_par_item: dict[str, list[int]] = {}
    dernier = 0  # plus grand numéro déjà utilisé
    precedent: list[int] = []  # numéros de l'item précédent (fusion autorisée)
    reparations = 0
    for it in items:
        nums = sorted(set(par_id.get(it.item_id, [])))
        ok = bool(nums) and all(n in valides for n in nums)
        if ok:
            # Croissance : soit après tout ce qui précède, soit partage exact du
            # dernier mot de l'item précédent (fusion « nousles »).
            fusion = nums == [precedent[-1]] if precedent else False
            ok = fusion or nums[0] > dernier
            # Les numéros d'un même item doivent être consécutifs (mot coupé).
            ok = ok and nums == list(range(nums[0], nums[-1] + 1))
        if not ok:
            if nums:
                reparations += 1
            nums = []
        lus_par_item[it.item_id] = nums
        if nums:
            dernier = max(dernier, nums[-1])
            precedent = nums
    utilises = {n for nums in lus_par_item.values() for n in nums}
    en_trop = [m.numero for m in mots if m.numero not in utilises]
    return Alignement(lus_par_item, en_trop, reparations)
