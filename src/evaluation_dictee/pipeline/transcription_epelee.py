"""Transcription épelée en deux passes : reconstruction, questions fermées, application.

Passe 1 : le modèle épelle chaque mot (« p e r d u »), ligne par ligne. Le texte est
reconstruit ICI, jamais par le modèle : recoller les lettres lui-même lui redonne
l'occasion de corriger l'élève.
Passe 2 : pour chaque mot lu en passe 1, on propose quelques orthographes très proches
(lecture de la passe 1, mot attendu, fautes connues de la grille, variantes
morphologiques courantes), dans un ordre mélangé. Le texte de référence n'est JAMAIS
envoyé au modèle : il ne sert qu'à fabriquer les choix.
"""

from __future__ import annotations

import difflib
import random
import re
import unicodedata
import zlib
from dataclasses import dataclass, field
from typing import Any

from evaluation_dictee.data.grid import GridItem

MARQUE_INCERTAIN = "[?]"
CHOIX_AUTRE = "AUTRE"
LETTRES_CHOIX = "ABCDEFGH"
# Nombre maximal d'options par question (exemple du prompt : A à D).
N_OPTIONS_MAX = 4

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
# Passe 2 : variantes et questions
# ─────────────────────────────────────────────────────────────────────────────

# Familles de terminaisons interchangeables (l'ordre compte : la plus longue d'abord).
_FAMILLES_TERMINAISONS: list[list[str]] = [
    ["ées", "és", "ée", "é", "er", "ez"],
    ["aient", "ait", "ais", "ai"],
    ["ues", "ue", "us", "u"],
    ["ies", "ie", "is", "it", "i"],
]

_HOMOPHONES: list[set[str]] = [
    {"a", "à", "as"},
    {"ou", "où"},
    {"et", "est", "es"},
    {"son", "sont"},
    {"on", "ont"},
    {"ce", "se"},
    {"ces", "ses", "c'est", "s'est"},
    {"leur", "leurs"},
    {"la", "là", "l'a"},
    {"mes", "mais", "met", "mets"},
    {"peu", "peut", "peux"},
    {"sans", "s'en", "sang"},
    {"dans", "d'en"},
    {"ni", "n'y"},
    {"si", "s'y"},
    {"tout", "tous", "toute", "toutes"},
    {"quel", "quels", "quelle", "quelles", "qu'elle", "qu'elles"},
]

_ACCENTS = {"é": "e", "è": "e", "ê": "e", "ë": "e", "à": "a", "â": "a", "î": "i", "ï": "i",
            "ô": "o", "û": "u", "ù": "u", "ç": "c"}  # fmt: skip


def normaliser(mot: str) -> str:
    """Clé de comparaison : minuscules, sans accents ni marque d'hésitation."""
    mot = sans_marque(mot).lower()
    decompose = unicodedata.normalize("NFD", mot)
    return "".join(c for c in decompose if unicodedata.category(c) != "Mn")


def variantes_courantes(mot: str) -> list[str]:
    """Fautes d'orthographe courantes autour d'un mot (accords, terminaisons, homophones).

    Args:
        mot: forme de départ (en général le mot attendu).

    Returns:
        Variantes distinctes du mot, de la plus probable à la moins probable.
    """
    if not mot or est_ponctuation(mot):
        return []
    out: list[str] = []
    bas = mot.lower()
    for groupe in _HOMOPHONES:
        if bas in groupe:
            out += sorted(groupe - {bas})
    terminaison_trouvee = False
    for famille in _FAMILLES_TERMINAISONS:
        for term in famille:
            if bas.endswith(term) and len(bas) > len(term) + 1:
                racine = mot[: -len(term)]
                out += [racine + t for t in famille if t != term]
                terminaison_trouvee = True
                break
        if terminaison_trouvee:
            break
    # Accord en genre et en nombre (perdu → perdue, perdus, perdues). Pas sur les mots
    # courts (« à » → « às ») ni sur les formes verbales en -nt (« sont » → « sonts »).
    if len(bas) >= 3 and not bas.endswith("nt") and not terminaison_trouvee:
        if bas.endswith("es") and len(bas) >= 4:
            out += [mot[:-2], mot[:-1]]
        elif bas.endswith(("s", "x")):
            out.append(mot[:-1])
        elif bas.endswith("e"):
            # « perdue » → « perdu », mais pas « école » → « écol ».
            out += [mot + "s"] + ([mot[:-1]] if bas[-2] in "aeiouéy" else [])
        elif not bas.endswith(("es", "é", "er", "ez")):
            out += [mot + "s", mot + "e"]
    # Accents : forme sans accent, et e final accentué/non accentué.
    sans_accent = "".join(_ACCENTS.get(c, c) for c in mot)
    out.append(sans_accent)
    return [v for v in dict.fromkeys(out) if v and v != mot]


@dataclass
class _Sous:
    """Sous-partie d'un élément de la passe 1 (découpe aux apostrophes d'élision)."""

    texte: str
    item: GridItem | None = None


def _decouper_elision(mot: str) -> list[str]:
    """« n'étaient » → [« n' », « étaient »] (les élisions sont des items à part)."""
    parts = re.findall(r"[^'’]+['’]|[^'’]+$", mot)
    return parts if "".join(parts) == mot else [mot]


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normaliser(a), normaliser(b)).ratio()


def aligner(sous_parties: list[_Sous], items: list[GridItem]) -> None:
    """Associe chaque sous-partie lue en passe 1 à un item de la référence (en place).

    `difflib` aligne sur des clés normalisées (sans accents ni casse) ; dans les blocs
    non identiques, on apparie à la position quand les longueurs concordent, sinon au
    meilleur score de ressemblance.

    Args:
        sous_parties: sous-parties de la passe 1, dans l'ordre.
        items: items de la grille, dans l'ordre.
    """
    cles_lues = [normaliser(s.texte) for s in sous_parties]
    cles_ref = [normaliser(it.attendu) for it in items]
    matcher = difflib.SequenceMatcher(None, cles_lues, cles_ref, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                sous_parties[i1 + k].item = items[j1 + k]
        elif op == "replace":
            if i2 - i1 == j2 - j1:
                for k in range(i2 - i1):
                    sous_parties[i1 + k].item = items[j1 + k]
                continue
            libres = list(range(j1, j2))
            for i in range(i1, i2):
                if not libres:
                    break
                j = max(libres, key=lambda j: _ratio(sous_parties[i].texte, items[j].attendu))
                if _ratio(sous_parties[i].texte, items[j].attendu) >= 0.5:
                    sous_parties[i].item = items[j]
                    # On garde l'ordre : les items à gauche de j ne sont plus disponibles.
                    libres = [x for x in libres if x > j]


@dataclass
class Question:
    """Question fermée sur un élément de la passe 1."""

    id: int
    ligne: int  # 1-indexée
    position: int  # 1-indexée
    lu: str  # lecture de la passe 1 (sans marque)
    options: list[str] = field(default_factory=list)


def _options_pour(sous: list[_Sous], variantes_p1: list[str]) -> list[str]:
    """Orthographes candidates d'un élément : une sous-partie varie, les autres restent.

    Ordre de priorité : lecture passe 1, mot attendu, variantes signalées en passe 1,
    fautes connues de la grille, variantes morphologiques ; plafonné à N_OPTIONS_MAX.
    """
    lu = "".join(s.texte for s in sous)
    candidats: list[str] = [lu]
    # Mot attendu complet (toutes sous-parties à leur forme attendue).
    if all(s.item is not None for s in sous):
        candidats.append("".join(s.item.attendu for s in sous if s.item is not None))
    candidats += [sans_marque(v) for v in variantes_p1]
    for idx, s in enumerate(sous):
        if s.texte.endswith(("'", "’")):
            continue  # élision : « l' » ne varie pas en « l's »
        base = s.item.attendu if s.item is not None else s.texte
        fautes: list[str] = []
        if s.item is not None:
            fautes = [*s.item.ex_lexicale, *s.item.ex_grammaticale, *s.item.ex_les_deux]
        for forme in [*fautes, *variantes_courantes(base), *variantes_courantes(s.texte)]:
            avant = "".join(x.texte for x in sous[:idx])
            apres = "".join(x.texte for x in sous[idx + 1 :])
            candidats.append(avant + forme + apres)
    uniques = [c for c in dict.fromkeys(candidats) if c and " " not in c]
    return uniques[:N_OPTIONS_MAX]


def construire_questions(
    lignes: list[list[str]],
    items: list[GridItem],
    incertains: list[dict[str, Any]] | None = None,
    graine: str = "",
) -> list[Question]:
    """Construit une question fermée par mot lu en passe 1.

    Les ponctuations et les éléments annotés (ratures, ajouts, illisibles) ne sont pas
    interrogés. L'ordre des options est mélangé de façon reproductible (graine = id de
    la copie), pour que la bonne orthographe ne soit pas toujours à la même place.

    Args:
        lignes: éléments recollés de la passe 1, ligne par ligne.
        items: items de la grille, dans l'ordre (référence : jamais montrée au modèle).
        incertains: entrées « incertains » de la passe 1 (variantes proposées).
        graine: chaîne servant à initialiser le mélange.

    Returns:
        Les questions, numérotées à partir de 1.
    """
    variantes_par_pos: dict[tuple[int, int], list[str]] = {}
    for inc in incertains or []:
        try:
            cle = (int(inc["ligne"]), int(inc["position"]))
        except (KeyError, TypeError, ValueError):
            continue
        variantes = inc.get("variantes") or []
        variantes_par_pos[cle] = [recoller(str(v)) for v in variantes if str(v).strip()]

    # Découpe en sous-parties et alignement global avec la référence.
    elements: list[tuple[int, int, list[_Sous]]] = []
    a_plat: list[_Sous] = []
    for li, ligne in enumerate(lignes, 1):
        for pos, el in enumerate(ligne, 1):
            if est_annotation(el):
                continue
            mot = sans_marque(el)
            sous = [_Sous(p) for p in _decouper_elision(mot)] if mot else []
            elements.append((li, pos, sous))
            a_plat += sous
    aligner(a_plat, items)

    rng = random.Random(zlib.crc32(graine.encode("utf-8")))
    questions: list[Question] = []
    for li, pos, sous in elements:
        lu = "".join(s.texte for s in sous)
        if not lu or est_ponctuation(lu):
            continue
        options = _options_pour(sous, variantes_par_pos.get((li, pos), []))
        if len(options) < 2:
            continue
        rng.shuffle(options)
        questions.append(Question(len(questions) + 1, li, pos, lu, options))
    return questions


def _voisin(lignes: list[list[str]], li: int, pos: int, delta: int) -> str | None:
    """Élément voisin dans la transcription de la passe 1 (en traversant les lignes)."""
    a_plat = [(i, p) for i, ligne in enumerate(lignes, 1) for p in range(1, len(ligne) + 1)]
    k = a_plat.index((li, pos)) + delta
    if 0 <= k < len(a_plat):
        i, p = a_plat[k]
        return sans_marque(lignes[i - 1][p - 1])
    return None


def formater_questions(questions: list[Question], lignes: list[list[str]]) -> str:
    """Texte des questions pour le prompt (voisins pris dans la passe 1, jamais la référence).

    Args:
        questions: questions construites par `construire_questions`.
        lignes: éléments recollés de la passe 1.

    Returns:
        Le bloc à injecter dans `{{questions}}`.
    """
    blocs = []
    for q in questions:
        avant = _voisin(lignes, q.ligne, q.position, -1)
        apres = _voisin(lignes, q.ligne, q.position, +1)
        if avant and apres:
            contexte = f" (entre « {avant} » et « {apres} »)"
        elif avant:
            contexte = f" (après « {avant} »)"
        elif apres:
            contexte = f" (avant « {apres} »)"
        else:
            contexte = ""
        choix = "   ".join(f"{LETTRES_CHOIX[k]}) {opt}" for k, opt in enumerate(q.options))
        blocs.append(f"Question {q.id} — ligne {q.ligne}, position {q.position}{contexte}\n{choix}")
    return "\n\n".join(blocs)


@dataclass
class ResultatQuestions:
    """Application des réponses de la passe 2 aux lignes de la passe 1."""

    lignes: list[list[str]]
    n_corrections: int
    n_incertains: int


def appliquer_reponses(
    lignes: list[list[str]], questions: list[Question], reponses: list[dict[str, Any]]
) -> ResultatQuestions:
    """Remplace chaque mot interrogé par l'orthographe choisie en passe 2.

    Un choix lettré prime sur « epele » (moins sujet aux fautes de recopie) ; « AUTRE »
    prend « epele ». Une question sans réponse exploitable garde la lecture de la passe 1.

    Args:
        lignes: éléments recollés de la passe 1.
        questions: questions posées.
        reponses: entrées « reponses » du JSON de la passe 2.

    Returns:
        Les lignes corrigées, le nombre de mots modifiés et de réponses « certain: false ».
    """
    par_id: dict[int, dict[str, Any]] = {}
    for rep in reponses:
        if isinstance(rep, dict):
            try:
                par_id[int(rep["id"])] = rep
            except (KeyError, TypeError, ValueError):
                continue
    nouvelles = [list(ligne) for ligne in lignes]
    n_corr = n_inc = 0
    for q in questions:
        r = par_id.get(q.id)
        if r is None:
            continue
        if r.get("certain") is False:
            n_inc += 1
        choix = str(r.get("choix", "")).strip().upper()
        retenu: str | None = None
        if len(choix) == 1 and choix in LETTRES_CHOIX[: len(q.options)]:
            retenu = q.options[LETTRES_CHOIX.index(choix)]
        elif r.get("epele"):
            retenu = recoller(str(r["epele"]))
        if retenu and retenu != q.lu:
            nouvelles[q.ligne - 1][q.position - 1] = retenu
            n_corr += 1
        elif retenu:
            # Même mot : on retire la marque [?] éventuelle, la passe 2 a tranché.
            nouvelles[q.ligne - 1][q.position - 1] = retenu
    return ResultatQuestions(nouvelles, n_corr, n_inc)
