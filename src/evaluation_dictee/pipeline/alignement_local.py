"""Alignement local (sans LLM) de la transcription d'une copie sur les items de la dictée.

Pourquoi en local : confié au LLM, l'alignement produisait des sauts aberrants qui
décalaient tous les items suivants (mesuré le 13/11/2026 sur 30 copies : biais de
+13,9 points sur le taux de faute). Un programme déterministe est reproductible et ne
saute jamais.

Algorithme : Needleman-Wunsch (programmation dynamique, coût minimal) entre la suite
des mots lus et la suite des items attendus, avec cinq mouvements :
- appariement mot lu ↔ item (coût = dissemblance des deux mots) ;
- item absent (l'élève ne l'a pas écrit) ;
- mot lu en trop (l'élève l'a ajouté) ;
- fusion : un mot lu couvre deux items consécutifs (« nousles », « nétaient ») ;
- coupure : deux mots lus forment un item (« re trouver »).
Un mot oublié coûte un seul « absent » : la suite reste alignée sur ses bons mots.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass

from evaluation_dictee.data.grid import GridItem

MARQUE_INCERTAIN = "[?]"

# Coûts (sur une échelle où 1 = deux mots sans aucune lettre commune).
COUT_ABSENT = 0.6
COUT_EN_TROP = 0.6
# Pénalité ajoutée à une fusion ou une coupure. Elle doit dépasser COUT_ABSENT moins
# le coût d'un appariement exact (0) à ressemblance imparfaite : sinon « et » oublié
# devant « maman » serait vu comme la fusion « etmaman » ≈ « maman » plutôt que comme
# un absent.
PENALITE_FUSION = 0.4
# Ressemblance minimale pour accepter une fusion ou une coupure.
RATIO_MIN_FUSION = 0.8
# Écart de longueur maximal (en lettres) entre le mot composé et sa contrepartie. Nul
# pour une fusion : avec un écart d'une lettre, « ils » passait pour « Ils » + « n' ».
ECART_LONGUEUR_FUSION = 0
ECART_LONGUEUR_COUPURE = 1
_INTERDIT = 1e9

# Mot (lettres, traits d'union internes) avec apostrophe d'élision éventuelle, nombre,
# ou signe de ponctuation isolé ; « ... » reste un seul signe.
_RE_TOKEN = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*['’]?|\d+|\.\.\.|…|[^\w\s]")


def normaliser(mot: str) -> str:
    """Clé de ressemblance : minuscules, sans accents, apostrophe typographique unifiée."""
    mot = mot.replace("’", "'").lower()
    decompose = unicodedata.normalize("NFD", mot)
    return "".join(c for c in decompose if unicodedata.category(c) != "Mn")


def est_ponctuation(token: str) -> bool:
    """Vrai si le token ne contient aucune lettre ni chiffre."""
    return not any(c.isalnum() for c in token)


@dataclass
class MotLu:
    """Un token lu sur la copie (mot, élision ou signe de ponctuation)."""

    numero: int  # 1-indexé, ordre de lecture
    ligne: int  # 1-indexée
    texte: str


def tokeniser(transcription: str) -> list[MotLu]:
    """Découpe la transcription en tokens comparables aux items de la grille.

    Les élisions sont séparées (« n'étaient » → « n' », « étaient ») comme dans la
    grille ; chaque signe de ponctuation est un token ; les marques « [?] » sont
    retirées. Un saut de ligne échappé deux fois (barre oblique littérale suivie de n)
    compte aussi comme un saut.

    Args:
        transcription: texte de la passe 1.

    Returns:
        Les tokens numérotés, avec leur ligne.
    """
    texte = transcription.replace("\\n", "\n").replace(MARQUE_INCERTAIN, " ")
    out: list[MotLu] = []
    for li, ligne in enumerate(texte.split("\n"), 1):
        for tok in _RE_TOKEN.findall(ligne):
            out.append(MotLu(len(out) + 1, li, tok))
    return out


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normaliser(a), normaliser(b)).ratio()


def cout_appariement(lu: str, attendu: str) -> float:
    """Coût d'apparier un token lu à un item attendu (0 = identiques).

    Une ponctuation ne s'apparie qu'à une ponctuation : un point lu à la place d'une
    virgule est une erreur de ponctuation (coût modéré), un mot face à un signe est
    interdit.

    Args:
        lu: token lu sur la copie.
        attendu: mot attendu de l'item.

    Returns:
        Le coût de l'appariement.
    """
    p_lu, p_att = est_ponctuation(lu), est_ponctuation(attendu)
    if p_lu or p_att:
        if not (p_lu and p_att):
            return _INTERDIT
        return 0.0 if lu == attendu else 0.5
    if lu == attendu:
        return 0.0
    if normaliser(lu) == normaliser(attendu):
        return 0.1  # accent ou majuscule seulement
    return max(0.2, 1.0 - _ratio(lu, attendu))


def _cout_compose(lus: list[str], attendus: list[str], ecart_max: int) -> float:
    """Coût d'une fusion (1 lu, 2 attendus) ou d'une coupure (2 lus, 1 attendu).

    Interdit si une des parties est une ponctuation, si les longueurs diffèrent de plus
    de `ecart_max` lettres, ou si la ressemblance est trop faible.
    """
    if any(est_ponctuation(t) for t in [*lus, *attendus]):
        return _INTERDIT
    a = normaliser("".join(lus)).replace("'", "")
    b = normaliser("".join(attendus)).replace("'", "")
    if abs(len(a) - len(b)) > ecart_max:
        return _INTERDIT
    ratio = 1.0 if a == b else difflib.SequenceMatcher(None, a, b).ratio()
    if ratio < RATIO_MIN_FUSION:
        return _INTERDIT
    return 1.0 - ratio + PENALITE_FUSION


@dataclass
class Alignement:
    """Résultat : pour chaque item, les numéros des tokens lus associés."""

    lus_par_item: dict[str, list[int]]
    mots_en_trop: list[int]
    mots: list[MotLu]

    def ecrit(self, item_id: str) -> str:
        """Ce que l'élève a écrit pour cet item ("" si absent)."""
        par_num = {m.numero: m.texte for m in self.mots}
        return " ".join(par_num[n] for n in self.lus_par_item.get(item_id, []))

    def ecrits(self) -> dict[str, str]:
        """Mot écrit par item, pour tous les items."""
        return {item_id: self.ecrit(item_id) for item_id in self.lus_par_item}

    def n_absents(self) -> int:
        """Nombre d'items sans aucun token lu associé."""
        return sum(1 for nums in self.lus_par_item.values() if not nums)


def aligner(mots: list[MotLu], items: list[GridItem]) -> Alignement:
    """Aligne les tokens lus sur les items par Needleman-Wunsch (coût minimal global).

    Args:
        mots: tokens lus (`tokeniser`).
        items: items de la grille, dans l'ordre.

    Returns:
        L'alignement : numéros des tokens par item, et tokens associés à aucun item.
    """
    n, m = len(mots), len(items)
    lus = [mt.texte for mt in mots]
    att = [it.attendu for it in items]
    inf = float("inf")
    cout = [[inf] * (m + 1) for _ in range(n + 1)]
    # Mouvement retenu en (i, j) : (di, dj) consommés côté lus / côté items.
    pas: list[list[tuple[int, int]]] = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    cout[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            # Ordre des candidats = départage à coût égal : appariement simple d'abord.
            candidats: list[tuple[float, tuple[int, int]]] = []
            if i >= 1 and j >= 1:
                candidats.append(
                    (cout[i - 1][j - 1] + cout_appariement(lus[i - 1], att[j - 1]), (1, 1))
                )
            if j >= 2 and i >= 1:
                c_fusion = _cout_compose(
                    [lus[i - 1]], [att[j - 2], att[j - 1]], ECART_LONGUEUR_FUSION
                )
                candidats.append((cout[i - 1][j - 2] + c_fusion, (1, 2)))
            if i >= 2 and j >= 1:
                c_coupure = _cout_compose(
                    [lus[i - 2], lus[i - 1]], [att[j - 1]], ECART_LONGUEUR_COUPURE
                )
                candidats.append((cout[i - 2][j - 1] + c_coupure, (2, 1)))
            if j >= 1:
                candidats.append((cout[i][j - 1] + COUT_ABSENT, (0, 1)))
            if i >= 1:
                candidats.append((cout[i - 1][j] + COUT_EN_TROP, (1, 0)))
            meilleur = min(candidats, key=lambda c: c[0])
            cout[i][j], pas[i][j] = meilleur

    lus_par_item: dict[str, list[int]] = {it.item_id: [] for it in items}
    en_trop: list[int] = []
    i, j = n, m
    while i > 0 or j > 0:
        di, dj = pas[i][j]
        nums = [mots[k].numero for k in range(i - di, i)]
        if dj == 0:
            en_trop = nums + en_trop
        else:
            for k in range(j - dj, j):
                lus_par_item[items[k].item_id] = nums
        i, j = i - di, j - dj
    return Alignement(lus_par_item, en_trop, mots)
