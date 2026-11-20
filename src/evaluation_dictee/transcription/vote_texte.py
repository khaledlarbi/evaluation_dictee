"""Vote entre plusieurs lectures d'une même copie pour produire UNE transcription.

Chaque lecture est alignée sur les items de la dictée (`pipeline/alignement_local`).
Pour chaque item (ou groupe d'items lus comme un seul mot), la forme écrite donnée par
au moins deux lecteurs l'emporte ; à défaut, celle du lecteur de base. Le texte est
reconstruit à partir de la lecture de base, dont on garde les retours à la ligne et les
mots hors dictée : seuls les mots tranchés autrement par le vote y sont remplacés.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from evaluation_dictee.data.grid import GridItem
from evaluation_dictee.pipeline.alignement_local import (
    _RE_TOKEN,
    MARQUE_INCERTAIN,
    Alignement,
    aligner,
    est_ponctuation,
    tokeniser,
)

# Signes collés au mot précédent quand on les insère.
_SANS_ESPACE_AVANT = {".", ",", "…", "...", ")"}


@dataclass
class Unite:
    """Un item, ou plusieurs items lus comme un seul mot par le lecteur de base."""

    item_ids: list[str]
    attendu: str
    lectures: list[str | None]  # forme écrite par lecteur (None = lecteur en échec)
    retenu: str
    n_accord: int  # nombre de lecteurs qui donnent la forme retenue

    @property
    def unanime(self) -> bool:
        """Vrai si tous les lecteurs disponibles donnent la même forme."""
        dispo = [lu for lu in self.lectures if lu is not None]
        return len(set(dispo)) == 1


@dataclass
class ResultatVote:
    """Transcription votée et détail du vote."""

    texte: str
    base: int  # indice du lecteur dont le texte sert de trame
    unites: list[Unite] = field(default_factory=list)

    def compte(self) -> dict[str, int]:
        """Nombre d'unités unanimes, tranchées à la majorité ou sans majorité."""
        n_unanimes = sum(u.unanime for u in self.unites)
        n_sans_majorite = sum(1 for u in self.unites if not u.unanime and u.n_accord < 2)
        return {
            "unanimes": n_unanimes,
            "majorite": len(self.unites) - n_unanimes - n_sans_majorite,
            "sans_majorite": n_sans_majorite,
        }


def _texte_normalise(transcription: str) -> str:
    """Même normalisation que `tokeniser`, pour que les positions correspondent."""
    return transcription.replace("\\n", "\n").replace(MARQUE_INCERTAIN, " ")


def _positions(texte: str) -> list[tuple[int, int]]:
    """Positions (début, fin) de chaque token de `tokeniser`, dans le même ordre."""
    out: list[tuple[int, int]] = []
    debut_ligne = 0
    for ligne in texte.split("\n"):
        out.extend(
            (debut_ligne + m.start(), debut_ligne + m.end()) for m in _RE_TOKEN.finditer(ligne)
        )
        debut_ligne += len(ligne) + 1
    return out


def _unites_base(
    alignement: Alignement, items: list[GridItem]
) -> list[tuple[list[str], list[int]]]:
    """Groupe les items consécutifs que la lecture de base associe aux mêmes tokens."""
    unites: list[tuple[list[str], list[int]]] = []
    for it in items:
        nums = alignement.lus_par_item.get(it.item_id, [])
        if unites and nums and unites[-1][1] == nums:
            unites[-1][0].append(it.item_id)
        else:
            unites.append(([it.item_id], nums))
    return unites


def _forme(alignement: Alignement, item_ids: list[str]) -> str:
    """Forme écrite par un lecteur pour un groupe d'items (sans doublon de fusion)."""
    formes: list[str] = []
    for item_id in item_ids:
        ecrit = alignement.ecrit(item_id)
        if ecrit and (not formes or formes[-1] != ecrit):
            formes.append(ecrit)
    return " ".join(formes)


def _choisir(lectures: list[str | None], base: int) -> tuple[str, int]:
    """Forme donnée par au moins deux lecteurs, sinon celle du lecteur de base."""
    compte = Counter(lu for lu in lectures if lu is not None)
    base_forme = lectures[base] or ""
    forme, n = compte.most_common(1)[0]
    # Égalité au sommet : le lecteur de base l'emporte.
    if n >= 2 and compte[base_forme] < n:
        return forme, n
    return base_forme, compte[base_forme]


def voter_transcription(
    transcriptions: list[str | None],
    items: list[GridItem],
    base: int = 0,
) -> ResultatVote:
    """Combine plusieurs lectures en une transcription, item par item.

    Args:
        transcriptions: une transcription par lecteur (None ou "" si le lecteur a échoué).
        items: items de la grille, dans l'ordre de la dictée.
        base: indice du lecteur dont le texte sert de trame (repli sur le premier
            lecteur disponible s'il a échoué).

    Returns:
        La transcription votée et le détail par unité. Texte vide si aucun lecteur
        n'a rien produit.
    """
    dispo = [i for i, t in enumerate(transcriptions) if t and t.strip()]
    if not dispo:
        return ResultatVote(texte="", base=base)
    if base not in dispo:
        base = dispo[0]

    alignements: list[Alignement | None] = []
    for t in transcriptions:
        mots = tokeniser(t) if t and t.strip() else []
        alignements.append(aligner(mots, items) if mots and items else None)

    texte = _texte_normalise(transcriptions[base] or "")
    al_base = alignements[base]
    if al_base is None:
        return ResultatVote(texte=texte.strip(), base=base)

    positions = _positions(texte)
    attendus = {it.item_id: it.attendu for it in items}
    unites: list[Unite] = []
    # Modifications du texte de base : (début, fin, remplacement, ordre de création).
    edits: list[tuple[int, int, str, int]] = []
    fin_precedente: int | None = None

    for item_ids, nums in _unites_base(al_base, items):
        lectures = [_forme(al, item_ids) if al is not None else None for al in alignements]
        retenu, n_accord = _choisir(lectures, base)
        unites.append(
            Unite(
                item_ids=item_ids,
                attendu=" ".join(dict.fromkeys(attendus[i] for i in item_ids)),
                lectures=lectures,
                retenu=retenu,
                n_accord=n_accord,
            )
        )
        if nums:
            debut, fin = positions[nums[0] - 1][0], positions[nums[-1] - 1][1]
            if retenu != lectures[base]:
                if retenu:
                    edits.append((debut, fin, retenu, len(edits)))
                else:
                    debut_sup = debut - 1 if debut > 0 and texte[debut - 1] == " " else debut
                    edits.append((debut_sup, fin, "", len(edits)))
            fin_precedente = fin
        elif retenu:
            # Absent de la lecture de base mais lu par la majorité : on l'insère.
            if fin_precedente is None:
                edits.append((0, 0, retenu + " ", len(edits)))
            else:
                colle = est_ponctuation(retenu) and retenu in _SANS_ESPACE_AVANT
                ajout = retenu if colle else " " + retenu
                edits.append((fin_precedente, fin_precedente, ajout, len(edits)))

    for debut, fin, remplacement, _ in sorted(edits, key=lambda e: (e[0], e[3]), reverse=True):
        texte = texte[:debut] + remplacement + texte[fin:]
    return ResultatVote(texte=texte.strip(), base=base, unites=unites)
