# Bilan des tentatives : transcription en deux étapes et vote entre lecteurs

*13/11/2026, branche `feat/passe2-alignement`. Grille simplifiée 1 / erreur / 0.*
*Détail des essais sur l'image : [pistes_amelioration_image.md](pistes_amelioration_image.md).*

## Meilleur résultat

**Vote à trois lecteurs** : qwen3-8-27b (lecteur principal), gemma4-26b-moe sur la
copie recadrée en 4 bandes, qwen3-6-35b-moe. Codage par qwen3-6-35b-moe à partir de
l'alignement local. Mesuré sur 50 copies (4 150 items), en recombinant des lectures
déjà faites.

| | Accord | Kappa | Ponct. | Non détectées | Biais |
|---|---|---|---|---|---|
| **Vote à trois, gemma en 4 bandes** | **90,1 %** | **0,739** | **0,612** | **15,9 %** | **+0,1 pt** |

- Renvoi humain des items où qwen3-8-27b et gemma divergent : 10,3 % des items
  renvoyés, 6,2 % d'erreur résiduelle (accord 93,8 %).
- Config : `configs/scoring/dictee_two_stage_vote.yaml`. Elle n'a pas encore été
  lancée telle quelle de bout en bout : le chiffre vient de la recombinaison.

Progression du kappa : 0,650 (qwen3-6-35b-moe seul) → 0,687 (qwen3-8-27b) → 0,728
(vote à trois) → **0,739** (vote avec gemma en bandes).

## 1. Codage : le problème est réglé

| Tentative | Résultat | Retenu |
|---|---|---|
| Passe 1 épelée lettre par lettre, puis questions à choix fermé en passe 2 (branche `feat/transcription-deux-passes`) | remplacée par l'alignement | non |
| Alignement des mots lus sur la dictée par le LLM | sauts qui décalent tous les items suivants : biais de +13,9 pt sur 30 copies | non |
| Alignement local (Needleman-Wunsch, sans LLM), puis codage des paires alignées par le LLM | 83 items sur 83 corrects sur le texte de référence ; plus de décalage en cascade | **oui** |
| Prompt de passe 1 d'origine, avec deux règles en plus (rond à la place du point sur le i, majuscule après un point) | prompt actuel | **oui** |
| Refus des générations tronquées, avec un nouvel essai | qwen était tombé à 0,451 (une boucle acceptée comme transcription) ; remonte à 0,635 | **oui** |
| Diagnostic de l'origine des désaccords (codage ou lecture) | 98 à 100 % viennent de la lecture : le levier est l'image | outil gardé |

## 2. Lecteurs et vote (50 copies)

| Tentative | Kappa | Retenu |
|---|---|---|
| qwen3-6-35b-moe seul | 0,650 | — |
| gemma4-26b-moe seul (page entière) | 0,629 | — |
| qwen3-8-27b seul | 0,687 | **lecteur principal** |
| Vote qwen3-6 + gemma + qwen3-6 à température 0,7 | 0,703 | non |
| Vote qwen3-6 + gemma + qwen3-8-27b | 0,728 | oui, puis amélioré par l'image |
| Deux lecteurs, renvoi de leurs désaccords à l'humain | 15,5 % de renvoi, accord 94,3 % | mesure de confiance gardée |
| Vote intégré au pipeline, contrôlé sur 10 copies | 0,723 | **oui** |

Règle du vote : le code donné par au moins deux lecteurs l'emporte ; le lecteur
principal ne tranche que sans majorité (0,4 à 1 % des items). Confiance d'un item =
part des lecteurs qui donnent le code retenu.

## 3. Image envoyée en passe 1 (50 copies)

Résolution mesurée (tokens d'image sur des images synthétiques) : les deux qwen
lisent en pleine résolution (32 px par token) ; gemma ramène toute image à environ
256 tokens, soit ≈ 113 px par token pour la page entière.

| Tentative | Kappa | Retenu |
|---|---|---|
| gemma, page entière | 0,629 | — |
| gemma, 4 bandes sans recadrage | 0,709 | — |
| **gemma, recadrage + 4 bandes** | **0,722** (biais de +5,1 à +2,4 pt) | **oui** |
| gemma, recadrage + 6 bandes | 0,713 : le gain plafonne | non |
| qwen3-8-27b, agrandissement ×2 | 0,669 | non |
| qwen3-8-27b, recadrage + ×2 | 0,688 : aucun gain | non |

## 4. Essais sans gain (30 copies, 2 490 items)

| Tentative | Résultat | Retenu |
|---|---|---|
| Deuxième lecture gemma à coupes décalées, à la place de qwen3-6 | kappa 0,759 contre 0,754 (bruit) ; erreurs trop corrélées à la première lecture gemma (73 % contre 57 % pour qwen3-6) | non |
| Relecture en zoom des items discordants (gemma) | moins juste que le vote sur les items relus (63 % contre 68 %), biais +4 pt quand elle tranche. Seul intérêt : filtrer le renvoi (7,4 % pour 5,7 % d'erreur), pour ≈ 15 appels de plus par copie | non |
| Épaississement du trait de 1 px, sur les trois lecteurs (49 copies) | tous les lecteurs perdent (vote 0,747 → 0,686 ; gemma 0,725 → 0,631), la ponctuation la première : un point se colle aux lettres, les boucles se referment | non |
| Même relecture avec qwen3-8-27b, bande agrandie ×2 | encore moins bon (testé seulement avant la correction de la localisation) | non |

## 5. Croisements lecteur × codeur (99 copies, 8 217 items, 14/11/2026)

Trois lecteurs (gemma recadré en 4 bandes, qwen3-8-27b et qwen3-6-35b-moe sur la page
entière) × trois codeurs (qwen3-6-35b-moe, gemma4-26b-moe, qwen3-8-27b, en texte seul
sur les paires alignées). Une copie absente d'un run : comparaison sur 99 copies.

| Lecteur \ Codeur | qwen3-6-35b-moe | gemma4-26b-moe | qwen3-8-27b |
|---|---|---|---|
| gemma, recadrage + 4 bandes | 0,715 | 0,715 | 0,716 |
| qwen3-8-27b | 0,676 | 0,687 | 0,675 |
| qwen3-6-35b-moe | 0,663 | 0,664 | 0,658 |
| **Vote à trois lecteurs** | **0,745** | **0,747** | **0,743** |

- **Le codeur ne change rien** (écarts de 0,01 au plus, dans le bruit) : 0 à 2 % des
  désaccords viennent du codage, quel que soit le codeur.
- **Tout se joue à la lecture.** gemma en bandes reste le meilleur lecteur (0,715,
  12,6 % de fautes ratées contre 21 % pour qwen3-8-27b et 27 % pour qwen3-6-35b-moe).
- Le vote à trois lecteurs (≈ 0,745, biais −0,4 pt, 18,7 % de non-unanimes, 3,6 à
  3,8 % d'erreur résiduelle) reste la meilleure approche, quel que soit le codeur.
  Le choix du codeur peut se faire sur le coût ou la vitesse.

## Reste à faire

- Lancer la config de vote telle quelle sur 50 copies, pour confirmer le 0,739 hors
  recombinaison.
- Demander à la DEPP des scans en niveaux de gris, s'ils existent en amont.
