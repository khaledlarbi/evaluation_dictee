# Pistes d'amélioration de l'image envoyée en passe 1

*Rédigé le 13/11/2026, branche `feat/passe2-alignement`.*

## Pourquoi l'image

Sur 50 copies (alignement local, codage des paires alignées), 98 à 100 % des
désaccords avec l'expert viennent de la **lecture**, presque aucun du **codage** :
le LLM code ce qu'il a lu dans 99,8 % des items (qwen3-6-35b-moe) et 100 %
(gemma4-26b-moe). La marge de progrès est donc dans ce que le modèle voit.

Une réserve : ce diagnostic range dans « lecture » trois causes qu'il ne distingue
pas. Le modèle a mal lu un mot, l'alignement s'est trompé d'appariement (cas qui
devrait être rare), ou la grille accepte une forme que la lecture distingue.

## Ce qui était envoyé

La page entière telle quelle : TIFF 1 bit de 1594 × 2044 pixels, converti en
niveaux de gris puis encodé en PNG (`data/loaders.py`, `models/vlm.py`).

## Résolution vue par chaque modèle (mesurée)

Mesure indirecte : nombre de tokens d'image (`usage.prompt_tokens` d'un appel à
1 token, moins le prompt seul) sur des images synthétiques, sans donnée d'élève.
Le « côté par token » suppose des tokens carrés. Repère : l'interligne de la copie
fait environ 68 px.

| Lecteur, image envoyée | Tokens d'image | Côté par token | Tokens par ligne |
|---|---|---|---|
| qwen3-8-27b / qwen3-6-35b-moe, page entière | ≈ 3 200 | 32 px (pleine résolution) | ≈ 2,1 |
| qwen, recadrée + ×2 | ≈ 7 100 | 16 px | ≈ 4,3 |
| gemma4-26b-moe, page entière | ≈ 254, quelle que soit la taille | ≈ 113 px | ≈ 0,6 |
| gemma, recadrée + 4 bandes | ≈ 4 × 263 | ≈ 42 px | ≈ 1,6 |
| gemma, recadrée + 6 bandes | ≈ 6 × 247 | ≈ 34 px | ≈ 2,0 |

- **Qwen** (résolution dynamique) découpe l'image d'origine en carrés de 32 px,
  sans plafond (12 800 tokens à 4 fois la taille de la page) : rien n'est réduit.
- **Gemma** condense toute image en environ 256 tokens. Le budget n'est pas
  réglable par requête sur llm.lab (`mm_processor_kwargs` refusé). Sur la page
  entière, une ligne tenait dans moins d'un token : points, virgules et accents
  disparaissaient, d'où une ponctuation à 0,445 et un biais de +5 points.
- Le serveur accepte plusieurs images par appel (8 testées) ; chacune reçoit son
  propre budget. Les bandes sont donc envoyées dans un même appel, avec une
  consigne de lecture dans l'ordre : une seule transcription, rien à recoller.

## Ce qui est implémenté

Option `image` propre à chaque lecteur (`ModelConfig.image`, `data/pretraitement.py`),
désactivée par défaut, reflétée dans le nom du run (`-crop`, `-b4`, `-x2`) :

- `recadrage` : intérieur du cadre pré-imprimé, repéré comme la plus longue suite
  d'encre continue (union de 5 rangées, pour tolérer un trait incliné ou interrompu).
  Trouvé sur 50 copies sur 50, zone stable de ≈ 1 185 × 1 540 px. Hors du cadre,
  l'encre est identique d'une copie à l'autre (pré-imprimé).
- `n_bandes` : coupes au creux du profil horizontal, autour des positions
  régulières. 90 % des rangées de coupe ne touchent aucun pixel d'encre, 1 % en
  touchent plus de 20 (sur 1 185).
- `agrandissement` : facteur Lanczos.

## Résultats (50 copies, 4 150 items, codage qwen3-6-35b-moe)

| Lecteur, image | Accord | Kappa | Ponct. | Non détectées | Biais |
|---|---|---|---|---|---|
| gemma, page entière | 84,5 % | 0,629 | 0,445 | 12,6 % | +5,1 pt |
| gemma, 4 bandes sans recadrage | 88,5 % | 0,709 | 0,587 | 12,6 % | +3,0 pt |
| **gemma, recadrage + 4 bandes** | **89,1 %** | **0,722** | 0,570 | **11,8 %** | **+2,4 pt** |
| gemma, recadrage + 6 bandes | 88,7 % | 0,713 | 0,557 | 12,6 % | +2,4 pt |
| qwen3-8-27b, page entière | 88,2 % | 0,687 | 0,571 | 20,8 % | +0,5 pt |
| qwen3-8-27b, ×2 | 87,4 % | 0,669 | 0,563 | 21,1 % | +0,7 pt |
| qwen3-8-27b, recadrage + ×2 | 88,0 % | 0,688 | 0,541 | 17,6 % | +1,1 pt |

Lecture :

- **Les bandes font gagner 0,09 de kappa à gemma**, qui devient le meilleur lecteur
  seul. Le gain plafonne dès 4 bandes : 6 bandes (34 px par token) ne font pas mieux.
  Au-delà de ≈ 42 px par token, gemma n'est plus limité par la résolution.
- Le recadrage ajoute peu (+0,013 avec 4 bandes, dans le bruit) mais ne nuit pas.
- **Qwen ne gagne rien** à l'agrandissement : il voit déjà la page en pleine
  résolution. Ses fautes non détectées (≈ 21 %) relèvent de la sur-correction
  (lecture du mot attendu), pas d'un manque de pixels.
- Retenu : gemma en recadrage + 4 bandes, les deux qwen sur la page telle quelle
  (`configs/scoring/dictee_two_stage_vote.yaml`).

## Pistes restantes

1. **Demander à la DEPP des scans en niveaux de gris**, s'ils existent en amont. La
   binarisation 1 bit a déjà effacé une partie des accents légers.

## Essais sans gain (30 copies, 2 490 items, 13/11/2026)

### Deuxième lecture de gemma, coupes décalées d'une demi-bande

`image: {recadrage: true, n_bandes: 4, decalage_bandes: 0.5}`, à la place de
qwen3-6-35b-moe dans le vote.

| Vote (qwen3-8-27b, gemma 4 bandes, + …) | Kappa | Ponct. | Non détectées | Biais | Renvoi des non-unanimes | Erreur résiduelle |
|---|---|---|---|---|---|---|
| qwen3-6-35b-moe | 0,754 | 0,648 | 16,3 % | +0,5 pt | 17,9 % | 3,5 % |
| gemma décalé | 0,759 | 0,620 | 12,9 % | +1,6 pt | 15,3 % | 3,9 % |

Écart dans le bruit. La lecture décalée seule vaut la première (0,725 contre 0,736),
mais elle commet 73 % des erreurs de gemma (contre 57 % pour qwen3-6-35b-moe) :
deux lectures d'un même modèle se trompent ensemble, le vote y gagne peu.
**On garde qwen3-6-35b-moe.**

### Relecture en zoom des items non unanimes

`pipeline/relecture_zoom.py`, `scripts/relecture_zoom.py` : pour chaque item où les
trois lecteurs divergent (17,9 % des items), une bande de ≈ 4 lignes autour du mot
(position déduite de sa ligne dans la transcription du lecteur principal, rapportée
à l'étendue de l'écriture), une question ciblée (« recopie ce que l'élève a écrit
entre … et … »), puis le codage habituel.

- Premier essai : position rapportée à toute la hauteur du cadre. Passage trouvé
  pour 100 % des items du premier quart, 6 % du dernier : l'écriture n'occupe en
  médiane que 50 % du cadre (25 à 93 %). Corrigé : 99 % trouvés.
- Sur les items relus, la relecture est **moins juste que le vote** (63 % contre
  68 % pour gemma ; qwen3-8-27b ×2 encore en dessous au premier essai) et
  invente des fautes (biais +4 pt quand elle tranche) : isolé de son contexte, un
  mot paraît plus fautif.

| Usage de la relecture (gemma) | Kappa | Non détectées | Biais |
|---|---|---|---|
| Aucun (vote à trois) | 0,754 | 16,3 % | +0,5 pt |
| Tranche les non-unanimes | 0,744 | 10,8 % | +4,1 pt |
| Quatrième voix | 0,758 | 10,8 % | +3,5 pt |

Pour le renvoi humain, ne renvoyer que les non-unanimes où la relecture contredit
la majorité donne 7,4 % de renvoi pour 5,7 % d'erreur résiduelle, contre 12,2 % et
4,8 % (renvoi des seuls désaccords qwen3-8-27b/gemma) ou 17,9 % et 3,5 % pour le
vote seul : un point un peu meilleur que l'interpolation entre ces deux-là
(≈ 6,6 %), pour ≈ 15 appels de plus par copie. **Non retenu en l'état.**

### Épaississement du trait de 1 px (49 copies, 4 067 items)

`image: {epaississement: 1}` (dilatation de l'encre après découpage), sur chacun
des trois lecteurs de la meilleure config. Une copie de qwen3-8-27b bloquée (boucle
de génération) : comparaison sur les 49 copies communes.

| Lecteur | Kappa sans → avec | Ponct. sans → avec | Ponctuation correcte non vue |
|---|---|---|---|
| qwen3-8-27b | 0,690 → 0,618 | 0,577 → 0,457 | 7,9 % → 11,6 % |
| gemma, recadrage + 4 bandes | 0,725 → 0,631 | 0,594 → 0,434 | 7,2 % → 14,5 % |
| qwen3-6-35b-moe | 0,653 → 0,608 | 0,616 → 0,538 | 6,8 % → 10,2 % |
| Vote à trois | 0,747 → 0,686 | 0,655 → 0,539 | 6,1 % → 10,3 % |

Tous les lecteurs perdent, la ponctuation la première, à l'inverse de l'effet
attendu : épaissi, un point se colle aux lettres voisines et les boucles des
lettres (e, a, o) se referment. **Non retenu** ; l'option reste à 0 par défaut.

## Points de référence du vote (50 copies, 4 150 items)

| Variante | Accord | Kappa | Ponct. | Non détectées | Biais |
|---|---|---|---|---|---|
| qwen3-6-35b-moe seul | 87,3 % | 0,650 | 0,600 | 28,0 % | −2,0 pt |
| gemma4-26b-moe seul | 84,5 % | 0,629 | 0,445 | 12,6 % | +5,1 pt |
| qwen3-8-27b seul (codage qwen3-6-35b-moe) | 88,2 % | 0,687 | 0,571 | 20,8 % | +0,5 pt |
| Vote à trois (qwen, gemma, qwen à T = 0,7) | 89,1 % | 0,703 | 0,611 | 22,6 % | −1,3 pt |
| Vote à trois (qwen3-6-35b-moe, gemma, qwen3-8-27b) | 89,8 % | 0,728 | 0,591 | 17,1 % | +0,0 pt |
| **Vote à trois, gemma en recadrage + 4 bandes** (recombinaison) | **90,1 %** | **0,739** | **0,612** | **15,9 %** | **+0,1 pt** |

Avec gemma en bandes, qwen3-8-27b et gemma divergent sur 10,3 % des items (contre
15,5 %) : renvoyer ces items à l'humain laisse 6,2 % d'erreur résiduelle (contre 5,7 %).
