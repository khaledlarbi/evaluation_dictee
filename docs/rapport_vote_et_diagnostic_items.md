# Correction automatique de la dictée CM2 : vote à trois lecteurs et origine des erreurs

*15/11/2026, branche `feat/passe2-alignement`. Corpus 2015 complet : 3 413 copies,
282 860 items évaluables. Grille simplifiée 1 / erreur / 0.*

Ce rapport présente les deux meilleures approches sur le corpus complet. Il localise
ensuite les items sur lesquels elles hésitent, puis explique d'où viennent les erreurs
sur les élisions et les majuscules. Le détail de toutes les tentatives antérieures
figure dans [bilan_tentatives_lecture.md](bilan_tentatives_lecture.md) et
[pistes_amelioration_image.md](pistes_amelioration_image.md).

## En bref

- **Le vote à trois lecteurs reste la meilleure approche.** Sur tout le corpus, il
  obtient un kappa de 0,716 et un biais quasi nul (−0,3 point). Le lecteur gemma seul
  obtient 0,693.
- **Pour 81 % des items, les trois lecteurs sont d'accord, et ils ont alors raison
  dans 95 % des cas.** Renvoyer à un correcteur humain les 19 % d'items restants
  ramènerait l'erreur résiduelle à 3,8 %.
- **Les items sans unanimité ne sont pas répartis au hasard.** La moitié d'entre eux
  tombe sur 25 des 83 items de la dictée.
- **Pour les élisions et les majuscules, les erreurs viennent de la lecture de
  l'écriture.** Ni l'alignement ni le codage ne sont en cause. Trois erreurs reviennent
  en masse :
  - « n' » est lu « m' » (80 à 85 % des écarts sur ces items) ;
  - « S'ils » est lu « Ils », ce qui fait tomber deux items à la fois ;
  - « A ce moment » est lu « À ce moment », que le codeur compte comme une faute alors
    que l'expert l'accepte.

## 1. Vocabulaire

| Terme | Définition |
|---|---|
| **Copie** | Une dictée scannée d'un élève (une image). |
| **Item** | Un mot ou un signe de ponctuation du texte dicté. La dictée compte 83 items, chacun repéré par sa position (1 à 83) et sa forme attendue (« n' », « S' », « A »…). Les élisions forment un item distinct du mot qui suit : « S'ils » donne l'item 33 « S' » et l'item 34 « ils ». |
| **Code** | Le jugement porté sur un item, selon la grille simplifiée : **1** = correct ; **9** = erreur (lexicale, grammaticale ou les deux, regroupées) ; **0** = absent (mot oublié ou ponctuation manquante). |
| **Expert** | Le correcteur de la DEPP. Son code sert de référence. On ne dispose que d'une correction par copie : l'accord entre deux correcteurs humains n'est pas mesurable sur ce corpus. |
| **Item évaluable** | Un item que l'expert a codé 1, 9 ou 0. Les copies vierges et les items illisibles (code expert `i`) sont exclus de toutes les mesures. |
| **Lecteur** | Le modèle de vision qui transcrit la copie (passe 1). |
| **Codeur** | Le modèle de langage qui attribue un code à chaque item, à partir du texte transcrit seulement (passe 2). |
| **Lecture** | Ce qu'un lecteur a écrit pour un item donné, après alignement (par exemple « m' » là où l'on attend « n' »). Une lecture vide signifie qu'aucun mot transcrit n'a été rattaché à l'item. |

## 2. Le pipeline commun

Chaque copie passe par trois étapes.

1. **Transcription à l'aveugle (passe 1).** Le lecteur reçoit l'image et transcrit ce
   qui est écrit, sans le texte de la dictée, pour ne pas être tenté de corriger
   l'élève. La consigne demande de lire l'état final des ratures, de garder les
   retours à la ligne, de lire un rond sur le i comme un point, et de mettre une
   majuscule après un point.
2. **Alignement.** Un algorithme classique (Needleman-Wunsch, sans modèle de langage)
   met en face chaque mot attendu et ce que l'élève a écrit. Un mot oublié ou un mot en
   trop ne décale pas les items suivants.
3. **Codage (passe 2).** Le codeur reçoit les paires (attendu, écrit) et attribue un
   code 1 / 9 / 0 à chaque item.

Le codeur est qwen3-6-35b-moe. Les essais croisés sur 99 copies montrent que changer
de codeur ne change rien (écarts de kappa ≤ 0,01). Tout se joue à la lecture.

## 3. Les deux approches comparées

### Approche A : un seul lecteur, gemma sur la copie recadrée en 4 bandes

- **Lecteur** : gemma4-26b-moe.
- **Préparation de l'image** :
  - la copie est **recadrée** à l'intérieur du cadre pré-imprimé, ce qui supprime
    l'en-tête et le pied de page ;
  - elle est ensuite **découpée en 4 bandes horizontales**, en coupant dans les
    interlignes, au creux du profil d'encre.

  Les 4 bandes sont envoyées comme 4 images dans le même appel, avec la consigne de
  les lire à la suite.
- **Pourquoi les bandes** : gemma réduit toute image à environ 256 tokens, quelle que
  soit sa taille. La page entière est alors vue à environ 113 pixels par token, ce
  qui est trop grossier pour une écriture manuscrite. Chaque bande reçoit son propre
  budget de tokens : 4 bandes donnent 4 fois plus de détail.
- **Config** : `configs/scoring/dictee_two_stage_gemma_bandes.yaml`.

### Approche B : vote à trois lecteurs

- **Lecteurs** :
  - qwen3-8-27b (lecteur principal), sur la page entière ;
  - gemma4-26b-moe, recadré et en 4 bandes comme dans l'approche A ;
  - qwen3-6-35b-moe, sur la page entière.

  Les deux qwen voient déjà l'image en pleine résolution, et les découper ne leur a
  rien apporté.
- **Chaque lecture est codée séparément**, ce qui donne trois codes par item.
- **Règle du vote** : le code donné par au moins deux lecteurs l'emporte. Le lecteur
  principal ne tranche que si les trois codes diffèrent (0,6 % des items).
- **Confiance d'un item** : la part des lecteurs qui donnent le code retenu, soit
  3/3, 2/3 ou 1/3.
- **Config** : `configs/scoring/dictee_two_stage_vote.yaml`.

## 4. Indicateurs

Tous les indicateurs sont calculés sur les **items évaluables communs aux deux
approches**. Ils sont mis en commun sur l'ensemble des copies : chaque item compte
autant, quelle que soit la copie.

| Indicateur | Définition | Lecture |
|---|---|---|
| **Accord** | Part des items où le code prédit est égal au code expert. | Plus haut = mieux. |
| **Kappa** | Kappa de Cohen entre codes prédits et codes experts (1 / 9 / 0). C'est l'accord corrigé de ce que donnerait le hasard, à fréquences de codes égales : 1 = accord parfait, 0 = pas mieux que le hasard. | Indicateur principal. |
| **Kappa mots** / **Kappa ponct.** | Même kappa, restreint aux items qui sont des mots ou à ceux qui sont des signes de ponctuation. Ce sont deux tâches différentes : un mot se rate surtout par une faute non vue, un signe surtout par une absence non vue. | |
| **Biais** | (part des items que le modèle code 9 − part des items que l'expert code 9) × 100, en points. Un biais négatif signifie que le modèle trouve moins de fautes que l'expert ; un biais positif, qu'il en trouve plus. | Proche de 0 = le taux de faute moyen des élèves est bien estimé, même si les items en cause ne sont pas toujours les mêmes. |
| **Fautes ratées** | Parmi les mots que l'expert code 9, part que le modèle code 1. | C'est la sous-détection. Plus bas = mieux. |
| **Fautes inventées** | Parmi les mots que l'expert code 1, part que le modèle code 9. | C'est la sur-détection. Plus bas = mieux. |
| **Ponct. non vue** | Parmi les signes de ponctuation que l'expert code 1, part que le modèle code 0 (absent). | Plus bas = mieux. |
| **Unanimité** (vote seulement) | Part des items où les trois lecteurs donnent le même code (confiance 3/3). | |
| **Taux de renvoi** | Part des items qu'on enverrait à un correcteur humain, ici les items non unanimes. | Coût humain. |
| **Erreur résiduelle** | Part des items mal codés une fois les items renvoyés corrigés par l'humain, rapportée à tous les items. On suppose que l'humain renvoie le code expert. | Qualité finale. |

## 5. Résultats sur le corpus complet

| | Accord | Kappa | Kappa mots | Kappa ponct. | Biais | Fautes ratées | Fautes inventées | Ponct. non vue |
|---|---|---|---|---|---|---|---|---|
| A. gemma, recadrage + 4 bandes | 88,3 % | 0,693 | 0,697 | 0,586 | +2,3 pt | **14,4 %** | 8,9 % | 6,7 % |
| **B. Vote à trois lecteurs** | **89,8 %** | **0,716** | **0,718** | **0,626** | **−0,3 pt** | 19,6 % | **6,2 %** | **5,6 %** |

Les élèves font des fautes sur 19,6 % des items selon l'expert. gemma seul en compte
21,9 %, et le vote 19,3 %.

- **gemma seul rate moins de fautes, mais en invente davantage.** Il lit l'écriture de
  près, au point de prendre parfois une lettre mal formée pour une faute.
- **Le vote est meilleur sur tous les autres indicateurs.** Son taux de faute global
  est quasi exact.

**Confiance du vote, et ce qu'elle permet :**

| Lecteurs d'accord | Part des items | Codes justes |
|---|---|---|
| 3 sur 3 | 81,1 % | 95,3 % |
| 2 sur 3 | 18,4 % | 66,9 % |
| Tous différents | 0,6 % | 32,5 % |

La confiance est donc un bon signal. Renvoyer à l'humain tous les items non unanimes
reviendrait à renvoyer **18,9 % des items**, pour une **erreur résiduelle de 3,8 %**
(accord de 96,2 %).

Deux remarques :

- Les kappas du corpus complet sont 0,02 à 0,03 en dessous de ceux mesurés sur les
  échantillons de 50 à 100 copies. Les premières copies étaient un peu plus faciles
  que la moyenne.
- gemma n'a pas pu transcrire 11 copies (3 essais sans transcription exploitable).
  Ces copies manquent à l'approche A. Le vote les couvre, sauf une copie qui reste en
  échec.

## 6. Où se concentrent les items non unanimes

Les 19 % d'items non unanimes ne sont pas répartis au hasard sur les 83 items.
**La moitié d'entre eux tombe sur 25 items.** Ces items forment trois familles.

1. **Élisions et majuscules** : S' (33), n' (17 et 35), A (73), ce (74), ils (34),
   Aussitôt (67), Martine (59). Ce sont aussi les items les plus mal tranchés.
2. **Mots où les élèves font souvent des fautes** : verrons, retrouvé, inquiets,
   perdus, rentrés… Le taux de non-unanimité d'un item est corrélé à 0,60 avec son
   taux de faute selon l'expert. C'est attendu : quand l'élève hésite, sa graphie
   hésite aussi, et les lecteurs divergent.
3. **Le tiret** (item 22).

La première famille pose un problème à part : la langue française n'y est pas
difficile pour un élève de CM2, et pourtant les lecteurs s'y trompent. C'est l'objet
du diagnostic qui suit.

## 7. Diagnostic : élisions et majuscules

### 7.1 Question posée

Quand un lecteur s'écarte de l'expert sur ces items, la cause peut être de trois
sortes :

- **la lecture** : le lecteur a mal transcrit l'écriture ;
- **l'alignement** : la transcription est juste, mais l'item a été rattaché au
  mauvais mot, par exemple « S' » et « ils » fusionnés en un seul mot ;
- **le codage** : la lecture et l'alignement sont justes, mais le code attribué est
  faux.

### 7.2 Méthode

- **Unité de compte : une lecture.** Les trois lecteurs sont comptés séparément. Une
  copie donne donc trois lectures par item, soit environ 10 240 lectures par item sur
  le corpus.
- **Écart** : une lecture dont le code (celui donné par le codeur à cette lecture)
  diffère du code expert. Il y en a de deux sortes principales.
  - **Faux écart** : l'expert code 1 (correct), et le code donné à la lecture n'est
    pas 1. Le modèle voit une faute ou une absence là où il n'y en a pas.
  - **Faute ratée** : l'expert code 9 ou 0, et le code donné à la lecture est 1. Le
    modèle ne voit pas la faute.
- Chaque lecture en écart est **classée selon ce qui a été lu pour l'item**, comparé
  à la forme attendue :

| Classe | Ce qui a été lu | Origine probable |
|---|---|---|
| identique | Exactement la forme attendue. | Pour un faux écart, le codage (la lecture est juste, le code non). Pour une faute ratée, la lecture : le lecteur a transcrit la bonne forme là où l'élève a fait une faute. |
| majuscule seule | La forme attendue à la casse près (« Ils » pour « ils »). | Lecture |
| accent/casse | La forme attendue à l'accent près (« À » pour « A »). | Lecture ou règle de codage, selon le cas |
| apostrophe seule | La forme attendue sans l'apostrophe. | Lecture |
| fusion avec un voisin | Un même mot transcrit rattaché à deux items. | Alignement |
| vide | Aucun mot rattaché à l'item. | Lecture (mot non vu) |
| vide, mot en trop voisin | Aucun mot rattaché, mais un mot non aligné juste à côté. | Alignement possible |
| autre lecture | Une autre forme (« m' » pour « n' »). | Lecture |

- Pour les classes « vide » et « autre lecture », on a ensuite compté les **formes
  lues les plus fréquentes** (au moins 20 occurrences) et leur contexte immédiat.
- Seuls des agrégats ont été calculés : aucune transcription d'élève n'est sortie du
  SSP Cloud.

### 7.3 Résultat d'ensemble : ni l'alignement ni le codage ne sont en cause

Sur les 8 items étudiés (17, 33, 34, 35, 59, 67, 73, 74) :

| Type d'écart | Lectures | Répartition |
|---|---|---|
| Faux écart (expert 1) | 15 146 | autre lecture 62 %, accent/casse 16 %, majuscule seule 10 %, vide 9 %, vide avec mot en trop voisin 2 %, **fusion 0 %**, **identique 0 %** |
| Faute ratée (expert 9 ou 0) | 4 832 | **identique 100 %** |

- **Fusion 0 %** et mot en trop voisin 2 % : l'alignement ne joue presque aucun rôle.
- **Identique 0 % pour les faux écarts** : quand le lecteur lit juste, le codeur code
  juste.
- **Les fautes ratées sont toutes des lectures « identiques »** : le lecteur a écrit
  la forme correcte là où l'expert voit une faute. C'est une sur-correction à la
  lecture : le modèle « redresse » ce que l'élève a écrit.

### 7.4 Résultats par item

« Faux écarts » est la part des lectures d'items que l'expert code correct.

| Pos. | Attendu | Faux écarts | Fautes ratées | Ce que les lecteurs écrivent à la place (part des faux écarts) |
|---|---|---|---|---|
| 17 | n' (« garçons n'étaient ») | 18,2 % | 221 | **« m' » 85 %**, vide 6 % |
| 33 | S' (« S'ils ») | 18,0 % | 389 | **vide 48 %**, dont « . ∅ Ils n' » 35 % ; s' 14 %, L' 7 %, Y' 6 %, Si 6 %, J' 6 % |
| 34 | ils (« S'ils ») | 11,0 % | 366 | **« Ils » 59 %**, dont « ∅ Ils n' » 49 % |
| 35 | n' (« ils n'ont ») | 15,8 % | 520 | **« m' » 80 %**, vide 9 % |
| 59 | Martine | 14,0 % | 160 | autres lectures du nom propre 94 % |
| 67 | Aussitôt | 18,7 % | 471 | majuscule seule 21 %, accent 12 %, autres lectures 64 % |
| 73 | A (« A ce moment ») | **49,6 %** | 123 | **« À » 39 %**, « Et » 24 %, « et » 13 %, « à » 4 %, « Au » 3 % |
| 74 | ce (« A ce moment ») | 2,4 % | **2 582** | faux écarts rares ; mais 2 582 fautes ratées, toutes lues « ce » |

### 7.5 Interprétation

**1. « n' » est lu « m' » : la plus grosse source d'erreur de cette famille.**

- En cursive, un n se lit comme un m : un jambage de trop suffit.
- Sur les deux items 17 et 35, cela représente **environ 2 900 lectures erronées**.
- Le contexte rend pourtant « m' » impossible ici : « garçons m'étaient » n'a pas de
  sens, et « ils m'ont » ne convient pas à la suite de la phrase.

**2. « S'ils » est lu « Ils ».**

- Le lecteur ne voit pas le « S' » en début de phrase et met la majuscule sur « ils ».
- **Une seule erreur de lecture fait ainsi tomber deux items** : 33 est jugé absent,
  et 34 reçoit une majuscule fautive.

**3. « A ce moment » : surtout une question de règle de codage.**

- Le texte de référence écrit « A ce moment », sans accent sur la majuscule.
- « À » est pourtant l'orthographe correcte aujourd'hui, et **l'expert l'accepte** : il
  code « correct » quand les lecteurs lisent « À ».
- La grille ne prévoit pas ce cas. Elle ne liste que « a » et « à » comme fautes. Le
  codeur compte donc « À » comme une faute.
- C'est ce qui fait de cet item le plus mal tranché de la dictée (49,6 % de faux
  écarts).
- Les lectures « Et » / « et » (37 % des faux écarts), elles, sont de vraies erreurs
  de lecture du A cursif.

**4. « ce » : le problème inverse.**

- Les lecteurs lisent « ce » alors que l'expert voit une faute, que la grille donne
  comme « se ». C'est une sur-correction à la lecture : le contexte « À ce moment »
  pousse le modèle vers la forme attendue.
- Ce cas ne se corrige ni par l'alignement ni par le codage.

**5. Martine et Aussitôt** : ce sont des lectures variées du mot, sans erreur
dominante. Aucune correction ciblée ne s'impose.

## 8. Pistes de correction

Les deux premières pistes ont été testées le 15/11/2026 (section 9).

| Piste | Ce qu'elle vise | Comment la tester | Point d'attention |
|---|---|---|---|
| Accepter « À » comme correct sur l'item 73 | ≈ 39 % des faux écarts de l'item 73 | Recoder les fichiers existants, sans nouvel appel aux modèles | Modifie la grille : décision méthodologique, à valider avec la DEPP |
| Règle de lecture n' / m' dans la consigne de passe 1 (compter les jambages : deux pour un n, trois pour un m) | ≈ 2 900 lectures sur les items 17 et 35 | 100 copies, trois lecteurs ; comparer ces items et le kappa global | La règle doit rester générale : la lecture se fait à l'aveugle, sans le texte de la dictée |
| Règle de lecture sur la lettre élidée en début de phrase (« S'… », « L'… ») | Items 33 et 34 | Même test | Même contrainte |
| Seuils de renvoi propres à chaque item | Renvoyer davantage sur les items connus pour être difficiles, moins sur les autres | Recalcul sur les fichiers existants | — |

Pour « ce » (item 74), aucune piste simple ne se dégage. Ajouter une règle de lecture
risquerait d'augmenter les fautes inventées ailleurs.

### Piste à explorer : remplacer le vote par une agrégation apprise par item

*Notée le 15/11/2026, non mise en œuvre.*

Le vote à la majorité traite tous les items de la même façon. Or la section 7 montre
que les erreurs des lecteurs dépendent de l'item : sur « n' », un 9 venu d'une
lecture « m' » est presque toujours un faux écart ; sur « ce », un 1 unanime cache
souvent une faute. Le vote ne peut pas en tenir compte, un modèle appris le peut.

- **Idée** : pour chaque item, apprendre un modèle (par exemple une régression
  logistique multinomiale) qui donne la probabilité que l'item soit correct (1),
  fautif (9) ou absent (0), à partir des trois codes des lecteurs.
- **Autres variables possibles** :
  - la lecture elle-même, ou sa classe (« m' », vide, majuscule seule…) ;
  - les codes des items voisins (« S' » et « ils » tombent ensemble) ;
  - des indicateurs de la copie : taux de non-unanimité sur toute la copie, mots non
    alignés, échec d'un lecteur.
- **Structure** : un modèle par item risque d'être instable sur les items rares.
  Un modèle commun avec un effet par item (ou une régularisation vers le modèle
  commun) est plus sûr.
- **Ce que cela apporte en plus du code** : des probabilités calibrées. Le renvoi à
  l'humain se ferait sur la probabilité d'erreur, et non plus sur la seule
  unanimité, ce qui rejoint la piste des seuils de renvoi propres à chaque item.
- **Comment le tester** : sur les fichiers existants, sans nouvel appel aux modèles.
  Validation croisée par copie (une copie entière dans le même pli). Comparer au vote
  le kappa, le biais et l'erreur résiduelle à taux de renvoi égal.
- **Réutilisation pour la dictée 2025** : les modèles appris sur 2015 corrigeraient
  directement 2025.
- **Variante sans code expert** : le modèle de Dawid & Skene, déjà prévu dans la
  méthodologie, estime la fiabilité de chaque lecteur par item à partir de leurs
  seuls désaccords. Il pourrait être réajusté sur 2025 sans nouvelle correction
  humaine, et servir de point de comparaison au modèle supervisé.
- **Points d'attention** :
  - les modèles par item ne se transfèrent que si le texte dicté est le même. Sinon,
    il faut des variables qui décrivent l'item (élision, majuscule, ponctuation,
    fréquence du mot) plutôt que son identité ;
  - les scans, la cohorte et les versions des lecteurs peuvent changer d'une année à
    l'autre. Il faudra vérifier la calibration sur un échantillon de 2025 corrigé à la
    main ; le corpus 2021 peut servir de premier test de transfert ;
  - le modèle apprend à reproduire l'expert 2015, y compris ses choix sur les cas
    ambigus (« À » sur l'item 73).

## 9. Résultats des tests (15/11/2026)

### 9.1 Accepter « À » sur l'item 73 : gain net, mais la décision revient à la DEPP

Le test recode les fichiers existants du corpus complet, sans nouvel appel aux
modèles. Une lecture « À » de l'item 73 reçoit le code 1 ; les autres lectures gardent
leur code. Le vote est ensuite recalculé. Cela touche 2 221 lectures du vote et 935
items de gemma seul.

| | Kappa | Accord item 73 | Biais | Fautes inventées | Erreur résiduelle (renvoi 18,9 %) |
|---|---|---|---|---|---|
| Vote, grille actuelle | 0,716 | 47,5 % | −0,3 pt | 6,2 % | 3,81 % |
| **Vote, « À » accepté** | **0,721** | **67,0 %** | −0,6 pt | **5,8 %** | **3,64 %** |
| gemma seul, grille actuelle | 0,693 | 40,2 % | +2,3 pt | 8,9 % | — |
| gemma seul, « À » accepté | 0,699 | 62,7 % | +1,9 pt | 8,4 % | — |

- L'item 73 passe de 47,5 % à 67 % d'accord. Le kappa global gagne 0,005, ce qui
  correspond à un item sur 83.
- Ce n'est **pas encore appliqué** dans la grille ni dans le code. C'est une
  décision méthodologique, à valider avec la DEPP.
- Une fois la décision prise, le plus simple serait d'ajouter « À » aux formes
  acceptées de l'item dans `configs/grille_dictee_2015.json`.

### 9.2 Règles de lecture n' / m' et lettre élidée : aucun effet, non retenues

Deux règles ont été ajoutées à la consigne de passe 1 (option
`prompt.regles_elisions`, config `dictee_two_stage_vote_elisions.yaml`) :

- **Règle 10** : compter les jambages pour distinguer n' de m'. Devant un verbe,
  n' est plus fréquent que m'.
- **Règle 11** : une lettre suivie d'une apostrophe est un mot à part entière, à ne
  pas oublier.

Le test porte sur 100 copies (8 300 items), avec trois lecteurs. La comparaison se fait
avec deux runs de référence sur les mêmes copies : le run complet, et la recombinaison
des runs de croisement. L'écart entre ces deux références donne le bruit d'un run à
l'autre, soit 0,006 de kappa pour le vote.

| | Vote | qwen3-8-27b | gemma b4 | qwen3-6 |
|---|---|---|---|---|
| Référence (run complet) | 0,740 | 0,672 | 0,716 | 0,652 |
| Référence bis (croisements) | 0,746 | 0,679 | 0,717 | 0,664 |
| Règles d'élision | 0,739 | 0,686 | **0,694** | 0,659 |

Part de chaque forme parmi toutes les lectures, les trois lecteurs cumulés :

| Item | Forme | Référence | Règles d'élision |
|---|---|---|---|
| 17 « n' » | lu « m' » | 19 % | 21 % |
| 35 « n' » | lu « m' » | 15 % | 14 % |
| 33 « S' » | vide | 14 % | 12 % |
| 34 « ils » | lu « Ils » | 8 % | 7 % |

- **Les lecteurs n'appliquent pas les règles.** La confusion n / m et l'oubli du S'
  restent au même niveau.
- **Le vote ne bouge pas** : 0,739, contre 0,740 et 0,746 pour les références.
- **gemma perd 0,022**, surtout sur la ponctuation (0,621 → 0,587). Une consigne plus
  longue semble le distraire. Les deux qwen varient dans le bruit.
- **Conclusion** : l'option reste dans le code, désactivée par défaut, mais n'est pas
  retenue. La confusion n / m relève de ce que le modèle voit, pas de ce qu'on lui
  demande. Des scans en niveaux de gris seraient la piste la plus directe.

## 10. Limites

- **Le code expert est l'unique référence.** Faute de double correction, on ne sait
  pas quelle part des écarts relève d'une hésitation légitime entre correcteurs humains.
  Le fichier `correction_dictee_2021.csv` ne contient lui aussi qu'une correction par
  copie.
- **Le diagnostic porte sur le corpus 2015 et sur le texte de cette dictée.** Les
  confusions n / m et la lettre élidée oubliée devraient se retrouver ailleurs ; le cas
  « A » / « À » est propre à ce texte.
- **Les scans sont binarisés en noir et blanc (1 bit).** Cela rend les accents et les
  jambages plus difficiles à distinguer. Des scans en niveaux de gris, s'ils existent,
  aideraient directement les erreurs décrites ici.
