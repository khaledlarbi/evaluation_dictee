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

## Ce qui est envoyé aujourd'hui

La page entière telle quelle : TIFF 1 bit de 1594 × 2044 pixels, converti en
niveaux de gris puis encodé en PNG (`data/loaders.py`, `models/vlm.py`). Il n'y a ni
recadrage, ni découpage, ni prétraitement.

La résolution que le modèle voit réellement est fixée par le serveur (llm.lab /
vLLM), selon les limites de chaque modèle : une page de 3,3 millions de pixels est
probablement réduite avant lecture. Les accents, les points et les virgules ne
font alors plus que quelques pixels. C'est une hypothèse, non vérifiée.

## Pistes, par ordre de priorité

1. **Mesurer la résolution vue par chaque modèle.** Le nombre de tokens d'entrée
   d'un appel (Langfuse ou champ `usage.prompt_tokens`) donne le nombre de tokens
   de l'image, donc sa taille après réduction. Mesure gratuite, qui dit si les
   pistes suivantes valent le coup.
2. **Recadrer sur la zone d'écriture.** Retirer les marges, le titre, le cadre et
   le pied de page pré-imprimés, en délimitant la zone d'encre manuscrite. Les
   pixels envoyés portent alors plus d'écriture. Sans appel supplémentaire et sans
   risque pour l'alignement.
3. **Découper la copie en bandes de quelques lignes** (probablement le gain le plus
   fort). Repérer les interlignes (profil de projection horizontal : rangées
   presque sans encre) et couper à ces endroits, sans chevauchement, en 3 ou 4
   bandes. Le modèle lit chaque bande séparément, puis on recolle les
   transcriptions dans l'ordre ; l'alignement local fait le reste. Chaque bande est
   vue à une résolution 3 à 4 fois plus grande.
   - Coût : 3 ou 4 appels de passe 1 par copie au lieu d'un, chacun plus court.
   - Risque : couper un mot qui déborde sur la ligne voisine ; couper au creux de
     l'interligne le limite.
4. **Rendre le trait plus lisible pour le modèle** (à tester en A/B). Le scan 1 bit
   a des bords en escalier, très différents des images d'entraînement des modèles.
   Un léger flou (adoucir les bords) et un léger épaississement des traits fins
   (accents, points) peuvent aider. Gain incertain, essai presque gratuit.
5. **Demander à la DEPP des scans en niveaux de gris**, s'ils existent en amont. La
   binarisation 1 bit a déjà effacé une partie des accents légers ; aucun
   prétraitement ne les récupérera.

## Protocole de test proposé

- Chaque piste devient une option de config, **désactivée par défaut**.
- Comparaison sur les **50 mêmes copies**, avec qwen3-6-35b-moe et gemma4-26b-moe,
  d'abord sur 10 copies pour vérifier que tout marche.
- Indicateurs à suivre : kappa global, kappa ponctuation, et surtout la part des
  désaccords due à la lecture (ligne « Origine des désaccords » du log). Si l'image
  aide, c'est cette part qui doit baisser.

## Points de référence (50 copies, 4 150 items)

| Variante | Accord | Kappa | Ponct. | Non détectées | Biais |
|---|---|---|---|---|---|
| qwen3-6-35b-moe seul | 87,3 % | 0,650 | 0,600 | 28,0 % | −2,0 pt |
| gemma4-26b-moe seul | 84,5 % | 0,629 | 0,445 | 12,6 % | +5,1 pt |
| Vote à trois (qwen, gemma, qwen à T = 0,7) | 89,1 % | 0,703 | 0,611 | 22,6 % | −1,3 pt |
