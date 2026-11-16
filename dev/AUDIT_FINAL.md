# Audit final — 15 novembre 2026

La chaîne complète a été exécutée depuis une copie sans résultats préexistants. Les résultats et figures régénérés sont présents dans ce dossier. Les corrections suivantes ont été appliquées.

## Corrections ayant un effet numérique

**Profils des capteurs, runs 5 et 6.** Certaines fenêtres de borate dépassaient la grille extraite. L’interpolation répétait alors la valeur du bord, ce qui déformait les profils utilisés dans la PCA. Les grilles sont élargies aux données effectivement mesurées et un contrôle interdit désormais ce dépassement. Les 684 acquisitions brutes couvrent les fenêtres requises. Le calcul des pressions à partir des ajustements des raies n’était pas concerné.

Les cinq intervalles de rupture et leur classement restent identiques :

| Run | Intervalle (°C) | Probabilité maximale avant → après |
| --- | --- | --- |
| 2 | 262–286 | 0,025 → 0,026 |
| 3 | 270–277 | 0,111 → 0,092 |
| 4 | 256–262 | 0,001 → 0,001 |
| 5 | 313–332 | 0,001 → 0,003 |
| 6 | 300–306 | 0,001 → 0,001 |

Le run 3 reste un candidat au seuil retenu de 0,05. Les cinq familles, leurs poids et la méthode de rupture sont conservés.

**IR.** Conformément à votre confirmation, ε4500 = 0,44 ± 0,03 est désormais lu dans le CSV, au lieu de l’incertitude 0,01 inscrite dans le programme. Les teneurs moyennes restent identiques ; leurs incertitudes augmentent :

| Échantillon | Eau totale (% massique) | Incertitude avant → après (1σ) |
| --- | --- | --- |
| ANN 3 | 2,9719 | 0,1599 → 0,1811 |
| ANN 5 | 4,8014 | 0,2030 → 0,2224 |
| ANN 8 | 8,3085 | 1,0886 → 1,0935 |

Les coefficients et paramètres utilisés sont enregistrés dans `results/IR/IR_metadata.json`.

## Cohérence et fonctionnement

- Référence NBO/T harmonisée à **0,59**, conformément à votre confirmation ; les valeurs calculées de la figure 8 restent inchangées.
- Notes actives et inventaire des figures réécrits : figures principales 2–4 et 6–10 ; DSC 5 fournie séparément ; suppléments S1–S4 et S6–S11. L’ancien texte d’analyse est conservé dans `notes/` avec une mention historique.
- Correction de l’exécution avec un chemin Python contenant des espaces, de la création du dossier de sortie pour la viscosité seule et du chargement des spectres ambiants à une colonne. Suppression de fonctions auxiliaires inutilisées et de commentaires périmés.
- `make test` exécute aussi l’IR. Des contrôles supplémentaires couvrent les fenêtres des capteurs, les résultats enregistrés, la propagation des incertitudes IR et la cohérence des paramètres.

## Vérifications et limites

- **94 tests réussis** après reproduction complète sous Python 3.13.5 avec les dépendances verrouillées.
- **931 empreintes de fichiers d’entrée inchangées** ; aucune modification des mesures brutes.
- **18 figures du manuscrit**, plus deux diagnostics IR et un diagnostic de centroïde : les **21 PDF** ont été rendus et inspectés visuellement. Aucun défaut de mise en page repéré.
- Fichiers JSON contrôlés, sans valeurs numériques non finies. Figures et résultats régénérés sous les noms courants. Le dossier `test_rutpures` reste supprimé.

La compatibilité n’a pas été exécutée dans un environnement Python 3.12 neuf. Cet audit vérifie le fonctionnement et la cohérence interne ; il ne constitue pas une validation indépendante des calibrations scientifiques ni des hypothèses physiques.

Références : [méthodes et paramètres](../ANALYSIS.md), [inventaire des figures](../FIGURES.md), [historique de validation](VALIDATION.md).
