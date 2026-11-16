# Numérotation courante — 15 novembre 2026

Après insertion de la figure DSC en position 5, les anciennes figures
principales 5–9 deviennent 6–10. Les figures supplémentaires ne changent pas.
Les comptes rendus datés ci-dessous conservent leur numérotation historique ;
la table actuelle est dans [FIGURES.md](../FIGURES.md). La validation de cette
migration figure à la fin du présent fichier.

# Validation après réparation — 9 novembre 2026

Cause : le déplacement des fonctions avait transformé le chemin peak_fits.h5 en peak_fih5 dans tg_spectra_ruptures.py. Les familles Qn étaient donc absentes et manuscript_data échouait lors de leur accès. Chemin corrigé, test de non-régression du chemin par défaut ajouté. Une erreur explicite remplace désormais le KeyError si les observables d’un run sont absentes.

Validation exécutée dans for_publication avec Python ../.venv/bin/python :
- make figures : succès, code retour 0 ; 14 PDF et 14 PNG produits.
- python -m src.centroid_lf : succès.
- python -m unittest discover -s tests -v : 68 tests passent.
- Les cinq transitions (bornes, p-values, statuts et métadonnées) sont exactement identiques au JSON du projet original.
- JSON des figures 6 et 9 exactement identiques aux références actuelles.
- Figure 5 : mêmes bornes et p-values ; écart absolu de 2.46e-6 sur le gain ACP (et le gain total), autres familles identiques. L’ACP automatique peut varier numériquement entre exécutions ; aucune modification de solveur dans cette réparation.
- 931 fichiers d’entrée vérifiés avec INPUTS.sha256, tous intacts.

Les tests de figure 9 utilisent désormais les fonctions publiées et des valeurs numériques de référence ; aucun import ni fichier test_fig9_models externe. make test régénère ses prérequis figures et diagnostics de centroïde afin que les tests d’exports ne dépendent pas de fichiers absents.

Pas de modification des méthodes ni des acquisitions du projet original.


# Simplification de temperature.py — 9 novembre 2026

Périmètre : for_publication uniquement. Le module conserve le calcul des offsets par acquisition pour l’histogramme de la figure 2d, les positions médianes des grains et les statistiques publiées. Suppression des régressions par run, sensibilités, comparaisons de quatre modèles, profils, AICc, graphiques diagnostiques et sorties associées. Méthodes de pression et solveur de racines inchangés. Documentation et commentaires en français.

Sorties : results/temperature_steps.csv (run, step, T_C, ruby_nm, borate_nm, dT_step_K) et results/temperature_summary.json. summary reçoit désormais uniquement la liste des acquisitions ; analyse_runs retourne cette liste. Les anciennes options CLI d’analyses exploratoires sont retirées ; --help décrit le nouveau périmètre. Le lanceur et le Makefile n’ont pas besoin de changer.

Résultats : 62 offsets finis ; moyenne 18.75506607796965 K ; médiane 17.15357158643019 K ; intervalle empirique central à 68.268949 % : [-8.648270057511494, 45.062901344211504] K. Les bornes sont les quantiles Phi(-1)/Phi(+1), pas un intervalle de confiance d’un estimateur. Pas de correction appliquée aux températures ou pressions publiées.

Validation :
- Comparaison avant/après des 62 lignes : identifiants, positions, températures et offsets exactement identiques.
- Huit tests température : quantiles, valeurs manquantes, CSV/JSON, valeurs publiées, calibration à 296 K, limites/absence de racines, choix de racine et médianes des grains.
- MPLCONFIGDIR=/tmp/pub-mpl make test PYTHON=../.venv/bin/python exécuté depuis for_publication : code retour 0 ; chaîne complète régénérée, 14 PNG et 14 PDF, diagnostics de centroïde et 74 tests réussis.
- Figure 2 PNG identique pixel par pixel à la version avant simplification.
- Aucun ancien fichier temperature_models_*, temperature_runs.csv, temperature.h5 ni figures/temperature n’est recréé. Anciennes sorties copiées temporairement sous /tmp/temperature-simplification-before avant leur retrait du paquet.
- README.md et ANALYSIS.md mis à jour ; code original hors for_publication préservé.


# Suppression de run_overview.py — copie publication

Les fonctions graphiques state_bands et style, ainsi que les constantes BANDS et TICKS, sont déplacées dans centroid_lf.py, leur seul consommateur. Docstrings en français. Aucun calcul scientifique modifié. Le script run_overview.py et son cache sont supprimés de for_publication ; tableau du README actualisé. La version originale hors for_publication reste intacte.

Validation : python -m src.centroid_lf réussit ; 13 PNG diagnostiques sont identiques pixel par pixel aux fichiers précédents, JSON inchangé. La suite complète unittest passe : 74 tests. Aucune référence résiduelle au module supprimé dans les sources, tests ou README.


# Simplification de qn_pressure_step — 9 novembre 2026

Modifications limitées à for_publication. Le module ne conserve que compare_pressure_gls et ses dépendances NumPy/SciPy. Suppression des analyses indépendantes Q2/Q3, modèles de mélange, assemblage des données, tracé diagnostique et main autonome. Interface et calculs GLS utilisés par la figure 7 conservés. Docstrings/commentaires français et mise en forme lisible. README, ANALYSIS et FIGURES actualisés.

Les tests des fonctions supprimées sont remplacés par cinq tests du GLS : solutions analytiques à covariance diagonale, covariance additive commune, relation linéaire connue, constante sans effet, entrées invalides/rang insuffisant/covariance singulière.

Validation : figure 7 régénérée avec succès et inspectée visuellement ; JSON strictement identique à la version antérieure, dont les tests de pression. Suite complète : 74 tests réussis ; les cinq tests ciblés repassent après mise en forme finale. Le PNG régénéré par invocation directe diffère pixel par pixel de celui du pipeline complet ; aucune identité d’image n’est revendiquée. Aucun changement du code de tracé. Pas de modification des calibrations ni des données ; code original hors for_publication préservé.


# Déplacement du GLS dans raman_treatment — 9 novembre 2026

Dans for_publication, compare_pressure_gls est déplacée sans changement de calcul dans src/raman_treatment.py (nom en minuscules conservé). Imports de la figure 7 et des tests adaptés. qn_pressure_step.py supprimé avec son cache. Tests renommés test_raman_pressure_gls.py ; README et ANALYSIS actualisés. Le projet original est préservé.

Validation : figure 7 régénérée avec succès ; JSON identique et PNG identique pixel par pixel à l’invocation directe précédente. Suite complète : 74 tests passent. Aucun import résiduel de qn_pressure_step dans les sources ou tests de la copie publication.


# Corrections de l’audit avant publication — 9 novembre 2026

Cette section décrit l’état courant et remplace les valeurs antérieures ci-dessus lorsqu’elles diffèrent.

- Composition conservée : SiO₂/Al₂O₃/Na₂O/H₂O = 58,425 / 11,875 / 24,70 / 5,00 % massiques (somme 100). Les paramètres du code étaient déjà corrects ; les commentaires et descriptions contradictoires sont corrigés.
- FWHM : mesure aux deux croisements de la moitié de la hauteur du pic corrigé de son fond, avec interpolation en nm. Le premier pic détecté et les seuils de détection sont conservés. La largeur initialise la pseudo-Voigt du borate et définit sa fenêtre et ses bornes.
- Contrôle de `success` après l’ajustement du borate : en cas d’échec, avertissement, compteur incrémenté et paramètres ajustés conservés à NaN, sans remplacement par le maximum détecté pour les pressions. Aucun échec observé sur les acquisitions actuelles.
- Suppression de `fit_peaks`, `treat_data` et de leurs imports devenus inutiles. Suppression des constantes inutilisées `TG_1ATM`, `TG_1ATM_DRY`, `TC_MANUSCRIPT`.
- Toutes les PCA Raman/jauges utilisent `random_state=0`, avec sélection automatique du solveur conservée. La graine est enregistrée dans les transitions et le JSON de la figure 5.
- Prérequis documentaire corrigé : Python ≥3.12. Cible `ir` déclarée dans `.PHONY`. Le fichier de licence reste à ajouter par le propriétaire.

Comparaison avec les données et références numériques antérieures à la correction :

| Quantité | Avant | Après |
| --- | ---: | ---: |
| Extrapolation à 0 GPa (°C) | 239,690068 | 239,733171 |
| Pente (°C/GPa) | 6,92095 | 6,929199 |
| Moyenne des offsets diagnostiques (K) | 18,755066 | 19,429255 |
| Médiane des offsets diagnostiques (K) | 17,153572 | 17,529003 |

Écart absolu maximal des pressions médianes du borate : 0,085685 GPa ; de leur dispersion P_std : 0,029073 GPa. Écart maximal des positions ajustées du borate par grain : 0,022479 nm. Pressions médianes du rubis inchangées à l’arrondi numérique près. Les calibrations et références de longueur d’onde sont conservées. Les offsets restent diagnostiques et ne corrigent pas les pressions.

Les cinq intervalles restent 262–286, 270–277, 256–262, 313–332 et 300–306 °C pour les Runs 2–6. Le Run 3 reste candidat et les quatre autres restent soutenus par les deux tests. Les probabilités changent légèrement : Run 2, p linéaire 0,023 → 0,025 et p quadratique 0,020 → 0,019 ; Run 3, p linéaire 0,112 → 0,111. Les autres probabilités sont inchangées.

Les références numériques des tests de température et des guides de la figure 9 sont actualisées après comparaison. La température conserve sa tolérance absolue de 10⁻⁵ K. Le test numérique des guides utilise une tolérance absolue de 10⁻⁶ cm⁻¹ (ou cm⁻¹/GPa pour la pente) ; la pente attendue est désormais exprimée directement par GPa, sans division implicite par cinq. Les contrôles analytiques et de covariance sont conservés.

Validation avec Python 3.13.5 et l’environnement existant :

- Reproduction dans une copie temporaire initialement sans résultats ; actualisation des références dérivées puis `make test ir` : code retour 0, 81 tests réussis.
- Sept nouveaux tests : largeur gaussienne indépendante du seuil, interpolation sur l’axe réel, largeur du premier pic sélectionné, rejet d’un ajustement déclaré non convergé, récupération d’une pseudo-Voigt connue et répétabilité des PCA Raman/jauges.
- 14 PNG et 14 PDF du manuscrit, diagnostics de centroïde et résultats IR des trois échantillons générés.
- Figure 5 inspectée ; deux générations en processus distincts donnent un PNG identique pixel par pixel et des résultats numériques JSON identiques, hors ajout explicite de la graine dans les métadonnées.
- 931 fichiers enregistrés dans INPUTS.sha256 vérifiés, aucune différence.
- Sorties validées recopiées dans results/ et figures/ ; anciens noms de diagnostics individuels devenus obsolètes retirés. Suite relancée dans le dossier de publication : 81 tests réussis.
- L’installation dans un nouvel environnement Python 3.12 n’a pas été exécutée.


# Figures supplémentaires des ruptures — 13 novembre 2026

S7, S8, S9 et S10 présentent respectivement les Runs 2, 3, 5 et 6. Ils réutilisent l’analyse et le tracé de la figure 5, paramétrés par run et nom de sortie : mêmes cinq familles, sélection des paliers, standardisation, graines et 999 permutations par modèle nul. Les acquisitions cristallisées restent affichées mais exclues de l’ajustement de rupture.

L’ancienne figure de viscosité S7 est renumérotée S11 : module, PNG, PDF, JSON, CSV, sélection dans le lanceur, tests et documentation actualisés. L’ancien CSV S7 est retiré pour éviter de l’associer au nouveau diagnostic du Run 2.

Validation :

- S7–S11 générées avec succès via `python -m src.figures` et inspectées pour S7–S10.
- Le test d’intégration vérifie les correspondances run/numéro, les fichiers PNG/PDF/JSON et l’identité des bornes et probabilités avec les transitions existantes.
- Figure 5 régénérée : PNG identique pixel par pixel et JSON identique à la version précédente.
- Viscosité S11 : PNG et JSON identiques aux anciens fichiers S7.
- Le dossier du manuscrit contient désormais 18 PNG et 18 PDF.
- Après régénération du diagnostic de centroïde manquant, la suite complète passe : 82 tests, Python 3.13.5.


# Lisibilité des figures de rupture — 14 novembre 2026

Figure 5 et S7–S10 : les limites Y sont calculées sur les seuls canaux standardisés des acquisitions glass/melt retenues pour l’ajustement, avec une marge de 5 % de leur étendue. Les points cristallisés ne peuvent plus étendre les axes ; ceux hors limites sont masqués par le cadre. Une note dans chaque figure explicite ce choix. La standardisation et les calculs de rupture ne sont pas modifiés.

Les canaux individuels sont noirs, avec de petits cercles, sous le libellé « Input channels ». La moyenne est bleue et désignée « Channel mean (guide) ». Le coût de rupture utilise chaque canal séparément, pas cette moyenne. Les losanges rouges ouverts des points cristallisés sont conservés. La légende adopte « Crystallised (not fitted) » et l’axe « Standardised signal », en anglais britannique.

Validation : cinq PNG/PDF régénérés ; contrôle visuel du Run 5 (S9), auparavant fortement comprimé par les cristallisés. Les cinq JSON numériques sont identiques aux versions précédentes. Un test vérifie que des cristallisés arbitrairement extrêmes ne modifient pas les limites et que les signaux ajustés restent dans le cadre. Suite complète : 83 tests réussis, Python 3.13.5.


# Référence de température fictive limite du verre reçu — 14 novembre 2026

Figure 6 : les deux anciens points de pics DSC sont remplacés par une seule mesure de température fictive limite, 219 ±4 °C à 25 K/min, libellée « T_f' at 25 K/min ». Il s’agit du verre ANN +5 wt% H2O tel que reçu après synthèse au piston-cylindre, puis chargé dans la DAC. Le point est affiché à pression ambiante (0 GPa sur le graphique), avec une barre verticale de ±4 °C. Aucun niveau de confiance n’a été supposé.

Les valeurs, la vitesse de chauffe, la provenance et la description de l’incertitude sont centralisées dans `src/config.py`, sous `ANN_FICTIVE_REFERENCE`. `DSC_PEAKS_C` et les anciennes métadonnées `DSC_comparisons` sont supprimés. Les JSON de la figure 6, de S6 et de la régression enregistrent désormais `limiting_fictive_temperature_reference`, explicitement exclue de la régression. Documentation scientifique et table des figures actualisées.

Validation : figure 6 et S6 régénérées en PNG/PDF/JSON ; figure 6 inspectée visuellement. Comparaison avec le JSON antérieur : tous les résultats de régression sont strictement identiques, seule la référence de comparaison change. Les paramètres enregistrés sont bien 219 °C, ±4 °C et 25 K/min. Suite complète : 83 tests réussis, Python 3.13.5.


# Décalage après insertion de la figure DSC — 15 novembre 2026

La figure DSC insérée dans le manuscrit prend le numéro 5 et reste fournie séparément. Les figures principales générées sont renumérotées ainsi :

| Ancien nom | Nouveau nom |
| --- | --- |
| fig5_ruptures | fig6_ruptures |
| fig6_transitions | fig7_transitions |
| fig7_melt_areas | fig8_melt_areas |
| fig8_delta_tg | fig9_delta_tg |
| fig9_raman | fig10_raman |

Les modules Python, imports, noms de sortie PNG/PDF/JSON, sélecteurs du lanceur, noms des tests concernés et références actives dans README.md, FIGURES.md et ANALYSIS.md sont actualisés. `results/fig6_interval_regression.json` devient `results/fig7_interval_regression.json`, y compris dans les métadonnées et les tests consommateurs. Les anciens modules et sorties ne sont pas conservés sous forme de doublons ou de relais. Les figures supplémentaires gardent leurs numéros ; S6 partage désormais la génération de la figure principale 7, et S7–S10 utilisent le tracé de la figure principale 6.

Le dossier expérimental `test_rutpures` et tout son contenu sont supprimés. L’analyse de rupture initiale à cinq familles reste active. Les comptes rendus historiques datés conservent leurs numéros d’époque, comme précisé en tête de ce fichier et d’ANALYSIS.md.

Validation avec Python 3.13.5 et l’environnement existant :

- Comparaison syntaxique de toutes les sources avec la sauvegarde avant migration, après neutralisation des chaînes et normalisation des imports renommés : aucun changement de logique de calcul.
- 931 empreintes d’entrée vérifiées, aucune différence.
- Reproduction dans une copie temporaire sans résultats préexistants : `make test ir`, code retour 0. Toute la chaîne d’extraction, pression, bandes Raman, température, transitions, viscosité, figures, centroïdes et IR a été exécutée.
- 84 tests réussis. Le nouveau contrôle rejette les anciens sélecteurs et vérifie l’absence d’anciens modules, sorties et fichier de régression. Les tests vérifient la déduplication figure 7/S6 et la cohérence des figures supplémentaires.
- 18 PNG et 18 PDF du manuscrit générés sous les noms actuels, ainsi que les sorties IR et diagnostics.
- Les 14 JSON de figures sont comparés aux versions précédentes après mise à jour des références de chemins : aucune différence numérique hors tolérance absolue 10⁻⁵ ou relative 10⁻⁶. Les champs textuels restent identiques après normalisation des noms renommés.
- Les sorties validées sont recopiées dans le dossier de publication. Suite complète relancée dans ce dossier : 84 tests réussis.
- Aucun ancien nom actif dans les sources, résultats JSON ou métadonnées des figures. Aucun changement de méthode, de données ou de pondération des ruptures.

# Audit final — 15 novembre 2026

Cette entrée remplace les indications antérieures lorsqu’elles divergent. Le rapport concis figure dans [AUDIT_FINAL.md](AUDIT_FINAL.md).

- Confirmation utilisateur : NBO/T du verre hydraté = 0,59 ; ε4500 = 0,44 ± 0,03. La référence NBO/T est centralisée dans la configuration. Les coefficients IR sont lus et validés dans le CSV fourni, et leur provenance est enregistrée dans les résultats.
- Correction des grilles de borate des runs 5 (686–690 nm) et 6 (685,5–689 nm). Auparavant, des fenêtres alignées dépassaient les données extraites et l’interpolation répétait les bords. Un contrôle bloque désormais cette situation. Vérification de couverture sur les 684 acquisitions brutes. Les cinq familles et leurs poids restent inchangés.
- Les intervalles des runs 2–6 restent respectivement 262–286, 270–277, 256–262, 313–332 et 300–306 °C. Les probabilités maximales finales sont 0,026 ; 0,092 ; 0,001 ; 0,003 ; 0,001. Aucun changement du classement des ruptures ; le run 3 reste candidat.
- Les teneurs IR moyennes ne changent pas. Les écarts-types des teneurs totales passent à 0,181062 ; 0,222392 ; 1,093518 % massiques pour ANN 3, 5 et 8.
- Correction des chemins Python avec espaces, du lancement autonome de la viscosité, du chargement d’un spectre ambiant à une colonne ; suppression d’auxiliaires inutilisés. `make test` inclut désormais l’IR.
- Documentation active actualisée ; ancien texte ANALYSIS archivé explicitement comme historique. Références locales de la documentation active contrôlées.
- Exécution complète depuis une copie sans résultats : `make test` avec Python 3.13.5, code retour 0, 94 tests réussis. Dépendances conformes au verrouillage et `pip check` sans erreur. Pas d’installation neuve sous Python 3.12.
- 931 empreintes d’entrée vérifiées. Résultats et figures validés recopiés dans le dossier de publication ; sources Python identiques à la copie testée après analyse syntaxique, hors commentaires. Inspection visuelle des 21 PDF régénérés : 18 figures du manuscrit, deux diagnostics IR, un diagnostic de centroïde.
- Suite complète relancée dans le dossier de publication après remise des sorties : 94 tests réussis en 4,932 s, code retour 0. Les JSON des résultats et figures passent également une lecture stricte interdisant NaN et Infinity.

# Intégration de S5 — 16 novembre 2026

Le nouveau script `src/ctx.py` cherchait des dossiers inexistants : dates au format jour-mois-année au lieu d’année-mois-jour, et nom incomplet pour la session d’avril. Les chemins sont corrigés en `session_20250220_ANN`, `session_20250303_ANN` et `session_20250408_ANN_5H2O`. Le chargement inutilisé de la session du 30 avril, qui pointait aussi vers le mauvais dossier, est supprimé.

Le script expose maintenant `NAME` et `main()` pour le lanceur général, sans chargement ni génération à l’import. Les spectres sélectionnés, régions de ligne de base cubique, normalisation, marqueurs et annotations de pression/température sont conservés. Les annotations fournies ne sont pas recalculées. Une erreur explicite signale les fichiers manquants. Les sorties PNG, PDF et JSON sont enregistrées par le mécanisme commun, qui crée le dossier de destination. README, FIGURES et ANALYSIS incluent désormais S5.

Vérifications sous Python 3.13.5 : lancement autonome puis lancement général `python -m src.figures` réussis ; 96 tests réussis, dont deux contrôles nouveaux sur les données S5 et l’erreur de dossier vide. Aperçu PNG S5 inspecté. Aucun changement des calculs des autres figures.

# Déplacement du script S5 — 16 novembre 2026

Le script précédemment nommé `src/ctx.py` est déplacé vers `src/manuscript/supplementary_fig_5.py`. Les imports relatifs, le registre du lanceur général, les tests et les références actives de README, FIGURES et ANALYSIS utilisent ce nouvel emplacement. Les mentions antérieures dans cet historique décrivent le chemin à leur date. La commande autonome devient `python -m src.manuscript.supplementary_fig_5`. Le nom des sorties et le traitement des spectres restent identiques.

Validation : génération de S5 par le lanceur général réussie ; suite complète de 96 tests réussie sous Python 3.13.5. Aucune référence active à l’ancien module dans les sources, tests et guides courants.
