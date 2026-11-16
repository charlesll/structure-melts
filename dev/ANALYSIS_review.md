# Propositions pour ANALYSIS.md — à relire manuellement

Aucune des modifications de contenu proposées ci-dessous n’a été appliquée. Seul le lien vers VALIDATION.md a été adapté à son déplacement dans dev/.

1. **Introduction** : remplacer le paragraphe daté et les renvois à l’historique par « This guide describes the analysis methods and numerical conventions used in this publication package. » => DONE
2. **Référence fictive** : supprimer « The former two DSC peak markers are removed. » Le point actuel est déjà décrit juste avant. => DONE
3. **Référence de température** : remplacer « historical 25C filenames and workbook temperatures are not relabelled » par « filenames containing 25C and workbook temperatures retain their recorded labels ». Cela conserve une convention utile sans récit historique. => DONE
4. **Seuil à 7 GPa** : remplacer « or correcting the historical choice of the 7 GPa threshold » par «; the 7 GPa threshold is fixed ». Les limites statistiques précédentes restent explicites. => DONE
5. **Affichage des ruptures** : supprimer « as noted on each figure ». Cette mention n’est plus exacte : le texte sous les figures a été retiré. Conserver l’explication des limites verticales et des points hors cadre. => DONE
6. **Arrondis de la régression** : remplacer « b =6.92 °C/GPa » par « b =6.93 °C/GPa », et « [210.8, 270.6] °C » par « [210.9, 270.6] °C ». Le fichier actuel results/fig7_interval_regression.json donne 6.929199468841899 et [210.86227372349904, 270.6224596035264]. Aucun calcul ne doit changer. => DONE
7. **IR** : supprimer « The corrected 4500 SD supersedes the formerly hard-coded ±0.01. » La valeur actuelle ±0.03 est déjà indiquée. => DONE
8. **S5** : ajouter une courte description de src/manuscript/supplementary_fig_5.py : trois spectres Ctx sélectionnés ; ligne de base cubique sur 500–520, 800–840 et 1200–1300 cm⁻¹ ; normalisation au maximum ; marqueurs et annotations descriptives définis dans le script ; paramètres enregistrés dans le JSON. Ce contenu est déjà documenté dans FIGURES.md. => DONE

Ces propositions concernent la documentation uniquement. Les fichiers historiques déplacés dans dev/ conservent leurs descriptions datées.
