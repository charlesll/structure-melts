"""Figure 6: common-break analysis of Run 4, with crystals shown for context.

Rebuild the exact five feature families used by tg_spectra_ruptures. Fit only
non-crystallised heating observations, at or above 150 C. Display crystallised
points without using them to centre, scale or fit the channels. Global Raman
and sensor PCAs retain the existing all-run training population.
The same renderer and analysis also generate S7–S10 for Runs 2, 3, 5 and 6.

python -m src.figures.fig6_ruptures [--n-permutations 999]
"""
import argparse
import json

import h5py
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from ..config import RESULTS
from ..figure_data import RUN_LABEL, save, style
from ..pressure import load_dataset
from ..figure_data import DATASET
from .. import tg_spectra_ruptures as tr

NAME = "fig6_ruptures"
# Numérotation du manuscrit : Run 4, et non ancien indice informatique 4.
RUN_ID = "run04_20250303_34bar"
TITLES = ("Raman spectra (4 PCs)", "Raman area ratio (400–830 / 830–1270)",
          "Fitted-band area ratios (3 channels)", "Fitted-band positions (5 channels)",
          "Ruby and borate sensors (9 channels)")


def standardize_for_display(values, fitted):
    """Use fit-only means/SDs to display both fitted and excluded points."""
    values = np.asarray(values, float)
    # values : matrice (paliers, canaux) ; fitted : masque booléen des paliers
    # autorisés dans la segmentation. axis=0 calcule une valeur par canal.
    center = values[fitted].mean(axis=0)
    scale = values[fitted].std(axis=0)
    # Appliquer ensuite CES mêmes paramètres à toutes les lignes, y compris
    # les cristallisés. Ils restent comparables sans influencer la référence.
    # Un canal constant reçoit un diviseur 1 pour éviter une division par zéro.
    return (values - center) / np.where(scale > 0, scale, 1)


def fitted_y_limits(values, fitted):
    """Vertical display range from fitted glass/melt channels only.

    Crystallised points outside this range are clipped, without changing the
    channels or their standardisation. A 5% margin keeps fitted symbols clear
    of the panel borders; a constant signal receives a nonzero range.
    """
    observed = np.asarray(values, float)[fitted]
    lower, upper = float(np.min(observed)), float(np.max(observed))
    margin = .05 * (upper-lower) if upper > lower else .5
    return lower-margin, upper+margin


def build_analysis(n_permutations=999, run_id=RUN_ID):
    """Recompute current-data results, avoiding stale saved p-values/features."""
    datasets = load_dataset()
    # Les résultats corrigés viennent de dataset.h5 ; les formes des raies
    # viennent des fichiers markers et les bandes ajustées de peak_fits.h5.
    with h5py.File(DATASET, "r") as f:
        x = f["x_axis"][()]
    full = tr._run_families(datasets, x, RESULTS / "markers", tr.T_MIN,
                            drop_crystal=False)
    # Conserver les cristallisés POUR LE TRACÉ. Le filtre de chauffage et le
    # seuil commun T_MIN = 150 °C restent actifs.
    T, labels, families = full[run_id]
    crystals = np.array([str(label).endswith("Ctx") for label in labels])
    fitted = ~crystals
    # Deux jeux de données : families contient les points à montrer ;
    # fit_families ne contient que les points qui détermineront la rupture.
    fit_families = {name: (Y[fitted], flag) for name, (Y, flag) in families.items()}
    result = tr.find_common_break(T[fitted], fit_families)
    # Même test que le script d'analyse : 999 permutations par modèle nul,
    # avec graines fixes pour les permutations et pour l'entraînement PCA.
    for degree, seed, key in [(1, 0, "p_linear"), (2, 1, "p_quadratic")]:
        result[key] = tr.permutation_test(T[fitted], fit_families,
                                          n_permutations=n_permutations,
                                          seed=seed, null_degree=degree)["p_value"]
    result["p_conservative"] = max(result["p_linear"], result["p_quadratic"])
    return T, labels, families, fitted, result


def main(d=None, outdir="manuscript", n_permutations=999, *, run_id=RUN_ID, output_name=NAME):
    """The runner's shared flat table is unused: features need marker files."""
    style()
    T, labels, families, fitted, result = build_analysis(n_permutations, run_id=run_id)
    crystals = ~fitted
    fig, axes = plt.subplots(5, 1, figsize=(6.8, 7.8), sharex=True)
    # Un panneau par famille, axe T commun. La mise en page et les couleurs
    # peuvent être modifiées ici sans toucher au calcul dans build_analysis().
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.91, hspace=0.38)
    for k, (ax, name, title) in enumerate(zip(axes, tr.FAMILY_NAMES, TITLES)):
        Z = standardize_for_display(families[name][0], fitted)
        # Z : autant de lignes que de températures et une colonne par canal.
        # Z.T permet de parcourir les trajectoires de chaque canal.
        # Individual channels retain their signs: the mean is descriptive,
        # and can cancel; it is not the multivariate detection statistic.
        mean = Z.mean(axis=1)
        for channel in Z.T:
            # Courbes grises et petits symboles : canaux utilisés dans la rupture. Losanges rouges ouverts :
            # points cristallisés, non reliés pour éviter de suggérer un fit.
            ax.plot(T[fitted], channel[fitted], "o-", color="0.6",
                    ms=1.8, lw=0.65)
            ax.scatter(T[crystals], channel[crystals], marker="D", s=11,
                       facecolor="none", edgecolor="#B23A48", alpha=0.5, linewidth=0.6)
        ax.plot(T[fitted], mean[fitted], "o-", color="#0072B2", lw=1, ms=3, zorder=4)
        # Les gros symboles résument les moyennes ; les traits fins et petits
        # losanges montrent la dispersion entre canaux au sein de la famille.
        ax.scatter(T[crystals], mean[crystals], marker="D", s=26,
                   facecolor="white", edgecolor="#B23A48", linewidth=1, zorder=5)
        ax.axvspan(result["T_left"], result["T_right"], color="0.85", alpha=0.6, lw=0)
        # Bande grise = les deux paliers encadrant la séparation retenue.
        # Trait pointillé = leur milieu, pas une température directement mesurée.
        ax.axvline(result["T_break"], color="0.25", ls="--", lw=0.9)
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
        ax.set_ylim(*fitted_y_limits(Z, fitted))
        ax.set_title(f"{chr(97 + k)}) {title}", loc="left", fontsize=8)
        ax.tick_params(top=False, right=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="0.9", lw=0.5)
        ax.set_axisbelow(True)
    axes[-1].set(xlabel="Thermocouple temperature (°C)", xlim=(T.min() - 5, T.max() + 5))
    fig.supylabel("Standardised signal (fit-only mean and SD)", fontsize=9, x=0.02)
    fig.suptitle(f"{RUN_LABEL[run_id]} · common split: {result['T_left']:g}–{result['T_right']:g} °C "
                 f"(midpoint {result['T_break']:g} °C)", fontsize=10, y=0.975)
    handles = [Line2D([], [], color="0.6", marker="o", ms=1.8, lw=0.65, label="Input channels"),
               Line2D([], [], color="#0072B2", marker="o", ms=3, lw=1, label="Channel mean (guide)"),
               Line2D([], [], color="#B23A48", marker="D", mfc="white", ms=4,
                      ls="none", label="Crystallised (not fitted)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.54, 0.96),
               fontsize=7)
    png, _ = save(fig, output_name, outdir)
    # save() écrit PNG et PDF. Le JSON voisin garde les chiffres nécessaires
    # à la légende du manuscrit et explicite le statut des points exclus.
    metadata = {"run_id": run_id, "n_fitted": int(fitted.sum()),
                "n_crystallized_displayed": int(crystals.sum()),
                "n_permutations": n_permutations,
                "t_min_C": tr.T_MIN,
                "fitted_temperatures_C": T[fitted].tolist(),
                "crystallized_temperatures_C": T[crystals].tolist(),
                "split_interval_is_confidence_interval": False,
                "crystallized_used_for_break_fit": False,
                "global_pca_training_includes_crystallized": True,
                "pca_random_state": tr.PCA_RANDOM_STATE,
                **{key: value for key, value in result.items() if key != "improvement"}}
    metadata["improvement"] = result["improvement"]
    png.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[figures] {png}\n{json.dumps(metadata, indent=2)}")
    return png


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-permutations", type=int, default=999)
    main(n_permutations=parser.parse_args().n_permutations)
