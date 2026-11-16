"""Raman centroids versus temperature and Melt 830-1270 cm-1 centroid versus P.

Spectra are baseline-corrected and area-normalised exactly as in
src.raman_treatment.prepare_glass; the centroid is the intensity-weighted mean
Raman shift over the window, integrated with the trapezoid rule. The graphical style and state bands are defined locally.

Usage: python -m src.centroid_lf [--window LO HI]
Output: figures/raman/centroids/<run>_centroid.png,
        figures/raman/centroids/area_ratio/<run>_area_ratio.png,
        figures/raman/centroids/melt_centroid_830_1270_vs_pressure.{png,pdf,json},
        results/centroid_lf.csv and results/melt_centroid_830_1270.csv

The pressure plot always uses 830-1270 cm-1, independently of --window, which
selects the window for the per-run temperature plots. Only Etat == Melt enters
the pressure plot; glass and crystal-bearing acquisitions are excluded.
"""
import argparse
import csv
import json

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import raman_treatment as ts
from .figure_data import DATASET
from .config import FIGURES, RESULTS
from .pressure import load_dataset

from .raman_treatment import centroid, area_ratio

OUT_DIR = FIGURES / "raman" / "centroids"
AREA_RATIO_DIR = OUT_DIR / "area_ratio"


# Conventions graphiques propres aux diagnostics de centroïde.
BANDS = (("Melt", "#FFE0B0"), ("Melt+Ctx", "#FFB3B3"))
TICKS = np.arange(210, 400, 30)


def state_bands(ax, T, labels, steps=None):
    """Colore les domaines Melt (orange) et Melt+Ctx (rouge).

    Seuls les paliers de chauffage définissent les limites des domaines.
    Les limites sont placées à mi-distance des températures voisines ;
    le dernier domaine est prolongé de 5 °C en absence de palier suivant."""
    T = np.asarray(T, float)
    labels = np.asarray([str(s) for s in labels])
    heating = np.array([np.isfinite(t) and t >= np.nanmax(T[:i + 1]) for i, t in enumerate(T)])
    Th, Lh = T[heating], labels[heating]
    for state, colour in BANDS:
        sel = np.where(Lh == state)[0]
        if not len(sel):
            continue
        lo = Th[sel[0]] if sel[0] == 0 else 0.5 * (Th[sel[0] - 1] + Th[sel[0]])
        hi = Th[sel[-1]] + 0.5 * (Th[sel[-1] + 1] - Th[sel[-1]]) if sel[-1] + 1 < len(Th) \
            else Th[sel[-1]] + 5.0
        ax.axvspan(lo, hi, color=colour, lw=0, zorder=0)


def style(ax, title, xlim):
    """Applique les titres, graduations et grille des diagnostics en température."""
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel("Temperature, °C")
    ax.set_xticks([t for t in TICKS if xlim[0] <= t <= xlim[1]])
    ax.set_xlim(xlim)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)


def plot_melt_pressure(rows):
    """Save the 830–1270 cm^-1 centroid versus median Sm2+:SrB4O7 pressure.

    Uses the same signed, baseline-corrected spectra and trapezoid moment as
    the per-run LF diagnostic. Horizontal bars are P_std (grain dispersion plus propagated grain errors).
    Centroid errors are not estimated; no regression or confidence band is added.
    """
    from .figure_data import RUN_LABEL, save, style as figure_style
    from .manuscript_data import RUN_COLORS

    figure_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.6), layout='constrained')
    for run in dict.fromkeys(row['run_id'] for row in rows):
        points = [row for row in rows if row['run_id'] == run]
        ax.errorbar(
            [row['P_GPa'] for row in points],
            [row['centroid_830_1270_cm-1'] for row in points],
            xerr=[row['P_std_GPa'] for row in points],
            fmt='o', color=RUN_COLORS[run], markersize=4, linewidth=.7,
            label=RUN_LABEL[run],
        )
    ax.set(
        xlabel='Sm$^{2+}$:SrB$_4$O$_7$ median pressure (GPa)',
        ylabel='830–1270 cm$^{-1}$ centroid (cm$^{-1}$)',
        title='Melt-only',
    )
    ax.legend(fontsize=7, ncol=2)
    png, _ = save(fig, 'melt_centroid_830_1270_vs_pressure', 'raman/centroids')
    metadata = {
        'window_cm-1': [830., 1270.], 'window_endpoints_included': True,
        'definition': 'Integral(x*I dx)/integral(I dx), trapezoid rule',
        'preprocessing': 'prepare_glass: two-part linear baseline and full-spectrum area normalization; signed intensities retained',
        'selection': 'Etat == Melt; all Melt steps, no crystal-bearing spectra',
        'horizontal_errors': 'P_std, descriptive grain dispersion plus propagated errors; not median standard error',
        'vertical_errors': 'Centroid uncertainty not estimated',
        'points': rows,
    }
    png.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    with (RESULTS / 'melt_centroid_830_1270.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return png


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", type=float, nargs=2, default=(400.0, 830.0), metavar=("LO", "HI"))
    a = ap.parse_args(argv)
    D = load_dataset()
    with h5py.File(DATASET, "r") as f:
        x = f["x_axis"][()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AREA_RATIO_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    melt_rows = []
    for r, d in D.items():
        spectra = ts.prepare_glass(x, d["spectra"])
        c = centroid(x, spectra, a.window)
        ratio = area_ratio(x, spectra)
        high_centroid = centroid(x, spectra, (830., 1270.))
        for step in np.flatnonzero(d['Etat'] == 'Melt'):
            melt_rows.append({
                'run_id': r, 'step': int(step), 'T_C': float(d['T_C'][step]),
                'Etat': str(d['Etat'][step]),
                'P_GPa': float(d['borate_P_median'][step]),
                'P_std_GPa': float(d['borate_P_std'][step]),
                'centroid_830_1270_cm-1': float(high_centroid[step]),
            })
        T = d["T_C"]
        rows += [[r, i, T[i], d["Etat"][i], c[i], ratio[i]] for i in range(len(T))]
        pad = 0.04 * np.ptp(T) if np.ptp(T) > 0 else 5.0
        fig, ax = plt.subplots(figsize=(4.6, 4.2), constrained_layout=True)
        state_bands(ax, T, d["Etat"])
        ax.plot(T, c, "o", ms=6, mfc="#0072B2", mec="k", mew=0.8, zorder=3)
        ax.set(ylabel=f"Centroid {a.window[0]:g}-{a.window[1]:g} cm$^{{-1}}$")
        style(ax, r, (np.nanmin(T) - pad, np.nanmax(T) + pad))
        ax.title.set_fontsize(9)
        fig.savefig(OUT_DIR / f"{r}_centroid.png", dpi=250)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(4.6, 4.2), constrained_layout=True)
        state_bands(ax, T, d["Etat"])
        ax.plot(T, ratio, "o", ms=6, mfc="#0072B2", mec="k", mew=0.8, zorder=3)
        ax.set(ylabel="Area ratio 400-830 / 830-1270 cm$^{-1}$")
        style(ax, r, (np.nanmin(T) - pad, np.nanmax(T) + pad))
        fig.savefig(AREA_RATIO_DIR / f"{r}_area_ratio.png", dpi=250)
        plt.close(fig)
    with open(RESULTS / "centroid_lf.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "step", "T_C", "Etat", f"centroid_{a.window[0]:g}_{a.window[1]:g}",
                    "area_ratio_400_830_over_830_1270"])
        w.writerows(rows)
    plot_melt_pressure(melt_rows)
    print("[centroid_lf] ->", OUT_DIR)


if __name__ == "__main__":
    main()
