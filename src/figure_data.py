"""Shared data assembly for the manuscript figures (Python port of Processing.jl).

Every figure script (`src/figures/*.py`) imports from here so that the figure
set reads one dataset, one state convention and one panel ordering. The port
replaces the Julia/Makie chain of `old_code/Processing.jl`; numbers are read from
`results/dataset.h5` and `results/peak_fits.h5`, so
the figures are regenerated from the pipeline instead of a frozen HDF5.

Conventions kept identical to the Julia code, so that a figure can be compared
point by point with the published version:

- pressure is the *borate* median unless stated otherwise (`P_b`);
- error bars on P use P_std (grain dispersion plus propagated errors) -- see `ANALYSIS.md`;
- error bars on T are +/- SIGMA_T (thermocouple only; excludes sample–thermocouple bias);
- `Etat` labels drive the marker shapes: Verre = square, Melt = circle,
  Melt+Ctx = diamond;
- bands: orange = [Tg, Tc] (melt), red = [Tc, Tmax] (melt + crystals).

Where the reviewers asked for more (R1: annotate P/T on spectra, report P at the
bracket points, distinguish model curves from data), the panels add the
information rather than change the convention.

Literature values used by the comparison panels are defined here once.
"""
from pathlib import Path

import h5py
import numpy as np

from .config import DATA, FIGURES, RESULTS, SIGMA_T

DATASET = RESULTS / "dataset.h5"
PEAK_FITS = RESULTS / "peak_fits.h5"

# Descriptive linear extrapolation of the five spectral brackets (T >=150 °C),
# including the Run 3 candidate. Profile support is approximate 95%, NOT 1 sigma.
# Updated 9 Nov 2026 after correction of fluorescence FWHM measurement.
# Source: results/fig7_interval_regression.json, src/tg_pressure_regression.py.
TG_EXTRAPOLATED_0GPA_C = 239.73317122038492
TG_EXTRAPOLATED_0GPA_PROFILE95_C = (210.86227372349904, 270.6224596035264)

PEAK_ORDER = ("TOH", "Q2", "T2S", "Q3", "Q4")
PEAK_LABELS = {"TOH": "T–OH", "Q2": "Q$^2$", "T2S": "T$_{2S}$", "Q3": "Q$^3$", "Q4": "Q$^4$"}
STATE_STYLE = {"Verre": ("grey", "s"), 
               "Melt": ("orange", "o"),
               "Melt+Ctx": ("red", "D")}
RUN_ORDER = ["run01_20250220_25C", "run02_20250220", "run03_20250303_32bar",
             "run04_20250303_34bar", "run05_20250408", "run06_20250430"]
RUN_LABEL = {r: f"Run {i + 1}" for i, r in enumerate(RUN_ORDER)}

def load_dataset(path=DATASET):
    """Return the flat per-step table as a dict of arrays, sorted by run then step."""
    with h5py.File(path, "r") as f:
        g = f["flat"]
        d = {k: g[k][()] for k in g}
        d["run_id"] = np.array([s.decode() for s in d["run_id"]])
        d["Etat"] = np.array([s.decode() for s in d["Etat"]])
    return d


def run_slices(d):
    """{run_id: index array} in manuscript order."""
    return {r: np.flatnonzero(d["run_id"] == r) for r in RUN_ORDER if (d["run_id"] == r).any()}


def per_run_median(d, key):
    """Median of `key` over the steps of each run, in manuscript order."""
    idx = run_slices(d)
    return {r: float(np.nanmedian(d[key][i])) for r, i in idx.items()}


def melt_table(d):
    """Rows of `Melt` spectra sorted by borate pressure (as in Processing.jl)."""
    m = d["Etat"] == "Melt"
    order = np.argsort(d["borate_P_median"][m])
    out = {k: d[k][m][order] for k in ("run_id", "T_C", "borate_P_median", "borate_P_std",
                                       "ruby_P_median", "ruby_P_std")}
    out["index"] = np.flatnonzero(m)[order]   # row index in the flat dataset
    return out


def run_step_index(d, run_id):
    """Position of each row inside its own run (0-based), for step-indexed lookups."""
    idx = np.zeros(len(d["T_C"]), int)
    for i in run_slices(d).values():
        idx[i] = np.arange(i.size)
    return idx


def ambient_spectrum(path=None):
    """Starting-glass Raman spectrum at 25 °C and ambient pressure.

    Defaults explicitly to ANN_5prctH2O_LF.asc, avoiding the other compositions
    in the same directory. Two-column input is wavelength (nm), intensity;
    one-column input is intensity on a 1 cm^-1 grid starting at 400 cm^-1.
    Applies despiking, constant baseline subtraction on 1220–1300 cm^-1 and
    Long correction at 25 °C; returns (x_cm, intensity, discrete LF centroid).
    """
    import rampy as rp
    path = Path(path) if path is not None else DATA / "session_ambient_ANN" / "ANN_5prctH2O_LF.asc"
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    if raw.shape[1] < 2:            # one-column intensity export: 1 cm-1 grid from 400
        x = np.arange(400.0, 400.0 + raw.size)
        y = raw.ravel()
    else:
        x_nm, y = raw[:, 0], raw[:, 1]
        x = 1e7 / 532.0 - 1e7 / x_nm        # Raman shift, cm-1
        order = np.argsort(x)
        x, y = x[order], y[order]
    # despike + constant baseline on the 1220-1300 cm-1 window, as in Processing.jl
    y = rp.despiking(x, y, neigh=5, threshold=3) if hasattr(rp, "despiking") else y
    if not np.isfinite(y).all():
        raise ValueError("Despiking produced nonfinite starting-glass intensities")
    roi = np.array([[1220.0, 1300.0]])
    y_cor = rp.baseline(x.reshape(-1, 1), y.reshape(-1, 1), roi=roi,
                        method="poly", polynomial_order=0)[0]
    y = np.asarray(y_cor).ravel()
    y = np.asarray(rp.tlcorrection(x, y, 25.0, 532.0)[1]).ravel()
    lf = x < 830
    centroid = float(np.sum(x[lf] * y[lf]) / np.sum(y[lf]))
    return x, y, centroid


def figure_path(name, subdir):
    """Absolute output path for a figure, with its directory created."""
    out = FIGURES / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out / name


def save(fig, name, subdir, dpi=300):
    """Save a figure as PNG (300 dpi) and PDF next to it; return the paths."""
    import matplotlib.pyplot as plt
    base = figure_path(name, subdir)
    png, pdf = base.with_suffix(".png"), base.with_suffix(".pdf")
    # Préserver exactement 180 mm pour les nouvelles figures double colonne.
    bbox = None if abs(fig.get_figwidth() - 180 / 25.4) < 1e-6 else "tight"
    fig.savefig(png, dpi=dpi, bbox_inches=bbox)
    fig.savefig(pdf, bbox_inches=bbox)
    plt.close(fig)
    return png, pdf


def style():
    """Manuscript-like matplotlib defaults (single-column width, small fonts)."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "legend.frameon": False,
        "figure.figsize": (3.25, 2.6), "mathtext.default": "regular",
    })
