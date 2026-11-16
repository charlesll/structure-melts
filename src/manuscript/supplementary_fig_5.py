"""Supplementary Figure 5: selected crystallised Raman spectra.

Reproduce with ``python -m src.manuscript.supplementary_fig_5`` or ``python -m src.figures Supplementary_Fig_5``.
The spectra, annotations and cubic baseline regions are those supplied for S5.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rampy as rp

from ..config import DATA, ROOT
from ..figure_data import save, style

NAME = "Supplementary_Fig_5"
EXAMPLES = (
    ("session_20250220_ANN", "15bar_322C_Ctx_LF", "322 °C, 2.4 GPa", (695., 1095.)),
    ("session_20250303_ANN", "34bar_312C_Ctx_3", "312 °C, 5.5 GPa", (695., 1091.)),
    ("session_20250408_ANN_5H2O", "50bar_394C_Ctx_LF_1", "394 °C, 10.6 GPa",
     (447., 392., 670., 720., 992., 1032., 1118.)),
)
BASELINE_ROI = np.array([[500., 520.], [800., 840.], [1200., 1300.]])


def load_ctx_spectra(folder):
    """Read Ctx ASC files as wavelength (nm), intensity; fail on missing data."""
    folder = Path(folder)
    files = sorted(folder.glob("*Ctx*.asc"))
    if not files:
        raise FileNotFoundError(f"No Ctx spectra found in {folder}")
    spectra = {}
    for file in files:
        values = np.loadtxt(file, delimiter=",", usecols=(0, 1), ndmin=2)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(f"Invalid spectrum: {file}")
        spectra[file.stem] = values
    return spectra


def main(d=None):
    """Generate S5 directly from raw spectra, without modifying pipeline results."""
    style()
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 4.8))
    sources = []
    for ax, (session, stem, label, markers) in zip(axes, EXAMPLES):
        spectra = load_ctx_spectra(DATA / session)
        if stem not in spectra:
            raise FileNotFoundError(f"Required spectrum missing: {DATA / session / (stem + '.asc')}")
        spectrum = spectra[stem]
        x = rp.nm_to_invcm(spectrum[:, 0])
        corrected, _ = rp.baseline(x, spectrum[:, 1], method="poly",
                                   roi=BASELINE_ROI, polynomial_order=3)
        maximum = np.max(corrected)
        if not np.isfinite(corrected).all() or maximum <= 0:
            raise ValueError(f"Invalid baseline-corrected spectrum: {stem}")
        ax.plot(x, corrected / maximum, linewidth=.5, color="k")
        ax.set_ylabel("Intensity (a.u.)")
        ax.annotate(label, (.05, .8), xycoords="axes fraction")
        ax.grid()
        ax.vlines(markers, 0, 1.9, linestyles="--", color="k", linewidth=.5)
        sources.append(dict(source_file=str((DATA / session / (stem + '.asc')).relative_to(ROOT)),
                            annotation=label, markers_cm_inverse=list(markers)))
    axes[-1].set_xlabel("Raman shift (cm$^{-1}$)")
    fig.tight_layout()
    png, _ = save(fig, NAME, "manuscript", dpi=300)
    png.with_suffix(".json").write_text(json.dumps(dict(
        examples=sources, baseline_roi_cm_inverse=BASELINE_ROI.tolist(),
        baseline_polynomial_order=3, normalisation="Maximum after baseline subtraction",
        annotation_note="Temperature and pressure labels supplied with the original S5 script; not recalculated",
    ), indent=2) + "\n")
    print(f"[figures] {png}")
    return png


if __name__ == "__main__":
    main()
