"""Figure 3: corrected silicate and O-H spectra during manuscript Run 5.

Run 5 is run05_20250408 (legacy exp_nb=4). Preserve acquisition order,
including crystallized steps, and the legacy 0.0005 vertical offset and
perceptually uniform temperature colours. LF spectra already include the Long correction;
HF spectra are baseline-corrected only, as in the original treatment.
"""
import h5py
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from ..config import RAMAN_AXIS, RESULTS
from ..figure_data import load_dataset, save, style

NAME = "fig3_Raman"
RUN_ID = "run05_20250408"
OFFSET = 0.0005


def main(d=None, outdir="manuscript"):
    """Plot existing corrected results; never reprocess raw acquisitions.

    HF is area-normalized over its finite acquired range, independently of
    the 3200-3900 display limits. Integral normalization avoids dependence on
    resampling step (the stored HF grid has 2 cm-1 spacing). Missing detector
    coverage remains NaN: no extrapolation beyond measured wavelengths.
    """
    style()
    d = load_dataset() if d is None else d
    rows = np.flatnonzero(d["run_id"] == RUN_ID)
    if rows.size == 0:
        raise ValueError(f"No spectra for {RUN_ID}")
    with h5py.File(RESULTS / "markers" / f"{RUN_ID}.h5", "r") as f:
        x_hf = f["Glass/x_HF"][()]
        if not np.array_equal(f["T_C"][()], d["T_C"][rows]):
            raise ValueError("Marker and assembled temperatures differ; regenerate upstream results")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.70, wspace=0.20)
    # Use only the spectra displayed, with limits rounded out to 10 degrees.
    temperatures = np.asarray(d["T_C"][rows], dtype=float)
    if not np.isfinite(temperatures).all():
        raise ValueError("Displayed spectra must have finite temperatures")
    t_min = 10 * np.floor(temperatures.min() / 10)
    t_max = 10 * np.ceil(temperatures.max() / 10)
    norm = Normalize(t_min, max(t_max, t_min + 10))
    try:
        from cmcrameri import cm
        cmap = cm.batlow
    except ImportError:
        warnings.warn("cmcrameri unavailable: using cividis. Install cmcrameri "
                      "for the batlow palette.", stacklevel=2)
        cmap = plt.get_cmap("cividis")
    for step, row in enumerate(rows):
        temperature = float(d["T_C"][row])
        lf, hf = d["spectra"][row], d["spectra_HF"][row]
        finite = np.isfinite(hf)
        if finite.sum() < 2 or not np.isfinite(lf).all():
            raise ValueError(f"Missing corrected spectrum at step {step}")
        area = np.trapezoid(hf[finite], x_hf[finite])
        if not np.isfinite(area) or area <= 0:
            raise ValueError(f"Invalid O-H spectral area at step {step}")
        color = cmap(norm(temperature))
        axes[0].plot(RAMAN_AXIS, lf + step * OFFSET, color=color, lw=0.65)
        axes[1].plot(x_hf, hf / area + step * OFFSET, color=color, lw=0.65)

    for ax, limits, title in zip(axes, [(400, 1270), (3200, 3850)],
                                ["a) Silicate signals", "b) O-H stretching signal"]):
        ax.set(xlim=limits, 
               ylim=(0, None), 
               xlabel=r"Raman shift (cm$^{-1}$)", title=title)
        ax.grid(True, color="0.8", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Intensity (a.u.)")
    cax = fig.add_axes([0.29, 0.78, 0.46, 0.033])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                       orientation="horizontal")
    bar.set_label("Thermocouple temperature (°C)")
    bar.ax.xaxis.set_ticks_position("top")
    bar.ax.xaxis.set_label_position("top")
    png, _ = save(fig, NAME, outdir)
    print(f"[figures] {png}")
    return png


if __name__ == "__main__":
    main()
