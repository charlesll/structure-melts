"""Figure 4: starting glass, cold compression, melt and quenched HP glass.

Select by run ID and measured temperature, never by legacy flat row index.
Unlike the historical script, assign the 302 C melt and 197 C quench to
their actual curves, and select the cold-compressed step closest to 5 GPa.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import DATA, RAMAN_AXIS
from ..figure_data import ambient_spectrum, load_dataset, save, style

NAME = "fig4_comparison"


def select_rows(d):
    """Return cold-compressed, melt and quench rows with validated states."""
    cold = np.flatnonzero((d["run_id"] == "run01_20250220_25C")
                          & (d["T_C"] == 25) & (d["Etat"] == "Verre")
                          & np.isfinite(d["borate_P_median"]))
    if not cold.size:
        raise ValueError("No cold-compressed glass measurements")
    rows = [cold[np.argmin(np.abs(d["borate_P_median"][cold] - 5))]]
    for temperature, state in [(302, "Melt"), (197, "Verre")]:
        match = np.flatnonzero((d["run_id"] == "run03_20250303_32bar")
                               & (d["T_C"] == temperature) & (d["Etat"] == state))
        if len(match) != 1:
            raise ValueError(f"Expected one {state} spectrum at {temperature} C")
        rows.append(match[0])
    return rows


def main(d=None, outdir="manuscript"):
    style()
    d = load_dataset() if d is None else d
    rows = select_rows(d)
    # Explicit composition and spectral window: the directory also holds 3%
    # and 8% H2O glasses and an HF acquisition.
    x, ambient, _ = ambient_spectrum(DATA / "session_ambient_ANN" / "ANN_5prctH2O_LF.asc")
    fig, ax = plt.subplots(figsize=(3.6, 4.0))
    ax.plot(x, ambient, color="black", lw=0.8)
    ax.text(800, 0.0011, "1 atm\n25 °C", ha="center", va="top", fontsize=7)
    ax.text(1230, 0.0003, "Glass", ha="center", fontsize=7)
    for row, offset, color, label in zip(
            rows, [0.0015, 0.003, 0.0045], ["purple", "red", "orange"],
            ["Glass", "Melt", "HP\nGlass"]):
        y = d["spectra"][row]
        if not np.isfinite(y).all():
            raise ValueError(f"Non-finite spectrum in row {row}")
        ax.plot(RAMAN_AXIS, y + offset, color=color, lw=0.8)
        ax.text(830, offset + 0.0013,
                f'~{d["borate_P_median"][row]:.1f} GPa\n{d["T_C"][row]:.0f} °C',
                ha="center", va="top", fontsize=7)
        ax.text(1240, offset + 0.0003, label, ha="center", fontsize=7)

    # Qualitative frequency guides retained from the manuscript, not fits.
    for xs, ys in [([520, 550], [0.0016, 0.0032]),
                   ([600, 635, 650, 650], [0.0016, 0.0032, 0.005, 0.0063]),
                   ([950, 950], [0.0013, 0.0058]),
                   ([1070, 1080, 1085, 1085], [0.0051, 0.0054, 0.0062, 0.0075])]:
        ax.plot(xs, ys, color="black", ls=":", lw=0.9)
    ax.set(xlim=(400, 1300), ylim=(-0.0003, 0.0082),
           xlabel=r"Raman shift (cm$^{-1}$)", 
           ylabel="Intensity (area normalised)")
    ax.set_xticks(np.arange(400, 1301, 200))
    ax.grid(True, color="0.85", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    png, _ = save(fig, NAME, outdir)
    print(f"[figures] {png}")
    for row in rows:
        print(f'  {d["run_id"][row]}: {d["T_C"][row]:g} C, '
              f'{d["borate_P_median"][row]:.3f} GPa, {d["Etat"][row]}')
    return png


if __name__ == "__main__":
    main()
