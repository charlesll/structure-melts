"""Near-IR speciation; absorbance is decadic, thickness in micrometres."""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rampy as rp
import gvar
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from .config import DATA, RESULTS, FIGURES, IR_EXTINCTION_REFERENCE_COMPOSITION

BIR = [[3800, 3860],[4250, 4280], [4690, 4720], [5400, 6000]]
DENSITIES = {'ANN-3prct-1.CSV': (2458, 13), 
             'ANN-5prct-1.CSV': (2449, 13),
             'ANN-8prct-1.CSV': (2434, 13)}

def peak_height(x, y, bounds, seed=42):
    """Return absorbance maximum and conditional noise SD on a 2 cm-1 grid.

    Independent Gaussian residual noise; baseline uncertainty is excluded.
    """
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 41 or not np.all(np.isfinite(y)) or not np.all(np.diff(x) > 0):
        raise ValueError('Expected finite spectrum on an increasing axis')
    # Reject isolated spikes before smoothing the broad absorption bands.
    local_median = median_filter(y, size=5, mode='nearest')
    residual = y-local_median
    noise = 1.4826*np.median(np.abs(residual-np.median(residual)))
    cleaned_absorbance = y.copy()
    outliers = np.abs(residual) > 6*max(noise, np.finfo(float).eps)
    isolated_spikes = outliers & ~np.roll(outliers, 1) & ~np.roll(outliers, -1)
    cleaned_absorbance[isolated_spikes] = local_median[isolated_spikes]
    # The 31-point window spans about 60 cm-1 on the resampled axis.
    smooth = savgol_filter(cleaned_absorbance, 31, 3)
    smoothing_residual = cleaned_absorbance-smooth
    noise = 1.4826*np.median(np.abs(smoothing_residual-np.median(smoothing_residual)))
    peak_window = (x >= bounds[0]) & (x <= bounds[1])
    if peak_window.sum() < 3:
        raise ValueError('Insufficient peak coverage')
    peak_indices = np.flatnonzero(peak_window)
    maximum_index = peak_indices[np.argmax(smooth[peak_window])]
    # Repeat the complete maximum search for each noise realization.
    random_generator = np.random.default_rng(seed)
    simulated_spectra = smooth + random_generator.normal(0, noise, (500, len(x)))
    simulated_heights = savgol_filter(simulated_spectra, 31, 3, axis=1)[:, peak_window].max(axis=1)
    smoothing_sensitivity = max(abs(savgol_filter(cleaned_absorbance, w, 3)[peak_window].max()-smooth[maximum_index]) for w in (21, 41))
    return dict(height=float(smooth[maximum_index]), ese_height=float(simulated_heights.std(ddof=1)),
                position=float(x[maximum_index]), noise=float(noise), smoothing_sensitivity=float(smoothing_sensitivity),
                edge=bool(maximum_index in (peak_indices[0], peak_indices[-1])), low_signal=bool(smooth[maximum_index] < 3*noise)), smooth


def extinction_coefficients():
    """Read means and standard uncertainties from the adopted NS4 table row.

    A missing or duplicate composition raises an error, rather than silently
    selecting a different set of coefficients. Coefficients are shared across
    samples when their GVars are constructed in main().
    """
    table = pd.read_csv(DATA / "IR/IR_extinction_coeffs.csv")
    selected = np.ones(len(table), dtype=bool)
    for oxide, value in IR_EXTINCTION_REFERENCE_COMPOSITION.items():
        selected &= np.isclose(table[oxide].to_numpy(float), value, rtol=0, atol=1e-8)
    if selected.sum() != 1:
        raise ValueError("Expected exactly one IR extinction-coefficient reference composition")
    row = table.loc[selected].iloc[0]
    coefficients = {band: (float(row['E_'+band]), float(row['ese_E_'+band]))
                    for band in ('4500', '5200')}
    if any(not np.isfinite(pair).all() or pair[0] <= 0 or pair[1] < 0
           for pair in coefficients.values()):
        raise ValueError("Invalid IR extinction coefficients or standard uncertainties")
    return coefficients


def main(interactive=False):
    # 1. Paths and sample list
    IR_FIGURES = FIGURES / "IR"
    RESULTS_IR = RESULTS / "IR"
    IR_FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS_IR.mkdir(parents=True, exist_ok=True)

    # Get the list of spectra
    list_sp = pd.read_csv(DATA / "IR/ANN_IR_list.csv")

    # Adopted NS4 peak-height coefficients from the supplied comparison table.
    # The confirmed 4500 cm^-1 coefficient is 0.44 ±0.03 (1 SD).
    coefficients = extinction_coefficients()
    extinction_heights = {band: gvar.gvar(*pair) for band, pair in coefficients.items()}

    # One panel per sample, with the same absorbance scale.
    figure, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True, sharey=True)
    diagnostic, diagnostic_axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

    for count, name in enumerate(list_sp.name):
        # 2. Read the spectrum and fit the sixth-order baseline
        sp = pd.read_csv(DATA / "IR" / name, delimiter=";", decimal=",", header=None).values
        sp = sp[sp[:, 0] > 3775, :]
        sp = sp[np.argsort(sp[:, 0])]

        corrIR, baseline = rp.baseline(
            sp[:, 0], sp[:, 1], BIR, method="poly", polynomial_order=6
        ) # 6 order poly like in Yamashita

        ax = axes[count]
        ax.plot(sp[:, 0], sp[:, 1], label=name)
        ax.plot(sp[:, 0], baseline[:, 0], "--", label="Baseline")
        for lower, upper in BIR:
            ax.axvspan(lower, min(upper, sp[:, 0].max()), color="grey", alpha=0.08)
        ax.set_ylabel("Absorbance")
        ax.legend()

        if interactive:
            working_plot = rp.plot_spectrum(sp[:, 0], sp[:, 1], baselines=[baseline])
            working_plot.show()

        # 3. Resample at 2 cm-1, then measure the two peak heights
        x = np.arange(4250, 5400, 2.0)
        corrected_absorbance = np.interp(x, sp[:, 0], corrIR[:, 0])

        peak_4500, smooth_4500 = peak_height(x, corrected_absorbance, (4400, 4600))
        peak_5200, smooth_5200 = peak_height(x, corrected_absorbance, (5000, 5350))

        # Keep raw absorbance heights and quality indicators in the output table.
        for band, measurements in [("4500", peak_4500), ("5200", peak_5200)]:
            for quantity, value in measurements.items():
                list_sp.loc[count, quantity + "_" + band] = value

        ax_diagnostic = diagnostic_axes[count]
        ax_diagnostic.plot(x, corrected_absorbance, color="grey", lw=0.6, label=name)
        for measurements, smoothed, bounds in [
            (peak_4500, smooth_4500, (4400, 4600)),
            (peak_5200, smooth_5200, (5000, 5350)),
        ]:
            peak_window = (x >= bounds[0]) & (x <= bounds[1])
            ax_diagnostic.plot(x[peak_window], smoothed[peak_window])
            ax_diagnostic.plot(measurements["position"], measurements["height"], "o")
        ax_diagnostic.set_ylabel("Corrected absorbance")
        ax_diagnostic.legend()

        # 4. Correct heights and areas for thickness (micrometres -> cm)
        thickness = gvar.gvar(
            list_sp.loc[count, "epaisseur_micrometers"],
            list_sp.loc[count, "ese_epaisseur"],
        ) * 1.0e-4

        height_4500 = gvar.gvar(peak_4500["height"], peak_4500["ese_height"]) / thickness
        height_5200 = gvar.gvar(peak_5200["height"], peak_5200["ese_height"]) / thickness

        window_4500 = (x >= 4250) & (x < 4740)
        window_5200 = (x >= 4700) & (x < 5400)
        y_4500 = gvar.gvar(
            corrected_absorbance[window_4500],
            np.full(window_4500.sum(), peak_4500["noise"]),
        ) / thickness
        y_5200 = gvar.gvar(
            corrected_absorbance[window_5200],
            np.full(window_5200.sum(), peak_5200["noise"]),
        ) / thickness
        area_4500 = np.trapezoid(y_4500, x=x[window_4500])
        area_5200 = np.trapezoid(y_5200, x=x[window_5200])

        # 5. Water contents from areas (original extinction coefficients)
        # Density in kg/m3; concentrations in wt% as H2O equivalent.
        rho = gvar.gvar(*DENSITIES[name])
        # E_area_4500 = extinction_areas["4500"]
        # E_area_5200 = extinction_areas["5200"]
        # C_OH_area = 1802 * area_4500 / (rho * E_area_4500)
        # C_H2Omol_area = 1802 * area_5200 / (rho * E_area_5200)
        # C_H2O_total_area = C_OH_area + C_H2Omol_area

        # 6. Water contents from heights (composition-dependent coefficients)
        E_height_4500 = extinction_heights["4500"]
        E_height_5200 = extinction_heights["5200"]
        C_OH_height = 1802 * height_4500 / (rho * E_height_4500)
        C_H2Omol_height = 1802 * height_5200 / (rho * E_height_5200)
        C_H2O_total_height = C_OH_height + C_H2Omol_height

        # Store means and standard uncertainties separately for CSV export.
        results = {
            "height_cm_4500": height_4500,
            "height_cm_5200": height_5200,
            "area_4500": area_4500,
            "area_5200": area_5200,
            # "C_area_4500": C_OH_area,
            # "C_area_5200": C_H2Omol_area,
            # "C_area_total": C_H2O_total_area,
            "C_height_4500": C_OH_height,
            "C_height_5200": C_H2Omol_height,
            "C_height_total": C_H2O_total_height,
        }
        for quantity, value in results.items():
            list_sp.loc[count, quantity] = gvar.mean(value)
            list_sp.loc[count, "ese_" + quantity] = gvar.sdev(value)

        print(name)
        print(f"  C_OH from heights (wt%) = {gvar.mean(C_OH_height):.3f} ± {gvar.sdev(C_OH_height):.3f} (1 SD)")
        # print("  C_OH from areas (wt%) =", C_OH_area)
        print(f"  C_H2Omol from heights (wt%) = {gvar.mean(C_H2Omol_height):.3f} ± {gvar.sdev(C_H2Omol_height):.3f} (1 SD)")
        # print("  C_H2Omol from areas (wt%) =", C_H2Omol_area)
        print(f"  C_H2O total from heights (wt%) = {gvar.mean(C_H2O_total_height):.3f} ± {gvar.sdev(C_H2O_total_height):.3f} (1 SD)")
        # print("  C_H2O total from areas (wt%) =", C_H2O_total_area)

    # 7. Save figures and numerical results
    axes[-1].set_xlabel("Wavenumber (cm$^{-1}$)")
    figure.tight_layout()
    figure.savefig(IR_FIGURES / "IR_spectra_baselines.pdf")
    figure.savefig(IR_FIGURES / "IR_spectra_baselines.png", dpi=300)
    plt.close(figure)

    diagnostic_axes[-1].set_xlabel("Wavenumber (cm$^{-1}$)")
    diagnostic.tight_layout()
    diagnostic.savefig(IR_FIGURES / "IR_peak_diagnostics.pdf")
    diagnostic.savefig(IR_FIGURES / "IR_peak_diagnostics.png", dpi=300)
    plt.close(diagnostic)

    list_sp.to_csv(RESULTS_IR / "IR_results.csv", index=False)
    metadata = dict(
        coefficient_table="data/IR/IR_extinction_coeffs.csv",
        reference_composition=IR_EXTINCTION_REFERENCE_COMPOSITION,
        peak_height_coefficients={band: dict(mean=pair[0], sd=pair[1])
                                  for band, pair in coefficients.items()},
        coefficient_units="L mol^-1 cm^-1", uncertainty="standard deviations (1 sigma)",
        densities_kg_m3={name: dict(mean=value[0], sd=value[1]) for name, value in DENSITIES.items()},
        thickness_units="micrometres converted to cm by 1e-4",
        method="Peak heights; sixth-order polynomial baseline; 2 cm^-1 grid; Savitzky-Golay 31 points/order 3",
        limitations="Conditional noise/parameter propagation; baseline and coefficient-transfer errors not quantified")
    (RESULTS_IR / "IR_metadata.json").write_text(json.dumps(metadata, indent=2)+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    arguments = parser.parse_args()
    main(interactive=arguments.interactive)
