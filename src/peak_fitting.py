"""Gaussian-prior Raman peak fitting for the silicate region.

The five-component model follows the final analysis in the legacy
``Peak_fitting.ipynb``: TOH, Q2, T2S, Q3, and Q4. Fits are saved in HDF5 and
publication-sized diagnostics are written under ``figures/raman``.
"""
import argparse
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


from .config import FIGURES, RESULTS

DATASET = RESULTS / "dataset.h5"
FIT_RESULTS = RESULTS / "peak_fits.h5"
FIT_FIGURES = FIGURES / "raman"
PEAK_NAMES = ("TOH", "Q2", "T2S", "Q3", "Q4")
PEAK_PRIORS = (
    {"kind": "gaussian", "amplitude": 0.05, "center": 900.0, "width": 34.0,
     "center_sd": 10.0, "width_sd": 2.0, "amplitude_sd": 0.1},
    {"kind": "gaussian", "amplitude": 0.25, "center": 950.0, "width": 37.0,
     "center_sd": 5.0, "width_sd": 2.0, "amplitude_sd": 0.1},
    {"kind": "gaussian", "amplitude": 0.30, "center": 1020.0, "width": 35.0,
     "center_sd": 5.0, "width_sd": 2.0, "amplitude_sd": 0.2},
    {"kind": "pseudovoigt", "amplitude": 0.70, "center": 1085.0, "width": 27.1,
     "fraction": 0.25, "center_sd": 3.0, "width_sd": 2.0, "amplitude_sd": 0.02,
     "fraction_sd": 0.02},
    {"kind": "gaussian", "amplitude": 0.30, "center": 1135.0, "width": 35.0,
     "center_sd": 2.0, "width_sd": 1.5, "amplitude_sd": 0.2},
)


def _component(x, amplitude, center, width, kind, fraction=0.0):
    """Evaluate a Gaussian or pseudo-Voigt peak parameterized by HWHM."""
    scaled = (x - center) / width
    gaussian = np.exp(-np.log(2.0) * scaled**2)
    if kind == "gaussian":
        return amplitude * gaussian
    lorentzian = 1.0 / (1.0 + scaled**2)
    return amplitude * ((1.0 - fraction) * gaussian + fraction * lorentzian)


def _model(x, parameters):
    """Sum the five components from a flat parameter vector."""
    y = np.zeros_like(x, dtype=float)
    offset = 0
    for prior in PEAK_PRIORS:
        amplitude, center, width = parameters[offset:offset + 3]
        offset += 3
        fraction = 0.0
        if prior["kind"] == "pseudovoigt":
            fraction = parameters[offset]
            offset += 1
        y = y + _component(x, amplitude, center, width, prior["kind"], fraction)
    return y


def _peak_quantities(parameters):
    """Calculate peak centers and integrated areas from fitted parameters."""
    positions, areas = [], []
    offset = 0
    for prior in PEAK_PRIORS:
        amplitude, center, width = parameters[offset:offset + 3]
        offset += 3
        fraction = 0.0
        if prior["kind"] == "pseudovoigt":
            fraction = parameters[offset]
            offset += 1
        gaussian_area = width * np.sqrt(np.pi / np.log(2.0))
        lorentzian_area = np.pi * width
        area = amplitude * (
            (1.0 - fraction) * gaussian_area + fraction * lorentzian_area
        )
        positions.append(center)
        areas.append(area)
    return np.asarray(positions), np.asarray(areas)


def _prior_vectors():
    """Moyennes et SD des 16 priors gaussiens, sans bornes dures."""
    initial, uncertainty = [], []
    for prior in PEAK_PRIORS:
        initial.extend((prior['amplitude'],prior['center'],prior['width']))
        uncertainty.extend((prior['amplitude_sd'],prior['center_sd'],prior['width_sd']))
        if prior['kind'] == 'pseudovoigt':
            initial.append(prior['fraction']); uncertainty.append(prior['fraction_sd'])
    return np.asarray(initial), np.asarray(uncertainty)


def _prepare_spectrum(x, y):
    """Select 820–1270 cm⁻¹, remove endpoint baseline, and scale by its maximum."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 820.0) & (x <= 1270.0)
    x, y = x[valid], y[valid]
    if len(x) < 100:
        raise ValueError("Fewer than 100 finite Raman channels in the fitting region.")
    endpoint = (x <= 830.0) | (x >= 1220.0)
    baseline = np.polyval(np.polyfit(x[endpoint], y[endpoint], 1), x)
    corrected = y - baseline
    scale = float(np.max(corrected))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Spectrum has no positive intensity after baseline subtraction.")
    return x, corrected / scale


def fit_spectrum(x, y):
    """Ajustement MAP gaussien lsqfit et propagation gvar (16 paramètres).

    Même fond et estimation du bruit que la version Python précédente afin
    d'isoler le changement de fonction objectif. Priors indépendants par
    paramètre, sans bornes dures. Covariance postérieure locale non rééchelonnée.
    """
    import gvar as gv
    import lsqfit
    x_fit, y_fit = _prepare_spectrum(x, y)
    initial, prior_sd = _prior_vectors()
    noise = max(float(np.std(y_fit-gaussian_filter1d(y_fit, sigma=12.0))), 1e-4)
    # Les GVar de données sont indépendants; le fit construit les corrélations
    # postérieures entre paramètres par le recouvrement des bandes.
    prior = gv.gvar(initial, prior_sd)
    observed = gv.gvar(y_fit, np.full_like(y_fit, noise))
    fit = lsqfit.nonlinear_fit(data=(x_fit, observed), prior=prior, p0=initial,
                              fcn=_model, fitter='scipy_least_squares',
                              maxit=3000, tol=1e-9)
    if fit.stopping_criterion == 0:
        raise RuntimeError('Gaussian-prior fit reached its iteration limit')
    posterior = np.asarray(fit.p)
    positions, areas = _peak_quantities(posterior)
    parameters = np.asarray(gv.mean(posterior), float)
    if not np.isfinite(parameters).all():
        raise RuntimeError('Nonfinite posterior parameters')
    widths, mixtures = [], []
    offset = 0
    for component in PEAK_PRIORS:
        widths.append(posterior[offset+2]); offset += 3
        fraction = 0.
        if component['kind'] == 'pseudovoigt':
            fraction = posterior[offset]; offset += 1
        mixtures.append(fraction)
    # Les priors sont gaussiens réels : signaler explicitement un optimum
    # non physique plutôt que masquer une contrainte par troncature.
    physical = (np.all(gv.mean(areas) > 0) and np.all(gv.mean(widths) > 0)
                and 0 <= gv.mean(mixtures[3]) <= 1)
    if not physical:
        raise RuntimeError('Nonphysical Gaussian posterior mean (area/width/mixture)')
    area_fractions = areas/sum(areas)
    ratio = areas[1]/areas[3]
    return {
        'positions': gv.mean(positions), 'areas': gv.mean(areas),
        'position_sd': gv.sdev(positions), 'area_sd': gv.sdev(areas),
        'area_covariance': gv.evalcov(areas),
        'parameter_covariance': gv.evalcov(posterior),
        'area_fractions': gv.mean(area_fractions),
        'area_fraction_covariance': gv.evalcov(area_fractions),
        'qn_ratio': float(gv.mean(ratio)), 'qn_ratio_sd': float(gv.sdev(ratio)),
        'widths': gv.mean(widths), 'fractions': gv.mean(mixtures),
        'x': x_fit, 'observed': y_fit, 'fitted': _model(x_fit,parameters),
        'parameters': parameters, 'success': True, 'cost': float(fit.chi2/2),
        'noise': noise, 'chi2': float(fit.chi2), 'dof': int(fit.dof),
        'Q': float(fit.Q), 'stopping_criterion': int(fit.stopping_criterion),
    }


def _fit_run(run_id, group, x_axis):
    """Fit all glass spectra from one HDF5 run."""
    spectra = group["spectra"][()]
    pressure = group["borate_P_median"][()]
    temperature = group["T_C"][()]
    n_steps = len(temperature)
    positions = np.full((n_steps, len(PEAK_NAMES)), np.nan)
    areas = np.full_like(positions, np.nan)
    position_sd = np.full_like(positions, np.nan)
    area_sd = np.full_like(positions, np.nan)
    widths = np.full_like(positions, np.nan)
    fractions = np.full_like(positions, np.nan)
    area_covariance = np.full((n_steps, 5, 5), np.nan)
    posterior_fields = {
        'parameters': np.full((n_steps,16),np.nan),
        'parameter_covariance': np.full((n_steps,16,16),np.nan),
        'area_fractions': np.full((n_steps,5),np.nan),
        'area_fraction_covariance': np.full((n_steps,5,5),np.nan),
        **{k:np.full(n_steps,np.nan) for k in
           ['qn_ratio','qn_ratio_sd','noise','chi2','dof','Q','stopping_criterion']},
    }
    costs = np.full(n_steps, np.nan)
    fitted_spectra = []
    for i, spectrum in enumerate(spectra):
        try:
            fit = fit_spectrum(x_axis, spectrum)
        except (ValueError, RuntimeError) as error:
            print(f"[peak_fitting] {run_id} step {i}: {error}")
            fitted_spectra.append(None)
            continue
        positions[i], areas[i] = fit["positions"], fit["areas"]
        position_sd[i], area_sd[i] = fit["position_sd"], fit["area_sd"]
        widths[i], fractions[i] = fit["widths"], fit["fractions"]
        area_covariance[i] = fit["area_covariance"]
        for key in posterior_fields:
            posterior_fields[key][i] = fit[key]
        costs[i] = fit["cost"]
        fitted_spectra.append(fit)
    return {
        "run_id": run_id,
        "temperature": temperature,
        "pressure": pressure,
        "positions": positions,
        "areas": areas,
        "position_sd": position_sd,
        "area_sd": area_sd,
        "widths": widths,
        "fractions": fractions,
        "costs": costs,
        "area_covariance": area_covariance,
        "fitted_spectra": fitted_spectra,
        **posterior_fields,
    }


def _plot_run(result, path):
    """Plot fitted spectral envelopes and peak-area fractions versus temperature."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    valid = np.isfinite(result["positions"]).all(axis=1)
    temperatures = result["temperature"][valid]
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(np.nanmin(temperatures), np.nanmax(temperatures))
    for index, fit in enumerate(result["fitted_spectra"]):
        if fit is None:
            continue
        color = cmap(norm(result["temperature"][index]))
        axes[0].plot(fit["x"], fit["observed"], color=color, alpha=0.55, linewidth=0.8)
        axes[0].plot(fit["x"], fit["fitted"], color=color, linewidth=0.8)
    axes[0].set(xlabel="Raman shift (cm$^{-1}$)", ylabel="Normalised intensity",
                title="Silicate-band fits")
    axes[0].grid(alpha=0.2)
    if len(temperatures):
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[0],
                     label="Temperature (°C)")

    fractions = result['area_fractions']
    fraction_sd = np.sqrt(np.maximum(np.diagonal(result['area_fraction_covariance'], axis1=1, axis2=2),0))
    markers = ("o", "s", "^", "D", "P")
    for column, name in enumerate(PEAK_NAMES):
        axes[1].errorbar(result['temperature'], fractions[:, column],
                         yerr=fraction_sd[:,column], fmt='none', ecolor='0.6', lw=.5)
        axes[1].scatter(result["temperature"], fractions[:, column],
                        c=result["temperature"], cmap=cmap, norm=norm,
                        s=28, marker=markers[column], edgecolor="black",
                        linewidth=0.25, label=name)
    axes[1].set(xlabel="Temperature (°C)", ylabel="Fitted area fraction",
                title="Peak areas versus temperature")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(result["run_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_single_fit(result, index, path):
    """One spectrum: data, total fit, every band and the residual."""
    fit = result["fitted_spectra"][index]
    x, parameters = fit["x"], fit["parameters"]
    fig, (ax, res) = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True,
                                  gridspec_kw={"height_ratios": [4, 1]}, constrained_layout=True)
    ax.plot(x, fit["observed"], "k.", ms=2.5, label="Measured")
    ax.plot(x, fit["fitted"], "r-", lw=1.4, label="Total fit")
    offset = 0
    for column, prior in enumerate(PEAK_PRIORS):
        amplitude, center, width = parameters[offset:offset + 3]
        offset += 3
        fraction = 0.0
        if prior["kind"] == "pseudovoigt":
            fraction = parameters[offset]
            offset += 1
        band = _component(x, amplitude, center, width, prior["kind"], fraction)
        ax.fill_between(x, band, alpha=0.3, color=f"C{column}")
        ax.plot(x, band, color=f"C{column}", lw=1,
                label=f"{PEAK_NAMES[column]} ({center:.0f} cm$^{{-1}}$)")
    ax.set(ylabel="Normalised intensity")
    ax.legend(frameon=False, fontsize=8)
    res.plot(x, fit["observed"] - fit["fitted"], "k-", lw=0.7)
    res.axhline(0, color="0.6", lw=0.5)
    res.set(xlabel="Raman shift (cm$^{-1}$)", ylabel="Residual")
    ax.set_title(f'{result["run_id"]}, step {index:02d}: T = {result["temperature"][index]:.0f} °C, '
                 f'P = {result["pressure"][index]:.2f} GPa', fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_analysis(dataset_path=DATASET, output_path=FIT_RESULTS, figure_dir=FIT_FIGURES):
    """Fit all runs and write peak parameters and figures."""
    import h5py
    import gvar
    import lsqfit
    fits = []
    with h5py.File(dataset_path, "r") as data:
        x_axis = data["x_axis"][()]
        for run_id, group in data.items():
            if run_id in ("flat", "x_axis"):
                continue
            result = _fit_run(run_id, group, x_axis)
            fits.append(result)

    failures = sum(np.isnan(r['costs']).sum() for r in fits)
    if failures:
        raise RuntimeError(f'{failures} peak fits failed; existing output preserved')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Écriture atomique : ne remplacer le fichier actif qu'après succès complet.
    fd, temporary = tempfile.mkstemp(prefix=output_path.stem+'_', suffix='.h5', dir=output_path.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, "w") as output:
            import gvar as gv
            import lsqfit
            output.attrs['method'] = 'lsqfit Gaussian priors; gvar propagation'
            output.attrs['gvar_version'] = gv.__version__
            output.attrs['lsqfit_version'] = lsqfit.__version__
            output.attrs['uncertainty'] = 'local Gaussian posterior; conditional on model and fixed noise/baseline'
            output.attrs['preprocessing'] = '820-1270 cm-1; linear endpoint baseline then max normalization'
            output.attrs['noise_method'] = 'std(data - gaussian_filter1d(data, sigma=12)), floor 1e-4'
            output.attrs['hard_bounds'] = False
            initial, prior_sd = _prior_vectors()
            output.attrs['prior_mean'] = initial
            output.attrs['prior_sd'] = prior_sd
            names=[]
            for name, component in zip(PEAK_NAMES,PEAK_PRIORS):
                names.extend(name+'.'+k for k in ['amplitude','center','hwhm'])
                if component['kind'] == 'pseudovoigt': names.append(name+'.lorentzian_fraction')
            output.attrs['parameter_names'] = np.asarray(names,dtype='S')
            output.attrs["peak_names"] = np.asarray(PEAK_NAMES, dtype="S")
            for result in fits:
                group = output.create_group(result["run_id"])
                for key in ("temperature", "pressure", "positions", "position_sd", "areas",
                            "area_sd", "area_covariance", "widths", "fractions", "costs"):
                    group.create_dataset(key, data=result[key])
                for key in ['parameters','parameter_covariance','area_fractions',
                            'area_fraction_covariance','qn_ratio','qn_ratio_sd','noise',
                            'chi2','dof','Q','stopping_criterion']:
                    group.create_dataset(key, data=result[key])
                group.attrs["failed_fits"] = int(np.isnan(result["costs"]).sum())
                _plot_run(result, Path(figure_dir) / f'{result["run_id"]}_peak_analysis.png')
                for index, fit in enumerate(result["fitted_spectra"]):
                    if fit is None:
                        continue
                    name = (f'{result["run_id"]}_step{index:02d}_T{result["temperature"][index]:.0f}C'
                            f'_P{result["pressure"][index]:.2f}GPa_fit.png')
                    _plot_single_fit(result, index, Path(figure_dir) / "individual_fits" / result["run_id"] / name)
        Path(temporary).replace(output_path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    print(f"[peak_fitting] results -> {output_path}")
    return fits


def main(argv=None):
    """Command-line entry point for Raman peak fitting."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=FIT_RESULTS)
    parser.add_argument("--figures", type=Path, default=FIT_FIGURES)
    args = parser.parse_args(argv)
    run_analysis(args.dataset, args.output, args.figures)


if __name__ == "__main__":
    main()
