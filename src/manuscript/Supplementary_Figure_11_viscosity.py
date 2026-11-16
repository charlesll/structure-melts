"""Supplementary Figure 11: pressure dependence of ANN melt viscosity.

Run from the repository root:
    python -m src.manuscript.Supplementary_Figure_11_viscosity

Solid curves: MYEGA scenarios at 1400, 1500 and 1600 °C, using the same
linear spectral Tg(P) regression as the absolute T–P diagnostic. Dashed
curves: direct gpvisc predictions for the supplied hydrous ANN composition.
Pressure increases downward; viscosity is log10(eta / Pa s) on the top axis.
"""
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ..ann_viscosity import load_gpvisc_result, myega_fragility_interval
from ..config import ANN_MYEGA_A, ANN_MYEGA_FRAGILITY, ANN_MYEGA_FRAGILITY_SD, RESULTS
from ..figure_data import save, style
from ..manuscript_data import transitions
from ..tg_pressure_regression import fit_interval_line

NAME = 'Supplementary_Figure_11'
TEMPERATURES_C = (1400., 1500., 1600.)
COLORS = ('#0072B2', '#009E73', '#D55E00')


def main(d=None):
    """Write the figure, a numerical CSV and a provenance JSON sidecar."""
    style()
    gp_result = load_gpvisc_result()
    pressure = np.asarray(gp_result['P_GPa'])

    # Refit the current acquisition intervals with the established likelihood.
    # Do not replace the absolute intercept by the GP isoviscous temperature:
    # these remain two distinct definitions of Tg.
    fit = fit_interval_line(transitions(), profile=False)
    tg_c = fit['intercept_C'] + fit['slope_C_per_GPa'] * pressure
    tg_k = tg_c + 273.15

    fig, ax = plt.subplots(figsize=(180 / 25.4, 4.8), layout='constrained')
    output = []
    for temperature_c, color in zip(TEMPERATURES_C, COLORS):
        # Transform fragility quantiles directly: no symmetric-error approximation.
        log_eta_myega, lower, upper = myega_fragility_interval(
            temperature_c + 273.15, tg_k)
        direct = gp_result['high_temperature_C'][str(int(temperature_c))]
        log_eta_gp = np.asarray(direct['log10_eta'])

        ax.fill_betweenx(pressure, lower, upper, color=color, alpha=.16, linewidth=0)
        ax.plot(log_eta_myega, pressure, color=color, linewidth=1.5)
        ax.plot(log_eta_gp, pressure, color=color, linewidth=1.3, linestyle='--')
        for index, p in enumerate(pressure):
            output.append({
                'T_C': temperature_c, 'P_GPa': float(p),
                'spectral_linear_Tg_K': float(tg_k[index]),
                'MYEGA_log10_eta': float(log_eta_myega[index]),
                'MYEGA_fragility_lower95_log10_eta': float(lower[index]),
                'MYEGA_fragility_upper95_log10_eta': float(upper[index]),
                'gpvisc_log10_eta': float(log_eta_gp[index]),
                'gpvisc_predictive_sd_log10_eta': direct['predictive_sd_log10_eta'][index]})

    # Separate temperature colors from model line styles to avoid six labels.
    temperature_handles = [Line2D([], [], color=color, label=f'{temperature:g} °C')
                           for temperature, color in zip(TEMPERATURES_C, COLORS)]
    model_handles = [Line2D([], [], color='0.25', label='MYEGA · i-Melt parameters'),
                     Line2D([], [], color='0.25', linestyle='--', label='gpvisc · direct GP')]
    ax.legend(handles=temperature_handles + model_handles, loc='lower left', fontsize=8)
    ax.set(xlabel=r'Viscosity, $\log_{10}(\eta/\mathrm{Pa\,s})$',
           ylabel='Pressure (GPa)', ylim=(13., 0.))
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.grid(alpha=.15, linewidth=.5)
    fig.supxlabel(
        f'i-Melt: Ae = {ANN_MYEGA_A:g}, m = {ANN_MYEGA_FRAGILITY:g} ± {ANN_MYEGA_FRAGILITY_SD:g} (1σ)\n'
        'Shading: 95% fragility-only interval; Tg(P) and Ae fixed', fontsize=8)
    png, _ = save(fig, NAME, 'manuscript', dpi=600)

    with (RESULTS / f'{NAME}.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    metadata = {
        'MYEGA_Ae': ANN_MYEGA_A, 'MYEGA_fragility': ANN_MYEGA_FRAGILITY,
        'MYEGA_fragility_sd': ANN_MYEGA_FRAGILITY_SD,
        'MYEGA_parameter_source': 'i-Melt values supplied by user',
        'MYEGA_interval': '95% Gaussian fragility quantiles transformed through MYEGA; Tg(P) and Ae fixed; no joint uncertainty band',
        'temperatures_C': TEMPERATURES_C, 'linear_Tg_fit': fit,
        'Tg_source': 'Current interval regression, same as figures/manuscript/Supplementary_Figure_6',
        'gpvisc_source': 'results/ann_gpvisc.json',
        'gpvisc_version': gp_result['gpvisc_version'],
        'gpvisc_model_number': gp_result['model_number'],
        'composition_wt_supplied': gp_result['composition_wt_supplied'],
        'limitations': 'MYEGA uses fixed Ae and spectral Tg(P), with fragility-only uncertainty. GP and spectral Tg definitions differ. No viscosity measurements or full uncertainty bands. High-pressure hydrous ANN may be extrapolative.',
        'values': output}
    png.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(png)
    return png


if __name__ == '__main__':
    main()
