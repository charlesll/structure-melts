"""Figure 7: P–T paths, split brackets and conditional ambient extrapolation."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ..config import ANN_FICTIVE_REFERENCE, SIGMA_T, RESULTS
from ..pressure import load_dataset
from ..figure_data import save, style, STATE_STYLE, RUN_LABEL
from ..manuscript_data import WIDTH, RUN_COLORS, transitions
from ..tg_pressure_regression import fit_interval_line
from .supplementary_fig6 import NAME as SUPPLEMENTARY_NAME, plot_interval_regression
NAME = 'fig7_transitions'
# The runner maps either output name to this single generation step.
ADDITIONAL_OUTPUTS = (SUPPLEMENTARY_NAME,)


def regression_summary(rows):
    """Fit only observed split intervals; the fictive-temperature reference is not a fit anchor."""
    result = fit_interval_line(rows)
    result['without_pressure_spread'] = fit_interval_line(rows, pressure_spread=False)
    result['leave_one_run_out'] = []
    for row in rows:
        remaining = [r for r in rows if r['run_id'] != row['run_id']]
        fit = fit_interval_line(remaining, profile=False)
        result['leave_one_run_out'].append(dict(omitted_run=row['run_id'],
                                                intercept_C=fit['intercept_C'],
                                                slope_C_per_GPa=fit['slope_C_per_GPa'],
                                                residual_scatter_C=fit['residual_scatter_C']))
    # Resolution sensitivity: endpoints are configurations, not random errors.
    import itertools
    slopes, intercepts = [], []
    for bits in itertools.product([0, 1], repeat=len(rows)):
        temperature = [row['T_right'] if bit else row['T_left']
                       for row, bit in zip(rows, bits)]
        slope, intercept = np.polyfit([r['P'] for r in rows], temperature, 1)
        slopes.append(slope)
        intercepts.append(intercept)
    result['endpoint_OLS_sensitivity'] = dict(intercept_range_C=[min(intercepts), max(intercepts)],
                                               slope_range_C_per_GPa=[min(slopes), max(slopes)],
                                               note='Deterministic resolution sensitivity, not a confidence interval; no pressure errors in this control')
    result['limiting_fictive_temperature_reference'] = dict(
        **ANN_FICTIVE_REFERENCE, plotted_pressure_GPa=0.0,
        quantity="Limiting fictive temperature", used_in_regression=False)
    return result


def main(d=None):
    style()
    datasets = load_dataset()
    rows = transitions()
    regression = regression_summary(rows)
    # S6 reuses this exact fit, including its profiles; do not fit it a second time.
    plot_interval_regression(rows, regression)
    intercept = regression['intercept_C']
    lower, upper = regression['intercept_profile95_C']
    (RESULTS / 'fig7_interval_regression.json').write_text(json.dumps(regression, indent=2) + '\n')
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 4.3),
                               gridspec_kw={'width_ratios': [2, 1]}, layout='constrained')
    for run, data in datasets.items():
        for state, (marker_color, marker) in STATE_STYLE.items():
            mask = data['Etat'] == state
            a.errorbar(data['borate_P_median'][mask], data['T_C'][mask],
                       xerr=data['borate_P_std'][mask], yerr=SIGMA_T,
                       fmt=marker, ms=3.5, color=marker_color, mec='k', mew=.5, lw=.5)
    for row in rows:
        color = RUN_COLORS[row['run_id']]
        supported = row['status'] == 'supported'
        face = color if supported else 'white'
        # Preserve the actual bracket polygons, separate from TC error bars.
        x_fill = [row['P_left']-row['P_std_left'], row['P_left']+row['P_std_left'],
                  row['P_right']+row['P_std_right'], row['P_right']-row['P_std_right']]
        y_fill = [row['T_left'], row['T_left'], row['T_right'], row['T_right']]
        a.fill(x_fill, y_fill, color='orange', alpha=.3, ec='none')
        b.plot([row['Tc'], row['Tc']], [row['T_left'], row['T_right']],
               color=color, ls='-' if supported else ':', lw=3)
        b.errorbar(row['Tc'], row['Tg'], xerr=SIGMA_T, yerr=SIGMA_T,
                   fmt='o', mfc=face, color=color, ms=4,
                   label=RUN_LABEL[row['run_id']]+(' (candidate)' if not supported else ''))
    # Only the extrapolated intercept is plotted; no connecting orange line.
    a.errorbar(0., intercept, yerr=[[intercept-lower], [upper-intercept]],
               fmt='k*', ms=8, capsize=3, lw=1.2, zorder=6)
    a.annotate(f'$T_g(0)$ = {intercept:.0f} °C', xy=(0, intercept),
               xytext=(.55, 267), fontsize=6,
               arrowprops=dict(arrowstyle='-', color='k', lw=.6))
    # Independent as-received glass reference, shown at ambient pressure.
    reference = ANN_FICTIVE_REFERENCE
    reference_label = f"$T_f'$ at {reference['heating_rate_K_min']:g} K/min"
    a.errorbar(0., reference['temperature_C'],
               yerr=reference['error_half_width_C'], fmt='X',
               color='#009E73', mec='white', mew=.4, ms=6,
               capsize=3, lw=1, zorder=7)
    state_labels = {'Verre': 'Glass or Glass-like', 'Melt': 'Melt', 'Melt+Ctx': 'Melt+Ctx'}
    handles = [Line2D([], [], marker=marker, color='0.4', ls='',
                       label=state_labels.get(state, state))
               for state, (_, marker) in STATE_STYLE.items()]
    handles += [Line2D([], [], marker='*', color='k', ls='',
                        label=f'Spectral extrapolation: {intercept:.0f} °C [{lower:.0f}, {upper:.0f}]'),
                Line2D([], [], marker='X', color='#009E73', ls='',
                       label=reference_label)]
    a.legend(handles=handles, fontsize=5.4, loc='upper left', framealpha=.9,
             handletextpad=.4, borderpad=.4, labelspacing=.35)
    a.set(xlabel='SrB$_4$O$_7$:Sm$^{2+}$ median P (GPa)',
          ylabel='Thermocouple temperature (°C)', title='a)', xlim=(-1.1, 13.5), ylim=(0, 440))
    b.set(xlabel='$T_c$ (°C)', ylabel='$T_g$ split midpoint (°C)', title='b)')
    b.legend(fontsize=6)
    png, _ = save(fig, NAME, 'manuscript')
    png.with_suffix('.json').write_text(json.dumps(regression, indent=2) + '\n')
    return png


if __name__ == '__main__':
    main()
