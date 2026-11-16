"""Supplementary Figure 6: absolute temperature–pressure interval regression.

This renderer has no fitting or standalone entry point. Figure 7 owns the fit
and calls it with the same rows and fitted parameters, ensuring the two figures
cannot use different regressions. Run python -m src.figures.fig7_transitions.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from ..figure_data import RUN_LABEL, save, style
from ..manuscript_data import WIDTH
from ..tg_pressure_regression import absolute_line_envelope

NAME = 'Supplementary_Figure_6'


def plot_interval_regression(rows, fit):
    """Render the shared absolute fit, sampling brackets and mean-fit support.

    Pressure is in GPa, temperatures in °C. The profile envelope includes
    intercept–slope dependence; it is neither a prediction interval nor a
    confidence interval for temperatures independently measured as Tg.
    """
    pressure = np.linspace(0., 13., 27)
    mean, lower, upper = absolute_line_envelope(pressure, rows, fit)

    style()
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(WIDTH, 4.2), layout='constrained')
    for index, row in enumerate(rows):
        # The polygon preserves both actual acquisition temperatures/pressures.
        # Its temperature bounds are sampling limits, not Gaussian errors.
        pressure_vertices = [
            row['P_left'] - row['P_std_left'], row['P_left'] + row['P_std_left'],
            row['P_right'] + row['P_std_right'], row['P_right'] - row['P_std_right'],
        ]
        temperature_vertices = [row['T_left']] * 2 + [row['T_right']] * 2
        candidate = row.get('status') == 'candidate'
        ax.fill(
            pressure_vertices, temperature_vertices, color='orange', alpha=.3,
            zorder=2, edgecolor='darkorange' if candidate else 'none',
            hatch='///' if candidate else None,
            label='Observed split brackets' if index == 0 else None,
        )
        label = RUN_LABEL[row['run_id']] + (' (candidate)' if candidate else '')
        high_pressure = row['P'] > 10
        # Keep the longer candidate label clear of the fitted line.
        label_offset = (-4, 12) if high_pressure else (4, 24 if candidate else 8)
        ax.annotate(
            label, (row['P'], (row['T_left'] + row['T_right']) / 2),
            xytext=label_offset,
            ha='right' if high_pressure else 'left',
            textcoords='offset points', fontsize=7,
        )

    ax.fill_between(
        pressure, lower, upper, color='#6A3D9A', alpha=.18, zorder=1,
        label='Approximate 95% pointwise profile support',
    )
    ax.plot(
        pressure, mean, color='#6A3D9A', linewidth=1.4, zorder=3,
        label=f'Linear fit: T = {fit["intercept_C"]:.1f} + {fit["slope_C_per_GPa"]:.2f} P',
    )
    ax.errorbar(
        0., mean[0], yerr=[[mean[0] - lower[0]], [upper[0] - mean[0]]],
        fmt='k*', markersize=8, capsize=3, zorder=4,
        label=f'Intercept: {mean[0]:.1f} °C [{lower[0]:.1f}, {upper[0]:.1f}]',
    )
    ax.set(
        xlabel='SrB$_4$O$_7$:Sm$^{2+}$ median pressure (GPa)',
        ylabel='Thermocouple temperature (°C)', xlim=(-.4, 13.5),
        title='Hydrous ANN: spectral-change interval regression',
    )
    ax.legend(loc='upper left', fontsize=7)
    # Restore the exact physical canvas after native-backend pixel rounding.
    fig.set_size_inches(WIDTH, 4.2, forward=False)
    # Preserve that canvas rather than cropping it to the artists.
    with plt.rc_context({'savefig.bbox': None}):
        png, pdf = save(fig, NAME, 'manuscript', dpi=600)
    metadata = {
        'fit': fit,
        'absolute_temperature_envelope': {
            'P_GPa': pressure.tolist(), 'mean_C': mean.tolist(),
            'lower_C': lower.tolist(), 'upper_C': upper.tolist(),
        },
        'interpretation': 'Pointwise approximate 95% profile support of absolute mean T(P); not calibrated at n=5; no fictive-temperature anchor',
        'construction': 'Constrain q=a+bP at each P and reoptimize slope and residual scatter',
    }
    png.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return png, pdf
