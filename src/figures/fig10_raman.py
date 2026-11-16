"""Figure 10: LF centroid versus pressure and fitted T2S/Q4 area ratio.

Only acquisitions in the current dataset are shown. In particular, the new
run01 point at 0 GPa replaces the separate ambient reference formerly drawn
as a star. Model and source-alignment checks are in tests/test_fig10.py.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from ..figure_data import RUN_LABEL, save, style
from ..manuscript_data import RUN_COLORS, WIDTH
from ..raman_observables import acquisition_table, parallel_centroid_guides

NAME = 'fig10_raman'


def plot_pressure_centroids(ax, rows, melt_guides=False):
    """Plot Melt and 25 °C glass acquisitions, returning the displayed data.

    Pressure bars are P_std (between-grain dispersion plus propagated errors), not calibrated errors
    on a pressure mean. The retained glass quadratic is a descriptive guide;
    optional parallel Melt guides are limited to sampled domains. No shading.
    """
    melt = [row for row in rows if row['state'] == 'Melt']
    glass = [row for row in rows if row['state'] == 'Verre' and row['T_C'] == 25.]
    for selected, marker in ((melt, 'o'), (glass, '^')):
        for run in dict.fromkeys(row['run_id'] for row in selected):
            points = [row for row in selected if row['run_id'] == run]
            ax.errorbar(
                [row['P_GPa'] for row in points],
                [row['centroid_cm-1'] for row in points],
                xerr=[row['P_std_GPa'] for row in points],
                fmt=marker, color=RUN_COLORS[run], markersize=4, linewidth=.6,
            )

    # Fit only the five run01 observations, including its actual 0 GPa spectrum.
    pressure = np.array([row['P_GPa'] for row in glass])
    centroid = np.array([row['centroid_cm-1'] for row in glass])
    coefficients = np.polyfit(pressure, centroid, 2)
    grid = np.linspace(0., 6.5, 100)  # Historical display extent retained.
    ax.plot(grid, np.polyval(coefficients, grid), ':', color='#882255',
            label='Glass quadratic guide')
    ax.plot([], [], '^', color='0.3', label='Glass at 25 °C')
    ax.plot([], [], 'o', color='0.3', label='Melt-only')
    fit = None
    if melt_guides:
        fit = parallel_centroid_guides(melt)
        for group, (lower, upper) in enumerate(fit['domains_GPa']):
            grid = np.linspace(lower, upper, 100)
            prediction = (fit['low_intercept_at_threshold_cm-1']
                          + fit['common_slope_cm-1_per_GPa']*(grid-fit['threshold_GPa'])
                          + group*fit['high_offset_cm-1'])
            ax.plot(grid, prediction, '--', color='.2', linewidth=1,
                    label='Melt parallel guides' if group == 0 else None)
    ax.legend(fontsize=6)
    ax.set(xlabel='SrB$_4$O$_7$:Sm$^{2+}$ median pressure (GPa)', ylabel='LF centroid (cm$^{-1}$)',
           xlim=(0, 14), title='a) Glass and melt')
    return {'glass': glass, 'melt': melt, 'glass_quadratic': coefficients.tolist(),
            'melt_fit': fit}


def plot_t2s_q4_centroid(ax, melt):
    """Plot centroid against fitted T2S/Q4, with local ratio errors at 1 sigma.

    Ratio errors include covariance of the two fitted areas. Centroid errors
    and cross-covariance with the ratio are not available; no relation is fitted.
    """
    for run in dict.fromkeys(row['run_id'] for row in melt):
        points = [row for row in melt if row['run_id'] == run]
        ax.errorbar(
            [row['T2S_Q4'] for row in points],
            [row['centroid_cm-1'] for row in points],
            xerr=[row['T2S_Q4_sd'] for row in points],
            fmt='o', color=RUN_COLORS[run], markersize=4, linewidth=.6,
            label=RUN_LABEL[run],
        )
    ax.set(xlabel='Fitted T$_{2S}$/Q$^4$ area ratio',
           ylabel='LF centroid (cm$^{-1}$)', title='b) Melt-only')
    ax.legend(fontsize=6, ncol=2)


def main(d=None):
    """Regenerate PNG/PDF and the acquisition-level numerical sidecar."""
    style()
    rows = acquisition_table()
    fig, (ax_pressure, ax_ratio) = plt.subplots(
        1, 2, figsize=(WIDTH, 3.4), layout='constrained')
    displayed = plot_pressure_centroids(ax_pressure, rows, melt_guides=True)
    plot_t2s_q4_centroid(ax_ratio, displayed['melt'])
    metadata = {
        **displayed,
        'glass_n': len(displayed['glass']), 'melt_n': len(displayed['melt']),
        'ambient_reference': 'Separate ambient reference omitted; run01 0 GPa point retained',
        'model_checks': 'tests/test_fig10.py',
        'panel_b': 'LF centroid versus fitted T2S/Q4 area ratio, 22 Melt acquisitions',
        'preprocessing': 'prepare_glass: two-part baseline and area normalization; signed LF moment for x < 830 cm^-1',
        'vertical_errors': 'Centroid uncertainty not estimated; ratio horizontal errors are local 1-sigma from full area covariance',
    }
    png, _ = save(fig, NAME, 'manuscript')
    png.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return png


if __name__ == '__main__':
    main()
