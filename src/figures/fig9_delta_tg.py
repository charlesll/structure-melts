"""Figure 9: pressure-induced Tg changes relative to each series' reference.

The ANN band retains absolute temperature uncertainty: mean and bounds
are shifted by a fixed nominal Tg(0). Albite/anorthite differences instead use
their shared first measurement, with its covariance propagated explicitly.
Run from the repository root: python -m src.figures.fig9_delta_tg.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from ..figure_data import (save, style, TG_EXTRAPOLATED_0GPA_C,
                           TG_EXTRAPOLATED_0GPA_PROFILE95_C)
from ..manuscript_data import WIDTH, transitions
from ..tg_pressure_regression import fit_interval_line, nominal_reference_envelope
NAME = 'fig9_delta_tg'


def difference_covariance(sd):
    """Covariance des différences à la première valeur; annulation exacte en 0.

    Mesures originales indépendantes; la référence partagée crée les termes
    hors diagonale. Unités K²; différences °C et K numériquement identiques.
    """
    sd = np.asarray(sd, float)
    jac = np.eye(len(sd))
    jac[:, 0] -= 1
    return (jac * sd**2) @ jac.T


def _plot_ann(ax):
    """Plot acquisition polygons and the shifted absolute ANN fit support."""
    rows = transitions()
    fit = fit_interval_line(rows, profile=False)
    if not np.isclose(fit['intercept_C'], TG_EXTRAPOLATED_0GPA_C, atol=1e-6, rtol=0):
        raise ValueError('Stored ANN Tg(0) does not match the current interval regression')
    fit_pressure = np.linspace(0., 13., 27)
    fit_delta, fit_lower, fit_upper = nominal_reference_envelope(
        fit_pressure, rows, fit, TG_EXTRAPOLATED_0GPA_C)
    metadata = {
        'provenance': ('old_code/Processing.jl lines 866–902; values transcribed, '
                       'original papers not re-digitized'),
        'ANN_reference_C': TG_EXTRAPOLATED_0GPA_C,
        'ANN_reference_profile95_C': list(TG_EXTRAPOLATED_0GPA_PROFILE95_C),
        'ANN_reference_source': ('results/fig7_interval_regression.json; '
                                 'stored in src/figure_data.py'),
        'ANN_reference_covariance_K2': None,
        'ANN_reference_uncertainty_note': (
            f'Absolute fit support is shifted by the fixed nominal Tg(0), '
            f'{TG_EXTRAPOLATED_0GPA_C} C. Finite intercept uncertainty remains visible. '
            'This is not the within-line random difference T(P)-T(0); '
            'no independent reference error is added.'),
        'ANN_linear_fit': fit,
        'ANN_fit_envelope': {
            'P_GPa': fit_pressure.tolist(),
            'delta_T_K': fit_delta.tolist(),
            'lower_K': fit_lower.tolist(),
            'upper_K': fit_upper.tolist(),
            'interpretation': (
                'Absolute mean-fit pointwise approximate 95% support, '
                'translated by a fixed nominal reference; not support of the '
                'within-line random difference or a new-acquisition prediction '
                'interval; small n=5'),
            'construction': (
                'Profile absolute q=a+bP over slope and residual scatter at each P, '
                'then subtract the same fixed nominal Tg(0) from mean and both bounds'),
        },
        'ANN_brackets': [],
    }
    for k, t in enumerate(rows):
        # Même polygone que Figure7 : pressions ±P_std aux deux paliers.
        # Les bornes en T sont des acquisitions, pas un intervalle de confiance.
        x = [t['P_left']-t['P_std_left'], t['P_left']+t['P_std_left'],
             t['P_right']+t['P_std_right'], t['P_right']-t['P_std_right']]
        lower_temperature = t['T_left'] - TG_EXTRAPOLATED_0GPA_C
        upper_temperature = t['T_right'] - TG_EXTRAPOLATED_0GPA_C
        y = [lower_temperature] * 2 + [upper_temperature] * 2
        ax.fill(x, y, color='orange', alpha=.3, zorder=2,
                edgecolor='none', linewidth=.5,
                label='ANN + 5wt% H$_2$O acquisition brackets' if k == 0 else None)
        metadata['ANN_brackets'].append({
            'run_id': t['run_id'], 'status': t['status'],
            'P_vertices_GPa': x, 'delta_T_vertices_K': y,
            'is_confidence_interval': False})
    ax.fill_between(fit_pressure, fit_lower, fit_upper, color='#6A3D9A',
                    alpha=.18, linewidth=0, zorder=3,
                    label='ANN + 5wt% H$_2$O fit with 95% support')
    ax.plot(fit_pressure, fit_delta, color='#6A3D9A', lw=1.1, zorder=4)
    return metadata


def _plot_literature_measurements(ax):
    """Plot independent albite/anorthite measurements relative to their first point."""
    metadata = {}
    series = [
        ('Albite · B2004', [0.1, 0.5, 1, 3, 6],
         [683, 690, 676, 663, 630], [4, 3, 5, 5, 15], '#0072B2', 'D'),
        ('Anorthite · B2004', [0.1, 0.3, 0.6, 3, 4.5, 6],
         [844, 847, 863, 859, 873, 881], [5, 6, 6, 6, 6, 15], 'black', 'o'),
    ]
    for name, pressure, temperature, sd, color, marker in series:
        covariance = difference_covariance(sd)
        delta_temperature = np.array(temperature) - temperature[0]
        ax.errorbar(
            pressure, delta_temperature, yerr=np.sqrt(np.diag(covariance)),
            fmt=marker, color=color, markersize=3, linewidth=0.7, label=name,
        )
        metadata[name] = {
            'P_GPa': pressure,
            'Tg_original': temperature,
            'sd_original': sd,
            'difference_covariance_K2': covariance.tolist(),
            'reference': ('first point at 0.1 GPa; original absolute temperature '
                          'units unspecified in Julia, differences in K'),
        }
    return metadata


def _plot_peridotite(ax):
    """Plot the historical peridotite isoviscosity prediction."""
    pressure = np.linspace(0.0001, 15, 100)
    # Legacy Russell (2024) parameters, with log10 viscosity = 12.
    # Reference pressure is 0.0001 GPa; retained Julia water fraction is 0.029.
    delta = 84.21 * (pressure - 0.0001) / (12 - (-5.36))
    ax.plot(pressure, delta, '--', color='#E69F00', label='Peridotite · R2024 model')


def main(d=None):
    """Draw the comparison and save PNG, PDF and numerical JSON sidecar.

    The optional dataset argument follows the shared manuscript figure runner;
    this figure reads the current transition table rather than raw spectra.
    """
    style()
    fig, ax = plt.subplots(figsize=(0.5 * WIDTH, 3.5), layout='constrained')

    metadata = _plot_ann(ax)
    metadata['literature'] = _plot_literature_measurements(ax)
    _plot_peridotite(ax)

    ax.axhline(0, color='0.4', linewidth=0.5)
    ax.set(
        xlabel='Pressure (GPa)',
        ylabel=r'$\Delta T_g = T_g(P)-T_g(0)$ (K)',
        xlim=(-0.25, 15),
    )
    ax.legend(fontsize=6, ncol=1, loc='lower center', bbox_to_anchor=(0.5, 1.02))
    png, _ = save(fig, NAME, 'manuscript')
    png.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return png


if __name__ == '__main__':
    main()
