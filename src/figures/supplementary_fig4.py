"""Supplementary Figure 4: fitted band area fractions.

Reproduce: python -m src.figures.supplementary_fig4
Scientific selections and covariance unchanged.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ..pressure import load_dataset
from ..figure_data import save, style, PEAK_ORDER, PEAK_LABELS, RUN_LABEL
from ..manuscript_data import WIDTH, BAND_COLORS, BAND_MARKERS, fits_for, transitions, heating_indices
NAME = 'Supplementary_Fig_4'


def main(d=None):
    style()
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    D = load_dataset()
    titles_ = ['a) ', 'b) ', 'c) ', 'd) ', 'e) ', 'f) ']
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.9), layout='constrained')
    for title_prefix, ax, t in zip(titles_, axes.flat, transitions()):
        run = t['run_id']; r = D[run]
        f, cov, _, _ = fits_for(run, r)
        sd = np.sqrt(np.maximum(np.diagonal(cov, axis1=-2, axis2=-1), 0))
        heating = np.zeros(len(f), bool); heating[heating_indices(r)] = True
        ax.axvspan(t['Tg'], t['Tc'], color='#E69F00', alpha=.12)
        ax.axvspan(t['Tc'], max(r['T_C']), color="#D50000", alpha=.10)
        ax.axvline(t['Tg'], color='0.3', ls=':' if t['status']=='candidate' else '--', lw=.8)
        for j, (color, marker) in enumerate(zip(BAND_COLORS, BAND_MARKERS)):
            # Refroidissement ouvert et indépendant des trajectoires de montée.
            for m, face in [(heating, color), (~heating, 'white')]:
                ax.errorbar(r['T_C'][m], f[m, j], yerr=sd[m, j], fmt=marker, mfc=face, color=color, ms=3, lw=.5)
        ax.set(xlabel='Thermocouple temperature (°C)', 
               ylabel='Area fraction', 
               ylim=(0, .4), 
               title=title_prefix + RUN_LABEL[run]+(' · candidate' if t['status']=='candidate' else ''))
    ax = axes.flat[-1]; ax.axis('off')
    handles = [Line2D([], [], marker=m, color=c, ls='', label=PEAK_LABELS[n]) for n, c, m in zip(PEAK_ORDER, BAND_COLORS, BAND_MARKERS)]
    handles += [Line2D([], [], marker='o', mfc='white', color='0.3', ls='', label='Quench')]
    ax.legend(handles=handles, loc='center')
    ax.text(.5, .03,
            '1σ errors conditional on fit model\n'
            'Orange shading: melt; red shading: crystallised\n'
            'Dashed boundary: supported split\n'
            'Dotted boundary: candidate split',
            ha='center', transform=ax.transAxes, fontsize=7)
    return save(fig, NAME, 'manuscript', dpi=600)[0]


if __name__ == '__main__':
    main()
