"""Figure 2 : diagnostics des deux jauges à la température du thermocouple."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from ..pressure import load_dataset
from ..figure_data import save, style, RUN_LABEL, RESULTS
from ..manuscript_data import WIDTH, RUN_COLORS
NAME = 'fig2_PT'


def main(d=None, outdir='manuscript'):
    style()
    D = load_dataset()
    fig = plt.figure(figsize=(WIDTH, 5.6), layout='constrained')
    grid = fig.add_gridspec(2, 2)
    a = fig.add_subplot(grid[0, 0])
    sub = grid[0, 1].subgridspec(2, 1, hspace=.08)
    b = fig.add_subplot(sub[0]); c = fig.add_subplot(sub[1], sharex=b)
    e = fig.add_subplot(grid[1, 0]); h = fig.add_subplot(grid[1, 1])
    etiquettes_pos = dict(zip(RUN_COLORS.keys(), [(30, 5), (330, 2.5), (320, 4.8), (180, 6.3), (80, 11.0), (150, 8.0)]))
    for run, color in RUN_COLORS.items():
        r = D[run]
        for gauge, marker, ls in [('ruby', 'o', '-'), ('borate', 's', '--')]:
            a.errorbar(r['T_C'], r[gauge+'_P_median'], yerr=r[gauge+'_P_std'],
                        color=color, marker=marker, ls=ls, ms=2, lw=.6,
                        label=RUN_LABEL[run] if gauge == 'ruby' else None)
        e.errorbar(r['borate_P_median'], r['ruby_P_median'],
                   xerr=r['borate_P_std'], yerr=r['ruby_P_std'], fmt='o', ms=2, color=color, lw=.5)
        a.text(*etiquettes_pos[run], RUN_LABEL[run], color=color, fontsize=6)
    # Run 5 diamond failure at 350 °C; abbreviation defined in the caption.
    failure_T = 350.0
    a.vlines(failure_T, 10., 13., color='black', linestyle=':',
             linewidth=1.2, zorder=7)
    a.annotate('D.F.', xy=(failure_T, 13.), xytext=(0, 4),
               textcoords='offset points', ha='center', va='bottom',
               fontsize=6.5, color='black', zorder=8)
    r = D['run04_20250303_34bar']
    for ax, gauge, marker in [(b, 'ruby', 'o'), (c, 'borate', 's')]:
        # Une colonne par grain; tous les états, cristallisés compris.
        pressures=r[gauge+'_P']
        errors=r[gauge+'_P_err']
        if pressures.shape != errors.shape:
            raise ValueError(f'Run 4 {gauge}: dimensions pression/erreur incompatibles')
        # Erreurs propres à chaque grain; ne pas réutiliser le P_std du palier.
        for grain,grain_error in zip(pressures.T,errors.T):
            ax.errorbar(r['T_C'],grain,yerr=grain_error,fmt=marker,
                        ms=1.8,elinewidth=.75,capsize=2,capthick=.65,
                        barsabove=True)
        if gauge == 'ruby':
            ax.set_ylabel(gauge.capitalize()+' P (GPa)')
        else:
            ax.set_ylabel('SrB$_4$O$_7$:Sm$^{2+}$ P (GPa)')
    b.tick_params(labelbottom=False)
    b.set_title('b) Run 4 · individual grains')
    c.set_xlabel('Thermocouple temperature (°C)')
    a.set(xlabel='Thermocouple temperature (°C)', 
          ylabel='Median pressure (GPa)', 
          title='a) Ruby: circles / solid; SrB$_4$O$_7$:Sm$^{2+}$: squares / dashed')
    #a.legend(ncol=3, fontsize=6)
    lim = [min(e.get_xlim()[0], e.get_ylim()[0]), max(e.get_xlim()[1], e.get_ylim()[1])]
    e.plot(lim, lim, 'k--', lw=.7)
    e.set(xlabel='SrB$_4$O$_7$:Sm$^{2+}$ median P (GPa)', 
          ylabel='Ruby median P (GPa)', 
          title='c) Gauge comparison')
    with open(RESULTS / 'temperature_steps.csv') as f:
        values = np.array([float(row['dT_step_K']) for row in csv.DictReader(f)])
    values = values[np.isfinite(values)]
    h.hist(values, bins=20, color='0.75', edgecolor='white')
    for value, label, color in [(0, 'Zero', 'k'), (values.mean(), 'Mean', '#D55E00'), (np.median(values), 'Median', '#0072B2')]:
        h.axvline(value, label=f'{label}: {value:.0f}', color=color, lw=1)
    h.legend(fontsize=6)
    h.set(xlabel='Equivalent temperature offset (K)', 
          ylabel='Count', 
          title=f'd) Gauge-equivalent temperature offset · n={len(values)}')
    #h.text(.5, .96, 'Diagnostic only; no correction applied', transform=h.transAxes, ha='center', va='top', fontsize=6)
    return save(fig, NAME, outdir)[0]


if __name__ == '__main__':
    main()
