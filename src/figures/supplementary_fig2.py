"""Publication version of fitted Raman band positions versus borate pressure.

Reproduce: python -m src.figures.supplementary_fig2
All Glass/Verre and Melt acquisitions; existing fit positions and conditional
standard deviations (cm^-1). No refitting, preprocessing, or state reassignment.
"""
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ..pressure import load_dataset
from ..figure_data import save, style, PEAK_FITS, PEAK_ORDER, PEAK_LABELS
from ..manuscript_data import WIDTH

NAME = 'Supplementary_Fig_2'


def main(d=None):
    style()
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    dataset = load_dataset()
    rows = []
    with h5py.File(PEAK_FITS, 'r') as fits:
        band_order = tuple(v.decode() if isinstance(v, bytes) else str(v)
                           for v in fits.attrs['peak_names'])
        if band_order != PEAK_ORDER:
            raise ValueError('Unexpected fitted-band ordering')
        for run, data in dataset.items():
            group = fits[run]
            np.testing.assert_allclose(group['temperature'][()], data['T_C'])
            np.testing.assert_allclose(group['pressure'][()], data['borate_P_median'])
            for i in np.flatnonzero(np.isin(data['Etat'], ['Glass', 'Verre', 'Melt'])):
                rows.append(dict(run=run, step=int(i), state=str(data['Etat'][i]),
                                 P_GPa=float(data['borate_P_median'][i]),
                                 T_C=float(data['T_C'][i]),
                                 position_cm1=group['positions'][i].tolist(),
                                 position_sd_cm1=group['position_sd'][i].tolist()))
    pressure = np.array([r['P_GPa'] for r in rows])
    temperature = np.array([r['T_C'] for r in rows])
    position = np.array([r['position_cm1'] for r in rows])
    sd = np.array([r['position_sd_cm1'] for r in rows])
    states = np.array([r['state'] for r in rows])
    assert np.isfinite(np.c_[pressure, temperature, position, sd]).all()
    norm = plt.Normalize(temperature.min(), temperature.max())
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.6), layout='constrained')
    for j, (ax, band) in enumerate(zip(axes.flat, PEAK_ORDER)):
        ax.errorbar(pressure, position[:, j], yerr=sd[:, j], fmt='none',
                    ecolor='0.68', elinewidth=.5, alpha=.75, zorder=1)
        for mask, marker in [(states != 'Melt', 's'), (states == 'Melt', 'o')]:
            ax.scatter(pressure[mask], position[mask, j], c=temperature[mask],
                       cmap='plasma', norm=norm, marker=marker, s=18,
                       edgecolors='0.2', linewidths=.3, zorder=2)
        ax.set_title(f'({chr(97+j)}) {PEAK_LABELS[band]}', loc='left', pad=5)
        ax.set_xlim(min(-.3, pressure.min()-.3), max(13., pressure.max()+.3))
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    info = axes.flat[-1]
    info.axis('off')
    info.legend(handles=[Line2D([], [], marker='s', color='0.35', linestyle='none',
                                markersize=4, label='Glass or Glass-like'),
                         Line2D([], [], marker='o', color='0.35', linestyle='none',
                                markersize=4, label='Melt')],
                loc='upper left', bbox_to_anchor=(.03, .98), fontsize=7)
    info.text(.08, .64, f'{len(rows)} spectra · {len(set(r["run"] for r in rows))} runs',
              transform=info.transAxes, fontsize=7)
    color_axis = info.inset_axes([.08, .38, .82, .065])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='plasma'), cax=color_axis,
                 orientation='horizontal', ticks=[25, 100, 200, 300, 360],
                 label='Temperature (°C)')
    info.text(.08, .1, 'Vertical bars: conditional fit ±1σ',
              transform=info.transAxes, fontsize=6.5)
    fig.supxlabel('SrB$_4$O$_7$:Sm$^{2+}$ median pressure (GPa)', fontsize=8)
    fig.supylabel('Raman band position (cm$^{-1}$)', fontsize=8)
    png, _ = save(fig, NAME, 'manuscript', dpi=600)
    metadata = dict(bands=list(PEAK_ORDER), observations=rows,
                    sources=['results/dataset.h5', 'results/peak_fits.h5'],
                    selection='All acquisitions labelled Glass, Verre or Melt; no temperature cutoff',
                    uncertainties='Vertical: conditional fit SD only, not full experimental uncertainty; horizontal pressure bars not shown',
                    dimensions_mm=[180, 4.6*25.4], PNG_dpi=600, PDF='vector, embedded TrueType fonts')
    png.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return png


if __name__ == '__main__':
    main()
