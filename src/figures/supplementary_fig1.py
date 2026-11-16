"""Two existing Raman deconvolutions closest to 3 and 12 GPa.

Reproduce: python -m src.figures.supplementary_fig1
Selection uses pressure only among Glass/Verre and Melt acquisitions, never
fit quality. Existing posterior means are evaluated without any refitting.
"""
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ..pressure import load_dataset
from ..peak_fitting import _prepare_spectrum, _component, _model, _peak_quantities, PEAK_PRIORS
from ..figure_data import save, style, DATASET, PEAK_FITS, PEAK_ORDER, PEAK_LABELS
from ..manuscript_data import WIDTH, BAND_COLORS

NAME = 'Supplementary_Fig_1'


def main(d=None):
    style()
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    dataset = load_dataset()
    candidates = [(float(p), run, i) for run, data in dataset.items()
                  for i, p in enumerate(data['borate_P_median'])
                  if data['Etat'][i] in ['Verre', 'Glass', 'Melt'] and np.isfinite(p)]
    examples = []
    with h5py.File(DATASET, 'r') as data_file, h5py.File(PEAK_FITS, 'r') as fits:
        axis = data_file['x_axis'][()]
        stored_bands = tuple(v.decode() if isinstance(v, bytes) else str(v)
                             for v in fits.attrs['peak_names'])
        if stored_bands != PEAK_ORDER:
            raise ValueError('Unexpected component ordering')
        for target in [3., 12.]:
            pressure, run, step = min(candidates, key=lambda v: (abs(v[0]-target), v[1], v[2]))
            data = dataset[run]
            group = fits[run]
            np.testing.assert_allclose(group['temperature'][()], data['T_C'])
            np.testing.assert_allclose(group['pressure'][()], data['borate_P_median'])
            x, observed = _prepare_spectrum(axis, data['spectra'][step])
            parameters = group['parameters'][step]
            centers, areas = _peak_quantities(parameters)
            np.testing.assert_allclose(centers, group['positions'][step], atol=1e-9)
            np.testing.assert_allclose(areas, group['areas'][step], atol=1e-9)
            components = []
            offset = 0
            for prior in PEAK_PRIORS:
                amplitude, center, width = parameters[offset:offset+3]
                offset += 3
                fraction = 0.
                if prior['kind'] == 'pseudovoigt':
                    fraction = parameters[offset]
                    offset += 1
                components.append(_component(x, amplitude, center, width, prior['kind'], fraction))
            components = np.array(components)
            fitted = _model(x, parameters)
            np.testing.assert_allclose(components.sum(axis=0), fitted, atol=1e-12)
            examples.append(dict(run=run, step=int(step), target_pressure_GPa=target,
                                 pressure_GPa=pressure, temperature_C=float(data['T_C'][step]),
                                 state=str(data['Etat'][step]), x=x, observed=observed,
                                 fitted=fitted, components=components,
                                 residual=observed-fitted, parameters=parameters))
    fig = plt.figure(figsize=(WIDTH, 4.25), layout='constrained')
    grid = fig.add_gridspec(3, 2, height_ratios=[.45, 3., .8], hspace=.04, wspace=.08)
    legend_axis = fig.add_subplot(grid[0, :])
    legend_axis.axis('off')
    handles = [Line2D([], [], color='0.2', marker='.', linestyle='none', markersize=3,
                      label='Measured spectrum'),
               Line2D([], [], color='black', lw=1.2, label='Total fit')]
    handles += [Line2D([], [], color=color, lw=1.2, label=PEAK_LABELS[band])
                for band, color in zip(PEAK_ORDER, BAND_COLORS)]
    legend_axis.legend(handles=handles, ncol=4, loc='center', fontsize=7,
                       columnspacing=1.5, handlelength=1.8)
    residual_limit = max(abs(e['residual']).max() for e in examples) * 1.12
    ymin = min(-.03, min(e['observed'].min() for e in examples)-.025)
    ymax = max(1.05, max(e['fitted'].max() for e in examples)+.04)
    for column, example in enumerate(examples):
        ax = fig.add_subplot(grid[1, column])
        res = fig.add_subplot(grid[2, column], sharex=ax)
        x = example['x']
        for component, color in zip(example['components'], BAND_COLORS):
            ax.fill_between(x, component, color=color, alpha=.15, lw=0)
            ax.plot(x, component, color=color, lw=.8)
        ax.plot(x, example['observed'], '.', color='0.25', ms=1.6, alpha=.65, zorder=4)
        ax.plot(x, example['fitted'], color='black', lw=1.15, zorder=5)
        ax.set_title(f'({chr(97+column)}) {example["pressure_GPa"]:.2f} GPa, '
                     f'{example["temperature_C"]:.0f} °C', loc='left')
        ax.set(xlim=(820, 1270), ylim=(ymin, ymax))
        ax.tick_params(labelbottom=False)
        res.plot(x, example['residual'], color='0.25', lw=.55)
        res.axhline(0, color='0.6', lw=.5)
        res.set(ylim=(-residual_limit, residual_limit), xlabel='Raman shift (cm$^{-1}$)')
        res.set_xticks([850, 950, 1050, 1150, 1250])
        if column == 0:
            ax.set_ylabel('Normalised intensity')
            res.set_ylabel('Residual')
        else:
            ax.tick_params(labelleft=False)
            res.tick_params(labelleft=False)
    png, _ = save(fig, NAME, 'manuscript', dpi=600)
    numerical = {}
    metadata = []
    for j, example in enumerate(examples):
        metadata.append({key: value for key, value in example.items() if not isinstance(value, np.ndarray)})
        for key, value in example.items():
            if isinstance(value, np.ndarray):
                numerical[f'panel_{j+1}_{key}'] = value
    np.savez_compressed(png.with_suffix('.npz'), **numerical)
    png.with_suffix('.json').write_text(json.dumps(dict(
        selection='Closest borate pressure to 3 and 12 GPa among Glass/Verre and Melt; independent of fit quality',
        examples=metadata, bands=list(PEAK_ORDER),
        source_files=['results/dataset.h5', 'results/peak_fits.h5'],
        preprocessing='Existing 820–1270 cm^-1 endpoint baseline and maximum normalization',
        model='Four Gaussian components and one Q3 pseudo-Voigt; stored posterior means; no refitting',
        residual='Observed minus total fit, same normalized units and scale in both panels',
        dimensions_mm=[180, 4.25*25.4], PNG_dpi=600, PDF='vector, embedded TrueType fonts'
    ), indent=2)+'\n')
    print(metadata)
    return png


if __name__ == '__main__':
    main()
