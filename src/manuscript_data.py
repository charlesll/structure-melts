"""Données et incertitudes communes aux figures 2 et 7–10 et la figure supplémentaire 4."""
import json
import h5py
import numpy as np
from cmcrameri import cm
from .config import SIGMA_T
from .figure_data import RUN_ORDER, DATASET, PEAK_FITS, RESULTS

WIDTH = 180 / 25.4
RUN_COLORS = dict(zip(RUN_ORDER, ['#332288', '#0072B2', '#009E73', '#D55E00', '#CC79A7', '#882255']))
BAND_COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#332288']
BAND_MARKERS = ['h', 's', 'D', '^', 'p']
CMAP = cm.batlow
TRANSITIONS = RESULTS / 'manuscript_transitions.json'


def propagate_areas(areas, covariance):
    """Fractions, covariance des fractions et Q2/Q3 avec écart-type conditionnel.

    Entrées (...,5) et (...,5,5), dans l'ordre TOH,Q2,T2S,Q3,Q4.
    """
    from .peak_uncertainty import propagate_area_fractions, propagate_area_ratio
    f, fc = propagate_area_fractions(areas, covariance)
    r, rs = propagate_area_ratio(areas, covariance)
    return f, fc, r, rs


def heating_indices(d):
    """Indices sources sur la montée, sans supprimer les cristallisés."""
    from .tg_spectra_ruptures import _clean_ladder
    return _clean_ladder(d['T_C'], d['Etat'], np.arange(len(d['T_C'])), drop_crystal=False)[2].astype(int)


def first_crystal(d):
    idx = heating_indices(d)
    hits = idx[d['Etat'][idx] == 'Melt+Ctx']
    return int(hits[0]) if len(hits) else None


def fits_for(run, d):
    """Valide les correspondances avant de fournir les résultats des bandes."""
    with h5py.File(PEAK_FITS) as f:
        g = f[run]
        if not np.array_equal(g['temperature'][()], d['T_C']) or not np.allclose(g['pressure'][()], d['borate_P_median'], equal_nan=True):
            raise ValueError(f'{run}: dataset/peak_fits incompatibles; régénérer les fits')
        if 'area_covariance' not in g:
            raise ValueError(f'{run}: covariance absente; régénérer les fits')
        return propagate_areas(g['areas'][()], g['area_covariance'][()])


def transition_status(p_linear, p_quadratic):
    """Soutien sous les deux nuls, seuil strict de 0,05."""
    return 'supported' if max(p_linear, p_quadratic) < .05 else 'candidate'


def build_transitions():
    """999 permutations par nul, graines historiques 0 et 1; indices zéro-based."""
    from .pressure import load_dataset
    from . import tg_spectra_ruptures as tr
    D = load_dataset()
    with h5py.File(DATASET) as f:
        x = f['x_axis'][()]
    # Vérification explicite des paliers et de la dimension des spectres capteurs.
    for run, d in D.items():
        with h5py.File(RESULTS / 'markers' / (run + '.h5')) as markers:
            if not np.array_equal(markers['T_C'][()], d['T_C']):
                raise ValueError(f'{run}: températures markers/dataset incompatibles')
            for sensor in ['Ruby', 'Borate']:
                if markers[sensor]['spectra'].shape[0] != len(d['T_C']):
                    raise ValueError(f'{run}: dimensions {sensor} incompatibles')
    families = tr._run_families(D, x, RESULTS / 'markers', tr.T_MIN)
    rows = []
    for run in RUN_ORDER[1:]:
        d = D[run]
        fits_for(run, d)
        if run not in families:
            raise ValueError(
                f"{run}: observables de rupture absentes ; vérifier les ajustements "
                f"Raman dans {tr.PEAK_FITS} et les paliers de chauffage admissibles"
            )
        T, labels, fam = families[run]
        result = tr.find_common_break(T, fam)
        row = {'run_id': run, 'n_permutations_per_null': 999, 't_min_C': tr.T_MIN,
                              'pca_random_state': tr.PCA_RANDOM_STATE}
        for degree, seed, name in [(1, 0, 'p_linear'), (2, 1, 'p_quadratic')]:
            row[name] = tr.permutation_test(T, fam, n_permutations=999, seed=seed, null_degree=degree)['p_value']
        row['p_max'] = max(row['p_linear'], row['p_quadratic'])
        row['status'] = transition_status(row['p_linear'], row['p_quadratic'])
        idx = heating_indices(d)
        idx = idx[(d['Etat'][idx] != 'Melt+Ctx') & (d['T_C'][idx] >= tr.T_MIN)]
        if not np.array_equal(d['T_C'][idx], T):
            raise ValueError(f'{run}: indices/features incompatibles')
        left = int(idx[np.flatnonzero(T == result['T_left'])[0]])
        right = int(idx[np.flatnonzero(T == result['T_right'])[0]])
        tc = first_crystal(d)
        row.update(step_left=left, step_right=right, step_tc=tc,
                   T_left=float(d['T_C'][left]), T_right=float(d['T_C'][right]),
                   Tg=float(result['T_break']), Tc=float(d['T_C'][tc]), sigma_T=SIGMA_T)
        for label, i in [('left', left), ('right', right)]:
            row['P_' + label] = float(d['borate_P_median'][i])
            row['P_std_' + label] = float(d['borate_P_std'][i])
        row['P'] = (row['P_left'] + row['P_right']) / 2
        row['split_bracket_is_confidence_interval'] = False
        row['physical_transition_demonstrated'] = False
        rows.append(row)
    TRANSITIONS.write_text(json.dumps(rows, indent=2) + '\n')
    return rows


def transitions():
    if not TRANSITIONS.exists():
        raise FileNotFoundError('Run python -m src.manuscript_data before plotting')
    return json.loads(TRANSITIONS.read_text())


if __name__ == '__main__':
    for row in build_transitions():
        print(row)
