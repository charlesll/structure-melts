"""Aligned Raman observables for manuscript Figure 10 and its model checks.

Centroids use the historical signed intensity moment for x < 830 cm^-1.
Spectra retain negative baseline residuals; no clipping or spectral refitting
is performed here. Ratio uncertainties are conditional on the five-band fit.
"""
import h5py
import numpy as np

from .raman_treatment import area_ratio
from .figure_data import DATASET, PEAK_FITS
from .manuscript_data import fits_for, transitions
from .peak_uncertainty import propagate_area_ratio
from .pressure import load_dataset
from .raman_treatment import prepare_glass


from .raman_treatment import lf_centroid


def acquisition_table():
    """Return all current acquisitions with source run/step and aligned features.

    Q4/T2S and T2S/Q4 are fitted area ratios (indices 4 and 2), without additional
    normalization. gvar propagation includes the area covariance. It does
    not estimate its covariance with the centroid from the same spectrum.
    """
    dataset = load_dataset()
    with h5py.File(DATASET) as handle:
        x = handle['x_axis'][()]
    rows = []
    with h5py.File(PEAK_FITS) as fits:
        for run, data in dataset.items():
            fractions, fraction_covariance, _, _ = fits_for(run, data)
            # fits_for also validates source temperature, pressure and covariance.
            spectra = prepare_glass(x, data['spectra'])
            centroid = lf_centroid(x, spectra)
            integrated_ratio = area_ratio(x, spectra)
            group = fits[run]
            areas = group['areas'][()]
            area_covariance = group['area_covariance'][()]
            ratio, ratio_sd = propagate_area_ratio(
                areas, area_covariance,
                numerator=4, denominator=2)
            inverse_ratio, inverse_ratio_sd = propagate_area_ratio(
                areas, area_covariance, numerator=2, denominator=4)
            for step in range(len(centroid)):
                rows.append({
                    'run_id': run, 'step': step, 'state': str(data['Etat'][step]),
                    'T_C': float(data['T_C'][step]),
                    'P_GPa': float(data['borate_P_median'][step]),
                    'P_std_GPa': float(data['borate_P_std'][step]),
                    'centroid_cm-1': float(centroid[step]),
                    'LF_HF_area_ratio': float(integrated_ratio[step]),
                    'Q4_T2S': float(ratio[step]), 'Q4_T2S_sd': float(ratio_sd[step]),
                    'T2S_Q4': float(inverse_ratio[step]),
                    'T2S_Q4_sd': float(inverse_ratio_sd[step]),
                    # Integrated fitted areas of max-normalized spectra: arbitrary
                    # area units, not calibrated intensity or species abundance.
                    'Q4_area': float(areas[step, 4]),
                    'Q4_area_sd': float(np.sqrt(area_covariance[step, 4, 4])),
                    'T2S_area': float(areas[step, 2]),
                    'T2S_area_sd': float(np.sqrt(area_covariance[step, 2, 2])),
                    'T2S_fraction': float(fractions[step, 2]),
                    'T2S_fraction_sd': float(np.sqrt(fraction_covariance[step, 2, 2])),
                })
    return rows


def transition_observables(rows):
    """Attach the actual paired endpoint spectra to each temperature bracket.

    Do not use nearest pressures or interpolate a Melt-only centroid into a
    heating bracket: its earlier acquisition may still be labelled Glass.
    """
    lookup = {(row['run_id'], row['step']): row for row in rows}
    output = []
    for transition in transitions():
        endpoints = []
        for side in ('left', 'right'):
            row = lookup[(transition['run_id'], transition['step_' + side])]
            if row['T_C'] != transition['T_' + side]:
                raise ValueError('Transition/source temperature mismatch')
            endpoints.append(row)
        output.append({**transition, 'endpoints': endpoints})
    return output


def parallel_centroid_guides(melt, threshold_gpa=6.):
    """Fit descriptive parallel lines with equal total weight per run.

    C(P)=a+b(P-threshold)+d*I(P>=threshold), with C in cm^-1 and P in GPa.
    The nominal threshold partitions the sampled groups; it is not an inferred
    physical transition pressure. Return plotting domains separately so no line
    is drawn through the unsampled interval. No confidence interval is inferred
    from these repeated observations across only five experimental runs.
    """
    pressure = np.array([row['P_GPa'] for row in melt])
    centroid = np.array([row['centroid_cm-1'] for row in melt])
    runs = np.array([row['run_id'] for row in melt])
    high = pressure >= threshold_gpa
    matrix = np.column_stack([np.ones(len(melt)), pressure-threshold_gpa, high])
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(centroid)):
        raise ValueError('Parallel guides require finite pressures and centroids')
    if np.linalg.matrix_rank(matrix) != 3:
        raise ValueError('Parallel guides require two groups and a pressure span')
    weights = np.array([1./np.count_nonzero(runs == run) for run in runs])
    root_weight = np.sqrt(weights)
    coefficients = np.linalg.lstsq(matrix*root_weight[:, None],
                                   centroid*root_weight, rcond=None)[0]
    residual = centroid - matrix @ coefficients
    return {
        'model': 'Parallel descriptive lines, equal total weight per run',
        'threshold_GPa': threshold_gpa,
        'low_intercept_at_threshold_cm-1': float(coefficients[0]),
        'common_slope_cm-1_per_GPa': float(coefficients[1]),
        'high_offset_cm-1': float(coefficients[2]),
        'weighted_RMSE_cm-1': float(np.sqrt(np.sum(weights*residual**2)/weights.sum())),
        'n_acquisitions': len(melt), 'n_runs': len(np.unique(runs)),
        'domains_GPa': [[float(pressure[mask].min()), float(pressure[mask].max())]
                        for mask in (~high, high)],
        'interpretation': 'Descriptive group guides; no discontinuity or precisely located transition established. No fit or connection drawn across the pressure gap.',
    }
