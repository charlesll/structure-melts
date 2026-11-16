"""ANN viscosity calculations and predictions for Supplementary Figure 11.

The supplied composition is in weight percent, not mole percent. Its sum is
100; gpvisc normalizes it before converting to oxide mole fractions.
Temperatures passed to all prediction functions are kelvin, pressures GPa.
"""
import json
from importlib.metadata import version

import numpy as np

from .config import ANN_COMPOSITION_WT, ANN_MYEGA_A, ANN_MYEGA_FRAGILITY, ANN_MYEGA_FRAGILITY_SD, RESULTS

GPVISC_RESULT = RESULTS / 'ann_gpvisc.json'


def myega(temperature_k, tg_k, fragility=ANN_MYEGA_FRAGILITY, A=ANN_MYEGA_A):
    """Return log10(eta / Pa s), broadcasting temperature and Tg arrays.

    Pressure acts only through Tg(P). Fragility and A are fixed scenario
    parameters, not independently fitted viscosity measurements.
    """
    temperature = np.asarray(temperature_k, dtype=float)
    tg = np.asarray(tg_k, dtype=float)
    if np.any(temperature <= 0) or np.any(tg <= 0):
        raise ValueError('Temperatures must be positive kelvin')
    ratio = tg / temperature
    return A + (12 - A) * ratio * np.exp((fragility / (12 - A) - 1) * (ratio - 1))


def myega_fragility_interval(temperature_k, tg_k):
    """Return nominal log viscosity and pointwise 95% fragility-only bounds.

    The i-Melt fragility is Gaussian with mean 34.5 and sd 1.2; Ae=-1.48
    and the supplied Tg(P) are fixed. MYEGA is monotonic in fragility at fixed
    positive T and Tg (constant at T=Tg), so transforming the two Gaussian
    quantiles gives exact asymmetric bounds without linearizing the exponential.
    These bounds exclude uncertainty in Tg, Ae and the viscosity model.
    """
    from scipy.stats import norm

    offset = float(norm.ppf(.975)) * ANN_MYEGA_FRAGILITY_SD
    first = myega(temperature_k, tg_k, fragility=ANN_MYEGA_FRAGILITY - offset)
    second = myega(temperature_k, tg_k, fragility=ANN_MYEGA_FRAGILITY + offset)
    center = myega(temperature_k, tg_k)
    return center, np.minimum(first, second), np.maximum(first, second)


class ANNViscosityPredictor:
    """Load one supplied gpvisc GP model once and reuse its prediction cache."""

    def __init__(self, model_number=1):
        import gpvisc
        import torch

        # A small thread count avoids excessive CPU overhead for small queries.
        torch.set_num_threads(2)
        self.gpvisc = gpvisc
        self.model_number = model_number
        self.model, self.likelihood = gpvisc.load_gp_model(model_number, device='cpu')
        self.composition = gpvisc.generate_query_single(
            **ANN_COMPOSITION_WT, composition_mole=False, nb_values=1)

    def predict(self, temperature_k, pressure_gpa):
        """Return GP mean and predictive sd in log10(Pa s), flattened arrays.

        Uses the full GP, including its likelihood, rather than the ANN mean
        function alone. Pressure and temperature broadcast to a common shape.
        """
        temperature, pressure = np.broadcast_arrays(temperature_k, pressure_gpa)
        if np.any(temperature <= 0) or not np.all(np.isfinite(temperature)):
            raise ValueError('Temperature must be finite and positive kelvin')
        if not np.all(np.isfinite(pressure)):
            raise ValueError('Pressure must be finite GPa')
        inputs = self.gpvisc.scale_for_gaussianprocess(
            temperature.ravel(), pressure.ravel(), self.composition)
        mean, sd = self.gpvisc.predict(inputs, self.model, self.likelihood, device='cpu')
        return np.asarray(mean, float).ravel(), np.asarray(sd, float).ravel()


def solve_viscous_tg(predictor, pressure_gpa, target=12., tolerance_k=.002):
    """Solve mean log10 viscosity = target by simultaneous bracketed bisection.

    First scan 250–2000 K at each pressure and require exactly one descending
    crossing. Fail explicitly if the scan has no root or ambiguous roots.
    Refine every bracket together for efficient GP inference. Predictive sd
    is returned at each root; it is not a Tg confidence interval.
    """
    pressure = np.asarray(pressure_gpa, float)
    if pressure.ndim != 1 or not np.all(np.isfinite(pressure)):
        raise ValueError('Pressures must be a finite one-dimensional array')
    if not np.isfinite(tolerance_k) or tolerance_k <= 0:
        raise ValueError('Root tolerance must be finite and positive')
    scan = np.linspace(250., 2000., 36)
    temperatures = np.tile(scan, (len(pressure), 1))
    pressures = np.broadcast_to(pressure[:, None], temperatures.shape)
    mean, _ = predictor.predict(temperatures, pressures)
    residual = mean.reshape(temperatures.shape) - target
    crossings = (residual[:, :-1] >= 0) & (residual[:, 1:] < 0)
    if not np.all(crossings.sum(axis=1) == 1):
        raise ValueError('Expected one descending viscosity crossing at every pressure')
    index = crossings.argmax(axis=1)
    lower, upper = scan[index].copy(), scan[index + 1].copy()
    while np.max(upper - lower) > tolerance_k:
        midpoint = (lower + upper) / 2
        value, _ = predictor.predict(midpoint, pressure)
        cold = value > target
        lower = np.where(cold, midpoint, lower)
        upper = np.where(cold, upper, midpoint)
    tg = (lower + upper) / 2
    mean, sd = predictor.predict(tg, pressure)
    if np.max(np.abs(mean - target)) > .002:
        raise ValueError('Viscosity root residual exceeds 0.002 log10 Pa s')
    return tg, mean, sd


def calculate_gpvisc(model_number=1):
    """Calculate Tg over 0–13 GPa and direct high-temperature viscosities."""
    predictor = ANNViscosityPredictor(model_number)
    pressure = np.linspace(0., 13., 53)
    tg, log_eta, sd = solve_viscous_tg(predictor, pressure)
    high_temperature = {}
    for temperature_c in (1400., 1500., 1600.):
        mean, error = predictor.predict(temperature_c + 273.15, pressure)
        high_temperature[str(int(temperature_c))] = {
            'log10_eta': mean.tolist(), 'predictive_sd_log10_eta': error.tolist()}
    total = sum(ANN_COMPOSITION_WT.values())
    result = {
        'gpvisc_version': version('gpvisc'), 'model_number': model_number,
        'device': 'cpu', 'prediction': 'full GP including likelihood',
        'composition_wt_supplied': ANN_COMPOSITION_WT,
        'composition_wt_sum': total,
        'composition_wt_normalized': {key: 100 * value / total for key, value in ANN_COMPOSITION_WT.items()},
        'composition_mole_fraction': predictor.composition[predictor.gpvisc.list_oxides()].iloc[0].to_dict(),
        'P_GPa': pressure.tolist(), 'Tg_K': tg.tolist(),
        'Tg_C': (tg - 273.15).tolist(), 'delta_Tg_K': (tg - tg[0]).tolist(),
        'log10_eta_at_Tg': log_eta.tolist(), 'predictive_sd_log10_eta_at_Tg': sd.tolist(),
        'target_log10_eta': 12., 'root_tolerance_K': .002,
        'high_temperature_C': high_temperature,
        'limitations': 'Viscous Tg is a GP mean isoviscosity, not measured Tg. No Tg uncertainty band inferred from marginal viscosity sd. High-pressure hydrous ANN may be extrapolative.'}
    GPVISC_RESULT.parent.mkdir(parents=True, exist_ok=True)
    GPVISC_RESULT.write_text(json.dumps(result, indent=2) + '\n')
    print('gpvisc Tg endpoints (C):', result['Tg_C'][0], result['Tg_C'][-1], flush=True)
    return result


def load_gpvisc_result():
    """Read the reproducible prediction table; calculate it if absent."""
    if not GPVISC_RESULT.exists():
        return calculate_gpvisc()
    result = json.loads(GPVISC_RESULT.read_text())
    if result['composition_wt_supplied'] != ANN_COMPOSITION_WT:
        raise ValueError('Stale gpvisc composition: rerun python -m src.ann_viscosity')
    return result


def check_model_sensitivity():
    """Compare independent pretrained models at 0, 6.5 and 13 GPa.

    Model disagreement diagnoses extrapolation; it is not a confidence band.
    """
    pressure = np.array([0., 6.5, 13.])
    output = {}
    for number in (2, 3):
        predictor = ANNViscosityPredictor(number)
        tg, mean, _ = solve_viscous_tg(predictor, pressure)
        output[str(number)] = {
            'P_GPa': pressure.tolist(), 'Tg_C': (tg - 273.15).tolist(),
            'delta_Tg_K': (tg - tg[0]).tolist(), 'root_log10_eta': mean.tolist()}
        print('Model', number, output[str(number)], flush=True)
    (RESULTS / 'ann_gpvisc_model_sensitivity.json').write_text(
        json.dumps(output, indent=2) + '\n')
    return output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-models', action='store_true',
                        help='Also compare Tg from GP models 2 and 3')
    arguments = parser.parse_args()
    calculate_gpvisc()
    if arguments.check_models:
        check_model_sensitivity()
