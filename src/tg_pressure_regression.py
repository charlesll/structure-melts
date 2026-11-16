"""Descriptive interval regression of spectral split brackets against P.

The brackets are observed sampling intervals, not Gaussian errors. A normal
latent between-run residual is assumed; the likelihood integrates over each
T interval. Pressure spread enters as an explicit Gaussian sensitivity proxy,
not as a calibrated standard error of the median. The independent fictive-temperature reference does not constrain the fit.

Numerical methods shared by Figures 7, 9 and Supplementary Figure 6.
Run python -m src.figures.fig7_transitions to generate Figures 7 and S6
from one common fit; plotting is kept separate from these numerical methods.
"""
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import log_ndtr


def interval_log_probability(lower, upper):
    """Stable log[Phi(upper)-Phi(lower)] for standardized finite endpoints."""
    lower, upper = np.broadcast_arrays(lower, upper)
    positive = lower > 0
    a = log_ndtr(np.where(positive, -lower, upper))
    b = log_ndtr(np.where(positive, -upper, lower))
    return a + np.log(-np.expm1(np.minimum(b-a, -1e-15)))


def _interval_likelihood(rows, pressure_spread=True):
    """Shared validated likelihood for fitting and absolute-temperature profiles."""
    if len(rows) < 4:
        raise ValueError('At least four runs required for this descriptive fit')
    p = np.array([r['P'] for r in rows], float)
    lo = np.array([r['T_left'] for r in rows], float)
    hi = np.array([r['T_right'] for r in rows], float)
    st = np.array([r['sigma_T'] for r in rows], float)
    endpoint_sd = np.array([[r['P_std_left'], r['P_std_right']] for r in rows], float)
    if not np.isfinite(endpoint_sd).all() or np.any(endpoint_sd < 0) or np.any(st < 0):
        raise ValueError('Invalid or missing endpoint pressure/temperature uncertainty')
    sp = endpoint_sd.max(axis=1)
    sp = np.hypot(sp, np.array([abs(r['P_right']-r['P_left'])/2 for r in rows]))
    if not pressure_spread:
        sp[:] = 0.
    if np.any(hi <= lo) or not np.isfinite(np.r_[p, lo, hi, st, sp]).all():
        raise ValueError('Invalid event intervals or missing uncertainty inputs')

    def nll(theta):
        intercept, slope, scatter = theta
        sd = np.sqrt(st**2 + scatter**2 + (slope*sp)**2)
        sd = np.maximum(sd, 1e-8)
        return float(-interval_log_probability((lo-intercept-slope*p)/sd,
                                               (hi-intercept-slope*p)/sd).sum())

    return nll, p, lo, hi, sp


def fit_interval_line(rows, pressure_spread=True, profile=True, slope_profile=False):
    """Fit T=a+bP to interval-censored descriptive events (P GPa, T °C).

    Profile interval uses delta(-2 log L)=3.84146 (approximate 95% support,
    asymptotic, uncalibrated with five runs). Residual scatter is ML estimated.
    sigma_T is propagated separately from the actual sampling brackets.
    slope_profile optionally profiles the slope, reoptimizing intercept/scatter.
    """
    nll, p, lo, hi, sp = _interval_likelihood(rows, pressure_spread)

    slope, intercept = np.polyfit(p, (lo+hi)/2, 1)
    solutions = [minimize(nll, [intercept, slope, scatter], method='Nelder-Mead',
                          bounds=[(None, None), (None, None), (0, None)],
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-9})
                 for scatter in [0., 10., 30.]]
    optimum = min(solutions, key=lambda result: result.fun)
    if not optimum.success:
        raise RuntimeError(optimum.message)
    theta = optimum.x
    result = dict(intercept_C=float(theta[0]), slope_C_per_GPa=float(theta[1]),
                  residual_scatter_C=float(theta[2]), nll=float(optimum.fun),
                  pressure_spread=pressure_spread, n_runs=len(rows),
                  pressure_proxy_GPa=sp.tolist(), inputs=rows,
                  interpretation='Descriptive extrapolation of spectral breaks, not independently established physical Tg',
                  likelihood='Normal latent residual integrated over observed temperature brackets; no uniform event distribution',
                  pressure_assumption='Gaussian proxy from max endpoint P_std plus half endpoint drift in quadrature; P_std is not median SE',
                  interval_note='Approximate asymptotic profile support, not guaranteed 95% coverage at n=5')
    if profile:
        def profile_cost(a):
            trials = [minimize(lambda z: nll([a, z[0], z[1]]), start,
                               method='Nelder-Mead', bounds=[(None, None), (0, None)],
                               options={'maxiter': 2500, 'xatol': 1e-7, 'fatol': 1e-8})
                      for start in [theta[1:], [theta[1], 25.]]]
            best = min(trials, key=lambda r: r.fun)
            if not best.success:
                raise RuntimeError('Profile optimization failed')
            return float(best.fun)
        threshold = 3.841458820694124 / 2
        roots = []
        for direction in [-1, 1]:
            width = 5.
            while profile_cost(theta[0]+direction*width)-optimum.fun < threshold:
                width *= 2
                if width > 2000:
                    raise RuntimeError('Intercept support interval unbounded in search range')
            roots.append(float(brentq(lambda a: profile_cost(a)-optimum.fun-threshold,
                                     *sorted([theta[0], theta[0]+direction*width]))))
        result['intercept_profile95_C'] = roots
        grid = np.linspace(roots[0]-10, roots[1]+10, 101)
        result['intercept_profile'] = dict(intercept_C=grid.tolist(),
                                            twice_delta_nll=[2*(profile_cost(a)-optimum.fun) for a in grid])
    if slope_profile:
        # Profiling b reoptimizes a and residual scatter jointly. For the SAME
        # fitted line, T(P)-T(0)=bP exactly; no independent intercept error is added.
        def slope_cost(b):
            trials = [minimize(lambda z: nll([z[0], b, z[1]]), start,
                               method='Nelder-Mead', bounds=[(None, None), (0, None)],
                               options={'maxiter': 2500, 'xatol': 1e-7, 'fatol': 1e-8})
                      for start in [[theta[0], theta[2]], [theta[0], 25.]]]
            best = min(trials, key=lambda r: r.fun)
            if not best.success:
                raise RuntimeError('Slope profile optimization failed')
            return float(best.fun)
        threshold = 3.841458820694124 / 2
        roots = []
        for direction in [-1, 1]:
            width = 1.
            while slope_cost(theta[1]+direction*width)-optimum.fun < threshold:
                width *= 2
                if width > 1000:
                    raise RuntimeError('Slope support interval unbounded in search range')
            roots.append(float(brentq(lambda b: slope_cost(b)-optimum.fun-threshold,
                                     *sorted([theta[1], theta[1]+direction*width]))))
        result['slope_profile95_C_per_GPa'] = roots
        grid = np.linspace(roots[0]-1, roots[1]+1, 101)
        result['slope_profile'] = dict(slope_C_per_GPa=grid.tolist(),
                                        twice_delta_nll=[2*(slope_cost(b)-optimum.fun) for b in grid])
    return result


def centered_line_envelope(pressure, fit):
    """Mean linear response T(P)-T(0)=bP and profile-support envelope (K).

    P is a finite nonnegative array in GPa. Intercept is profiled jointly with
    scatter to obtain slope support, then cancels algebraically in each line.
    This is not a prediction interval for a new acquisition or a Gaussian
    propagation treating the intercept as an independent measured reference.
    """
    pressure = np.asarray(pressure, float)
    if pressure.ndim != 1 or not np.isfinite(pressure).all() or np.any(pressure < 0):
        raise ValueError('Expected finite nonnegative pressure vector')
    low, high = fit['slope_profile95_C_per_GPa']
    return (pressure*fit['slope_C_per_GPa'], pressure*low, pressure*high)


def absolute_line_envelope(pressure, rows, fit):
    """Pointwise profile support for absolute mean T(P)=a+bP in °C.

    At each pressure, constrain q=a+bP and profile over b and residual scatter.
    This retains intercept–slope dependence and recovers the intercept support
    at P=0. It is approximate 95% pointwise mean support, not a simultaneous
    band or a prediction interval for new observations (especially at n=5).
    """
    pressure = np.asarray(pressure, float)
    if pressure.ndim != 1 or not np.isfinite(pressure).all() or np.any(pressure < 0):
        raise ValueError('Expected finite nonnegative pressure vector')
    nll, _, _, _, _ = _interval_likelihood(rows, fit['pressure_spread'])
    a, b, scatter = (fit[key] for key in ['intercept_C', 'slope_C_per_GPa', 'residual_scatter_C'])
    if not np.isclose(nll([a, b, scatter]), fit['nll'], atol=1e-7, rtol=0):
        raise ValueError('Fit parameters and profile input data do not match')
    mean = a+b*pressure
    lower, upper = [], []
    threshold = 3.841458820694124/2
    for p0, q0 in zip(pressure, mean):
        def cost(q):
            trials = [minimize(lambda z: nll([q-z[0]*p0, z[0], z[1]]), start,
                               method='Nelder-Mead', bounds=[(None, None), (0, None)],
                               options={'maxiter': 2500, 'xatol': 1e-7, 'fatol': 1e-8})
                      for start in [[b, scatter], [b, 25.]]]
            best = min(trials, key=lambda r: r.fun)
            if not best.success:
                raise RuntimeError('Absolute-temperature profile optimization failed')
            return float(best.fun)
        bounds = []
        for direction in [-1, 1]:
            width = 5.
            while cost(q0+direction*width)-fit['nll'] < threshold:
                width *= 2
                if width > 2000:
                    raise RuntimeError('Absolute-temperature profile is not bounded')
            bounds.append(brentq(lambda q: cost(q)-fit['nll']-threshold,
                                *sorted([q0, q0+direction*width])))
        lower.append(bounds[0])
        upper.append(bounds[1])
    return mean, np.array(lower), np.array(upper)


def nominal_reference_envelope(pressure, rows, fit, reference_C):
    """Display absolute fit support after subtracting a fixed nominal reference.

    This translates the absolute T(P) profile support without re-centering each
    admissible line on its own intercept. Finite intercept uncertainty remains
    visible at zero pressure. It is NOT support for the random within-line
    difference T(P)-T(0), and no independent reference error is added.
    """
    if not np.isscalar(reference_C) or not np.isfinite(reference_C):
        raise ValueError('Expected a finite scalar reference temperature')
    absolute = absolute_line_envelope(pressure, rows, fit)
    return tuple(values-reference_C for values in absolute)

