"""Ruby R1 and SrB4O7:Sm2+ 0-0 optical pressure gauges.

Wavelengths in nm, temperature in K, pressures in GPa.
Wei et al. (2024), J. Appl. Phys. 135, 105902, Eq. (10), Tables I, II, IV.
"""
import warnings

import numpy as np
from uncertainties import correlated_values
from uncertainties import unumpy as unp

from .config import (BORATE_SCALE, LBD0_BORATE_NM as CONFIG_LBD0_BORATE_NM,
                     LBD0_RUBY_NM as CONFIG_LBD0_RUBY_NM, RUBY_SCALE)


def ruby_T_corr(T_K, calibration="datchi2007"):
    """Zero-pressure ruby R1 thermal wavelength shift in nm, relative to 296 K."""
    dT = np.asarray(T_K, dtype=float) - 296.0
    if calibration == "datchi2007":
        return 0.00746*dT - 3.01e-6*dT**2 + 8.76e-9*dT**3
    if calibration == "wei2024":
        return 7.29e-3*dT - 3.47e-6*dT**2 + 1.09e-8*dT**3
    raise ValueError("calibration must be 'datchi2007' or 'wei2024'")


def borate_T_corr(T_K, calibration="datchi2007"):
    """Zero-pressure SrB4O7:Sm2+ 0-0 thermal wavelength shift in nm."""
    dT = np.asarray(T_K, dtype=float) - 296.0
    if calibration == "datchi2007":
        return -8.7e-5*dT + 4.62e-6*dT**2 - 2.38e-9*dT**3
    if calibration == "wei2024":
        return -0.70e-4*dT
    raise ValueError("calibration must be 'datchi2007' or 'wei2024'")


# Zero-pressure positions measured in the 23 degC Raman room and adopted
# directly at 296 K; the nominal 25C filenames are not acquisition thermometry.
# One set of references lives in config.py and is re-exported here for callers.
# These anchors affect every pressure and the gauge-equivalent thermal offset.
LBD0_RUBY_NM = CONFIG_LBD0_RUBY_NM
LBD0_BORATE_NM = CONFIG_LBD0_BORATE_NM


def pressure_sensor(lbd_nm, T_K, scale="ruby", lbd_0_ruby=LBD0_RUBY_NM,
                    lbd_0_borate=LBD0_BORATE_NM,
                    borate_scale=None, ruby_scale=None, apply_T_shift=True, *,
                    return_uncertainty=False, calibration_cov=None,
                    sigma_lbd_nm=0.0, sigma_T_K=0.0, sigma_lbd_0_nm=0.0):
    """Return pressure in GPa, or (P, u_P) if return_uncertainty=True.

    Wavelengths and their standard uncertainties are in nm; sample temperature
    and its standard uncertainty are in K. Inputs support NumPy broadcasting.

    Reference wavelengths are measured ambient zero-pressure values adopted
    directly at 296 K (23 degC Raman room, src.config.LBD_REFERENCE_*), not
    shifted from the historical nominal 25 degC filename labels. The default calibration is Wei2024 for both gauges.

    T_K is the temperature of the SAMPLE, not of the thermocouple and not 296 K:
    the function subtracts the thermal shift s(T) of the zero-pressure line, so
    a room-temperature T here inflates every pressure at high temperature.

    Calibration uncertainty is supported for Wei2024 only. Errors from Tables
    I, II and IV are treated as standard uncertainties of independent
    coefficients. This is an approximation, not a full accuracy budget:
    model discrepancy and pressure-reference systematics are excluded.

    calibration_cov optionally replaces this independence assumption. Supply
    the coefficient covariance in physical units, in the order
    (A0, A1, A2, B0, B1, t1, t2, t3) for ruby, or
    (A0, A1, A2, B0, B1, t1) for borate, where thermal shift = sum(ti*dT**i).
    The covariance must be symmetric and positive semidefinite.

    sigma_lbd_nm, sigma_T_K and sigma_lbd_0_nm default to zero and are
    independent of each other and the calibration. Returned uncertainties
    are pointwise: shared calibration errors must not be treated as
    independent across measurements in subsequent fits.

    Examples
    --------
    >>> P, u_P = pressure_sensor(700., 500., ruby_scale="wei2024",
    ...                          return_uncertainty=True)
    """
    if scale not in ("ruby", "borate"):
        raise ValueError("scale must be 'ruby' or 'borate'")
    # One calibration for the whole study: Wei2024 on both gauges, as set in
    # src.config. Never mix the two gauges' scales, the offset analysis would
    # absorb the difference.
    ruby_scale = RUBY_SCALE if ruby_scale is None else ruby_scale
    borate_scale = BORATE_SCALE if borate_scale is None else borate_scale
    selected = ruby_scale if scale == "ruby" else borate_scale
    if return_uncertainty and selected != "wei2024":
        raise ValueError("Calibration uncertainties require the wei2024 scale")
    if not return_uncertainty and (calibration_cov is not None or any(
        np.any(np.asarray(s) != 0)
        for s in (sigma_lbd_nm, sigma_T_K, sigma_lbd_0_nm)
    )):
        raise ValueError("Uncertainty inputs require return_uncertainty=True")

    wavelength = np.asarray(lbd_nm, dtype=float)
    T = np.asarray(T_K, dtype=float)
    reference = np.asarray(lbd_0_ruby if scale == "ruby" else lbd_0_borate,
                           dtype=float)
    for value in (wavelength, T, reference):
        if np.any(~np.isfinite(value)) or np.any(value <= 0):
            raise ValueError("Wavelengths and temperature must be finite and positive")

    if selected == "wei2024":
        if np.any((T < 296) | (T > 850)):
            warnings.warn("Wei2024 calibration range: 296-850 K, up to 55 GPa",
                          RuntimeWarning, stacklevel=2)

        # Wei et al. (2024), Tables I, II and IV: (value, uncertainty).
        if scale == "ruby":
            coefficients = [
                (1881., 1.), (.35, .07), (1.15e-3, .12e-3),
                (10.94, .03), (-.0381, .0023),
                (7.29e-3, .62e-3), (-3.47e-6, .38e-6), (1.09e-8, .57e-8),
            ]
        else:
            coefficients = [
                (2761., 20.), (-.36, .09), (2.07e-5, .14e-5),
                (-9.88, .82), (.0223, .0025), (-.70e-4, .08e-4),
            ]
        values, errors = np.asarray(coefficients).T
        if return_uncertainty:
            if calibration_cov is None:
                coefficients = unp.uarray(values, errors)
            else:
                cov = np.asarray(calibration_cov, dtype=float)
                if cov.shape != (len(values), len(values)) or not np.all(np.isfinite(cov)):
                    raise ValueError("Invalid calibration covariance shape or values")
                # Check in normalized units because coefficient scales differ.
                norms = np.sqrt(np.maximum(np.diag(cov), 0))
                norms = np.where(norms > 0, norms, 1.)
                corr = cov / np.outer(norms, norms)
                if (not np.allclose(corr, corr.T, rtol=1e-10, atol=1e-12)
                        or np.min(np.linalg.eigvalsh(corr)) < -1e-10
                        or np.any(np.diag(cov) < 0)):
                    raise ValueError("Calibration covariance must be symmetric and positive semidefinite")
                coefficients = correlated_values(values, cov)
            for sigma in (sigma_lbd_nm, sigma_T_K, sigma_lbd_0_nm):
                if np.any(~np.isfinite(sigma)) or np.any(np.asarray(sigma) < 0):
                    raise ValueError("Standard uncertainties must be finite and nonnegative")
            wavelength = unp.uarray(wavelength, sigma_lbd_nm)
            T = unp.uarray(T, sigma_T_K)
            reference = unp.uarray(reference, sigma_lbd_0_nm)
        else:
            coefficients = values

        A0, A1, A2, B0, B1, *thermal = coefficients
        dT = T - 296.
        A = A0 + A1*dT + A2*dT**2
        B = B0 + B1*dT
        shift = sum(c*dT**i for i, c in enumerate(thermal, 1))
        lbd_T = reference + (shift if apply_T_shift else 0.)
        if np.any(unp.nominal_values(lbd_T) <= 0):
            raise ValueError("Temperature-corrected reference must be positive")

        math = unp if return_uncertainty else np
        log_ratio = math.log(wavelength / lbd_T)
        z = B * log_ratio
        # Series avoids division by zero at B(T)=0 and retains its uncertainty.
        small = np.abs(unp.nominal_values(z)) < 1e-4
        safe_z = np.where(small, 1., z)
        pressure = A * log_ratio * np.where(
            small, 1 + z/2 + z**2/6 + z**3/24 + z**4/120,
            math.expm1(z)/safe_z)

    elif scale == "ruby" and selected == "shen2020":
        shift = ruby_T_corr(T) if apply_T_shift else 0.
        x = (wavelength - shift - reference) / reference
        pressure = 1.87e3*x*(1. + 5.63*x)
    elif scale == "borate" and selected in ("datchi2007", "rashchenko2015"):
        shift = borate_T_corr(T) if apply_T_shift else 0.
        dlbd = wavelength - shift - reference
        if selected == "datchi2007":
            pressure = 3.989*dlbd*(1. + .006915*dlbd)/(1. + .0166*dlbd)
        else:
            pressure = -2836./14.3*((1. + dlbd/685.51)**(-14.3) - 1.)
    else:
        raise ValueError(f"Unknown {scale} calibration: {selected}")

    if return_uncertainty:
        nominal, std = unp.nominal_values(pressure), unp.std_devs(pressure)
        return (float(nominal), float(std)) if nominal.ndim == 0 else (nominal, std)
    return float(pressure) if np.ndim(pressure) == 0 else pressure
