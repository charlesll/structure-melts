"""Pseudo-Voigt fit of a fluorescence line on the raw spectrum.

The fit gives line position, FWHM and 1-sigma uncertainties from the local
covariance (conditional on the model; excludes calibration and systematics).
"""
import numpy as np
from scipy.optimize import least_squares


def pseudo_voigt(x, area, x0, fwhm, eta):
    """Unit-area pseudo-Voigt (eta: Lorentzian fraction) times area."""
    sigma = fwhm / 2.3548200450309493
    gauss = np.exp(-0.5 * ((x - x0) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    lorentz = (fwhm / 2) / np.pi / ((x - x0) ** 2 + (fwhm / 2) ** 2)
    return area * (eta * lorentz + (1 - eta) * gauss)


def fit_line(x, y, position, fwhm, half_window=1.5):
    """Fit a pseudo-Voigt plus linear baseline around a detected line.

    x, y : raw spectrum (x in nm); position, fwhm : first guesses in x units.
    half_window : fit range in units of the guessed FWHM around ``position``.
    Returns dict with position, fwhm, position_err, fwhm_err, eta, rms,
    success and the fitted window (x_fit, y_fit normalised, y_model); NaN values when the fit fails or has too few points.
    """
    nan = {k: np.nan for k in ("position", "fwhm", "position_err", "fwhm_err", "eta", "rms")}
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.abs(x - position) <= half_window * max(fwhm, 0.1)
    if m.sum() < 12:
        return {**nan, "success": False}
    xs, ys = x[m], y[m] / np.max(y[m])
    xc = xs - position
    amp = np.trapezoid(ys - np.min(ys), xs)
    p0 = [amp, 0.0, max(fwhm, 0.05), 0.5, np.min(ys), 0.0]
    lo = [0.0, -0.5 * fwhm, 0.02, 0.0, -np.inf, -np.inf]
    hi = [np.inf, 0.5 * fwhm, 4 * max(fwhm, 0.1), 1.0, np.inf, np.inf]

    def resid(p):
        return pseudo_voigt(xc, *p[:4]) + p[4] + p[5] * xc - ys

    try:
        sol = least_squares(resid, p0, bounds=(lo, hi), loss="soft_l1", f_scale=0.02)
    except (ValueError, FloatingPointError):
        return {**nan, "success": False}
    r = resid(sol.x)
    dof = max(len(xs) - len(p0), 1)
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * (r @ r) / dof
        err = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        err = np.full(len(p0), np.nan)
    return {"position": position + sol.x[1], "fwhm": sol.x[2],
            "position_err": err[1], "fwhm_err": err[2], "eta": sol.x[3],
            "rms": float(np.sqrt(np.mean(r ** 2))), "success": bool(sol.success),
            "x_fit": xs, "y_fit": ys, "y_model": ys + r}
