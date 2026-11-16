"""Pressure from ruby and borate grains at the sample temperature.

By default the thermocouple temperature is used as a sample-temperature
estimate. process_run(sample_T_C=...) can supply an independent estimate.
The reference wavelength in pressure_sensor remains defined at 296 K.

P_std is a descriptive spread: sqrt(sample_variance + mean(u_grain**2)).
It is NOT the standard error of the median. Calibration errors are shared
between grains, so their contribution is not divided by the grain count.
Scatter and propagated uncertainty are also saved separately. The scatter
can already include measurement noise: this quadrature is a descriptive
budget, not a deconvolution of physical gradients and measurement errors.

Usage: python -m src.pressure
Output: results/dataset.h5
"""

import h5py
import numpy as np
import rampy as rp
from scipy.optimize import minimize_scalar

from .config import (BORATE_SCALE, LBD0_BORATE_NM, LBD0_RUBY_NM, RAMAN_AXIS,
                     RESULTS, RUBY_SCALE, RUNS)


def pressure_sensor(*args, **kwargs):
    """Load calibration dependencies only when converting pressures."""
    from .pressure_sensors import pressure_sensor as convert
    return convert(*args, **kwargs)


DATASET = RESULTS / "dataset.h5"
LASER_NM = 532.0


def grain_pressures(positions, T_C, sensor, scale, *,
                    position_err=None, sigma_T_K=0., sigma_lbd_0_nm=0.):
    """Return (P, u_P), shape (n_steps, n_grains), in GPa.

    T_C is the SAMPLE temperature in Celsius, one value per step.
    position_err is the standard uncertainty of the line centre in nm,
    not the linewidth. Missing errors (None) contribute zero; supplied NaNs
    leave u_P unknown. Temperature/reference uncertainties can be scalars,
    per-step vectors, or arrays broadcastable to positions.
    Wei2024 is required because other calibration uncertainties are unavailable.
    """
    if sensor not in ("ruby", "borate"):
        raise ValueError("sensor must be 'ruby' or 'borate'")
    if scale != "wei2024":
        raise ValueError(
            "Grain uncertainty propagation requires wei2024; "
            "set RUBY_SCALE and BORATE_SCALE accordingly in config.py")
    pos = np.asarray(positions, float)
    temperatures = np.asarray(T_C, float)
    if pos.ndim != 2 or temperatures.shape != (pos.shape[0],):
        raise ValueError("Expected positions (n_steps, n_grains) and T_C (n_steps,)")
    T_K = np.broadcast_to(temperatures[:, None] + 273.15, pos.shape)

    def error_array(value):
        value = np.asarray(value, float)
        if value.ndim == 1 and value.shape == (pos.shape[0],):
            value = value[:, None]
        value = np.broadcast_to(value, pos.shape)
        if np.any(value < 0) or np.any(np.isinf(value)):
            raise ValueError("Standard uncertainties must be nonnegative or NaN")
        return value

    errors = [error_array(0. if position_err is None else position_err),
              error_array(sigma_T_K), error_array(sigma_lbd_0_nm)]
    P, u_P = np.full(pos.shape, np.nan), np.full(pos.shape, np.nan)
    valid = np.isfinite(pos) & np.isfinite(T_K) & (pos > 0) & (T_K > 0)
    if valid.any():
        # The zero-pressure reference is our measured room-temperature anchor,
        # adopted at 296 K (23 °C room; nominal 25C filename, src.config). The scale is
        # Wei2024 for both gauges. Passed explicitly so that these two anchors
        # are visible at the call site and cannot drift with a default.
        lbd_0 = LBD0_RUBY_NM if sensor == "ruby" else LBD0_BORATE_NM
        kw = {f"{sensor}_scale": scale, f"lbd_0_{sensor}": lbd_0,
              "apply_T_shift": True}
        # Missing uncertainty does not prevent calculating nominal pressure.
        P[valid], u_P[valid] = pressure_sensor(
            pos[valid], T_K[valid], scale=sensor, return_uncertainty=True,
            sigma_lbd_nm=np.nan_to_num(errors[0][valid]),
            sigma_T_K=np.nan_to_num(errors[1][valid]),
            sigma_lbd_0_nm=np.nan_to_num(errors[2][valid]), **kw)
        known_errors = np.logical_and.reduce([np.isfinite(e) for e in errors])
        u_P[~known_errors] = np.nan
    return P, u_P


def summarise(values, uncertainties=None):
    """Row-wise median, spread and number of finite values.

    Without uncertainties: sample std (NaN for one grain).
    With uncertainties: sqrt(sample_std**2 + mean(u_i**2)); for one grain,
    return its uncertainty, since no between-grain scatter can be estimated.
    If a valid grain has an unknown uncertainty, the total is NaN.
    This total is not the uncertainty of the median.
    """
    values = np.asarray(values, float)
    if values.ndim != 2:
        raise ValueError("values must have shape (n_steps, n_grains)")
    valid = np.isfinite(values)
    clean = np.where(valid, values, np.nan)
    n = valid.sum(axis=1)
    med, std = np.full(len(values), np.nan), np.full(len(values), np.nan)
    has, many = n > 0, n > 1
    med[has] = np.nanmedian(clean[has], axis=1)
    std[many] = np.nanstd(clean[many], axis=1, ddof=1)
    if uncertainties is not None:
        u = np.asarray(uncertainties, float)
        if u.shape != values.shape or np.any(u < 0) or np.any(np.isinf(u)):
            raise ValueError("uncertainties must match values and be nonnegative or NaN")
        complete = has & np.all(~valid | np.isfinite(u), axis=1)
        variance = np.sum(np.where(valid, u**2, 0.), axis=1)
        scatter_variance = np.where(n == 1, 0., std**2)
        std[:] = np.nan
        std[complete] = np.sqrt(
            scatter_variance[complete] + variance[complete]/n[complete])
    return med, std, n


def temperature_correction(ruby_nm, borate_nm, T_C, sigma_P=0.4, sigma_T=60.0,
                           max_correction=120.0, ruby_scale=None,
                           borate_scale=None):
    """Temperature offset (K) that makes ruby and borate pressures agree.

    For each step minimise ((P_ruby - P_borate)/sigma_P)^2 + (dT/sigma_T)^2 over
    dT in [-max_correction, +max_correction]; the second term regularises
    steps where the two sensors cannot be reconciled. `sigma_P` acts as a soft
    tolerated mismatch (per step, in GPa) and `sigma_T` as the prior standard
    deviation of the offset: the compromise sits at
    dT = -slope*dP / (slope**2 + (sigma_P/sigma_T)**2) locally,
    where slope = d(P_ruby-P_borate)/dT. It is a regularised mismatch,
    not a solution of P_ruby = P_borate. `src.temperature` solves the exact
    balance and reports the distribution of offsets.

    The calibrations are only valid between 296 and 850 K, so the search window
    is clipped to it. This offset is a diagnostic: it is stored for reference
    in `results/dataset.h5` (`T_cor`) but it is NOT applied to the pressures
    reported in the dataset.
    """
    if not all(np.isfinite(x) and x > 0
               for x in (sigma_P, sigma_T, max_correction)):
        raise ValueError("sigma_P, sigma_T and max_correction must be positive")
    ruby_scale = ruby_scale or RUBY_SCALE
    borate_scale = borate_scale or BORATE_SCALE
    ruby_nm, borate_nm, T_C = (np.asarray(a, float) for a in (ruby_nm, borate_nm, T_C))
    if T_C.ndim != 1 or ruby_nm.shape != T_C.shape or borate_nm.shape != T_C.shape:
        raise ValueError("Sensor medians and T_C must be matching 1D arrays")
    corr = np.full(T_C.shape, np.nan)
    for i in range(len(T_C)):
        if not (np.isfinite(ruby_nm[i]) and np.isfinite(borate_nm[i])
                and np.isfinite(T_C[i])):
            continue
        T0 = T_C[i] + 273.15
        lo, hi = max(-max_correction, 296.0 - T0), min(max_correction, 850.0 - T0)
        if lo >= hi:
            continue

        def cost(dT):
            # Same adopted 296 K anchors and Wei2024 scales as above.
            pr = pressure_sensor(ruby_nm[i], T0 + dT, scale="ruby",
                                 ruby_scale=ruby_scale, lbd_0_ruby=LBD0_RUBY_NM,
                                 apply_T_shift=True)
            pb = pressure_sensor(borate_nm[i], T0 + dT, scale="borate",
                                 borate_scale=borate_scale,
                                 lbd_0_borate=LBD0_BORATE_NM, apply_T_shift=True)
            return ((pr - pb) / sigma_P) ** 2 + (dT / sigma_T) ** 2

        res = minimize_scalar(cost, bounds=(lo, hi), method="bounded")
        if res.success and np.isfinite(res.fun):
            corr[i] = res.x
    return corr


def long_correction(x, spectra, T_C):
    """Temperature/frequency (Long) correction of each spectrum at its own T."""
    spectra, T_C = np.asarray(spectra, float), np.asarray(T_C, float)
    if spectra.shape[0] != len(T_C):
        raise ValueError("One temperature is required per spectrum")
    out = np.full_like(spectra, np.nan, dtype=float)
    for i, (y, T) in enumerate(zip(spectra, T_C)):
        if np.isfinite(y).all() and np.isfinite(T) and T > -273.15:
            out[i] = rp.tlcorrection(x, y.copy(), float(T), LASER_NM)[1].ravel()
    return out


def process_run(markers_file, ruby_scale=RUBY_SCALE, borate_scale=BORATE_SCALE,
                *, sample_T_C=None, sigma_T_K=0.,
                sigma_lbd_0_ruby_nm=0., sigma_lbd_0_borate_nm=0.):
    """Process a run using sample_T_C, or thermocouple T_C by default.

    sigma_T_K is a standard uncertainty (scalar or one value per step).
    Marker position errors are assumed to be standard uncertainties in nm.
    T_cor remains a diagnostic offset relative to the original thermocouple
    T_C; it is not added automatically to sample_T_C or to the pressures.
    Missing position-error datasets contribute zero, with availability saved.
    """
    with h5py.File(markers_file, "r") as f:
        d = {k: f[k][()] for k in ("T_C", "Pmem")}
        d["Etat"] = f["Etat"].asstr()[()]
        for sensor, grp in (("ruby", "Ruby"), ("borate", "Borate")):
            # Prefer the pseudo-Voigt line fit (borate) over the peak-maximum estimate.
            fitted = "fit_positions" in f[grp]
            d[f"{sensor}_positions"] = f[grp]["fit_positions" if fitted else "positions"][()]
            d[f"{sensor}_widths"] = f[grp]["fit_widths" if fitted else "widths"][()]
            error_key = "fit_position_err" if fitted else "position_err"
            if error_key in f[grp]:
                d[f"{sensor}_position_err"] = f[grp][error_key][()]
        glass_lf, glass_hf = f["Glass/LF"][()], f["Glass/HF"][()]
        d["run_id"] = f.attrs["run_id"]
    d["T_sample_C"] = np.asarray(d["T_C"] if sample_T_C is None else sample_T_C, float)
    if d["T_sample_C"].shape != d["T_C"].shape:
        raise ValueError("sample_T_C must contain one temperature per step")
    d["sigma_T_K"] = np.broadcast_to(np.asarray(sigma_T_K, float), d["T_C"].shape)
    for sensor, scale in (("ruby", ruby_scale), ("borate", borate_scale)):
        pos = d[f"{sensor}_positions"]
        position_err = d.get(f"{sensor}_position_err")
        P, u_P = grain_pressures(
            pos, d["T_sample_C"], sensor, scale, position_err=position_err,
            sigma_T_K=sigma_T_K,
            sigma_lbd_0_nm=(sigma_lbd_0_ruby_nm if sensor == "ruby"
                            else sigma_lbd_0_borate_nm))
        med, std, n = summarise(P, u_P)
        _, scatter, _ = summarise(P)
        valid = np.isfinite(P)
        rms = np.full(len(P), np.nan)
        complete = (n > 0) & np.all(~valid | np.isfinite(u_P), axis=1)
        rms[complete] = np.sqrt(
            np.sum(np.where(valid, u_P**2, 0.), axis=1)[complete]/n[complete])
        w_med, w_std, _ = summarise(d[f"{sensor}_widths"])
        pos_med, _, _ = summarise(pos)
        d.update({f"{sensor}_P": P, f"{sensor}_P_err": u_P,
                  f"{sensor}_P_median": med, f"{sensor}_P_std": std,
                  f"{sensor}_P_scatter": scatter, f"{sensor}_P_err_rms": rms,
                  f"{sensor}_position_err_available":
                      np.zeros(pos.shape, bool) if position_err is None
                      else np.isfinite(position_err),
                  f"{sensor}_n": n, f"{sensor}_width_median": w_med,
                  f"{sensor}_width_std": w_std, f"{sensor}_position_median": pos_med})
    d["T_cor"] = temperature_correction(d["ruby_position_median"],
                                        d["borate_position_median"], d["T_C"],
                                        ruby_scale=ruby_scale, borate_scale=borate_scale)
    d["spectra"] = long_correction(RAMAN_AXIS, glass_lf, d["T_sample_C"])
    d["spectra_HF"] = glass_hf
    return d


def write_dataset(runs=RUNS, path=DATASET, **kw):
    """Process all runs and write the HDF5 dataset (per-run groups + flat arrays)."""
    runs = list(runs)
    if not runs:
        raise ValueError("No runs supplied")
    results = [process_run(r.markers_file, **kw) for r in runs]
    path.parent.mkdir(parents=True, exist_ok=True)
    scalar_keys = ("T_C", "T_sample_C", "sigma_T_K", "Pmem", "Etat", "T_cor", "spectra", "spectra_HF")
    with h5py.File(path, "w") as f:
        f.attrs["ruby_scale"] = kw.get("ruby_scale", RUBY_SCALE)
        f.attrs["borate_scale"] = kw.get("borate_scale", BORATE_SCALE)
        f.attrs["pressure_spread_definition"] = "sqrt(sample_std**2 + mean(grain_uncertainty**2)); not median standard error"
        f.attrs["temperature_correction_applied"] = False
        f.attrs["missing_position_errors"] = "Absent errors contribute zero; see position_err_available"
        f.create_dataset("x_axis", data=RAMAN_AXIS)
        for run, d in zip(runs, results):
            g = f.create_group(run.run_id)
            g.attrs["is_tg_series"] = run.is_tg_series
            for k, v in d.items():
                if k == "run_id":
                    continue
                g.create_dataset(k, data=v.astype("S") if v.dtype.kind == "U" else v)
        # Flat arrays, one row per (run, step), for the final analysis
        flat = f.create_group("flat")
        cat = lambda k: np.concatenate([d[k] for d in results])
        for k in scalar_keys:
            v = cat(k)
            flat.create_dataset(k, data=v.astype("S") if v.dtype.kind == "U" else v)
        for k in ("ruby_P_median", "ruby_P_std", "borate_P_median", "borate_P_std",
                  "ruby_P_scatter", "borate_P_scatter",
                  "ruby_P_err_rms", "borate_P_err_rms", "ruby_n", "borate_n",
                  "ruby_position_median", "borate_position_median"):
            flat.create_dataset(k, data=cat(k))
        flat.create_dataset("exp_nb", data=np.concatenate(
            [np.full(len(d["T_C"]), i) for i, d in enumerate(results)]))
        flat.create_dataset("run_id", data=np.concatenate(
            [np.full(len(d["T_C"]), r.run_id) for r, d in zip(runs, results)]).astype("S"))
    return path


def load_dataset(path=DATASET):
    """Return {run_id: dict of arrays} from the HDF5 dataset."""
    out = {}
    with h5py.File(path, "r") as f:
        for run_id, g in f.items():
            if run_id in ("flat", "x_axis"):
                continue
            d = {k: g[k][()] for k in g}
            d["Etat"] = g["Etat"].asstr()[()]
            d["is_tg_series"] = bool(g.attrs["is_tg_series"])
            out[run_id] = d
    return out


def main():
    print("[pressure] ->", write_dataset())


if __name__ == "__main__":
    main()
