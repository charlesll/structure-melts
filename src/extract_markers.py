"""Stage 1: fit Ruby / SrB4O7:Sm2+ lines and clean the glass Raman spectra.

For every temperature step of a run, each pressure-sensor grain gives one
spectrum. We fit its R1 (ruby) or 0-0 (borate) line and store position,
centroid, width and the baseline-corrected line on a common wavelength grid.
Glass/melt spectra are background-corrected and put on a common Raman axis.

Usage: python -m src.extract_markers [run_id ...] [--no-fig]
Output: results/markers/<run_id>.h5 ; figures/fits/<run_id>/{Ruby,Borate}/*.png
"""
import argparse
import warnings

import h5py
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import rampy as rp

from .config import (BORATE_DEFAULTS, FIGURES, RAMAN_AXIS, RUBY_DEFAULTS, RUNS,
                     get_run, REFERENCE_RUBY_PARAMS)
from .functions import estimate_peak_parameters
from .raman_treatment import treat_glass, nm_to_invcm
from .line_fit import fit_line

HF_AXIS = np.arange(3000.0, 4000.0, 2.0)  # Raman axis for the O-H stretching region


def _read_asc(path):
    """Read a two-column (x, intensity) .asc file exported by Solis."""
    return np.genfromtxt(path, delimiter=",")[:, :2]


def _fit_sensor(sheet, spectra_dir, grid, params, fig_dir, fit_lines=False, reference_params=None):
    """Fit one sensor family (ruby or borate) for all steps and grains.

    Returns dict of arrays (n_steps, n_grains) for centroid/position/width and
    (n_steps, n_grid, n_grains) for the cleaned lines. Failed fits give NaN.
    With fit_lines, a pseudo-Voigt fit of the raw line adds fit_positions,
    fit_widths (FWHM) and their 1-sigma errors fit_position_err, fit_width_err.
    """
    cols = list(sheet.columns[2:])
    n, k = len(sheet), len(cols)
    out = {name: np.full((n, k), np.nan) for name in ("centroids", "positions", "widths")}
    out["spectra"] = np.full((n, len(grid), k), np.nan)
    fit_keys = ("fit_positions", "fit_widths", "fit_position_err", "fit_width_err")
    if fit_lines:
        out.update({name: np.full((n, k), np.nan) for name in fit_keys})
    n_fail = 0
    for i in range(n):
        for j, col in enumerate(cols):
            name = sheet.loc[i, col]
            if pd.isna(name):
                continue
            try:
                sp = _read_asc(spectra_dir / f"{name}.asc")
                if fig_dir is not None:
                    fig_dir.mkdir(parents=True, exist_ok=True)
                c, p, w, x, y = estimate_peak_parameters(
                    sp[:, 0], sp[:, 1], figure=fig_dir is not None,
                    save_figure=fig_dir is not None,
                    save_name=str(fig_dir / f"{name}.png") if fig_dir else "",
                    close_figure=True, **{**params, **(reference_params or {}).get(str(name), {})})
            except (IndexError, ValueError, FileNotFoundError, RuntimeError) as err:
                warnings.warn(f"{name}: line fit failed ({err})")
                n_fail += 1
                continue
            out["centroids"][i, j], out["positions"][i, j], out["widths"][i, j] = c, p, w
            out["spectra"][i, :, j] = rp.resample(x, y, grid, fill_value="extrapolate")
            if fit_lines:
                fit = fit_line(sp[:, 0], sp[:, 1], p, w)
                if not fit["success"]:
                    warnings.warn(f"{name}: pseudo-Voigt fit did not converge; retaining NaN fit values")
                    n_fail += 1
                    continue
                for key, val in zip(fit_keys, (fit["position"], fit["fwhm"],
                                               fit["position_err"], fit["fwhm_err"])):
                    out[key][i, j] = val
    out["n_failed"] = n_fail
    return out


def _glass(sheet, spectra_dir, axis_units=None):
    """Clean glass spectra of every step: LF on RAMAN_AXIS, HF on HF_AXIS."""
    lf = np.full((len(sheet), len(RAMAN_AXIS)), np.nan)
    hf = np.full((len(sheet), len(HF_AXIS)), np.nan)
    for i in range(len(sheet)):
        try:
            sp = _read_asc(spectra_dir / f"{sheet.loc[i, 'Glass_LF']}.asc")
            hf_name = sheet.loc[i, 'Glass_HF']
            hf_path = spectra_dir / f"{hf_name}.asc"
            if pd.isna(hf_name) or not hf_path.exists():
                sp_hf = None
                warnings.warn(f"step {i}: HF spectrum unavailable ({hf_name}); retaining measured LF")
            else:
                sp_hf = _read_asc(hf_path)
            bkg = _read_asc(spectra_dir / f"{sheet.loc[i, 'Bkg_LF']}.asc")
            unit = (axis_units or {}).get(str(sheet.loc[i, 'Glass_LF']), 'wavelength_nm')
            if unit not in ['wavelength_nm', 'raman_cm1']:
                raise ValueError(f'Unsupported LF input axis unit: {unit}')
            raman = nm_to_invcm(sp[:, 0]) if unit == 'wavelength_nm' else sp[:, 0]
            if raman.min() > RAMAN_AXIS.min() or raman.max() < RAMAN_AXIS.max():
                raise ValueError(f'LF axis does not cover the common Raman window; check units ({unit})')
            x, y, x_hf, y_hf = treat_glass(sp, sp_hf, bkg, convert_in_cm=unit == 'wavelength_nm')
        except (ValueError, FileNotFoundError, RuntimeError) as err:
            warnings.warn(f"glass spectrum of step {i}: {err}")
            continue
        lf[i] = rp.resample(x.ravel(), y.ravel(), RAMAN_AXIS, fill_value="extrapolate")
        if len(x_hf):
            order = np.argsort(x_hf.ravel())
            hf[i] = np.interp(HF_AXIS, x_hf.ravel()[order], y_hf.ravel()[order],
                              left=np.nan, right=np.nan)
    return lf, hf


def align_workbook_sheets(run, sheets):
    """Align all rows to Glass using (T_C, Pmem, repeated-key occurrence).

    Never zip unequal sheets by row index. Missing ambient sensor rows use one
    explicitly registered reference spectrum, not replicated pseudo-grains.
    Original workbooks are read-only. Missing non-reference sensors remain NaN.
    """
    def keyed(table):
        counts, out = {}, {}
        for _, row in table.iterrows():
            key = (float(row['T_C']), float(row['Pmem']))
            count = counts.get(key, 0)
            counts[key] = count+1
            out[key+(count,)] = row
        return out
    master = sheets['Glass'].reset_index(drop=True)
    master_keys = list(keyed(master))
    result = {'Glass': master}
    for label in ['Ruby', 'Borate', 'Tg']:
        source = sheets[label]
        lookup = keyed(source)
        extra = set(lookup)-set(master_keys)
        if extra:
            raise ValueError(f'{run.run_id}: {label} has acquisitions absent from Glass: {extra}')
        rows = []
        for key in master_keys:
            if key in lookup:
                rows.append(lookup[key].to_dict())
                continue
            row = {col: np.nan for col in source.columns}
            row.update(T_C=key[0], Pmem=key[1])
            if label == 'Tg':
                raise ValueError(f'{run.run_id}: missing experimental state for {key}')
            reference = run.ambient_reference
            if (reference and key[:2] == (reference['T_C'], reference['Pmem'])
                    and key[2] == 0):
                stem = reference[label]
                if not (run.spectra_dir / f'{stem}.asc').exists():
                    raise FileNotFoundError(stem)
                row[source.columns[2]] = stem
            else:
                warnings.warn(f'{run.run_id}: no {label} measurements for {key}')
            rows.append(row)
        result[label] = pd.DataFrame(rows, columns=source.columns)
    return result


def extract_run(run, make_figures=True):
    """Process one run and write its marker file. Returns the output path."""
    sheets = align_workbook_sheets(run, pd.read_excel(run.table, sheet_name=None))
    fig_dir = FIGURES / "fits" / run.run_id if make_figures else None
    x_ruby = np.arange(*run.x_ruby, 0.01)
    x_borate = np.arange(*run.x_borate, 0.01)

    ruby = _fit_sensor(sheets["Ruby"], run.spectra_dir, x_ruby,
                       {**RUBY_DEFAULTS, **run.ruby},
                       fig_dir / "Ruby" if fig_dir else None,
                       reference_params={run.ambient_reference['Ruby']: REFERENCE_RUBY_PARAMS}
                           if run.ambient_reference else None)
    borate = _fit_sensor(sheets["Borate"], run.spectra_dir, x_borate,
                         {**BORATE_DEFAULTS, **run.borate},
                         fig_dir / "Borate" if fig_dir else None, fit_lines=True)
    glass_lf, glass_hf = _glass(sheets["Glass"], run.spectra_dir, axis_units=run.glass_axis_units)

    run.markers_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(run.markers_file, "w") as f:
        f.attrs["run_id"] = run.run_id
        f.attrs["row_alignment"] = "Glass master; T_C/Pmem/occurrence keys; no workbook edits"
        f.create_dataset("T_C", data=sheets["Glass"]["T_C"].to_numpy(float))
        f.create_dataset("Pmem", data=sheets["Glass"]["Pmem"].to_numpy(float))
        f.create_dataset("Etat", data=sheets["Tg"]["Etat"].astype(str).to_numpy("S"))
        for label, res, grid in (("Ruby", ruby, x_ruby), ("Borate", borate, x_borate)):
            g = f.create_group(label)
            for key in ("centroids", "positions", "widths", "spectra", "fit_positions",
                        "fit_widths", "fit_position_err", "fit_width_err"):
                if key in res:
                    g.create_dataset(key, data=res[key], compression="gzip")
            g.create_dataset("x", data=grid)
            g.attrs["n_failed"] = res["n_failed"]
        g = f.create_group("Glass")
        g.create_dataset("LF", data=glass_lf, compression="gzip")
        g.create_dataset("HF", data=glass_hf, compression="gzip")
        g.create_dataset("x_LF", data=RAMAN_AXIS)
        g.create_dataset("x_HF", data=HF_AXIS)
        for key in ['Glass_LF', 'Glass_HF', 'Bkg_LF']:
            g.create_dataset(key+'_source', data=sheets['Glass'][key].fillna('').astype(str).to_numpy('S'))
        g.create_dataset('HF_available', data=np.isfinite(glass_hf).any(axis=1))
        g.create_dataset('input_axis_unit', data=np.array([run.glass_axis_units.get(str(v), 'wavelength_nm')
                                                          for v in sheets['Glass']['Glass_LF']], dtype='S'))
    return run.markers_file


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*", help="run ids (default: all)")
    ap.add_argument("--no-fig", action="store_true", help="skip per-spectrum figures")
    args = ap.parse_args(argv)
    runs = [get_run(r) for r in args.runs] if args.runs else RUNS
    for run in runs:
        print(f"[extract_markers] {run.run_id}")
        print("   ->", extract_run(run, make_figures=not args.no_fig))


if __name__ == "__main__":
    main()
