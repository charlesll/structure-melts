"""Offsets équivalents de température pour l’histogramme de la figure 2d.

Pour chaque acquisition, chercher ΔT tel que les pressions rubis et borate
coïncident à T_thermocouple + ΔT. Les positions mesurées sont fixes.
Ce diagnostic dépend des références et des équations des deux jauges :
il ne mesure pas indépendamment la température et ne corrige pas les données.

Sorties : temperature_steps.csv et temperature_summary.json dans results/.
L’histogramme est tracé uniquement par src.figures.fig2_pt.
"""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from .config import (BORATE_SCALE, LBD0_BORATE_NM, LBD0_RUBY_NM,
                     RESULTS, RUBY_SCALE, RUNS)
from .pressure_sensors import pressure_sensor

T_MIN_K, T_MAX_K = 296.0, 850.0  # Domaine thermique des calibrations Wei2024.
WINDOW_K = 150.0  # Recherche de ΔT dans ±150 K, limitée au domaine ci-dessus.
STEPS_CSV = RESULTS / "temperature_steps.csv"
SUMMARY_JSON = RESULTS / "temperature_summary.json"

def _P(pos, T_K, sensor, scale):
    """Pression en GPa pour une position en nm et une température en K.

    Une position manquante ou une température hors calibration donne NaN."""
    if not np.isfinite(pos) or not np.isfinite(T_K) or not (T_MIN_K <= T_K <= T_MAX_K):
        return np.nan
    # Références mesurées à 23 °C adoptées directement à 296 K.
    kw = (dict(ruby_scale=scale, lbd_0_ruby=LBD0_RUBY_NM) if sensor == "ruby"
          else dict(borate_scale=scale, lbd_0_borate=LBD0_BORATE_NM))
    return float(pressure_sensor(pos, T_K, scale=sensor, **kw))

def step_offsets(ruby_nm, borate_nm, T_C, window=WINDOW_K, n=601,
                 ruby_scale=RUBY_SCALE, borate_scale=BORATE_SCALE):
    """Calcule ΔT (K) annulant P_rubis − P_borate pour chaque acquisition.

    Entrées : trois vecteurs de même taille, positions médianes des raies
    en nm et températures nominales en °C. Recherche sur une grille de n
    points, puis résolution de chaque changement de signe par Brent.
    Si plusieurs racines existent, conserve celle de plus faible |ΔT|.
    Sans racine ou sans mesure valide : NaN. Retour : (offsets, pressions
    communes en GPa), deux vecteurs de même taille que les entrées."""
    r, b, T_C = (np.asarray(x, float) for x in (ruby_nm, borate_nm, T_C))
    if T_C.ndim != 1 or r.shape != T_C.shape or b.shape != T_C.shape:
        raise ValueError("Positions médianes et températures : trois vecteurs de même taille requis")
    if not np.isfinite(window) or window <= 0 or n < 3:
        raise ValueError("La fenêtre doit être finie et positive, et n doit être au moins égal à 3")
    T_K = T_C + 273.15
    out, p_out = np.full(T_K.shape, np.nan), np.full(T_K.shape, np.nan)
    for i in range(len(T_K)):
        if not (np.isfinite(r[i]) and np.isfinite(b[i]) and np.isfinite(T_K[i])
                and r[i] > 0 and b[i] > 0):
            continue
        lo, hi = max(-window, T_MIN_K-T_K[i]), min(window, T_MAX_K-T_K[i])
        if lo >= hi:
            continue
        def f(d):
            return (_P(r[i], T_K[i]+d, "ruby", ruby_scale)
                    - _P(b[i], T_K[i]+d, "borate", borate_scale))
        # Le balayage repère les changements de signe avant la résolution fine.
        grid = np.linspace(lo, hi, n)
        values = np.array([f(d) for d in grid])
        roots = list(grid[values == 0])
        for j in range(n-1):
            if np.isfinite(values[j:j+2]).all() and values[j]*values[j+1] < 0:
                roots.append(brentq(f, grid[j], grid[j+1]))
        if roots:
            out[i] = min(roots, key=abs)
            p_out[i] = _P(r[i], T_K[i]+out[i], "ruby", ruby_scale)
    return out, p_out

def _row_median(values):
    """Médiane par acquisition sur les grains finis et strictement positifs.

    Entrée (acquisitions, grains) ; sans grain valide, la médiane reste NaN.
    """
    values = np.asarray(values, float)
    values = np.where(np.isfinite(values) & (values > 0), values, np.nan)
    out = np.full(values.shape[0], np.nan)
    valid = np.isfinite(values).any(axis=1)
    out[valid] = np.nanmedian(values[valid], axis=1)
    return out

def analyse_runs(runs=RUNS):
    """Lit les positions des jauges et retourne une ligne par acquisition.

    Les positions ajustées sont préférées lorsqu’elles sont disponibles.
    Tous les états et toutes les séries sont conservés, comme dans la figure
    publiée. Chaque acquisition reçoit le même poids, quel que soit son run.
    """
    steps = []
    for run in runs:
        with h5py.File(run.markers_file, "r") as handle:
            temperatures = handle["T_C"][()]
            positions = {}
            for sensor in ("Ruby", "Borate"):
                group = handle[sensor]
                key = "fit_positions" if "fit_positions" in group else "positions"
                positions[sensor] = _row_median(group[key][()])
        offsets, _ = step_offsets(positions["Ruby"], positions["Borate"], temperatures)
        for index, temperature in enumerate(temperatures):
            steps.append(dict(run=run.run_id, step=index, T_C=float(temperature),
                              ruby_nm=float(positions["Ruby"][index]),
                              borate_nm=float(positions["Borate"][index]),
                              dT_step_K=float(offsets[index])))
    return steps


def summary(steps):
    """Statistiques descriptives des offsets finis, exprimées en K.

    Les quantiles Φ(−1) et Φ(+1), soit 15,8655 % et 84,1345 %, délimitent
    les 68,27 % centraux de la distribution empirique. Aucune hypothèse de
    normalité des offsets n’est nécessaire. Ce n’est pas un intervalle de
    confiance sur la moyenne ou la médiane. Les acquisitions non résolues
    sont exclues des statistiques mais restent présentes dans le CSV.
    """
    values = np.array([row["dT_step_K"] for row in steps], dtype=float)
    values = values[np.isfinite(values)]
    low, high = np.quantile(values, ndtr([-1., 1.])) if values.size else (np.nan, np.nan)
    return dict(n_steps=int(values.size),
                step_mean_K=float(values.mean()) if values.size else np.nan,
                step_median_K=float(np.median(values)) if values.size else np.nan,
                step_q16_K=float(low), step_q84_K=float(high))


def write_outputs(steps, summ, directory=RESULTS):
    """Écrit le CSV de la figure et le résumé numérique lisible par l’humain.

    Les statistiques non définies sont enregistrées comme null dans le JSON.
    """
    if not steps:
        raise ValueError("Aucune acquisition à enregistrer")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / STEPS_CSV.name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(steps[0]))
        writer.writeheader()
        writer.writerows(steps)
    result = {key: (value if np.isfinite(value) else None) for key, value in summ.items()}
    result.update(central_fraction=float(ndtr(1.) - ndtr(-1.)),
                  interpretation="Dispersion empirique des offsets, pas un intervalle de confiance.",
                  ruby_scale=RUBY_SCALE, borate_scale=BORATE_SCALE)
    path = directory / SUMMARY_JSON.name
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return path


def main(argv=None):
    """Recalcule les offsets et affiche les statistiques du manuscrit."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    steps = analyse_runs()
    summ = summary(steps)
    path = write_outputs(steps, summ)
    print(f"[temperature] {summ['n_steps']} offsets finis ; "
          f"moyenne {summ['step_mean_K']:.6f} K ; "
          f"médiane {summ['step_median_K']:.6f} K")
    print(f"  Intervalle empirique central à 68,27 % : "
          f"[{summ['step_q16_K']:.6f}, {summ['step_q84_K']:.6f}] K")
    print(f"  Résumé : {path}")


if __name__ == "__main__":
    main()
