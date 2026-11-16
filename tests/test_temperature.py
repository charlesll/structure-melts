"""Solveur et statistiques nécessaires à la figure 2d."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from src.config import LBD0_RUBY_NM, LBD0_BORATE_NM
from src.temperature import (STEPS_CSV, SUMMARY_JSON, _row_median,
                             step_offsets, summary, write_outputs)


class TemperatureSummaryTests(unittest.TestCase):
    def test_empirical_quantiles_exclude_missing_offsets(self):
        result = summary([{"dT_step_K": v} for v in (0., 10., 20., np.nan, np.inf)])
        self.assertEqual(result["n_steps"], 3)
        self.assertEqual(result["step_mean_K"], 10.)
        self.assertEqual(result["step_median_K"], 10.)
        self.assertAlmostEqual(result["step_q16_K"], 3.1731050786291415)
        self.assertAlmostEqual(result["step_q84_K"], 16.82689492137086)

    def test_empty_and_single_offset(self):
        for steps in ([], [{"dT_step_K": np.nan}]):
            result = summary(steps)
            self.assertEqual(result["n_steps"], 0)
            for name in ("mean", "median", "q16", "q84"):
                self.assertTrue(np.isnan(result[f"step_{name}_K"]))
        result = summary([{"dT_step_K": 5.}])
        for name in ("mean", "median", "q16", "q84"):
            self.assertEqual(result[f"step_{name}_K"], 5.)

    def test_outputs_preserve_missing_steps_and_write_valid_json(self):
        steps = [{"dT_step_K": np.nan}]
        with tempfile.TemporaryDirectory() as directory:
            path = write_outputs(steps, summary(steps), directory)
            result = json.loads(path.read_text())
            self.assertIsNone(result['step_mean_K'])
            self.assertEqual(result['n_steps'], 0)
            self.assertAlmostEqual(result['central_fraction'], .6826894921370859)
            with (Path(directory) / STEPS_CSV.name).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertTrue(np.isnan(float(rows[0]['dT_step_K'])))
            with self.assertRaises(ValueError):
                write_outputs([], summary([]), directory)

    def test_published_population_and_statistics(self):
        # Référence régénérée après correction de la FWHM des raies (9 novembre 2026).
        with STEPS_CSV.open() as handle:
            steps = [{"dT_step_K": float(row['dT_step_K'])} for row in csv.DictReader(handle)]
        result = summary(steps)
        self.assertEqual(len(steps), 62)
        self.assertEqual(result['n_steps'], 62)
        # La chaîne complète inclut des ajustements numériques des raies.
        # Leurs derniers chiffres peuvent varier selon les bibliothèques et
        # plateformes. Tolérance absolue de 0,00001 K sur les quatre statistiques,
        # très inférieure à la précision expérimentale et aux arrondis publiés.
        # Les tests analytiques du solveur et des quantiles restent plus stricts.
        np.testing.assert_allclose(
            [result['step_mean_K'], result['step_median_K'],
             result['step_q16_K'], result['step_q84_K']],
            [19.429255294420862, 17.52900326520816, -8.64362507635476, 46.232225581458046],
            atol=1e-5, rtol=0,
            err_msg="Écart supérieur à 0,00001 K : vérifier entrées, calibrations et ajustements",
        )
        saved = json.loads(SUMMARY_JSON.read_text())
        for key, value in result.items():
            self.assertAlmostEqual(saved[key], value)


class TemperatureSolverTests(unittest.TestCase):
    def test_known_reference_at_calibration_boundary(self):
        offsets, pressure = step_offsets([LBD0_RUBY_NM], [LBD0_BORATE_NM], [25.])
        # Les références sont adoptées à 296 K, et non à 298,15 K.
        np.testing.assert_allclose(offsets, [-2.15], atol=1e-8, rtol=0)
        np.testing.assert_allclose(pressure, [0.], atol=1e-10, rtol=0)

    def test_missing_measurements_and_unreachable_temperature(self):
        offsets, pressure = step_offsets([np.nan, -1., 694.3], [685.4]*3, [25., 25., -500.])
        self.assertTrue(np.isnan(offsets).all())
        self.assertTrue(np.isnan(pressure).all())
        with self.assertRaises(ValueError):
            step_offsets([694.3], [685.4, 685.5], [25.])
        with self.assertRaises(ValueError):
            step_offsets([694.3], [685.4], [25.], window=0)

    def test_nearest_root_and_no_root(self):
        def pressures(position, temperature, sensor, scale):
            # Deux racines : ΔT = −20 K et +50 K, pour T nominal = 500 K.
            return (temperature - 480.) * (temperature - 550.) if sensor == 'ruby' else 0.
        with patch('src.temperature._P', side_effect=pressures):
            offsets, _ = step_offsets([694.3], [685.4], [226.85])
        np.testing.assert_allclose(offsets, [-20.], atol=1e-8)
        with patch('src.temperature._P', side_effect=lambda p, t, sensor, scale: float(sensor == 'ruby')):
            offsets, _ = step_offsets([694.3], [685.4], [226.85])
        self.assertTrue(np.isnan(offsets[0]))

    def test_grain_medians_ignore_invalid_values_without_inventing_grains(self):
        result = _row_median([[694., np.nan, 696.], [np.nan, 0., -1.]])
        self.assertEqual(result[0], 695.)
        self.assertTrue(np.isnan(result[1]))
