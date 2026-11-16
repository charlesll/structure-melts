"""Ensure display-only crystals cannot change fit-channel standardization."""
import unittest

import numpy as np

from src.figures.fig6_ruptures import fitted_y_limits, standardize_for_display
from src.tg_spectra_ruptures import prepare_channels


class RupturesDisplayTests(unittest.TestCase):
    def test_excluded_crystals_do_not_change_training_standardization(self):
        training = np.array([[1., 2., 5.], [2., 4., 5.], [4., 3., 5.]])
        all_values = np.vstack([training, [1e8, -1e8, 6.]])
        fitted = np.array([True, True, True, False])
        transformed = standardize_for_display(all_values, fitted)
        np.testing.assert_allclose(transformed[:3], prepare_channels(training, True))
        self.assertTrue(np.isfinite(transformed).all())

    def test_crystals_use_the_same_units_as_the_fitted_points(self):
        values = np.array([[0.], [2.], [4.]])
        transformed = standardize_for_display(values, np.array([True, True, False]))
        np.testing.assert_allclose(transformed[:, 0], [-1., 1., 3.])

    def test_extreme_crystals_do_not_expand_vertical_display_range(self):
        values = np.array([[-2., 1.], [0., 3.], [1e8, -1e8]])
        fitted = np.array([True, True, False])
        limits = fitted_y_limits(values, fitted)
        changed = values.copy()
        changed[-1] *= 100
        self.assertEqual(fitted_y_limits(changed, fitted), limits)
        self.assertLess(limits[0], -2.)
        self.assertGreater(limits[1], 3.)
        self.assertGreater(limits[0], -3.)
        self.assertLess(limits[1], 4.)
        lower, upper = fitted_y_limits(np.zeros((3, 2)), fitted)
        self.assertLess(lower, 0.)
        self.assertGreater(upper, 0.)


class RupturesTemperatureTests(unittest.TestCase):
    def test_default_excludes_points_below_150_and_keeps_boundary(self):
        from pathlib import Path
        from unittest.mock import patch
        from src import tg_spectra_ruptures as tr
        from src import tg_spectra_ruptures as ts
        T = np.array([25., 120., 149., 150., 196., 220., 240., 260., 280., 300.])
        d = {'T_C': T, 'Etat': np.array(['Verre']*len(T)),
             'spectra': np.ones((len(T), 2)), 'is_tg_series': True}
        families = {name: (np.ones((len(T), 1)), True) for name in tr.FAMILY_NAMES[1:]}
        with patch.object(ts, 'sensor_channels', return_value={}), \
             patch.object(ts, '_sensor_pcas', return_value={}), \
             patch.object(ts, 'prepare_glass', return_value=d['spectra']), \
             patch.object(ts, 'fit_glass_pca', return_value=(np.ones((len(T), 4)), None)), \
             patch.object(ts, 'build_sensor_matrix', return_value=np.ones((len(T), 9))), \
             patch.object(ts, 'ratio_test_families', return_value=families):
            result = tr._run_families({'run': d}, np.arange(2), Path('.'))
            unrestricted = tr._run_families({'run': d}, np.arange(2), Path('.'), n_min=-np.inf)
        np.testing.assert_array_equal(result['run'][0], T[T >= 150.])
        np.testing.assert_array_equal(unrestricted['run'][0], T)
        self.assertEqual(tr.T_MIN, 150.)


class PcaReproducibilityTests(unittest.TestCase):
    def test_randomized_raman_scores_repeat(self):
        from src.tg_spectra_ruptures import fit_glass_pca
        rng = np.random.default_rng(17)
        # Cette dimension force le choix automatique du solveur aléatoire.
        data = {'run': rng.normal(size=(62, 880))}
        first, pca = fit_glass_pca(data, 'global', 'run')
        second, _ = fit_glass_pca(data, 'global', 'run')
        self.assertEqual(pca.random_state, 0)
        np.testing.assert_array_equal(first, second)

    def test_sensor_pca_uses_the_same_fixed_seed(self):
        from src.tg_spectra_ruptures import _sensor_pcas
        rng = np.random.default_rng(18)
        aligned = {'run': {name: rng.normal(size=(12, 3, 121))
                           for name in ('Ruby', 'Borate')}}
        first, second = _sensor_pcas(aligned), _sensor_pcas(aligned)
        for name in first:
            self.assertEqual(first[name].random_state, 0)
            np.testing.assert_array_equal(first[name].components_, second[name].components_)
