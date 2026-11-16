"""Largeurs mesurées et exclusion des ajustements de raies non convergés."""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.extract_markers import _fit_sensor
from src.functions import estimate_peak_parameters, estimate_fwhm


class MarkerFittingTests(unittest.TestCase):
    def test_gaussian_width_is_independent_of_detection_threshold(self):
        x = np.linspace(684., 690., 6001)
        y = np.exp(-.5*((x-687.)/.1)**2)
        expected = 2*np.sqrt(2*np.log(2))*.1
        for threshold in (.5, .7, .8, .9):
            result = estimate_peak_parameters(
                x, y.copy(), cut_lb=684., cut_hb=690., smooth=False,
                find_peaks_width=4, find_peaks_rel_height=threshold, figure=False)
            self.assertAlmostEqual(result[1], 687.)
            self.assertAlmostEqual(result[2], expected, delta=3e-6)

    def test_interpolated_width_uses_axis_units(self):
        x = np.array([0., .5, 1., 2., 3., 5.])
        y = np.array([0., 1., 4., 8., 3., 0.])
        self.assertAlmostEqual(estimate_fwhm(x, y), 1.8)
        with self.assertRaises(ValueError):
            estimate_fwhm(x, np.ones_like(x))

    def test_width_belongs_to_first_detected_peak(self):
        x = np.linspace(684., 690., 6001)
        y = np.exp(-.5*((x-686.)/.1)**2) + 2*np.exp(-.5*((x-688.)/.2)**2)
        result = estimate_peak_parameters(
            x, y, cut_lb=684., cut_hb=690., smooth=False,
            find_peaks_height=.4, find_peaks_width=4, figure=False)
        self.assertAlmostEqual(result[1], 686.)
        self.assertAlmostEqual(result[2], 2*np.sqrt(2*np.log(2))*.1, delta=3e-6)

    def test_unsuccessful_fit_is_counted_and_not_used_for_pressure(self):
        from src.pressure import grain_pressures
        x = np.linspace(685., 686., 101)
        sheet = pd.DataFrame(dict(T_C=[25.], Pmem=[1.], grain1=['test']))
        failed = dict(position=685.5, fwhm=.15, position_err=.01,
                      fwhm_err=.01, success=False)
        with patch('src.extract_markers._read_asc', return_value=np.column_stack([x, np.ones_like(x)])), \
             patch('src.extract_markers.estimate_peak_parameters', return_value=(685.5, 685.5, .15, x, np.ones_like(x))), \
             patch('src.extract_markers.fit_line', return_value=failed):
            with self.assertWarnsRegex(UserWarning, 'did not converge'):
                result = _fit_sensor(sheet, Path('.'), x, {}, None, fit_lines=True)
        self.assertEqual(result['n_failed'], 1)
        for key in ('fit_positions', 'fit_widths', 'fit_position_err', 'fit_width_err'):
            self.assertTrue(np.isnan(result[key]).all())
        pressure, _ = grain_pressures(result['fit_positions'], [25.], 'borate', 'wei2024')
        self.assertTrue(np.isnan(pressure).all())

    def test_pseudovoigt_recovers_known_center_and_width(self):
        from src.line_fit import fit_line, pseudo_voigt
        x = np.linspace(685., 686., 1001)
        y = pseudo_voigt(x, 2., 685.51, .12, .3) + .1 + .05*(x-685.5)
        fit = fit_line(x, y, 685.5, .15)
        self.assertTrue(fit['success'])
        self.assertAlmostEqual(fit['position'], 685.51, places=6)
        self.assertAlmostEqual(fit['fwhm'], .12, places=6)
