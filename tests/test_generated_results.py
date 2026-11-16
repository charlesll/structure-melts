"""Cross-stage consistency of the generated publication files."""
import json
import unittest

import h5py
import numpy as np

from src.config import RUNS, RESULTS, FIGURES, RAMAN_AXIS
from src.figure_data import RUN_ORDER


class GeneratedResultsTests(unittest.TestCase):
    def test_run_registry_and_acquisition_alignment(self):
        self.assertEqual([run.run_id for run in RUNS], RUN_ORDER)
        with h5py.File(RESULTS/'dataset.h5') as data, h5py.File(RESULTS/'peak_fits.h5') as fits:
            np.testing.assert_array_equal(data['x_axis'][()], RAMAN_AXIS)
            self.assertEqual(len(data['flat/T_C']), 62)
            self.assertEqual(set(fits), set(RUN_ORDER))
            states = data['flat/Etat'].asstr()[()]
            self.assertEqual(np.count_nonzero(states == 'Melt'), 22)
            self.assertEqual(np.count_nonzero(np.isin(states, ['Verre', 'Melt'])), 52)
            for run in RUNS:
                d, f = data[run.run_id], fits[run.run_id]
                with h5py.File(run.markers_file) as markers:
                    np.testing.assert_array_equal(markers['T_C'][()], d['T_C'][()])
                    self.assertEqual(markers['Ruby'].attrs['n_failed'], 0)
                    self.assertEqual(markers['Borate'].attrs['n_failed'], 0)
                    for sensor, half in [('Ruby', .8), ('Borate', .35)]:
                        grid = markers[sensor+'/x'][()]
                        centres = markers[sensor+'/positions'][()]
                        finite = centres[np.isfinite(centres)]
                        self.assertTrue((finite-half >= grid[0]-1e-9).all())
                        self.assertTrue((finite+half <= grid[-1]+1e-9).all())
                    available = markers['Glass/HF_available'][()]
                    np.testing.assert_array_equal(np.isfinite(d['spectra_HF'][()]).any(axis=1), available)
                np.testing.assert_array_equal(f['temperature'][()], d['T_C'][()])
                np.testing.assert_array_equal(f['pressure'][()], d['borate_P_median'][()])
                self.assertTrue(np.isfinite(d['spectra'][()]).all())
                for sensor in ('ruby', 'borate'):
                    self.assertTrue(np.isfinite(d[sensor+'_P_median'][()]).all())
                    self.assertTrue(np.isfinite(d[sensor+'_P_std'][()]).all())

    def test_band_fit_exports_are_physical_and_covariances_consistent(self):
        with h5py.File(RESULTS/'peak_fits.h5') as fits:
            self.assertEqual(tuple(fits.attrs['peak_names'].astype(str)), ('TOH', 'Q2', 'T2S', 'Q3', 'Q4'))
            for group in fits.values():
                self.assertEqual(group.attrs['failed_fits'], 0)
                for field in ('areas', 'widths', 'position_sd', 'area_sd', 'area_covariance',
                              'parameters', 'parameter_covariance', 'area_fraction_covariance'):
                    self.assertTrue(np.isfinite(group[field][()]).all(), field)
                self.assertTrue((group['areas'][()] > 0).all())
                self.assertTrue((group['widths'][()] > 0).all())
                np.testing.assert_allclose(group['area_fractions'][()].sum(axis=1), 1., atol=1e-12)
                covariance = group['area_covariance'][()]
                np.testing.assert_allclose(covariance, covariance.transpose(0, 2, 1), atol=1e-12)
                np.testing.assert_allclose(np.diagonal(covariance, axis1=1, axis2=2), group['area_sd'][()]**2, atol=1e-12)
                self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-10)
                fractions = group['area_fraction_covariance'][()]
                np.testing.assert_allclose(fractions.sum(axis=1), 0., atol=1e-10)

    def test_JSON_exports_contain_no_nonstandard_nan_or_infinity(self):
        def invalid(value):
            raise ValueError('Nonfinite JSON number: '+value)
        paths = list(RESULTS.rglob('*.json')) + list((FIGURES/'manuscript').glob('*.json'))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.name):
                json.loads(path.read_text(), parse_constant=invalid)
