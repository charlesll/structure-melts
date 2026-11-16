"""Regression checks for Figure 10 observables and exploratory run weighting."""
import unittest

import h5py
import numpy as np

from src.figure_data import PEAK_FITS
from src.raman_observables import acquisition_table, lf_centroid, transition_observables, parallel_centroid_guides


class FigureTenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = acquisition_table()

    def test_centroid_boundary_missing_and_signed_intensity(self):
        axis = np.array([400., 600., 829., 830., 900.])
        spectra = np.array([[1., 2., -0.5, 100., 100.], [0., 0., 0., 1., 1.]])
        result = lf_centroid(axis, spectra)
        self.assertAlmostEqual(result[0], (400.+1200.-.5*829.)/2.5)
        self.assertTrue(np.isnan(result[1]))
        with self.assertRaises(ValueError):
            lf_centroid(axis, np.ones((2, 3)))

    def test_current_ambient_is_a_single_run01_acquisition(self):
        glass = [r for r in self.rows if r['state'] == 'Verre' and r['T_C'] == 25.]
        self.assertEqual(len(glass), 5)
        ambient = [r for r in glass if r['run_id'] == 'run01_20250220_25C' and r['step'] == 0]
        self.assertEqual(len(ambient), 1)
        self.assertEqual(ambient[0]['step'], 0)
        self.assertEqual(ambient[0]['run_id'], 'run01_20250220_25C')
        self.assertEqual(sum(r['state'] == 'Melt' for r in self.rows), 22)

    def test_brackets_use_exact_source_spectra_and_q4_t2s_covariance(self):
        brackets = transition_observables(self.rows)
        self.assertEqual(len(brackets), 5)
        for bracket in brackets:
            for side, endpoint in zip(('left', 'right'), bracket['endpoints']):
                self.assertEqual(endpoint['step'], bracket['step_'+side])
                self.assertEqual(endpoint['T_C'], bracket['T_'+side])
        row = next(r for r in self.rows if r['state'] == 'Melt')
        with h5py.File(PEAK_FITS) as fits:
            group = fits[row['run_id']]
            area = group['areas'][row['step']]
            covariance = group['area_covariance'][row['step']]
        jacobian = np.zeros(5)
        jacobian[4] = 1./area[2]
        jacobian[2] = -area[4]/area[2]**2
        self.assertAlmostEqual(row['Q4_T2S'], area[4]/area[2])
        self.assertAlmostEqual(row['Q4_T2S_sd']**2, jacobian @ covariance @ jacobian)
        inverse_jacobian = np.zeros(5)
        inverse_jacobian[2] = 1./area[4]
        inverse_jacobian[4] = -area[2]/area[4]**2
        self.assertAlmostEqual(row['T2S_Q4'], area[2]/area[4])
        self.assertAlmostEqual(row['T2S_Q4_sd']**2,
                               inverse_jacobian @ covariance @ inverse_jacobian)


    def test_t2s_area_and_fraction_remain_distinct_with_source_errors(self):
        with h5py.File(PEAK_FITS) as fits:
            for row in self.rows:
                group = fits[row['run_id']]
                areas = group['areas'][row['step']]
                covariance = group['area_covariance'][row['step']]
                self.assertAlmostEqual(row['Q4_area'], areas[4])
                self.assertAlmostEqual(row['Q4_area_sd']**2, covariance[4, 4])
                self.assertAlmostEqual(row['T2S_area'], areas[2])
                self.assertAlmostEqual(row['T2S_area_sd']**2, covariance[2, 2])
                total = areas.sum()
                self.assertAlmostEqual(row['T2S_fraction'], areas[2]/total)
                jacobian = np.full(5, -areas[2]/total**2)
                jacobian[2] += 1./total
                self.assertAlmostEqual(row['T2S_fraction_sd']**2,
                                       jacobian @ covariance @ jacobian)

    def test_equal_run_weights_ignore_replicating_one_run(self):
        melt = [r for r in self.rows if r['state'] == 'Melt']
        def fit(rows):
            p = np.array([r['P_GPa'] for r in rows])
            y = np.array([r['centroid_cm-1'] for r in rows])
            run = np.array([r['run_id'] for r in rows])
            result = parallel_centroid_guides(rows)
            return np.array([result['low_intercept_at_threshold_cm-1'],
                             result['common_slope_cm-1_per_GPa'],
                             result['high_offset_cm-1']])
        duplicate = [r for r in melt if r['run_id'] == 'run06_20250430']
        np.testing.assert_allclose(fit(melt), fit(melt+duplicate), atol=1e-10)

    def test_manuscript_parallel_guides_match_exploration_and_skip_gap(self):
        melt = [r for r in self.rows if r['state'] == 'Melt']
        fit = parallel_centroid_guides(melt)
        # Référence après correction des largeurs de fluorescence (9 novembre 2026).
        # La pente est directement exprimée en cm⁻¹/GPa.
        expected = [624.9931522958191, 1.821671903594928, 12.717401834136759]
        np.testing.assert_allclose([fit['low_intercept_at_threshold_cm-1'],
                                   fit['common_slope_cm-1_per_GPa'],
                                   fit['high_offset_cm-1']], expected, atol=1e-6, rtol=0)
        self.assertLess(fit['domains_GPa'][0][1], 6.)
        self.assertGreater(fit['domains_GPa'][1][0], 6.)
        duplicate = [r for r in melt if r['run_id'] == 'run06_20250430']
        repeated = parallel_centroid_guides(melt+duplicate)
        self.assertAlmostEqual(fit['common_slope_cm-1_per_GPa'],
                               repeated['common_slope_cm-1_per_GPa'])

    def test_gap_does_not_identify_step_pressure_or_discontinuity(self):
        p = np.array([r['P_GPa'] for r in self.rows if r['state'] == 'Melt'])
        np.testing.assert_array_equal(p >= 5.6, p >= 7.3)
        lower, upper = p[p < 6].max(), p[p >= 6].min()
        continuous_ramp = np.clip((p-lower)/(upper-lower), 0., 1.)
        np.testing.assert_array_equal(continuous_ramp, (p >= 6).astype(float))
