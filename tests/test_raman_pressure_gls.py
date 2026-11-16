"""Tests indépendants du GLS utilisé par la figure 8."""
import unittest
import numpy as np
from scipy.stats import chi2
from src.raman_treatment import compare_pressure_gls


class PressureGLSTests(unittest.TestCase):
    def test_diagonal_covariance_matches_analytic_slope_and_plateaus(self):
        p = np.array([3., 4., 5., 8., 9., 12.])
        y = np.array([.5, .54, .53, .48, .49, .47])
        sigma = .03
        result = compare_pressure_gls(p, y, np.eye(len(p))*sigma**2)
        x = p-p.mean()
        slope = np.dot(x, y-y.mean())/np.dot(x, x)
        slope_error = sigma/np.sqrt(np.dot(x, x))
        self.assertAlmostEqual(result['linear']['effect'], slope)
        self.assertAlmostEqual(result['linear']['se'], slope_error)
        contrast = y[p >= 7].mean()-y[p < 7].mean()
        contrast_error = sigma*np.sqrt(1/np.sum(p >= 7)+1/np.sum(p < 7))
        self.assertAlmostEqual(result['fixed_7GPa']['effect'], contrast)
        self.assertAlmostEqual(result['fixed_7GPa']['se'], contrast_error)
        for name, value, error in [('linear', slope, slope_error),
                                    ('fixed_7GPa', contrast, contrast_error)]:
            self.assertAlmostEqual(result[name]['delta_chi2'], (value/error)**2)
            self.assertAlmostEqual(result[name]['p'], chi2.sf((value/error)**2, 1))
            np.testing.assert_allclose(result[name]['CI95'], [value-1.96*error, value+1.96*error])

    def test_shared_additive_error_does_not_inflate_slope_or_contrast(self):
        p = np.array([3., 4., 5., 8., 9., 12.])
        y = np.array([.5, .54, .53, .48, .49, .47])
        covariance = np.diag(np.array([.02, .03, .04, .02, .05, .03])**2)
        independent = compare_pressure_gls(p, y, covariance)
        shared = compare_pressure_gls(p, y, covariance+.02**2*np.ones((6, 6)))
        for name in ('linear', 'fixed_7GPa'):
            for key in ('effect', 'se', 'p'):
                self.assertAlmostEqual(independent[name][key], shared[name][key])

    def test_known_linear_relation(self):
        p = np.array([3., 4., 5., 8., 9., 12.])
        result = compare_pressure_gls(p, 1+.2*p, np.eye(6)*.01)
        self.assertAlmostEqual(result['linear']['effect'], .2)
        self.assertAlmostEqual(result['linear']['chi2'], 0.)
        self.assertEqual(result['linear']['df'], 4)
        self.assertEqual(result['df_constant'], 5)

    def test_constant_has_no_significant_effect(self):
        result = compare_pressure_gls(np.array([3., 4., 5., 8., 9., 12.]), np.ones(6), np.eye(6))
        for name in ('linear', 'fixed_7GPa'):
            self.assertAlmostEqual(result[name]['effect'], 0.)
            self.assertAlmostEqual(result[name]['p'], 1., places=6)

    def test_invalid_inputs_and_unidentifiable_models(self):
        p = np.array([3., 4., 5., 8., 9., 12.])
        y = np.arange(6.)
        for pressure, values, covariance in [
            (p, y[:-1], np.eye(6)), (p, y, np.eye(5)),
            (p, np.full(6, np.nan), np.eye(6)),
            (p, y, np.eye(6)+np.triu(np.ones((6, 6)), 1)),
            (np.ones(6), y, np.eye(6)),  # Pente non identifiable.
            (p+10, y, np.eye(6)),  # Un seul plateau échantillonné.
        ]:
            with self.assertRaises(ValueError):
                compare_pressure_gls(pressure, values, covariance)
        with self.assertRaises(np.linalg.LinAlgError):
            compare_pressure_gls(p, y, np.zeros((6, 6)))
