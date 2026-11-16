"""Numerical checks of bracket integration and descriptive extrapolation."""
import unittest
import numpy as np
from src.tg_pressure_regression import fit_interval_line, interval_log_probability, centered_line_envelope, absolute_line_envelope, nominal_reference_envelope


def synthetic_rows(width=2.):
    return [dict(run_id=str(i), P=float(i), P_left=float(i), P_right=float(i),
                 P_std_left=0., P_std_right=0., sigma_T=1.,
                 T_left=100.+3*i-width, T_right=100.+3*i+width) for i in range(1, 7)]


class IntervalRegressionTests(unittest.TestCase):
    def test_known_linear_intercept_and_slope(self):
        result = fit_interval_line(synthetic_rows(), profile=False)
        self.assertAlmostEqual(result['intercept_C'], 100., places=4)
        self.assertAlmostEqual(result['slope_C_per_GPa'], 3., places=4)

    def test_tail_probabilities_remain_finite_and_symmetric(self):
        np.testing.assert_allclose(interval_log_probability([20., -21.], [21., -20.]),
                                   [interval_log_probability(20., 21.)]*2)
        self.assertTrue(np.isfinite(interval_log_probability(20., 21.)))

    def test_missing_uncertainty_is_not_silently_imputed(self):
        for key in ['P_std_left', 'P_std_right']:
            rows = synthetic_rows()
            rows[0][key] = np.nan
            with self.assertRaises(ValueError):
                fit_interval_line(rows, profile=False)

    def test_centered_fit_cancels_intercept_exactly(self):
        # Deliberately broad intercept support cannot create width at zero.
        fit = dict(intercept_C=240., intercept_profile95_C=[200., 280.],
                   slope_C_per_GPa=7., slope_profile95_C_per_GPa=[3., 11.])
        mean, lo, hi = centered_line_envelope([0., 1., 13.], fit)
        np.testing.assert_array_equal([mean[0], lo[0], hi[0]], [0., 0., 0.])
        np.testing.assert_allclose([mean[-1], lo[-1], hi[-1]], [91., 39., 143.])

    def test_slope_profile_recovers_known_linear_response(self):
        result = fit_interval_line(synthetic_rows(), profile=False, slope_profile=True)
        lo, hi = result['slope_profile95_C_per_GPa']
        self.assertLess(lo, 3.)
        self.assertGreater(hi, 3.)
        self.assertAlmostEqual(result['intercept_C'], 100., places=4)

    def test_absolute_profile_at_zero_equals_intercept_profile(self):
        rows = synthetic_rows()
        fit = fit_interval_line(rows)
        mean, lo, hi = absolute_line_envelope([0., 3.], rows, fit)
        np.testing.assert_allclose([lo[0], hi[0]], fit['intercept_profile95_C'], atol=1e-5)
        np.testing.assert_allclose(mean, [100., 109.], atol=1e-4)
        self.assertTrue((lo < mean).all() and (mean < hi).all())
        self.assertLess(hi[1]-lo[1], hi[0]-lo[0])

    def test_nominal_reference_preserves_finite_intercept_support_at_zero(self):
        rows = synthetic_rows()
        fit = fit_interval_line(rows)
        ref = fit['intercept_C']
        mean, lo, hi = nominal_reference_envelope([0.], rows, fit, ref)
        self.assertAlmostEqual(mean[0], 0., places=8)
        self.assertLess(lo[0], 0.)
        self.assertGreater(hi[0], 0.)
        np.testing.assert_allclose([lo[0]+ref, hi[0]+ref],
                                   fit['intercept_profile95_C'], atol=1e-5)

    def test_wider_brackets_widen_intercept_support(self):
        narrow = fit_interval_line(synthetic_rows())['intercept_profile95_C']
        broad = fit_interval_line(synthetic_rows(15.))['intercept_profile95_C']
        self.assertGreater(broad[1]-broad[0], narrow[1]-narrow[0])


if __name__ == '__main__':
    unittest.main()
