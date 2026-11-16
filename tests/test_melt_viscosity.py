import unittest
import numpy as np
from src.ann_viscosity import myega, myega_fragility_interval, solve_viscous_tg


class MyegaTests(unittest.TestCase):
    def test_transition_viscosity_and_fragility_definition(self):
        Tg = 550.
        self.assertAlmostEqual(float(myega(Tg, Tg)), 12.)
        h = 1e-5
        slope = (myega(Tg/(1+h), Tg)-myega(Tg/(1-h), Tg))/(2*h)
        self.assertAlmostEqual(float(slope), 34.5, places=7)

    def test_high_temperature_limit_and_bracket_order(self):
        self.assertAlmostEqual(float(myega(1e15, 550.)), -1.48, places=10)
        values = myega(1773.15, np.array([529.15, 535.15]))
        self.assertGreater(values[1], values[0])
        with self.assertRaises(ValueError):
            myega(0., 550.)

    def test_fragility_interval_transforms_gaussian_quantiles(self):
        from scipy.stats import norm
        temperature = np.array([500., 1000., 1773.15])
        center, lower, upper = myega_fragility_interval(temperature, 500.)
        low_m = myega(temperature, 500., fragility=34.5-norm.ppf(.975)*1.2)
        high_m = myega(temperature, 500., fragility=34.5+norm.ppf(.975)*1.2)
        np.testing.assert_allclose(center, myega(temperature, 500.))
        np.testing.assert_allclose(lower, np.minimum(low_m, high_m))
        np.testing.assert_allclose(upper, np.maximum(low_m, high_m))
        self.assertEqual(lower[0], 12.)
        self.assertEqual(upper[0], 12.)
        self.assertTrue(np.all(lower <= center))
        self.assertTrue(np.all(center <= upper))



class ViscousTgTests(unittest.TestCase):
    def test_vector_inversion_and_missing_root(self):
        class Predictor:
            def predict(self, temperature, pressure):
                temperature, pressure = np.broadcast_arrays(temperature, pressure)
                # Exactly known roots: Tg = 500 + 10 P kelvin.
                mean = 12. + (500. + 10.*pressure - temperature)/100.
                return mean.ravel(), np.full(mean.size, .5)

        pressure = np.array([0., 5., 13.])
        tg, mean, sd = solve_viscous_tg(Predictor(), pressure)
        np.testing.assert_allclose(tg, 500.+10.*pressure, atol=.002)
        np.testing.assert_allclose(mean, 12., atol=.00002)
        np.testing.assert_allclose(sd, .5)
        with self.assertRaises(ValueError):
            solve_viscous_tg(Predictor(), np.array([200.]))
        with self.assertRaises(ValueError):
            solve_viscous_tg(Predictor(), np.array([np.nan]))

    def test_nonpositive_root_tolerance_is_rejected(self):
        for tolerance in (0., -1., np.nan, np.inf):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                solve_viscous_tg(None, np.array([0.]), tolerance_k=tolerance)

    def test_calculation_creates_missing_result_directory(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch
        import pandas as pd
        from src.ann_viscosity import calculate_gpvisc
        # Isoler l'écriture : ce substitut n'est pas un modèle physique.
        predictor = SimpleNamespace(
            composition=pd.DataFrame({'sio2': [1.]}),
            gpvisc=SimpleNamespace(list_oxides=lambda: ['sio2']),
            predict=lambda t, p: (np.zeros_like(p), np.ones_like(p)))
        root_result = (np.full(53, 500.), np.full(53, 12.), np.ones(53))
        with TemporaryDirectory() as directory:
            output = Path(directory)/'new_results'/'ann_gpvisc.json'
            with patch('src.ann_viscosity.GPVISC_RESULT', output), \
                 patch('src.ann_viscosity.ANNViscosityPredictor', return_value=predictor), \
                 patch('src.ann_viscosity.solve_viscous_tg', return_value=root_result):
                calculate_gpvisc()
            self.assertTrue(output.is_file())

    def test_saved_gp_inversion(self):
        import json
        from pathlib import Path
        from src.ann_viscosity import GPVISC_RESULT
        from src.config import ANN_COMPOSITION_WT
        result = json.loads(GPVISC_RESULT.read_text())
        self.assertEqual(result['composition_wt_supplied'], ANN_COMPOSITION_WT)
        self.assertAlmostEqual(sum(result['composition_wt_normalized'].values()), 100.)
        self.assertAlmostEqual(sum(result['composition_mole_fraction'].values()), 1.)
        self.assertEqual(result['P_GPa'][0], 0.)
        self.assertEqual(result['P_GPa'][-1], 13.)
        np.testing.assert_allclose(result['log10_eta_at_Tg'], 12., atol=.002)
        np.testing.assert_allclose(result['delta_Tg_K'], np.array(result['Tg_K'])-result['Tg_K'][0])

        supplementary = json.loads(Path('figures/manuscript/Supplementary_Figure_11.json').read_text())
        self.assertEqual(supplementary['MYEGA_Ae'], -1.48)
        self.assertEqual(supplementary['MYEGA_fragility'], 34.5)
        self.assertEqual(supplementary['MYEGA_fragility_sd'], 1.2)
        self.assertEqual(supplementary['temperatures_C'], [1400., 1500., 1600.])
        # Compare to the stored diagnostic without relying on rounded labels.
        fit = supplementary['linear_Tg_fit']
        diagnostic = json.loads(Path('figures/manuscript/Supplementary_Figure_6.json').read_text())['fit']
        self.assertAlmostEqual(fit['intercept_C'], diagnostic['intercept_C'], places=5)
        self.assertAlmostEqual(fit['slope_C_per_GPa'], diagnostic['slope_C_per_GPa'], places=5)
        for row in supplementary['values']:
            expected_tg = fit['intercept_C'] + fit['slope_C_per_GPa']*row['P_GPa'] + 273.15
            self.assertAlmostEqual(row['spectral_linear_Tg_K'], expected_tg)
            self.assertAlmostEqual(row['MYEGA_log10_eta'], float(myega(row['T_C']+273.15, expected_tg)))
