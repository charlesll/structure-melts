import unittest

import numpy as np

from src.peak_fitting import PEAK_PRIORS, _component, fit_spectrum


class PeakFittingTests(unittest.TestCase):
    def test_recovers_centers_from_synthetic_five_peak_spectrum(self):
        x = np.arange(820.0, 1271.0)
        from unittest.mock import patch
        from src.peak_fitting import _model, _prior_vectors
        parameters,_ = _prior_vectors()
        # Isoler le solveur : le fond d'une somme de pics possède des queues
        # non nulles dans les ROI et ne doit pas être confondu avec la vérité.
        truth=parameters.copy()
        truth[[1,4,7,10,14]] += [1.,-1.,1.,-.5,.3]
        y=_model(x,truth)
        with patch('src.peak_fitting._prepare_spectrum', return_value=(x,y)):
            result=fit_spectrum(x,y)

        np.testing.assert_allclose(
            result["positions"],
            truth[[1,4,7,10,14]],
            atol=0.1,
        )
        self.assertTrue(np.all(result["areas"] > 0))
        self.assertTrue(np.all(result["position_sd"] >= 0))
        self.assertTrue(np.all(result["area_sd"] >= 0))



class GaussianPriorFitTests(unittest.TestCase):
    def test_objective_and_posterior_area_covariance(self):
        import gvar as gv
        from src.peak_fitting import _model, _prior_vectors, _peak_quantities
        from src.peak_uncertainty import propagate_area_ratio
        x=np.arange(820.,1271.)
        initial,prior_sd=_prior_vectors()
        rng=np.random.default_rng(12)
        y=_model(x,initial)+rng.normal(0,.001,len(x))
        result=fit_spectrum(x,y)
        p=result['parameters']
        residual=(result['fitted']-result['observed'])/result['noise']
        penalty=(p-initial)/prior_sd
        self.assertAlmostEqual(result['chi2'],float(residual@residual+penalty@penalty),places=5)
        posterior=gv.gvar(p,result['parameter_covariance'])
        _,areas=_peak_quantities(posterior)
        np.testing.assert_allclose(gv.evalcov(areas),result['area_covariance'],rtol=1e-8,atol=1e-12)
        ratio,sd=propagate_area_ratio(result['areas'],result['area_covariance'])
        self.assertAlmostEqual(float(ratio),result['qn_ratio'])
        self.assertAlmostEqual(float(sd),result['qn_ratio_sd'])
        self.assertGreater(np.max(np.abs(result['area_covariance']-np.diag(np.diag(result['area_covariance'])))),0)
        self.assertAlmostEqual(result['area_fractions'].sum(),1.)
        self.assertGreaterEqual(np.linalg.eigvalsh(result['parameter_covariance']).min(),-1e-12)


if __name__ == "__main__":
    unittest.main()
