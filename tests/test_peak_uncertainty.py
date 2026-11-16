"""Propagation gvar, covariance complète et cas manquants."""
import unittest
import numpy as np
from src.peak_uncertainty import propagate_area_ratio, propagate_area_fractions, area_gvars


class PeakUncertaintyTests(unittest.TestCase):
    def test_correlated_fraction_and_ratio(self):
        a=np.array([1.,2.,3.,4.,5.])
        c=np.diag([.01,.02,.03,.04,.05])+.001
        f,fc=propagate_area_fractions(a,c)
        r,sd=propagate_area_ratio(a,c)
        def values(v): return np.r_[v/v.sum(),v[1]/v[3]]
        jac=np.column_stack([(values(a+np.eye(5)[j]*1e-5)-values(a-np.eye(5)[j]*1e-5))/2e-5 for j in range(5)])
        expected=jac@c@jac.T
        np.testing.assert_allclose(fc,expected[:5,:5],atol=1e-12)
        self.assertAlmostEqual(float(sd)**2,expected[-1,-1],places=12)
        np.testing.assert_allclose(fc.sum(0),0,atol=1e-16)
        self.assertAlmostEqual(f.sum(),1.)
        self.assertAlmostEqual(float(r),.5)

    def test_shared_scale_cancels_exactly(self):
        # Une erreur multiplicative commune aux aires ne change ni fractions ni rapport.
        a=np.arange(1.,6.)
        c=.01*np.outer(a,a)
        _,fc=propagate_area_fractions(a,c)
        _,sd=propagate_area_ratio(a,c)
        np.testing.assert_allclose(fc,0,atol=1e-16)
        self.assertAlmostEqual(float(sd),0.,places=8)

    def test_missing_and_zero_denominator(self):
        a=np.array([[1.,2.,3.,4.,5.],[1.,2.,3.,0.,5.],[np.nan]*5])
        c=np.array([np.eye(5)]*3)
        m,s=propagate_area_ratio(a,c)
        self.assertEqual(m[0],.5)
        self.assertTrue(np.isnan(m[1:]).all())
        f,fc=propagate_area_fractions(a,c)
        self.assertTrue(np.isnan(f[-1]).all())
        with self.assertRaises(ValueError): area_gvars(np.ones(5),-np.eye(5))
        with self.assertRaises(ValueError): propagate_area_ratio(np.ones(5),np.eye(4))


class NbotTests(unittest.TestCase):
    def test_shared_coefficients_and_fraction_covariance(self):
        from src.peak_uncertainty import nbot_from_fractions
        f=np.array([[.1,.2,.2,.3,.2],[.05,.15,.2,.4,.2]])
        j=np.eye(5)-np.ones((5,5))/5
        c=np.array([j@np.diag([.001]*5)@j.T]*2)
        mean,cov,fit=nbot_from_fractions(f,c)
        v=np.array([0,2*.84,0,1.06,0])
        expected_fit=np.diag([v@cc@v for cc in c])
        shared=np.outer(2*f[:,1],2*f[:,1])*.07**2+np.outer(f[:,3],f[:,3])*.04**2
        np.testing.assert_allclose(mean,2*.84*f[:,1]+1.06*f[:,3])
        np.testing.assert_allclose(fit,expected_fit,atol=1e-16)
        np.testing.assert_allclose(cov,expected_fit+shared,atol=1e-16)
        self.assertGreater(cov[0,1],0)
        with self.assertRaises(ValueError): nbot_from_fractions(f*2,c)
