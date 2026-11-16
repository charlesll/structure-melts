"""Régressions des sélections et propagations des figures manuscrit."""
import unittest
import numpy as np
from src.manuscript_data import propagate_areas, first_crystal, transition_status
from src.figure_data import melt_table
from src.figures.fig9_delta_tg import difference_covariance


class ManuscriptTests(unittest.TestCase):
    def test_covariance_jacobians(self):
        a = np.array([1., 2., 3., 4., 5.])
        c = np.diag([.01,.02,.03,.04,.05]) + .001
        f, fc, r, rs = propagate_areas(a,c)
        self.assertAlmostEqual(f.sum(),1.)
        def values(v):
            return np.r_[v/v.sum(),v[1]/v[3]]
        jac = np.column_stack([(values(a+np.eye(5)[j]*1e-5)-values(a-np.eye(5)[j]*1e-5))/2e-5 for j in range(5)])
        expected = jac@c@jac.T
        np.testing.assert_allclose(fc,expected[:5,:5],atol=1e-12)
        self.assertAlmostEqual(rs**2,expected[-1,-1],places=12)
        np.testing.assert_allclose(fc.sum(axis=0),0,atol=1e-16)
        f,fc,r,rs=propagate_areas(np.full(5,np.nan),np.full((5,5),np.nan))
        self.assertTrue(np.isnan(f).all())

    def test_first_crystal_heating(self):
        d={'T_C':np.array([25,220,200,250,260]),'Etat':np.array(['Verre','Melt','Melt+Ctx','Melt','Melt+Ctx'])}
        self.assertEqual(first_crystal(d),4)

    def test_melt_exact_selection(self):
        d={'Etat':np.array(['Verre','Melt','Melt+Ctx','Melt']), 'run_id':np.array(['run']*4)}
        for k in ['T_C','borate_P_median','borate_P_std','ruby_P_median','ruby_P_std']:
            d[k]=np.arange(4.)
        np.testing.assert_array_equal(melt_table(d)['index'],[1,3])

    def test_status_requires_both_nulls(self):
        self.assertEqual(transition_status(.001, .049), 'supported')
        self.assertEqual(transition_status(.001, .05), 'candidate')
        self.assertEqual(transition_status(.078, .01), 'candidate')

    def test_saved_ambient_reference_matches_regression_result(self):
        import json
        from src.figure_data import (RESULTS, TG_EXTRAPOLATED_0GPA_C,
                                     TG_EXTRAPOLATED_0GPA_PROFILE95_C)
        result = json.loads((RESULTS / 'fig7_interval_regression.json').read_text())
        self.assertAlmostEqual(TG_EXTRAPOLATED_0GPA_C, result['intercept_C'], places=6)
        np.testing.assert_allclose(TG_EXTRAPOLATED_0GPA_PROFILE95_C,
                                   result['intercept_profile95_C'], atol=1e-6)

    def test_generated_transitions_preserve_manuscript_brackets(self):
        # Référence scientifique validée avant réorganisation des modules.
        from src.manuscript_data import transitions
        expected = [
            ('run02_20250220', 262., 286., 'supported'),
            ('run03_20250303_32bar', 270., 277., 'candidate'),
            ('run04_20250303_34bar', 256., 262., 'supported'),
            ('run05_20250408', 313., 332., 'supported'),
            ('run06_20250430', 300., 306., 'supported'),
        ]
        actual = [(r['run_id'], r['T_left'], r['T_right'], r['status'])
                  for r in transitions()]
        self.assertEqual(actual, expected)

    def test_shared_reference(self):
        cov=difference_covariance([4,3,5])
        np.testing.assert_array_equal(cov,[[0,0,0],[0,25,16],[0,16,41]])


if __name__ == '__main__':
    unittest.main()
