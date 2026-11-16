"""Confirmed material inputs, IR uncertainties and publication metadata."""
import json
import tempfile
import unittest
from pathlib import Path

import gvar as gv
import numpy as np
import pandas as pd

from src.config import (ANN_COMPOSITION_WT, ANN_FICTIVE_REFERENCE,
                        ANN_HYDROUS_GLASS_NBO_T, RESULTS, FIGURES)
from src.ir_treatment import DENSITIES, extinction_coefficients
from src.figure_data import ambient_spectrum


class PublicationConsistencyTests(unittest.TestCase):
    def test_confirmed_material_inputs(self):
        self.assertEqual(ANN_COMPOSITION_WT,
                         dict(sio2=58.425, al2o3=11.875, na2o=24.70, h2o=5.00))
        self.assertAlmostEqual(sum(ANN_COMPOSITION_WT.values()), 100.)
        self.assertEqual(ANN_HYDROUS_GLASS_NBO_T, .59)
        self.assertEqual(ANN_FICTIVE_REFERENCE['temperature_C'], 219.)
        self.assertEqual(ANN_FICTIVE_REFERENCE['error_half_width_C'], 4.)
        self.assertEqual(ANN_FICTIVE_REFERENCE['heating_rate_K_min'], 25.)

    def test_exported_material_references_match_configuration(self):
        figure = json.loads((FIGURES/'manuscript'/'fig8_melt_areas.json').read_text())
        self.assertEqual(figure['hydrous_glass_reference_NBO_T'], ANN_HYDROUS_GLASS_NBO_T)
        regression = json.loads((RESULTS/'fig7_interval_regression.json').read_text())
        reference = regression['limiting_fictive_temperature_reference']
        for key, value in ANN_FICTIVE_REFERENCE.items():
            self.assertEqual(reference[key], value)
        self.assertFalse(reference['used_in_regression'])
        self.assertNotIn('DSC_comparisons', regression)

    def test_IR_coefficients_and_uncertainties_match_supplied_table(self):
        coefficients = extinction_coefficients()
        self.assertEqual(coefficients, {'4500': (.44, .03), '5200': (.64, .03)})
        metadata = json.loads((RESULTS/'IR'/'IR_metadata.json').read_text())
        for band, (mean, sd) in coefficients.items():
            self.assertEqual(metadata['peak_height_coefficients'][band], dict(mean=mean, sd=sd))
        table = pd.read_csv(RESULTS/'IR'/'IR_results.csv')
        self.assertEqual(len(table), 3)
        for _, row in table.iterrows():
            density = gv.gvar(*DENSITIES[row['name']])
            thickness = gv.gvar(row['epaisseur_micrometers'], row['ese_epaisseur'])*1e-4
            values = []
            for band in ('4500', '5200'):
                height = gv.gvar(row['height_'+band], row['ese_height_'+band])
                value = 1802*height/(thickness*density*gv.gvar(*coefficients[band]))
                values.append(value)
                self.assertAlmostEqual(float(gv.mean(value)), row['C_height_'+band], places=10)
                self.assertAlmostEqual(float(gv.sdev(value)), row['ese_C_height_'+band], places=10)
            self.assertAlmostEqual(float(gv.mean(sum(values))), row['C_height_total'], places=10)
            self.assertAlmostEqual(float(gv.sdev(sum(values))), row['ese_C_height_total'], places=10)

    def test_one_and_two_column_ambient_exports_agree(self):
        x = np.arange(400., 1401.)
        # Measurement-like noise avoids the despiker classifying smooth peak
        # curvature as a contiguous block of spikes with no clean neighbours.
        noise = np.random.default_rng(12).normal(0., .001, len(x))
        y = .2 + 1e-5*x + np.exp(-.5*((x-600.)/45.)**2) + noise
        with tempfile.TemporaryDirectory() as directory:
            intensity = Path(directory)/'one.asc'
            wavelength = Path(directory)/'two.asc'
            np.savetxt(intensity, y, delimiter=',')
            np.savetxt(wavelength, np.column_stack([1e7/(1e7/532.-x), y]), delimiter=',')
            one = ambient_spectrum(intensity)
            two = ambient_spectrum(wavelength)
        np.testing.assert_allclose(one[0], x, atol=0, rtol=0)
        np.testing.assert_allclose(one[0], two[0], atol=1e-10, rtol=0)
        np.testing.assert_allclose(one[1], two[1], atol=1e-10, rtol=1e-8)
        self.assertTrue(np.isfinite(one[1]).all())
        self.assertTrue(np.isfinite(one[2]))
        self.assertAlmostEqual(one[2], two[2], places=8)
