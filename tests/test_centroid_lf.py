import unittest

import numpy as np

from src.centroid_lf import OUT_DIR, area_ratio, centroid


class CentroidLfTests(unittest.TestCase):
    def test_area_ratio_integrates_the_two_regions(self):
        x = np.arange(400.0, 1271.0)
        spectra = np.vstack([np.where(x < 830.0, 2.0, 1.0),
                             np.where(x < 830.0, 4.0, 1.0)])

        ratios = area_ratio(x, spectra)

        low = (x >= 400.0) & (x <= 830.0)
        high = (x >= 830.0) & (x <= 1270.0)
        expected = (np.trapezoid(spectra[:, low], x[low], axis=1)
                    / np.trapezoid(spectra[:, high], x[high], axis=1))
        self.assertTrue(np.allclose(ratios, expected))

    def test_zero_high_frequency_area_returns_nan(self):
        x = np.arange(400.0, 1271.0)
        spectrum = np.where(x < 830.0, 1.0, 0.0)[None, :]

        ratio = area_ratio(x, spectrum)

        self.assertTrue(np.isnan(ratio[0]))

    def test_high_region_centroid_excludes_outside_intensity(self):
        x = np.arange(400., 1280.)
        spectrum = np.ones((1, len(x)))
        spectrum[:, (x < 830.) | (x > 1270.)] = 1e6
        # A constant intensity over [830,1270] has moment at the midpoint.
        self.assertAlmostEqual(centroid(x, spectrum, (830., 1270.))[0], 1050.)

    def test_melt_pressure_export_matches_actual_source_acquisitions(self):
        import json
        import h5py
        from src.pressure import load_dataset
        from src.figure_data import DATASET
        from src.raman_treatment import prepare_glass
        result = json.loads((OUT_DIR / 'melt_centroid_830_1270_vs_pressure.json').read_text())
        self.assertEqual(result['window_cm-1'], [830., 1270.])
        with h5py.File(DATASET) as handle:
            axis = handle['x_axis'][()]
        source = load_dataset()
        expected = set()
        for run, data in source.items():
            for step in np.flatnonzero(data['Etat'] == 'Melt'):
                expected.add((run, int(step)))
        actual = {(row['run_id'], row['step']) for row in result['points']}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), len(result['points']))
        for row in result['points']:
            data = source[row['run_id']]
            step = row['step']
            value = centroid(axis, prepare_glass(axis, data['spectra'][step:step+1]),
                             (830., 1270.))[0]
            self.assertAlmostEqual(row['centroid_830_1270_cm-1'], value)
            self.assertEqual(row['P_GPa'], data['borate_P_median'][step])
            self.assertEqual(row['Etat'], 'Melt')


if __name__ == "__main__":
    unittest.main()
