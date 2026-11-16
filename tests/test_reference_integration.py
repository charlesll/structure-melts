"""Ambient calibration and alignment: no row shifts or invented spectra/grains."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.config import (Run, LBD0_RUBY_NM, LBD0_BORATE_NM,
                        LBD_REFERENCE_RUBY_NM, LBD_REFERENCE_BORATE_NM, LBD_REFERENCE_T_K,
                        LBD_REFERENCE_ROOM_T_C)
from src.extract_markers import align_workbook_sheets, _glass
from src.pressure_sensors import pressure_sensor
from src.tg_spectra_ruptures import mean_sensor_shapes, shape_dispersion


class AmbientIntegrationTests(unittest.TestCase):
    def test_room_references_are_adopted_directly_at_296K(self):
        self.assertEqual(LBD_REFERENCE_ROOM_T_C, 23.)
        self.assertEqual(LBD_REFERENCE_T_K, 296.)
        self.assertEqual(LBD0_RUBY_NM, LBD_REFERENCE_RUBY_NM)
        self.assertEqual(LBD0_BORATE_NM, LBD_REFERENCE_BORATE_NM)
        for sensor, measured in [('ruby', LBD_REFERENCE_RUBY_NM), ('borate', LBD_REFERENCE_BORATE_NM)]:
            p = pressure_sensor(measured, LBD_REFERENCE_T_K, scale=sensor,
                                ruby_scale='wei2024', borate_scale='wei2024')
            self.assertAlmostEqual(float(p), 0., places=9)

    def test_alignment_uses_keys_and_only_one_reference_grain(self):
        with TemporaryDirectory() as directory:
            run = Run('test', directory, ambient_reference=dict(T_C=25., Pmem=1.,
                      Ruby='Rref', Borate='Bref'))
            for name in ['Rref', 'Bref']:
                (Path(directory)/(name+'.asc')).touch()
            # Sensor rows deliberately in the reverse order of the glass master.
            glass = pd.DataFrame(dict(T_C=[25,25,25], Pmem=[1,10,20]))
            sensors = pd.DataFrame(dict(T_C=[25,25], Pmem=[20,10], grain1=['20','10'], grain2=['b20','b10']))
            sheets = dict(Glass=glass, Ruby=sensors, Borate=sensors,
                          Tg=glass.assign(Etat=['Verre']*3))
            result = align_workbook_sheets(run, sheets)
            self.assertEqual(result['Ruby'].grain1.tolist(), ['Rref','10','20'])
            self.assertEqual(result['Borate'].grain1.tolist(), ['Bref','10','20'])
            self.assertTrue(pd.isna(result['Ruby'].grain2.iloc[0]))
            self.assertEqual(len(sensors), 2)  # original table untouched

    def test_missing_HF_does_not_remove_LF(self):
        sheet = pd.DataFrame(dict(Glass_LF=['LF'], Glass_HF=['missing'], Bkg_LF=['background']))
        x = np.array([400., 800., 1279.])
        with TemporaryDirectory() as directory, \
             patch('src.extract_markers._read_asc', return_value=np.array([[542.,1.],[556.,2.],[571.,1.]])), \
             patch('src.extract_markers.treat_glass', return_value=(x,np.ones(3),np.array([]),np.array([]))) as treat:
            with self.assertWarnsRegex(UserWarning, 'retaining measured LF'):
                lf, hf = _glass(sheet, Path(directory))
            self.assertTrue(np.isfinite(lf).all())
            self.assertTrue(np.isnan(hf).all())
            self.assertIsNone(treat.call_args.args[1])

    def test_wavenumber_input_is_not_converted_as_wavelength(self):
        sheet = pd.DataFrame(dict(Glass_LF=['LF_cm1'], Glass_HF=[np.nan], Bkg_LF=['background']))
        x = np.array([379., 800., 1281.])
        with TemporaryDirectory() as directory, \
             patch('src.extract_markers._read_asc', return_value=np.column_stack([x,np.ones(3)])), \
             patch('src.extract_markers.treat_glass', return_value=(x,np.ones(3),np.array([]),np.array([]))) as treat:
            with self.assertWarnsRegex(UserWarning, 'retaining measured LF'):
                lf, _ = _glass(sheet, Path(directory), axis_units={'LF_cm1':'raman_cm1'})
            self.assertFalse(treat.call_args.kwargs['convert_in_cm'])
            self.assertTrue(np.isfinite(lf).all())

    def test_single_grain_contributes_mean_but_not_dispersion(self):
        aligned = np.array([[[1.,2.,3.], [np.nan]*3], [[1.,2.,3.], [3.,4.,5.]]])
        np.testing.assert_allclose(mean_sensor_shapes(aligned), [[1,2,3],[2,3,4]])
        result = shape_dispersion(aligned)
        self.assertTrue(np.isnan(result[0]))
        self.assertAlmostEqual(result[1], 1.)

    def test_sensor_alignment_rejects_truncated_windows(self):
        from src.tg_spectra_ruptures import align_sensor
        x = np.linspace(685., 689., 401)
        spectra = np.ones((1, len(x), 2))
        # A missing grain stays missing; it is not extrapolated or invented.
        _, aligned = align_sensor(x, spectra, np.array([[687., np.nan]]), .35)
        self.assertTrue(np.isfinite(aligned[0, 0]).all())
        self.assertTrue(np.isnan(aligned[0, 1]).all())
        with self.assertRaisesRegex(ValueError, 'exceeds stored wavelength grid'):
            align_sensor(x, spectra, np.array([[688.9, np.nan]]), .35)


if __name__ == '__main__':
    unittest.main()
