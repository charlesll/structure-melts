import unittest

import numpy as np

from src.pressure_sensors import (LBD0_BORATE_NM, LBD0_RUBY_NM, borate_T_corr,
                                  pressure_sensor, ruby_T_corr)


class PressureCalibrationTests(unittest.TestCase):
    def test_zero_pressure_reference_tracks_thermal_shift(self):
        # build the wavelength from the same reference the function uses, so the
        # test checks the thermal-shift bookkeeping and not the choice of lbd_0
        temperature = np.array([296.0, 500.0, 800.0])
        ruby_wavelength = LBD0_RUBY_NM + ruby_T_corr(temperature, "wei2024")
        borate_wavelength = LBD0_BORATE_NM + borate_T_corr(temperature, "wei2024")

        ruby_pressure = pressure_sensor(
            ruby_wavelength, temperature, scale="ruby", ruby_scale="wei2024"
        )
        borate_pressure = pressure_sensor(
            borate_wavelength, temperature, scale="borate", borate_scale="wei2024"
        )

        np.testing.assert_allclose(ruby_pressure, 0.0, atol=1e-10)
        np.testing.assert_allclose(borate_pressure, 0.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
