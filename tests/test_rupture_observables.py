import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src import tg_spectra_ruptures as ts
from src.tg_spectra_ruptures import FAMILY_NAMES, find_common_break, permutation_test


class TgSpectraTests(unittest.TestCase):
    def test_default_peak_fit_path_matches_manuscript_dataset(self):
        # Régression : les chemins par défaut doivent viser le même fichier.
        from src.figure_data import PEAK_FITS
        self.assertEqual(ts.PEAK_FITS, PEAK_FITS)
        self.assertEqual(ts.qn_channels.__defaults__[0], PEAK_FITS)

    def test_ladder_drops_cooling_and_crystallised_points(self):
        T, lab, idx = ts._clean_ladder(
            [200, 250, 300, 320, 190], ["Verre", "Verre", "Melt", "Melt+Ctx", "Verre"],
            np.arange(5))
        self.assertEqual(list(T), [200, 250, 300])
        self.assertEqual(list(idx), [0, 1, 2])

    def test_prepare_glass_gives_unit_area(self):
        x = np.arange(400.0, 1280.0)
        y = 1e-3 + 1e-6 * x + np.exp(-0.5 * ((x - 1000) / 30) ** 2)
        out = ts.prepare_glass(x, y[None])
        self.assertAlmostEqual(float(np.trapezoid(out[0], x)), 1.0, places=6)



    def test_pca_modes_return_scores_for_the_run(self):
        rng = np.random.default_rng(5)
        data = {f"r{i}": rng.normal(size=(8, 50)) for i in range(3)}
        for mode in ts.MODES:
            scores, _ = ts.fit_glass_pca(data, mode, "r1", n_pc=3)
            self.assertEqual(scores.shape[0], 8)
        with self.assertRaises(ValueError):
            ts.fit_glass_pca(data, "bad", "r1")

    def test_qn_ratios_include_toh_and_t2s_q4_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "peak_fits.h5"
            areas = np.tile([1.0, 2.0, 4.0, 8.0, 16.0], (3, 1))
            with h5py.File(path, "w") as f:
                f.create_group("run").create_dataset("areas", data=areas)
                g = f["run"]
                g.create_dataset("positions", data=np.zeros_like(areas))
                g.create_dataset("widths", data=np.ones_like(areas))

            ratios = ts.qn_channels("run", path=path, kind="ratios")

        self.assertEqual(ratios.shape, (3, 10))
        self.assertAlmostEqual(ratios[0, 1], np.log(1.0 / 4.0))  # TOH/T2S
        self.assertAlmostEqual(ratios[0, 8], np.log(4.0 / 16.0))  # T2S/Q4

    def test_qn_channels_reject_unknown_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "peak_fits.h5"
            with h5py.File(path, "w") as f:
                g = f.create_group("run")
                for name in ("areas", "positions", "widths"):
                    g.create_dataset(name, data=np.ones((2, 5)))
            with self.assertRaisesRegex(ValueError, "unknown Qn channel set"):
                ts.qn_channels("run", path=path, kind="unknown")

    def test_ratio_test_families_use_requested_ratios_positions_and_sensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "peak_fits.h5"
            areas = np.tile([1.0, 2.0, 4.0, 8.0, 16.0], (4, 1))
            positions = np.tile([10.0, 20.0, 30.0, 40.0, 50.0], (4, 1))
            with h5py.File(path, "w") as f:
                g = f.create_group("run")
                g.create_dataset("areas", data=areas)
                g.create_dataset("positions", data=positions)
                g.create_dataset("widths", data=np.ones_like(areas))

            x = np.arange(400.0, 1271.0)
            spectra = np.tile(np.ones_like(x), (4, 1))
            sensors = np.ones((2, 9))
            families = ts.ratio_test_families(x, spectra, "run", sensors, np.array([1, 3]), path)

        self.assertEqual(set(families), {"raman_area_ratio", "qn_ratios", "peak_positions", "sensors"})
        self.assertEqual(families["raman_area_ratio"][0].shape, (2, 1))
        self.assertTrue(np.allclose(families["qn_ratios"][0], np.tile(
            [np.log(4.0 / 16.0), np.log(1.0 / 4.0), np.log(2.0 / 8.0)], (2, 1))))
        self.assertTrue(np.array_equal(families["peak_positions"][0], np.tile(positions[0], (2, 1))))
        self.assertTrue(np.array_equal(families["sensors"][0], sensors))

    def test_ruptures_finds_shared_piecewise_linear_change(self):
        T = np.arange(12, dtype=float)
        values = np.maximum(T - 6.0, 0.0)
        families = {
            "glass_pca": (np.column_stack([values * (i + 1) for i in range(4)]), True),
            "raman_area_ratio": (values[:, None], True),
            "qn_ratios": (np.column_stack((values, 2 * values)), True),
            "peak_positions": (np.column_stack((values, -values)), True),
            "sensors": (np.column_stack([values * (i + 1) for i in range(9)]), True),
        }

        result = find_common_break(T, families, min_side=3)

        self.assertIn(result["split_index"], (6, 7, 8))
        self.assertAlmostEqual(result["T_break"], (T[result["split_index"] - 1]
                                                   + T[result["split_index"]]) / 2)
        self.assertEqual(set(result["improvement"]), set(FAMILY_NAMES))
        self.assertGreater(result["total_improvement"], 0)

    def test_ruptures_permutation_test_researches_break(self):
        rng = np.random.default_rng(16)
        T = np.arange(18, dtype=float)
        trend = np.maximum(T - 9.0, 0.0)
        families = {
            "glass_pca": (np.column_stack([trend * (i + 1) + rng.normal(0, 0.02, len(T))
                                           for i in range(4)]), True),
            "raman_area_ratio": (trend[:, None] + rng.normal(0, 0.02, (len(T), 1)), True),
            "qn_ratios": (np.column_stack([trend, -trend, 2 * trend])
                          + rng.normal(0, 0.02, (len(T), 3)), True),
            "peak_positions": (np.column_stack([trend * (i + 1) for i in range(5)])
                               + rng.normal(0, 0.02, (len(T), 5)), True),
            "sensors": (np.column_stack([trend * (i + 1) for i in range(9)])
                        + rng.normal(0, 0.02, (len(T), 9)), True),
        }

        result = permutation_test(T, families, n_permutations=49, seed=5)

        self.assertGreater(result["observed_gain"], 0)
        self.assertGreaterEqual(result["p_value"], 1 / 50)
        self.assertLessEqual(result["p_value"], 1)
        self.assertEqual(len(result["null_gains"]), 49)


if __name__ == "__main__":
    unittest.main()
