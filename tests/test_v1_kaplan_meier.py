"""Kaplan-Meier helpers and representative v1 synthetic-trial capture."""

import unittest

import numpy as np

import regal_explorer as regal


class KaplanMeierTest(unittest.TestCase):
    def test_product_limit_curve_handles_tied_death_and_censor(self):
        curve = regal.kaplan_meier(
            np.array([2.0, 2.0, 3.0, 4.0]),
            np.array([1, 0, 1, 0]),
        )

        np.testing.assert_allclose(curve["times"], [0.0, 2.0, 3.0])
        np.testing.assert_allclose(curve["survival"], [1.0, 0.75, 0.375])
        np.testing.assert_allclose(curve["censor_times"], [2.0, 4.0])
        np.testing.assert_allclose(curve["censor_survival"], [0.75, 0.375])

    def test_capture_does_not_change_monte_carlo_summary(self):
        model = regal.build_plateau(regal.default_cfg())
        plain = regal.mc(model, nsim=80, seed=12345)
        captured = regal.mc(model, nsim=80, seed=12345, capture_km=True)

        for key in ("ps", "reach", "medHR", "medHR_IA", "aliveG", "aliveB"):
            self.assertEqual(plain[key], captured[key])
        np.testing.assert_array_equal(plain["hrsAll"], captured["hrsAll"])

    def test_representative_trial_is_complete_and_nearest_median_hr(self):
        cfg = regal.default_cfg()
        result = regal.mc(regal.build_plateau(cfg), nsim=120, seed=987654321,
                          capture_km=True)
        example = result["kmExample"]

        self.assertIsNotNone(example)
        self.assertEqual(len(example["time"]), cfg["N"])
        self.assertEqual(len(example["event"]), cfg["N"])
        self.assertEqual(len(example["arm"]), cfg["N"])
        self.assertEqual(int(example["event"].sum()), cfg["FINAL"])
        self.assertEqual(int(example["arm"].sum()), cfg["N"] // 2)
        self.assertTrue(np.all(example["time"] >= 0))

        nearest = min(abs(hr - result["medHR"]) for hr in result["hrsAll"])
        self.assertAlmostEqual(abs(example["hr"] - result["medHR"]), nearest)

        for arm_value in (0, 1):
            mask = example["arm"] == arm_value
            curve = regal.kaplan_meier(example["time"][mask], example["event"][mask])
            self.assertTrue(np.all(np.diff(curve["times"]) >= 0))
            self.assertTrue(np.all(np.diff(curve["survival"]) <= 0))
            self.assertGreaterEqual(curve["survival"].min(), 0.0)
            self.assertLessEqual(curve["survival"].max(), 1.0)

    def test_capture_returns_none_when_final_trigger_cannot_be_reached(self):
        cfg = regal.default_cfg(FINAL=200)
        result = regal.mc(regal.build_plateau(cfg), nsim=5, capture_km=True)
        self.assertEqual(result["reach"], 0.0)
        self.assertIsNone(result["kmExample"])


if __name__ == "__main__":
    unittest.main()
