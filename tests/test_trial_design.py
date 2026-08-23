"""Tests for legacy audit-boundary and interim-replay primitives."""

import math
import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from audit.interim_efficacy_replay import replay  # noqa: E402
import regal_explorer as regal  # noqa: E402
import trial_design  # noqa: E402
from trial_design import obrien_fleming_two_look  # noqa: E402


class TrialDesignTest(unittest.TestCase):
    def test_two_look_obrien_fleming_boundaries(self):
        boundary = obrien_fleming_two_look(alpha=0.025, interim_information=0.75)
        self.assertAlmostEqual(boundary["interim_z"], 2.32708, places=4)
        self.assertAlmostEqual(boundary["final_z"], 2.01531, places=4)

    def test_invalid_information_fraction(self):
        with self.assertRaises(ValueError):
            obrien_fleming_two_look(interim_information=1.0)

    def test_solver_realizes_alpha_outside_the_conventional_bracket(self):
        information = 0.75
        for alpha in (0.00001, 0.025, 0.2, 0.4):
            with self.subTest(alpha=alpha):
                boundary = obrien_fleming_two_look(
                    alpha=alpha, interim_information=information
                )
                realized = 1.0 - trial_design._bivariate_normal_cdf(
                    boundary["interim_z"],
                    boundary["final_z"],
                    math.sqrt(information),
                )
                self.assertAlmostEqual(realized, alpha, places=9)

    def test_cached_solver_does_not_share_mutable_results(self):
        trial_design._solve_obrien_fleming_two_look.cache_clear()
        first = obrien_fleming_two_look()
        first["interim_z"] = -1
        second = obrien_fleming_two_look()
        self.assertAlmostEqual(second["interim_z"], 2.32708, places=4)
        cache = trial_design._solve_obrien_fleming_two_look.cache_info()
        self.assertEqual((cache.misses, cache.hits), (1, 1))

    def test_nonpositive_interim_count_is_clamped(self):
        cfg = regal.default_cfg()
        cfg["IA"] = 0
        result = regal.mc(regal.build_plateau(cfg), nsim=5)
        self.assertTrue(math.isfinite(result["z_IA_efficacy"]))
        expected = obrien_fleming_two_look(interim_information=1 / cfg["FINAL"])
        self.assertAlmostEqual(result["z_IA_efficacy"], expected["interim_z"], places=8)

    def test_final_event_count_below_two_is_clamped(self):
        expected = obrien_fleming_two_look(interim_information=0.5)
        for final_events in (1, 0, -1):
            with self.subTest(final_events=final_events):
                cfg = regal.default_cfg()
                cfg["FINAL"] = final_events
                result = regal.mc(regal.build_plateau(cfg), nsim=5)
                self.assertTrue(math.isfinite(result["z_IA_efficacy"]))
                self.assertAlmostEqual(
                    result["z_IA_efficacy"], expected["interim_z"], places=8
                )

    def test_equal_strata_replay_characterization(self):
        result = replay(nsim=1000)
        self.assertEqual(result["bat_weights"], [135 / 7, 40 / 7, 25.0, 25.0, 25.0])
        self.assertAlmostEqual(result["interim_efficacy_crossing"], 0.941, delta=0.02)
        self.assertAlmostEqual(
            result["final_scenario_rejection_rate_given_reach"], 0.999, delta=0.01
        )
        self.assertAlmostEqual(result["median_final_hr"], 0.297, delta=0.03)


if __name__ == "__main__":
    unittest.main()
