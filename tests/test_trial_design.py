"""Tests for explicit protocol-boundary and interim-replay primitives."""

import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trial_design import obrien_fleming_two_look  # noqa: E402
from audit.interim_efficacy_replay import replay  # noqa: E402


class TrialDesignTest(unittest.TestCase):
    def test_two_look_obrien_fleming_boundaries(self):
        boundary = obrien_fleming_two_look(alpha=0.025, interim_information=0.75)
        self.assertAlmostEqual(boundary["interim_z"], 2.32708, places=4)
        self.assertAlmostEqual(boundary["final_z"], 2.01531, places=4)

    def test_invalid_information_fraction(self):
        with self.assertRaises(ValueError):
            obrien_fleming_two_look(interim_information=1.0)

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
