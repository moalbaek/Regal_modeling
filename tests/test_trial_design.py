"""Tests for legacy audit-boundary and interim-replay primitives."""

import math
import os
import sys
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from audit.interim_efficacy_replay import replay  # noqa: E402
import regal_explorer as regal  # noqa: E402
import trial_design  # noqa: E402
from trial_design import (  # noqa: E402
    HazardRatioFutilityRule,
    InterimDecision,
    classify_interim,
    lan_demets_obrien_fleming_spending,
    lan_demets_obrien_fleming_two_look,
    obrien_fleming_two_look,
    stratified_logrank,
    unstratified_logrank_diagnostic,
)


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
        self.assertAlmostEqual(result["interim_efficacy_crossing"], 0.633, delta=0.02)
        self.assertAlmostEqual(
            result["final_scenario_rejection_rate_given_reach"], 0.916, delta=0.03
        )
        self.assertAlmostEqual(result["median_final_hr"], 0.454, delta=0.03)


class V2LanDeMetsBoundaryTest(unittest.TestCase):
    def test_regal_spending_boundaries_match_protocol_validation_values(self):
        boundary = lan_demets_obrien_fleming_two_look(
            alpha=0.025, interim_information=60 / 80
        )
        self.assertAlmostEqual(boundary["interim_z"], 2.339711, places=6)
        self.assertAlmostEqual(boundary["final_z"], 2.011777, places=6)
        self.assertAlmostEqual(boundary["interim_alpha_spent"], 0.009649325)
        self.assertAlmostEqual(boundary["final_alpha_spent"], 0.025)
        self.assertEqual(
            boundary["spending_family"], "Lan-DeMets O'Brien-Fleming"
        )

    def test_spending_targets_realize_sequential_one_sided_alpha(self):
        information = 0.75
        boundary = lan_demets_obrien_fleming_two_look(0.025, information)
        first_crossing = 1.0 - trial_design.NormalDist().cdf(
            boundary["interim_z"]
        )
        overall_crossing = 1.0 - trial_design._bivariate_normal_cdf(
            boundary["interim_z"],
            boundary["final_z"],
            math.sqrt(information),
        )
        self.assertAlmostEqual(
            first_crossing, boundary["interim_alpha_spent"], places=12
        )
        self.assertAlmostEqual(overall_crossing, 0.025, places=12)
        self.assertAlmostEqual(
            lan_demets_obrien_fleming_spending(1.0, 0.025), 0.025, places=12
        )

    def test_realized_final_overshoot_preserves_sequential_alpha(self):
        boundary = lan_demets_obrien_fleming_two_look(
            alpha=0.025,
            interim_information=68 / 80,
            final_information=86 / 80,
        )
        expected_correlation = math.sqrt((68 / 80) / (86 / 80))
        self.assertAlmostEqual(
            boundary["canonical_correlation"], expected_correlation, places=12
        )
        overall_crossing = 1.0 - trial_design._bivariate_normal_cdf(
            boundary["interim_z"],
            boundary["final_z"],
            expected_correlation,
        )
        self.assertAlmostEqual(overall_crossing, 0.025, places=12)
        self.assertAlmostEqual(boundary["final_information"], 86 / 80)
        self.assertEqual(boundary["final_spending_information"], 1.0)
        self.assertAlmostEqual(boundary["final_alpha_spent"], 0.025)

    def test_legacy_and_v2_boundary_families_remain_distinct(self):
        legacy = obrien_fleming_two_look(0.025, 0.75)
        protocol = lan_demets_obrien_fleming_two_look(0.025, 0.75)
        self.assertAlmostEqual(legacy["interim_z"], 2.32708, places=4)
        self.assertAlmostEqual(protocol["interim_z"], 2.33971, places=4)
        self.assertNotAlmostEqual(
            legacy["interim_z"], protocol["interim_z"], places=3
        )

    def test_invalid_spending_inputs_fail_before_boundary_solving(self):
        for information in (0.0, -0.1, 1.1, float("nan")):
            with self.subTest(information=information):
                with self.assertRaises(ValueError):
                    lan_demets_obrien_fleming_spending(information)
        with self.assertRaises(ValueError):
            lan_demets_obrien_fleming_two_look(interim_information=1.0)


class StratifiedLogRankTest(unittest.TestCase):
    def test_tied_event_variance_and_sign_convention(self):
        result = stratified_logrank(
            time=[1.0, 1.0, 2.0, 2.0],
            event=[1, 1, 0, 0],
            treatment=[1, 0, 1, 0],
            strata=["all"] * 4,
        )
        self.assertEqual(result.score, 0.0)
        self.assertAlmostEqual(result.variance, 1.0 / 3.0)
        self.assertEqual(result.z, 0.0)
        self.assertEqual(result.one_step_hazard_ratio, 1.0)
        self.assertEqual(result.events, 2)
        self.assertEqual(result.informative_strata, 1)

        benefit = stratified_logrank(
            time=[3.0, 4.0, 1.0, 2.0],
            event=[1, 1, 1, 1],
            treatment=[1, 1, 0, 0],
            strata=[0, 0, 0, 0],
        )
        self.assertGreater(benefit.z, 0.0)
        self.assertLess(benefit.one_step_hazard_ratio, 1.0)

    def test_multiple_factor_columns_equal_precombined_strata(self):
        time = np.array([1, 3, 2, 4, 2, 5, 3, 6], dtype=float)
        event = np.array([1, 1, 1, 0, 1, 1, 0, 1])
        treatment = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        factors = np.array(
            [
                ["CR2", "MRD+"],
                ["CR2", "MRD+"],
                ["CR2", "MRD-"],
                ["CR2", "MRD-"],
                ["CR2p", "MRD+"],
                ["CR2p", "MRD+"],
                ["CR2p", "MRD-"],
                ["CR2p", "MRD-"],
            ],
            dtype=object,
        )
        # NumPy expands equal-length tuples into two columns, so retain them as
        # explicit scalar tuple labels for the one-dimensional comparison.
        combined_labels = np.empty(len(factors), dtype=object)
        combined_labels[:] = [tuple(row) for row in factors]
        matrix_result = stratified_logrank(time, event, treatment, factors)
        combined_result = stratified_logrank(
            time, event, treatment, combined_labels
        )
        self.assertEqual(matrix_result, combined_result)

    def test_stratification_removes_factor_mix_confounding(self):
        # Each stratum has proportional treated/control events (zero stratified
        # score), but the early high-risk stratum contains mostly GPS patients.
        treatment = np.array(
            [1] * 8 + [0] * 2 + [1] * 2 + [0] * 8,
            dtype=int,
        )
        event = np.array(
            [1] * 4 + [0] * 4 + [1] + [0] + [1] + [0] + [1] * 4 + [0] * 4,
            dtype=int,
        )
        time = np.array(
            [1] * 4 + [20] * 4 + [1, 20] + [10, 20] + [10] * 4 + [20] * 4,
            dtype=float,
        )
        strata = np.array(["early"] * 10 + ["late"] * 10)
        primary = stratified_logrank(time, event, treatment, strata)
        diagnostic = unstratified_logrank_diagnostic(time, event, treatment)
        self.assertAlmostEqual(primary.score, 0.0, places=12)
        self.assertAlmostEqual(primary.z, 0.0, places=12)
        self.assertGreater(diagnostic.score, 0.0)
        self.assertLess(diagnostic.z, 0.0)

    def test_interim_classification_prioritizes_efficacy_then_futility(self):
        favorable = stratified_logrank(
            [5, 6, 1, 2], [1, 1, 1, 1], [1, 1, 0, 0], [0] * 4
        )
        unfavorable = stratified_logrank(
            [1, 2, 5, 6], [1, 1, 1, 1], [1, 1, 0, 0], [0] * 4
        )
        rule = HazardRatioFutilityRule(1.0)
        self.assertIs(
            classify_interim(favorable, 0.5, rule),
            InterimDecision.EFFICACY_STOP,
        )
        self.assertIs(
            classify_interim(unfavorable, 10.0, rule),
            InterimDecision.FUTILITY_STOP,
        )
        self.assertIs(
            classify_interim(favorable, 10.0, rule), InterimDecision.CONTINUE
        )

    def test_analysis_inputs_and_futility_threshold_are_validated(self):
        with self.assertRaisesRegex(ValueError, "zero/one"):
            stratified_logrank([1, 2], [1, 2], [0, 1], [0, 0])
        with self.assertRaisesRegex(ValueError, "missing"):
            stratified_logrank([1, 2], [1, 0], [0, 1], [0, None])
        for threshold in (0, -1, True, float("nan")):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    HazardRatioFutilityRule(threshold)


if __name__ == "__main__":
    unittest.main()
