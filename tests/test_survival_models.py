"""Tests for the corrected REGAL v2 survival primitives."""

import os
import sys
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from survival_models import (  # noqa: E402
    CureMixtureComponent,
    ExponentialBackgroundMortality,
    FrailtyCaseMix,
    SurvivalScale,
    WeibullSurvival,
    marginal_survival,
)


class SurvivalScaleTest(unittest.TestCase):
    def setUp(self):
        self.uncured = WeibullSurvival(median_months=12.0, shape=1.0)
        self.background = ExponentialBackgroundMortality(annual_death_probability=0.02)

    def test_overall_scale_applies_background_only_to_cured_fraction(self):
        component = CureMixtureComponent(
            name="OS input",
            uncured=self.uncured,
            cure_fraction=0.2,
            survival_scale=SurvivalScale.OVERALL,
        )
        months = np.array([0.0, 12.0, 60.0])
        uncured = self.uncured.survival(months)
        background = self.background.survival(months)
        expected = 0.2 * background + 0.8 * uncured
        np.testing.assert_allclose(component.survival(months, self.background), expected)

        double_counted = background * (0.2 + 0.8 * uncured)
        self.assertGreater(abs(expected[-1] - double_counted[-1]), 0.001)

    def test_net_scale_applies_background_to_the_complete_mixture(self):
        component = CureMixtureComponent(
            name="net input",
            uncured=self.uncured,
            cure_fraction=0.2,
            survival_scale="net",
        )
        months = np.array([0.0, 12.0, 60.0])
        expected = self.background.survival(months) * (
            0.2 + 0.8 * self.uncured.survival(months)
        )
        np.testing.assert_allclose(component.survival(months, self.background), expected)
        self.assertIs(component.survival_scale, SurvivalScale.NET)

    def test_existing_inputs_default_to_overall_scale(self):
        component = CureMixtureComponent("legacy literature OS", self.uncured, 0.2)
        self.assertIs(component.survival_scale, SurvivalScale.OVERALL)

    def test_cured_patients_follow_population_mortality(self):
        component = CureMixtureComponent("all cured", self.uncured, 1.0)
        self.assertAlmostEqual(component.survival(240.0, self.background), 0.98**20)
        self.assertLess(component.survival(240.0, self.background), 1.0)

        times = component.sample_event_times(
            np.random.default_rng(11), self.background, np.ones(100)
        )
        self.assertTrue(np.all(np.isfinite(times)))

    def test_invalid_scale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overall.*net"):
            CureMixtureComponent("bad scale", self.uncured, 0.2, "disease-free")

    def test_sampling_matches_each_scale_analytic_survival(self):
        frailty = np.ones(100000)
        for scale in (SurvivalScale.OVERALL, SurvivalScale.NET):
            with self.subTest(scale=scale):
                component = CureMixtureComponent("sample check", self.uncured, 0.3, scale)
                times = component.sample_event_times(
                    np.random.default_rng(400 + int(scale is SurvivalScale.NET)),
                    self.background,
                    frailty,
                )
                empirical = np.mean(times > 24.0)
                expected = component.survival(24.0, self.background)
                self.assertAlmostEqual(empirical, expected, delta=0.005)


class FrailtyCaseMixTest(unittest.TestCase):
    def setUp(self):
        self.case_mix = FrailtyCaseMix(
            population_log_sd=0.7,
            eligibility_logit_intercept=0.0,
            eligibility_health_gradient=2.0,
        )

    def test_eligibility_enriches_baseline_risk_without_outcome_truncation(self):
        source = self.case_mix.draw_population(30000, np.random.default_rng(7))
        selected = self.case_mix.sample_enrolled(10000, np.random.default_rng(8))
        self.assertLess(np.mean(selected), np.mean(source) - 0.1)

        component = CureMixtureComponent(
            "no-cure event model", WeibullSurvival(12.0, 1.0), 0.0
        )
        no_background = ExponentialBackgroundMortality(0.0)
        self.assertLess(marginal_survival(component, 0.5, no_background, selected), 1.0)
        event_times = component.sample_event_times(
            np.random.default_rng(9), no_background, selected
        )
        self.assertTrue(np.any(event_times < 0.5))

    def test_selection_does_not_mechanically_inflate_cure_fraction(self):
        selected = self.case_mix.sample_enrolled(5000, np.random.default_rng(10))
        component = CureMixtureComponent(
            "cure mixture", WeibullSurvival(12.0, 1.0), 0.1
        )
        no_background = ExponentialBackgroundMortality(0.0)
        self.assertAlmostEqual(
            marginal_survival(component, 100000.0, no_background, selected),
            0.1,
            places=12,
        )

    def test_randomization_occurs_after_selection_and_balances_case_mix(self):
        cohort = self.case_mix.sample_randomized(10000, np.random.default_rng(12))
        self.assertEqual(np.count_nonzero(cohort.treatment), 5000)
        treated = cohort.frailty[cohort.treatment]
        control = cohort.frailty[~cohort.treatment]
        self.assertAlmostEqual(np.mean(treated), np.mean(control), delta=0.03)
        with self.assertRaises(ValueError):
            cohort.frailty[0] = 99.0

    def test_neutral_gradient_does_not_change_population_case_mix(self):
        neutral = FrailtyCaseMix(
            population_log_sd=0.7,
            eligibility_logit_intercept=0.0,
            eligibility_health_gradient=0.0,
        )
        source = neutral.draw_population(30000, np.random.default_rng(13))
        selected = neutral.sample_enrolled(10000, np.random.default_rng(14))
        self.assertAlmostEqual(np.mean(selected), np.mean(source), delta=0.03)

    def test_default_case_mix_leaves_the_reference_curve_unchanged(self):
        neutral = FrailtyCaseMix()
        selected = neutral.sample_enrolled(100, np.random.default_rng(15))
        np.testing.assert_array_equal(selected, np.ones(100))

        component = CureMixtureComponent(
            "reference", WeibullSurvival(12.0, 1.0), 0.1
        )
        background = ExponentialBackgroundMortality(0.02)
        self.assertAlmostEqual(
            marginal_survival(component, 24.0, background, selected),
            component.survival(24.0, background),
        )


if __name__ == "__main__":
    unittest.main()
