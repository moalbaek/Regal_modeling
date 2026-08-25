"""Tests for WP7 GPS effect families and posterior model averaging."""

from dataclasses import replace
from datetime import date
from math import exp, log
import os
import sys
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from event_likelihood import load_regal_public_history  # noqa: E402
from posterior import (  # noqa: E402
    BALANCED_MODEL_FAMILY_PRIOR,
    CURE_FAVORING_MODEL_FAMILY_PRIOR,
    DEFAULT_EFFECT_FAMILY_PRIORS,
    DEFAULT_MODEL_FAMILY_PRIOR_SENSITIVITY,
    SKEPTICAL_MODEL_FAMILY_PRIOR,
    ConditioningResult,
    DelayedCureEventTimeModel,
    EffectFamilyPrior,
    EffectFamilyProjection,
    GPSEffectFamily,
    GPSEffectScenarioSampler,
    LogLinearEnrollmentPrior,
    MAXIMUM_POSTERIOR_FORECAST_HISTORY_WEIGHT_SHARE,
    MINIMUM_POSTERIOR_FORECAST_ESS,
    ModelFamilyWeightPrior,
    PiecewiseMixtureHazardEventTimeModel,
    PiecewiseWeibullEventTimeModel,
    UniformPriorRange,
    condition_effect_families_futility_sensitivity_grid,
    default_regal_enrollment_prior,
    posterior_model_average,
    posterior_prior_sensitivity,
    _conditional_probability,
)
from trial_design import TrialDecisionDesign  # noqa: E402
from tests.test_posterior import small_enrollment_model, small_history  # noqa: E402


class FlexibleEventTimeModelTest(unittest.TestCase):
    def test_mixture_hazard_is_true_marginal_ph_and_round_trips(self):
        model = PiecewiseMixtureHazardEventTimeModel(
            scale_time=[10.0, 10.0],
            shape=[1.0, 1.0],
            cure_fraction=[0.25, 0.25],
            background_hazard=[0.0, 0.01],
            net_scale=[False, True],
            breakpoints=np.empty((2, 0)),
            hazard_multipliers=[[0.5], [0.5]],
        )
        time = np.array([10.0, 10.0])
        overall_baseline = 0.25 + 0.75 * exp(-1.0)
        net_baseline = exp(-0.1) * (0.25 + 0.75 * exp(-1.0))
        np.testing.assert_allclose(
            model.cdf(time),
            [1.0 - overall_baseline**0.5, 1.0 - net_baseline**0.5],
            rtol=0.0,
            atol=2e-15,
        )
        probabilities = np.array([0.25, 0.50])
        np.testing.assert_allclose(
            model.cdf(model.ppf(probabilities)),
            probabilities,
            rtol=0.0,
            atol=2e-13,
        )
        with self.assertRaises(ValueError):
            model.cure_fraction[0] = 0.0

    def test_mixture_hazard_preserves_defective_cure_mass_without_background(self):
        model = PiecewiseMixtureHazardEventTimeModel(
            scale_time=[10.0, 10.0],
            shape=[1.0, 1.0],
            cure_fraction=[0.36, 0.36],
            background_hazard=[0.0, 0.0],
            net_scale=[False, True],
            breakpoints=np.empty((2, 0)),
            hazard_multipliers=[[0.5], [0.5]],
        )
        limiting_event_mass = 1.0 - 0.36**0.5
        np.testing.assert_allclose(
            model.cdf(np.array([np.inf, np.inf])), limiting_event_mass
        )
        times = model.ppf(np.array([limiting_event_mass, 0.80]))
        self.assertTrue(np.all(np.isinf(times)))

    def test_piecewise_hazard_matches_closed_form_and_round_trips(self):
        model = PiecewiseWeibullEventTimeModel(
            scale_time=[10.0, 10.0, 12.0],
            shape=[1.0, 1.0, 1.2],
            constant_hazard=[0.0, 0.0, 0.01],
            weibull_weight=[1.0, 1.0, 1.0],
            breakpoints=[[5.0], [5.0], [4.0]],
            hazard_multipliers=[[1.0, 0.5], [1.0, 1.0], [0.8, 1.1]],
        )
        observed = model.cdf(np.array([10.0, 10.0, 9.0]))
        self.assertAlmostEqual(observed[0], 1.0 - exp(-0.75))
        self.assertAlmostEqual(observed[1], 1.0 - exp(-1.0))
        probabilities = np.array([0.25, 0.50, 0.70])
        np.testing.assert_allclose(
            model.cdf(model.ppf(probabilities)),
            probabilities,
            rtol=0.0,
            atol=2e-13,
        )
        with self.assertRaises(ValueError):
            model.hazard_multipliers[0, 0] = 2.0

    def test_delayed_cure_is_continuous_and_can_be_defective_at_zero_background(self):
        model = DelayedCureEventTimeModel(
            scale_time=[10.0, 10.0],
            shape=[1.0, 1.0],
            constant_hazard=[0.0, 0.0],
            weibull_weight=[1.0, 1.0],
            post_switch_hazard=[0.01, 0.0],
            switch_time=[5.0, 5.0],
            switch_to_background=[True, True],
        )
        at_landmark = 1.0 - exp(-0.5)
        observed = model.cdf(np.array([5.0, 5.0]))
        np.testing.assert_allclose(observed, at_landmark, rtol=0.0, atol=1e-15)
        probabilities = np.array([0.60, 0.60])
        times = model.ppf(probabilities)
        self.assertTrue(np.isfinite(times[0]))
        self.assertTrue(np.isinf(times[1]))
        self.assertAlmostEqual(model.cdf(times)[0], probabilities[0])
        self.assertAlmostEqual(model.cdf(times)[1], at_landmark)

    def test_event_models_reject_misaligned_piecewise_contracts(self):
        with self.assertRaisesRegex(ValueError, "one more column"):
            PiecewiseWeibullEventTimeModel(
                [10.0], [1.0], [0.0], [1.0], [[5.0]], [[1.0]]
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            PiecewiseWeibullEventTimeModel(
                [10.0], [1.0], [0.0], [1.0], [[5.0, 5.0]], [[1.0, 1.0, 1.0]]
            )


class EnrollmentPriorTest(unittest.TestCase):
    def test_default_wp7_prior_uses_only_open_and_close_as_calendar_support(self):
        history = load_regal_public_history()
        prior = default_regal_enrollment_prior(history)
        self.assertIsInstance(prior, LogLinearEnrollmentPrior)
        self.assertEqual(prior.study_start, history.study_start)
        self.assertEqual(prior.total_enrollment, history.target_enrollment)
        self.assertEqual(prior.enrollment_close, date(2024, 4, 30))
        draws = prior.sample_enrollment_dates(np.random.default_rng(7))
        self.assertEqual(len(draws), history.target_enrollment)
        self.assertGreaterEqual(min(draws), history.study_start)
        self.assertLessEqual(max(draws), prior.enrollment_close)

    def test_log_linear_prior_validates_bounds_and_rng_contract(self):
        with self.assertRaisesRegex(ValueError, "must not precede"):
            LogLinearEnrollmentPrior(8, date(2022, 1, 2), date(2022, 1, 1))
        with self.assertRaisesRegex(ValueError, "ordered"):
            LogLinearEnrollmentPrior(
                8,
                date(2022, 1, 1),
                date(2022, 2, 1),
                1.0,
                -1.0,
            )


class EffectFamilyTest(unittest.TestCase):
    def test_defaults_cover_every_required_family_with_valid_draws(self):
        self.assertEqual(
            {item.family for item in DEFAULT_EFFECT_FAMILY_PRIORS},
            set(GPSEffectFamily),
        )
        rng = np.random.default_rng(10)
        for prior in DEFAULT_EFFECT_FAMILY_PRIORS:
            self.assertEqual(prior.sample(rng).family, prior.family)
        with self.assertRaisesRegex(ValueError, "does not use"):
            EffectFamilyPrior(
                GPSEffectFamily.NO_EFFECT,
                hazard_ratio=UniformPriorRange(0.8, 1.0, log_scale=True),
            )

    def test_every_family_generates_estimable_stratified_patients_and_quantiles(self):
        entries = (date(2022, 1, 1),) * 126
        for index, prior in enumerate(DEFAULT_EFFECT_FAMILY_PRIORS):
            patients = GPSEffectScenarioSampler(prior)(
                entries, np.random.default_rng(100 + index)
            )
            self.assertEqual(patients.patient_count, 126)
            self.assertEqual(patients.strata.shape, (126, 4))
            _, cells = np.unique(
                patients.strata.astype(int), axis=0, return_inverse=True
            )
            for cell in range(int(cells.max()) + 1):
                selected = cells == cell
                imbalance = abs(
                    2 * int(patients.treatment[selected].sum())
                    - int(selected.sum())
                )
                self.assertLessEqual(imbalance, 1)
            probabilities = np.linspace(0.1, 0.9, 126)
            times = patients.event_time_model.ppf(probabilities)
            np.testing.assert_allclose(
                patients.event_time_model.cdf(times),
                probabilities,
                rtol=0.0,
                atol=3e-12,
                err_msg=prior.family.value,
            )

    def test_ph_at_one_is_exactly_nested_in_no_effect(self):
        entries = (date(2022, 1, 1),) * 64
        no_effect = EffectFamilyPrior(GPSEffectFamily.NO_EFFECT)
        neutral_ph = EffectFamilyPrior(
            GPSEffectFamily.PROPORTIONAL_HAZARDS,
            hazard_ratio=UniformPriorRange(1.0, 1.0, log_scale=True),
        )
        first = GPSEffectScenarioSampler(no_effect)(
            entries, np.random.default_rng(345)
        )
        second = GPSEffectScenarioSampler(neutral_ph)(
            entries, np.random.default_rng(345)
        )
        np.testing.assert_array_equal(first.treatment, second.treatment)
        np.testing.assert_array_equal(first.strata, second.strata)
        times = np.full(64, 500.0)
        np.testing.assert_array_equal(
            first.event_time_model.cdf(times), second.event_time_model.cdf(times)
        )

    def test_unit_extra_cure_probability_cures_every_treated_patient(self):
        prior = EffectFamilyPrior(
            GPSEffectFamily.CURE_FRACTION_DIFFERENCE,
            extra_cure_probability=UniformPriorRange(1.0, 1.0),
        )
        patients = GPSEffectScenarioSampler(prior)(
            (date(2022, 1, 1),) * 80,
            np.random.default_rng(991),
        )
        self.assertTrue(
            np.all(patients.event_time_model.weibull_weight[patients.treatment] == 0.0)
        )


def conditioning_result(
    family,
    *,
    log_history=log(0.4),
    continuation=0.5,
    rejection=0.5,
    reached=1.0,
    design=None,
):
    design = TrialDecisionDesign(interim_events=2, final_events=4) if design is None else design
    return ConditioningResult(
        scenario_name=family.value,
        design=design,
        importance_draws=1000,
        history_compatible_draws=500,
        continuation_compatible_draws=250,
        interim_efficacy_draws=150,
        interim_futility_draws=0,
        non_estimable_interim_draws=100,
        final_rejection_draws=100,
        final_non_rejection_draws=100,
        final_not_reached_draws=50,
        log_p_public_history=log_history,
        p_continue_given_public_history=continuation,
        p_final_rejection_given_public_history_and_continuation=rejection,
        p_final_reached_given_public_history_and_continuation=reached,
        history_effective_sample_size=400.0,
        continuation_effective_sample_size=200.0,
        maximum_history_weight_share=0.01,
        valid_disclosure_lag_mass=1.0,
        proposal_interim_z_targets=(0.0,),
        tilt_attempts=1000,
        tilt_fallbacks=0,
        draws_with_tilt_fallback=0,
        proposal_infeasible_draws=0,
        mean_tilt_iterations=3.0,
        maximum_tilt_error=1e-10,
    )


def family_projections(rejections=None):
    priors = {item.family: item for item in DEFAULT_EFFECT_FAMILY_PRIORS}
    if rejections is None:
        rejections = {
            family: (index + 1) / 10.0
            for index, family in enumerate(GPSEffectFamily)
        }
    return tuple(
        EffectFamilyProjection(
            family,
            priors[family],
            conditioning_result(family, rejection=rejections[family]),
        )
        for family in GPSEffectFamily
    )


class PosteriorModelAverageTest(unittest.TestCase):
    def test_complete_futility_grid_reuses_family_history_draws(self):
        history = small_history()
        rows = condition_effect_families_futility_sensitivity_grid(
            thresholds=(None, 1.0),
            history=history,
            enrollment_model=small_enrollment_model(history),
            base_design=TrialDecisionDesign(interim_events=2, final_events=4),
            nsim=30,
            seed=77,
            proposal_interim_z_targets=(0.0,),
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), len(GPSEffectFamily))
        for disabled, threshold in zip(*rows):
            self.assertEqual(disabled.family, threshold.family)
            self.assertEqual(
                disabled.conditioning.log_p_public_history,
                threshold.conditioning.log_p_public_history,
            )
            self.assertGreaterEqual(
                disabled.conditioning.p_continue_given_public_history,
                threshold.conditioning.p_continue_given_public_history,
            )

    def test_equal_family_evidence_preserves_prior_weights(self):
        projections = family_projections()
        forecast = posterior_model_average(
            projections, BALANCED_MODEL_FAMILY_PRIOR
        )
        self.assertTrue(forecast.is_posterior_forecast)
        self.assertFalse(projections[0].conditioning.is_posterior_forecast)
        for family, prior_weight in BALANCED_MODEL_FAMILY_PRIOR.weights:
            self.assertAlmostEqual(
                forecast.model_posterior_weights[family], prior_weight
            )
        expected = sum(
            forecast.model_prior_weights[projection.family]
            * projection.conditioning.p_final_rejection_given_public_history_and_continuation
            for projection in projections
        )
        self.assertAlmostEqual(
            forecast.p_final_rejection_given_public_history_and_continuation,
            expected,
        )
        self.assertAlmostEqual(
            forecast.p_public_history_and_continuation, 0.2
        )

    def test_roundoff_above_one_is_clamped_but_material_overshoot_fails(self):
        projections = list(family_projections())
        projections[0] = replace(
            projections[0],
            conditioning=replace(
                projections[0].conditioning,
                p_final_reached_given_public_history_and_continuation=(
                    1.0 + 0.5e-12
                ),
            ),
        )
        forecast = posterior_model_average(projections)
        self.assertEqual(
            forecast.p_final_reached_given_public_history_and_continuation,
            1.0,
        )
        projections[0] = replace(
            projections[0],
            conditioning=replace(
                projections[0].conditioning,
                p_final_reached_given_public_history_and_continuation=(
                    1.0 + 2.0e-12
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "final-reach probability"):
            posterior_model_average(projections)

    def test_zero_finite_denominator_mass_remains_a_readiness_diagnostic(self):
        self.assertTrue(
            np.isnan(_conditional_probability([], [float("-inf")]))
        )
        self.assertTrue(
            np.isnan(
                _conditional_probability(
                    [float("-inf")], [float("-inf")]
                )
            )
        )
        self.assertEqual(
            _conditional_probability([float("-inf")], [0.0]), 0.0
        )

        projections = list(family_projections())
        first = projections[0]
        degenerate_conditioning = replace(
            first.conditioning,
            log_p_public_history=float("-inf"),
            p_continue_given_public_history=float("nan"),
            p_final_rejection_given_public_history_and_continuation=float("nan"),
            p_final_reached_given_public_history_and_continuation=float("nan"),
            history_effective_sample_size=float("nan"),
            continuation_effective_sample_size=float("nan"),
            maximum_history_weight_share=float("nan"),
        )
        projections[0] = replace(
            first,
            conditioning=degenerate_conditioning,
        )
        result = posterior_model_average(projections)
        self.assertFalse(result.is_posterior_forecast)
        issues = result.forecast_readiness_issues
        self.assertTrue(
            any("public-history log evidence is not finite" in item for item in issues)
        )
        self.assertTrue(
            any("non-finite conditional probabilities" in item for item in issues)
        )
        self.assertTrue(any("history ESS is not finite" in item for item in issues))
        self.assertTrue(
            any("continuation ESS is not finite" in item for item in issues)
        )
        self.assertTrue(
            any("maximum history weight share is not finite" in item for item in issues)
        )

        all_degenerate = tuple(
            replace(
                projection,
                conditioning=replace(
                    degenerate_conditioning,
                    scenario_name=projection.conditioning.scenario_name,
                    design=projection.conditioning.design,
                ),
            )
            for projection in projections
        )
        with self.assertRaisesRegex(ValueError, "zero history/continuation evidence"):
            posterior_model_average(all_degenerate)

    def test_forecast_label_requires_monte_carlo_readiness_in_every_family(self):
        projections = list(family_projections())
        first = projections[0]
        cases = (
            (
                "history_effective_sample_size",
                MINIMUM_POSTERIOR_FORECAST_ESS - 1.0,
                "history ESS",
            ),
            (
                "continuation_effective_sample_size",
                MINIMUM_POSTERIOR_FORECAST_ESS - 1.0,
                "continuation ESS",
            ),
            (
                "maximum_history_weight_share",
                MAXIMUM_POSTERIOR_FORECAST_HISTORY_WEIGHT_SHARE + 0.01,
                "maximum history weight share",
            ),
        )
        for field_name, value, expected_issue in cases:
            with self.subTest(field_name=field_name):
                changed = list(projections)
                changed[0] = replace(
                    first,
                    conditioning=replace(
                        first.conditioning,
                        **{field_name: value},
                    ),
                )
                result = posterior_model_average(changed)
                self.assertFalse(result.is_posterior_forecast)
                self.assertTrue(
                    any(
                        expected_issue in issue
                        for issue in result.forecast_readiness_issues
                    )
                )

    def test_family_evidence_updates_model_weights_by_bayes_rule(self):
        projections = list(family_projections())
        no_effect = projections[0]
        projections[0] = replace(
            no_effect,
            conditioning=replace(
                no_effect.conditioning,
                log_p_public_history=log(0.8),
                p_continue_given_public_history=0.8,
            ),
        )
        forecast = posterior_model_average(
            projections, BALANCED_MODEL_FAMILY_PRIOR
        )
        self.assertGreater(
            forecast.model_posterior_weights[GPSEffectFamily.NO_EFFECT],
            forecast.model_prior_weights[GPSEffectFamily.NO_EFFECT],
        )
        self.assertAlmostEqual(sum(forecast.model_posterior_weights.values()), 1.0)

    def test_named_prior_sensitivity_reuses_likelihoods_and_moves_forecast(self):
        rejections = {
            GPSEffectFamily.NO_EFFECT: 0.05,
            GPSEffectFamily.PROPORTIONAL_HAZARDS: 0.35,
            GPSEffectFamily.DELAYED_PROPORTIONAL_HAZARDS: 0.30,
            GPSEffectFamily.CURE_FRACTION_DIFFERENCE: 0.80,
            GPSEffectFamily.DELAYED_CURE: 0.75,
            GPSEffectFamily.WANING_PIECEWISE: 0.40,
            GPSEffectFamily.RESPONDER_CURE: 0.90,
        }
        rows = posterior_prior_sensitivity(family_projections(rejections))
        self.assertEqual(
            [row.sensitivity_name for row in rows],
            [item.name for item in DEFAULT_MODEL_FAMILY_PRIOR_SENSITIVITY],
        )
        by_name = {row.sensitivity_name: row for row in rows}
        self.assertLess(
            by_name[SKEPTICAL_MODEL_FAMILY_PRIOR.name].p_final_rejection_given_public_history_and_continuation,
            by_name[BALANCED_MODEL_FAMILY_PRIOR.name].p_final_rejection_given_public_history_and_continuation,
        )
        self.assertLess(
            by_name[BALANCED_MODEL_FAMILY_PRIOR.name].p_final_rejection_given_public_history_and_continuation,
            by_name[CURE_FAVORING_MODEL_FAMILY_PRIOR.name].p_final_rejection_given_public_history_and_continuation,
        )

    def test_partial_or_mismatched_family_sets_cannot_claim_forecast_status(self):
        projections = family_projections()
        with self.assertRaisesRegex(ValueError, "every required"):
            posterior_model_average(projections[:-1])
        changed = list(projections)
        changed[-1] = replace(
            changed[-1],
            conditioning=replace(
                changed[-1].conditioning,
                design=TrialDecisionDesign(interim_events=3, final_events=5),
            ),
        )
        with self.assertRaisesRegex(ValueError, "same trial design"):
            posterior_model_average(changed)
        impossible = list(projections)
        impossible[0] = replace(
            impossible[0],
            conditioning=replace(
                impossible[0].conditioning,
                p_final_rejection_given_public_history_and_continuation=0.8,
                p_final_reached_given_public_history_and_continuation=0.7,
            ),
        )
        with self.assertRaisesRegex(ValueError, "cannot exceed final reach"):
            posterior_model_average(impossible)
        with self.assertRaisesRegex(ValueError, "positive prior mass"):
            ModelFamilyWeightPrior(
                "bad",
                tuple(
                    (family, 0.0 if index == 0 else 1.0 / 6.0)
                    for index, family in enumerate(GPSEffectFamily)
                ),
            )


if __name__ == "__main__":
    unittest.main()
