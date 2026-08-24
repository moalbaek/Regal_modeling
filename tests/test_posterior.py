"""Tests for WP6 latent-history and interim-continuation conditioning."""

from datetime import date
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import posterior as posterior_module  # noqa: E402
from event_likelihood import (  # noqa: E402
    CountConstraint,
    CountObservation,
    ObservationType,
    PiecewiseEnrollmentModel,
    PublicHistory,
    PublicHistoryLikelihood,
    ReportingLag,
    SourceRecord,
    load_regal_public_history,
)
from posterior import (  # noqa: E402
    ScenarioPatients,
    TiltProposalError,
    WeibullEventTimeModel,
    condition_futility_sensitivity_grid,
    condition_on_public_history,
    draw_history_importance_sample,
    exponential_tilt_event_intervals,
    public_history_constraint_branches,
)
from trial_design import TrialDecisionDesign  # noqa: E402


SOURCE = SourceRecord(
    "Synthetic WP6 fixture",
    "https://example.com/wp6",
    date(2026, 1, 1),
)
FIXED_LAG = ReportingLag("fixed", (0,), (1.0,))


def count_observation(identifier, cutoff, kind, count, lower, upper):
    return CountObservation(
        observation_id=identifier,
        observation_date=cutoff,
        announcement_date=(
            None
            if kind is ObservationType.THRESHOLD_NOT_ANNOUNCED
            else cutoff
        ),
        observation_type=kind,
        count=count,
        count_lower=lower,
        count_upper=upper,
        reporting_lag=FIXED_LAG,
        source=SOURCE,
        notes="Synthetic count used only to validate the conditioning engine.",
        accrual_anchor=identifier.startswith("enrollment"),
    )


def small_history():
    return PublicHistory(
        schema_version=1,
        registry_id="SYNTHETIC-WP6",
        study_start=date(2021, 1, 1),
        target_enrollment=8,
        interim_event_threshold=2,
        final_event_threshold=4,
        enrollment_observations=(
            count_observation(
                "enrollment_first_2",
                date(2021, 1, 2),
                ObservationType.THRESHOLD_REACHED_BY,
                2,
                2,
                8,
            ),
            count_observation(
                "enrollment_complete",
                date(2021, 1, 8),
                ObservationType.EXACT_AS_OF,
                8,
                8,
                8,
            ),
        ),
        event_observations=(
            count_observation(
                "interim_2",
                date(2021, 1, 20),
                ObservationType.THRESHOLD_HIT,
                2,
                2,
                2,
            ),
            count_observation(
                "events_3",
                date(2021, 1, 30),
                ObservationType.EXACT_AS_OF,
                3,
                3,
                3,
            ),
            count_observation(
                "event_4_not_announced",
                date(2021, 2, 5),
                ObservationType.THRESHOLD_NOT_ANNOUNCED,
                4,
                0,
                3,
            ),
        ),
    )


def small_enrollment_model(history=None):
    history = small_history() if history is None else history
    return PiecewiseEnrollmentModel(
        total_enrollment=history.target_enrollment,
        study_start=history.study_start,
        phase_end_dates=(date(2021, 1, 2), date(2021, 1, 8)),
        phase_probabilities=(0.25, 0.75),
    )


def small_scenario_sampler(entry_dates, rng):
    treatment = np.zeros(8, dtype=bool)
    strata = np.repeat(np.arange(2), 4)
    for stratum in range(2):
        indices = np.flatnonzero(strata == stratum)
        treatment[rng.permutation(indices)[:2]] = True
    scale = np.where(treatment, 32.0, 25.0)
    censoring = np.full(8, np.inf)
    censoring[0] = 60.0
    return ScenarioPatients(
        treatment=treatment,
        strata=strata,
        censoring_time=censoring,
        event_time_model=WeibullEventTimeModel(
            scale,
            np.ones(8),
            np.ones(8),
        ),
    )


class FixedEnrollmentModel(PiecewiseEnrollmentModel):
    fixed_dates = ()

    def sample_enrollment_dates(self, rng):
        return tuple(self.fixed_dates)


def fixed_model(entry_dates, history=None):
    history = small_history() if history is None else history
    model = FixedEnrollmentModel(
        history.target_enrollment,
        history.study_start,
        (date(2021, 1, 8),),
        (1.0,),
    )
    object.__setattr__(model, "fixed_dates", tuple(entry_dates))
    return model


def fixed_scenario(entry_dates, rng):
    treatment = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
    return ScenarioPatients(
        treatment,
        np.repeat(np.arange(2), 4),
        np.full(8, np.inf),
        WeibullEventTimeModel(
            np.where(treatment, 32.0, 25.0),
            np.ones(8),
            np.ones(8),
        ),
    )


class EventTimeContractTest(unittest.TestCase):
    def test_weibull_cdf_ppf_round_trip_and_defective_mass(self):
        model = WeibullEventTimeModel(
            [10.0, 20.0, 30.0],
            [1.0, 1.5, 0.8],
            [1.0, 0.8, 0.0],
        )
        probabilities = np.array([0.25, 0.40, 0.10])
        times = model.ppf(probabilities)
        self.assertTrue(np.isfinite(times[0]))
        self.assertTrue(np.isfinite(times[1]))
        self.assertTrue(np.isinf(times[2]))
        np.testing.assert_allclose(
            model.cdf(times)[:2], probabilities[:2], rtol=0.0, atol=1e-14
        )
        with self.assertRaises(ValueError):
            model.scale_time[0] = 4.0

    def test_scenario_contract_keeps_censoring_separate_and_immutable(self):
        patients = fixed_scenario((), np.random.default_rng(1))
        self.assertEqual(patients.patient_count, 8)
        self.assertTrue(np.all(np.isinf(patients.censoring_time)))
        with self.assertRaises(ValueError):
            patients.treatment[0] = False
        with self.assertRaisesRegex(ValueError, "one distribution per patient"):
            ScenarioPatients(
                np.zeros(8),
                np.zeros(8),
                np.full(8, np.inf),
                WeibullEventTimeModel(
                    np.ones(7), np.ones(7), np.ones(7)
                ),
            )


class HistoryBranchTest(unittest.TestCase):
    def test_actual_regal_history_has_nine_explicit_lag_branches(self):
        history = load_regal_public_history()
        branches = public_history_constraint_branches(history)
        self.assertEqual(len(branches), 9)
        self.assertAlmostEqual(sum(item.probability for item in branches), 1.0)
        for branch in branches:
            self.assertEqual(
                [(item.lower, item.upper) for item in branch.event_constraints],
                [(60, 60), (72, 72), (78, 78), (78, 79)],
            )

    def test_tilt_hits_count_and_continuation_proxy_moments(self):
        history = small_history()
        constraints = public_history_constraint_branches(history)[0].event_constraints
        cumulative = np.array(
            [
                [0.10, 0.30, 0.45],
                [0.15, 0.35, 0.50],
                [0.20, 0.40, 0.55],
                [0.25, 0.45, 0.60],
                [0.12, 0.32, 0.47],
                [0.18, 0.38, 0.53],
                [0.22, 0.42, 0.57],
                [0.28, 0.48, 0.63],
            ]
        )
        treatment = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=bool)
        neutral = exponential_tilt_event_intervals(
            cumulative,
            constraints,
            treatment,
            history=history,
            target_interim_z=0.0,
        )
        favorable = exponential_tilt_event_intervals(
            cumulative,
            constraints,
            treatment,
            history=history,
            target_interim_z=1.0,
        )
        category_means = neutral.probabilities.sum(axis=0)
        np.testing.assert_allclose(
            np.cumsum(category_means[:-1]),
            neutral.target_cumulative_counts,
            rtol=0.0,
            atol=1e-8,
        )
        # The exact 3 -> 3 right-censor interval is forced to zero.
        np.testing.assert_array_equal(neutral.probabilities[:, 2], 0.0)
        neutral_treated_early = neutral.probabilities[treatment, :1].sum()
        favorable_treated_early = favorable.probabilities[treatment, :1].sum()
        self.assertAlmostEqual(neutral_treated_early, 1.0, places=8)
        self.assertLess(favorable_treated_early, neutral_treated_early)


class ProposalRobustnessTest(unittest.TestCase):
    def test_tilt_support_uses_exact_positivity_not_numeric_tolerance(self):
        history = small_history()
        constraints = public_history_constraint_branches(history)[0].event_constraints
        intervals = np.tile(np.array([0.25, 0.125, 0.0, 0.625]), (8, 1))
        tiny = 1e-14
        intervals[0] = [tiny, tiny, 1.0 - 3.0 * tiny, tiny]
        _, _, positive_categories, _ = posterior_module._feature_tensor(
            intervals,
            np.array([2.0, 3.0, 3.0]),
            constraints,
            np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=bool),
            0,
            0.0,
        )
        self.assertEqual(tuple(positive_categories), (0, 1, 3))
        self.assertGreater(intervals[0, positive_categories].sum(), 0.0)
        self.assertLess(
            intervals[0, positive_categories].sum(),
            posterior_module.PROBABILITY_TOLERANCE,
        )

    def test_range_targets_preserve_every_possible_positive_increment(self):
        constraints = (
            CountConstraint(date(2021, 1, 1), 4, 6, "first range"),
            CountConstraint(date(2021, 1, 2), 5, 6, "second range"),
        )
        cumulative = np.tile(np.array([0.60, 0.62]), (10, 1))
        target = posterior_module._target_cumulative_counts(
            cumulative, constraints
        )
        increments = np.diff(np.concatenate(([0.0], target, [10.0])))
        self.assertTrue(4.0 <= target[0] <= 6.0)
        self.assertTrue(5.0 <= target[1] <= 6.0)
        self.assertGreater(increments[1], 0.0)
        self.assertGreaterEqual(
            increments[1], posterior_module.TARGET_CATEGORY_MARGIN
        )

    def test_quota_dps_apply_one_logical_cell_budget(self):
        probabilities = np.array(
            [
                [0.2, 0.8],
                [0.5, 0.5],
                [0.7, 0.3],
            ]
        )
        quotas = np.array([1, 2])
        for function in (
            posterior_module._quota_log_probability,
            posterior_module._quota_suffix_table,
        ):
            with self.assertRaisesRegex(ValueError, "8 logical DP cells"):
                function(probabilities, quotas, max_states=7)
            function(probabilities, quotas, max_states=8)

    def test_positive_quota_mass_underflow_is_not_structural_infeasibility(self):
        tiny = np.nextafter(0.0, 1.0)
        probabilities = np.array(
            [
                [tiny, 1.0],
                [tiny, 1.0],
            ]
        )
        quotas = np.array([2, 0])
        for function in (
            posterior_module._quota_log_probability,
            posterior_module._quota_suffix_table,
        ):
            with self.assertRaisesRegex(FloatingPointError, "underflowed"):
                function(probabilities, quotas, max_states=9)

    def test_real_newton_non_convergence_raises_public_tilt_error(self):
        history = small_history()
        constraints = public_history_constraint_branches(history)[0].event_constraints
        cumulative = np.array(
            [
                [0.10, 0.30, 0.45],
                [0.15, 0.35, 0.50],
                [0.20, 0.40, 0.55],
                [0.25, 0.45, 0.60],
                [0.12, 0.32, 0.47],
                [0.18, 0.38, 0.53],
                [0.22, 0.42, 0.57],
                [0.28, 0.48, 0.63],
            ]
        )
        treatment = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=bool)
        with self.assertRaisesRegex(TiltProposalError, "did not converge"):
            exponential_tilt_event_intervals(
                cumulative,
                constraints,
                treatment,
                history=history,
                target_interim_z=0.0,
                tolerance=1e-14,
                max_iterations=1,
            )
        self.assertIn("TiltProposalError", posterior_module.__all__)


class LatentHistoryConditioningTest(unittest.TestCase):
    def test_exact_quota_draws_always_satisfy_event_history_and_right_censor(self):
        history = small_history()
        model = small_enrollment_model(history)
        branches = public_history_constraint_branches(history)
        rng = np.random.default_rng(7)
        for _ in range(30):
            draw = draw_history_importance_sample(
                history,
                model,
                small_scenario_sampler,
                branches,
                (0.0,),
                rng,
            )
            self.assertTrue(draw.event_compatible)
            self.assertEqual(draw.event_counts, (2, 3, 3))
            self.assertTrue(np.isfinite(draw.log_importance_weight))

    def test_base_proposal_weight_matches_wp5_exact_event_likelihood(self):
        history = small_history()
        entry_dates = (
            date(2021, 1, 1),
            date(2021, 1, 2),
            date(2021, 1, 3),
            date(2021, 1, 4),
            date(2021, 1, 5),
            date(2021, 1, 6),
            date(2021, 1, 7),
            date(2021, 1, 8),
        )
        model = fixed_model(entry_dates, history)
        patients = fixed_scenario(entry_dates, np.random.default_rng(1))
        origin = history.study_start
        entry_time = np.asarray([(value - origin).days for value in entry_dates])

        def provider(cutoff):
            available = (cutoff - origin).days - entry_time
            active = available >= 0.0
            followup = np.maximum(available, 0.0)
            values = patients.event_time_model.cdf(followup)
            return np.where(active, values, 0.0)

        expected = PublicHistoryLikelihood(
            history, small_enrollment_model(history)
        ).event_log_likelihood(provider)
        draw = draw_history_importance_sample(
            history,
            model,
            fixed_scenario,
            public_history_constraint_branches(history),
            (),
            np.random.default_rng(99),
        )
        self.assertTrue(draw.public_history_compatible)
        # This fixture has one allowed event-count vector and one lag branch, so
        # the target/base-conditional importance ratio is its exact WP5 mass.
        self.assertAlmostEqual(draw.log_importance_weight, expected, places=11)

    def test_enrollment_and_event_evidence_use_the_same_latent_history(self):
        history = small_history()
        late_entries = (date(2021, 1, 8),) * 8
        draw = draw_history_importance_sample(
            history,
            fixed_model(late_entries, history),
            fixed_scenario,
            public_history_constraint_branches(history),
            (),
            np.random.default_rng(4),
        )
        self.assertFalse(draw.enrollment_compatible)
        self.assertTrue(draw.event_compatible)
        self.assertFalse(draw.public_history_compatible)

    def test_impossible_lag_branch_contributes_zero_weight_not_selection_bias(self):
        history = small_history()

        def no_event_scenario(entry_dates, rng):
            return ScenarioPatients(
                np.asarray([0, 1, 0, 1, 0, 1, 0, 1]),
                np.zeros(8),
                np.full(8, np.inf),
                WeibullEventTimeModel(
                    np.ones(8),
                    np.ones(8),
                    np.zeros(8),
                ),
            )

        draw = draw_history_importance_sample(
            history,
            small_enrollment_model(history),
            no_event_scenario,
            public_history_constraint_branches(history),
            (0.0,),
            np.random.default_rng(12),
        )
        self.assertEqual(draw.log_importance_weight, float("-inf"))
        self.assertFalse(draw.event_compatible)
        self.assertFalse(draw.public_history_compatible)
        self.assertFalse(draw.proposal_infeasible)

    def test_failed_tilt_falls_back_to_exact_base_proposal_and_is_reported(self):
        history = small_history()
        arguments = dict(
            scenario_name="forced tilt fallback",
            history=history,
            enrollment_model=small_enrollment_model(history),
            design=TrialDecisionDesign(interim_events=2, final_events=4),
            nsim=20,
            seed=19,
        )
        base = condition_on_public_history(
            small_scenario_sampler,
            proposal_interim_z_targets=(),
            **arguments,
        )
        with patch.object(
            posterior_module,
            "exponential_tilt_event_intervals",
            side_effect=TiltProposalError(
                "forced non-convergence"
            ),
        ):
            fallback = condition_on_public_history(
                small_scenario_sampler,
                proposal_interim_z_targets=(0.0,),
                **arguments,
            )
        self.assertEqual(fallback.tilt_attempts, 20)
        self.assertEqual(fallback.tilt_fallbacks, 20)
        self.assertEqual(fallback.draws_with_tilt_fallback, 20)
        self.assertEqual(fallback.tilt_fallback_rate, 1.0)
        self.assertIsNone(fallback.mean_tilt_iterations)
        self.assertIsNone(fallback.maximum_tilt_error)
        self.assertEqual(
            fallback.history_compatible_draws,
            base.history_compatible_draws,
        )
        self.assertEqual(fallback.log_p_public_history, base.log_p_public_history)

    def test_draw_reports_no_error_when_every_tilt_falls_back(self):
        history = small_history()
        with patch.object(
            posterior_module,
            "exponential_tilt_event_intervals",
            side_effect=TiltProposalError("forced non-convergence"),
        ):
            draw = draw_history_importance_sample(
                history,
                small_enrollment_model(history),
                small_scenario_sampler,
                public_history_constraint_branches(history),
                (0.0,),
                np.random.default_rng(19),
            )
        self.assertEqual(draw.tilt_attempts, 1)
        self.assertEqual(draw.tilt_fallbacks, 1)
        self.assertEqual(draw.tilt_iterations, ())
        self.assertIsNone(draw.maximum_tilt_error)

    def test_partial_tilt_fallback_renormalizes_the_realized_mixture(self):
        history = small_history()
        arguments = dict(
            scenario_name="partial tilt fallback",
            history=history,
            enrollment_model=small_enrollment_model(history),
            design=TrialDecisionDesign(interim_events=2, final_events=4),
            nsim=20,
            seed=19,
        )
        retained_only = condition_on_public_history(
            small_scenario_sampler,
            proposal_interim_z_targets=(-0.5,),
            **arguments,
        )
        real_tilt = posterior_module.exponential_tilt_event_intervals

        def fail_second_target(*args, **kwargs):
            if kwargs["target_interim_z"] == 0.75:
                raise TiltProposalError("forced partial fallback")
            return real_tilt(*args, **kwargs)

        with patch.object(
            posterior_module,
            "exponential_tilt_event_intervals",
            side_effect=fail_second_target,
        ):
            partial = condition_on_public_history(
                small_scenario_sampler,
                proposal_interim_z_targets=(-0.5, 0.75),
                **arguments,
            )
        self.assertEqual(partial.tilt_attempts, 40)
        self.assertEqual(partial.tilt_fallbacks, 20)
        self.assertEqual(partial.draws_with_tilt_fallback, 20)
        self.assertEqual(
            partial.log_p_public_history,
            retained_only.log_p_public_history,
        )
        self.assertEqual(
            partial.history_effective_sample_size,
            retained_only.history_effective_sample_size,
        )

    def test_real_newton_non_convergence_falls_back_without_aborting_run(self):
        history = small_history()
        arguments = dict(
            scenario_name="real Newton fallback",
            history=history,
            enrollment_model=small_enrollment_model(history),
            design=TrialDecisionDesign(interim_events=2, final_events=4),
            nsim=10,
            seed=23,
        )
        base = condition_on_public_history(
            small_scenario_sampler,
            proposal_interim_z_targets=(),
            **arguments,
        )
        fallback = condition_on_public_history(
            small_scenario_sampler,
            proposal_interim_z_targets=(0.0,),
            tilt_tolerance=1e-14,
            max_tilt_iterations=1,
            **arguments,
        )
        self.assertEqual(fallback.tilt_attempts, 10)
        self.assertEqual(fallback.tilt_fallbacks, 10)
        self.assertIsNone(fallback.mean_tilt_iterations)
        self.assertIsNone(fallback.maximum_tilt_error)
        self.assertEqual(fallback.log_p_public_history, base.log_p_public_history)
        self.assertEqual(
            fallback.history_effective_sample_size,
            base.history_effective_sample_size,
        )

    def test_zero_mass_component_preserves_the_full_proposal_law(self):
        history = small_history()
        entry_dates = (
            date(2021, 1, 1),
            date(2021, 1, 2),
            date(2021, 1, 3),
            date(2021, 1, 4),
            date(2021, 1, 5),
            date(2021, 1, 6),
            date(2021, 1, 7),
            date(2021, 1, 8),
        )
        arguments = (
            history,
            fixed_model(entry_dates, history),
            fixed_scenario,
            public_history_constraint_branches(history),
        )
        base = draw_history_importance_sample(
            *arguments,
            (),
            np.random.default_rng(0),
        )
        impossible_tilt = posterior_module.ExponentialTilt(
            np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (8, 1)),
            iterations=1,
            maximum_moment_error=0.0,
            target_cumulative_counts=np.array([2.0, 3.0, 3.0]),
            target_interim_z=0.0,
        )
        with patch.object(
            posterior_module,
            "exponential_tilt_event_intervals",
            return_value=impossible_tilt,
        ):
            surviving = draw_history_importance_sample(
                *arguments,
                (0.0,),
                np.random.default_rng(0),
            )
            failed = draw_history_importance_sample(
                *arguments,
                (0.0,),
                np.random.default_rng(1),
            )
        self.assertEqual(surviving.proposal_component, 0)
        self.assertFalse(surviving.proposal_infeasible)
        self.assertAlmostEqual(
            surviving.log_importance_weight,
            base.log_importance_weight + np.log(2.0),
            places=12,
        )
        self.assertEqual(failed.proposal_component, 1)
        self.assertTrue(failed.proposal_infeasible)
        self.assertEqual(failed.log_importance_weight, float("-inf"))
        self.assertFalse(failed.event_compatible)
        self.assertEqual(failed.tilt_attempts, 1)
        self.assertEqual(failed.tilt_fallbacks, 0)
        self.assertEqual(failed.tilt_iterations, (1,))
        self.assertEqual(failed.maximum_tilt_error, 0.0)
        # Averaging the successful and zero-weight component selections gives
        # exactly the same contribution as the surviving component alone.
        paired_log_mean = posterior_module._logsumexp(
            (surviving.log_importance_weight, failed.log_importance_weight)
        ) - np.log(2.0)
        self.assertAlmostEqual(
            paired_log_mean,
            base.log_importance_weight,
            places=12,
        )
        with patch.object(
            posterior_module,
            "draw_history_importance_sample",
            side_effect=(surviving, failed),
        ):
            result = condition_on_public_history(
                fixed_scenario,
                scenario_name="proposal infeasibility diagnostic",
                history=history,
                enrollment_model=fixed_model(entry_dates, history),
                design=TrialDecisionDesign(interim_events=2, final_events=4),
                nsim=2,
                seed=0,
                proposal_interim_z_targets=(0.0,),
            )
        self.assertEqual(result.proposal_infeasible_draws, 1)

    def test_selected_positive_mass_underflow_aborts_the_public_draw(self):
        history = small_history()
        entry_dates = tuple(
            date(2021, 1, day) for day in range(1, 9)
        )
        tiny = np.nextafter(0.0, 1.0)
        underflowing_tilt = posterior_module.ExponentialTilt(
            np.tile(np.array([tiny, tiny, 0.0, 1.0]), (8, 1)),
            iterations=1,
            maximum_moment_error=0.0,
            target_cumulative_counts=np.array([2.0, 3.0, 3.0]),
            target_interim_z=0.0,
        )
        with patch.object(
            posterior_module,
            "exponential_tilt_event_intervals",
            return_value=underflowing_tilt,
        ):
            with self.assertRaisesRegex(FloatingPointError, "underflowed"):
                draw_history_importance_sample(
                    history,
                    fixed_model(entry_dates, history),
                    fixed_scenario,
                    public_history_constraint_branches(history),
                    (0.0,),
                    np.random.default_rng(1),
                )

    def test_conditional_projection_conserves_every_weighted_branch(self):
        history = small_history()
        design = TrialDecisionDesign(interim_events=2, final_events=4)
        result = condition_on_public_history(
            small_scenario_sampler,
            scenario_name="synthetic fixed scenario",
            history=history,
            enrollment_model=small_enrollment_model(history),
            design=design,
            nsim=300,
            seed=20260824,
        )
        self.assertGreater(result.p_public_history, 0.0)
        self.assertLess(result.p_public_history, 1.0)
        self.assertGreater(result.history_compatible_draws, 100)
        self.assertEqual(result.proposal_infeasible_draws, 0)
        self.assertEqual(
            result.history_compatible_draws,
            result.continuation_compatible_draws
            + result.interim_efficacy_draws
            + result.interim_futility_draws
            + result.non_estimable_interim_draws,
        )
        self.assertEqual(
            result.continuation_compatible_draws,
            result.final_rejection_draws
            + result.final_non_rejection_draws
            + result.final_not_reached_draws,
        )
        self.assertGreater(result.history_effective_sample_size, 100.0)
        self.assertLess(result.maximum_history_weight_share, 0.05)
        self.assertTrue(0.0 <= result.p_continue_given_public_history <= 1.0)
        self.assertTrue(
            0.0
            <= result.p_final_rejection_given_public_history_and_continuation
            <= 1.0
        )
        self.assertFalse(result.is_posterior_forecast)

    def test_futility_grid_reuses_identical_history_draws(self):
        history = small_history()
        rows = condition_futility_sensitivity_grid(
            small_scenario_sampler,
            thresholds=(None, 1.0, 0.8),
            scenario_name="paired futility fixture",
            base_design=TrialDecisionDesign(interim_events=2, final_events=4),
            history=history,
            enrollment_model=small_enrollment_model(history),
            nsim=250,
            seed=17,
        )
        self.assertEqual(
            [row.assumed_futility_hr_threshold for row in rows],
            [None, 1.0, 0.8],
        )
        self.assertEqual(len({row.log_p_public_history for row in rows}), 1)
        self.assertEqual(len({row.history_compatible_draws for row in rows}), 1)
        self.assertGreaterEqual(
            rows[0].p_continue_given_public_history,
            rows[1].p_continue_given_public_history,
        )
        self.assertGreaterEqual(
            rows[1].p_continue_given_public_history,
            rows[2].p_continue_given_public_history,
        )

    def test_conditioning_inputs_reject_mismatched_designs_and_duplicate_grid(self):
        history = small_history()
        with self.assertRaisesRegex(ValueError, "event thresholds differ"):
            condition_on_public_history(
                small_scenario_sampler,
                scenario_name="bad design",
                history=history,
                enrollment_model=small_enrollment_model(history),
                design=TrialDecisionDesign(interim_events=3, final_events=5),
                nsim=1,
            )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            condition_futility_sensitivity_grid(
                small_scenario_sampler,
                thresholds=(None, None),
                scenario_name="duplicate grid",
                base_design=TrialDecisionDesign(interim_events=2, final_events=4),
                history=history,
                enrollment_model=small_enrollment_model(history),
                nsim=1,
            )


if __name__ == "__main__":
    unittest.main()
