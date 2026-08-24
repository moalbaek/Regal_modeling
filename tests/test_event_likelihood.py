"""Tests for the REGAL v2 public-history and joint count likelihood."""

from dataclasses import replace
from datetime import date, timedelta
from itertools import product
from math import exp, isfinite, log
import os
import sys
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from event_likelihood import (  # noqa: E402
    CalendarEventProbabilityProvider,
    CountObservation,
    DEFAULT_MAX_DP_STATES,
    ObservationType,
    PiecewiseEnrollmentModel,
    PublicHistory,
    PublicHistoryLikelihood,
    ReportingLag,
    SourceRecord,
    default_regal_enrollment_model,
    enrollment_anchor_checks,
    enrollment_log_likelihood,
    joint_cumulative_count_log_probability,
    joint_cumulative_count_probability,
    load_regal_public_history,
    sample_event_increment_trajectory,
)


def brute_force_count_probability(cumulative, lower, upper):
    """Enumerate every patient interval assignment for a tiny fixture."""

    cumulative = np.asarray(cumulative, dtype=float)
    interval_probabilities = np.column_stack(
        (
            cumulative[:, 0],
            np.diff(cumulative, axis=1),
            1.0 - cumulative[:, -1],
        )
    )
    cutoff_count = cumulative.shape[1]
    probability = 0.0
    for assignments in product(range(cutoff_count + 1), repeat=len(cumulative)):
        mass = 1.0
        increments = np.zeros(cutoff_count, dtype=int)
        for patient, category in enumerate(assignments):
            mass *= interval_probabilities[patient, category]
            if category < cutoff_count:
                increments[category] += 1
        counts = np.cumsum(increments)
        if np.all(counts >= lower) and np.all(counts <= upper):
            probability += mass
    return probability


class PublicHistoryDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = load_regal_public_history()

    def test_registry_start_and_fixed_trial_counts_are_versioned(self):
        self.assertEqual(self.history.schema_version, 1)
        self.assertEqual(self.history.registry_id, "NCT04229979")
        self.assertEqual(self.history.study_start, date(2021, 2, 8))
        self.assertEqual(self.history.target_enrollment, 126)
        self.assertEqual(self.history.interim_event_threshold, 60)
        self.assertEqual(self.history.final_event_threshold, 80)
        with self.assertRaisesRegex(ValueError, "unsupported.*schema"):
            replace(self.history, schema_version=2)

    def test_event_disclosures_preserve_distinct_observation_semantics(self):
        by_count = {item.count: item for item in self.history.event_observations}
        self.assertIs(by_count[60].observation_type, ObservationType.THRESHOLD_HIT)
        self.assertIsNone(by_count[60].observation_date)
        self.assertEqual(by_count[60].announcement_date, date(2024, 12, 10))
        expected_lag_days = (0, 7, 14)
        expected_lag_probabilities = (4 / 21, 13 / 21, 4 / 21)
        self.assertEqual(by_count[60].reporting_lag.days, expected_lag_days)
        self.assertEqual(
            [choice[0].cutoff_date for choice in by_count[60].cutoff_choices()],
            [
                date(2024, 12, 10) - timedelta(days=lag)
                for lag in expected_lag_days
            ],
        )
        self.assertIs(by_count[72].observation_type, ObservationType.EXACT_AS_OF)
        self.assertEqual(by_count[72].announcement_date, date(2025, 12, 29))
        self.assertEqual(by_count[72].reporting_lag.days, (3,))
        self.assertIs(by_count[78].observation_type, ObservationType.EXACT_AS_OF)
        self.assertEqual(by_count[78].reporting_lag.days, (1,))
        with self.assertRaisesRegex(ValueError, "must match"):
            replace(
                by_count[72],
                reporting_lag=ReportingLag("fixed", (2,), (1.0,)),
            )

        right_censor = by_count[80]
        self.assertIs(
            right_censor.observation_type,
            ObservationType.THRESHOLD_NOT_ANNOUNCED,
        )
        self.assertEqual(right_censor.observation_date, date(2026, 8, 11))
        self.assertIsNone(right_censor.announcement_date)
        self.assertEqual(right_censor.count_upper, 79)
        self.assertEqual(right_censor.reporting_lag.days, expected_lag_days)
        for observation in (by_count[60], right_censor):
            with self.subTest(observation=observation.observation_id):
                for actual, expected in zip(
                    observation.reporting_lag.probabilities,
                    expected_lag_probabilities,
                ):
                    self.assertAlmostEqual(actual, expected)
                mean = sum(
                    lag * probability
                    for lag, probability in observation.reporting_lag.choices
                )
                variance = sum(
                    probability * (lag - mean) ** 2
                    for lag, probability in observation.reporting_lag.choices
                )
                self.assertAlmostEqual(mean, 7.0)
                self.assertAlmostEqual(variance, 56 / 3)
                self.assertIn("independent draws", observation.notes)
        self.assertEqual(
            len(by_count[60].cutoff_choices())
            * len(right_censor.cutoff_choices()),
            9,
        )

    def test_undated_threshold_contributes_definite_calendar_bounds(self):
        threshold = next(
            item
            for item in self.history.event_observations
            if item.observation_id == "interim_60_threshold"
        )
        fixed = ReportingLag("fixed", (0,), (1.0,))
        later_lower_count = CountObservation(
            "contradictory_later_count",
            date(2025, 1, 15),
            date(2025, 1, 15),
            ObservationType.EXACT_AS_OF,
            55,
            55,
            55,
            fixed,
            threshold.source,
            "Synthetic count below a threshold already reached.",
        )
        earlier_higher_count = CountObservation(
            "contradictory_earlier_count",
            date(2024, 11, 20),
            date(2024, 11, 20),
            ObservationType.EXACT_AS_OF,
            61,
            61,
            61,
            fixed,
            threshold.source,
            "Synthetic count above a threshold before its earliest possible date.",
        )
        for contradictory in (later_lower_count, earlier_higher_count):
            with self.subTest(observation=contradictory.observation_id):
                with self.assertRaisesRegex(ValueError, "inconsistent over time"):
                    replace(
                        self.history,
                        event_observations=(
                            self.history.event_observations + (contradictory,)
                        ),
                    )

    def test_every_observation_carries_source_lag_and_notes(self):
        for observation in (
            self.history.enrollment_observations + self.history.event_observations
        ):
            with self.subTest(observation=observation.observation_id):
                self.assertTrue(observation.source.title)
                self.assertTrue(observation.source.url.startswith("https://"))
                self.assertTrue(observation.notes)
                self.assertTrue(observation.reporting_lag.choices)

    def test_november_anchor_is_explicitly_a_projection_not_likelihood_data(self):
        projection = next(
            item
            for item in self.history.enrollment_observations
            if item.observation_id == "ex_china_enrollment_projection"
        )
        self.assertIs(
            projection.observation_type,
            ObservationType.PROJECTED_COUNT_INTERVAL,
        )
        self.assertEqual((projection.count_lower, projection.count_upper), (101, 106))
        self.assertEqual(projection.count, 104)
        self.assertFalse(projection.use_in_likelihood)
        self.assertTrue(projection.accrual_anchor)

    def test_first_20_is_a_by_date_threshold_not_an_exact_month_end_count(self):
        first = next(
            item
            for item in self.history.enrollment_observations
            if item.observation_id == "first_20_before_protocol_v3"
        )
        self.assertIs(
            first.observation_type, ObservationType.THRESHOLD_REACHED_BY
        )
        self.assertEqual(first.count_lower, 20)
        self.assertEqual(first.count_upper, 126)


class JointCountLikelihoodTest(unittest.TestCase):
    def test_joint_probability_matches_brute_force_with_heterogeneous_patients(self):
        cumulative = np.array(
            [
                [0.10, 0.40],
                [0.30, 0.80],
                [0.60, 0.90],
            ]
        )
        lower = (1, 2)
        upper = (2, 2)
        expected = brute_force_count_probability(cumulative, lower, upper)
        actual = joint_cumulative_count_probability(cumulative, lower, upper)
        self.assertAlmostEqual(actual, expected, places=14)
        self.assertAlmostEqual(
            joint_cumulative_count_log_probability(cumulative, lower, upper),
            log(expected),
            places=14,
        )

    def test_cumulative_counts_are_not_multiplied_as_independent_marginals(self):
        cumulative = np.tile(np.array([0.20, 0.50]), (2, 1))
        joint = joint_cumulative_count_probability(cumulative, (1, 1), (1, 1))
        independent_marginals = (2 * 0.20 * 0.80) * (2 * 0.50 * 0.50)
        self.assertAlmostEqual(joint, 0.20)
        self.assertAlmostEqual(independent_marginals, 0.16)
        self.assertNotAlmostEqual(joint, independent_marginals)

    def test_impossible_or_invalid_count_contracts_fail_cleanly(self):
        cumulative = np.tile(np.array([0.25, 0.50]), (3, 1))
        self.assertEqual(
            joint_cumulative_count_log_probability(cumulative, (2, 1), (2, 1)),
            float("-inf"),
        )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            joint_cumulative_count_probability(cumulative, (0, 0), (4, 4))
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            joint_cumulative_count_probability(
                np.array([[0.5, 0.4], [0.2, 0.3]]), (0, 0), (2, 2)
            )
        with self.assertRaisesRegex(ValueError, "DP states"):
            joint_cumulative_count_probability(
                cumulative,
                (0, 0),
                (3, 3),
                max_states=3,
            )
        with self.assertRaisesRegex(ValueError, "integer"):
            joint_cumulative_count_probability(cumulative, (False, 0), (2, 2))

    def test_default_state_cap_clears_regal_three_cutoff_boundary(self):
        natural_boundary = 127**3
        self.assertEqual(DEFAULT_MAX_DP_STATES, 4_000_000)
        self.assertGreater(DEFAULT_MAX_DP_STATES, natural_boundary)
        with self.assertRaisesRegex(ValueError, "2,048,383 DP states"):
            joint_cumulative_count_log_probability(
                np.zeros((126, 3)),
                (0, 0, 0),
                (126, 126, 126),
                max_states=2_000_000,
            )

    def test_latent_trajectory_sampler_keeps_integer_increments(self):
        cumulative = np.tile(np.array([0.20, 0.70]), (3, 1))

        class ControlledRng:
            @staticmethod
            def random(size):
                if size != 3:
                    raise AssertionError("unexpected draw size")
                return np.array([0.05, 0.45, 0.95])

        trajectory = sample_event_increment_trajectory(cumulative, ControlledRng())
        np.testing.assert_array_equal(trajectory.increments, [1, 1])
        np.testing.assert_array_equal(trajectory.cumulative_counts, [1, 2])
        with self.assertRaises(ValueError):
            trajectory.increments[0] = 2


class EnrollmentModelTest(unittest.TestCase):
    def setUp(self):
        self.history = load_regal_public_history()
        self.model = default_regal_enrollment_model(self.history)

    def test_default_reference_is_centered_on_every_published_anchor(self):
        checks = enrollment_anchor_checks(self.history, self.model)
        self.assertEqual(
            [item.point_count for item in checks],
            [20, 104, 126],
        )
        self.assertTrue(all(item.reachable for item in checks))
        self.assertTrue(all(item.centered for item in checks))
        self.assertEqual(
            [item.expected_count for item in checks],
            [20.0, 104.0, 126.0],
        )

    def test_sampled_accrual_cannot_create_preopening_patients(self):
        dates = self.model.sample_enrollment_dates(np.random.default_rng(1208))
        self.assertEqual(len(dates), 126)
        self.assertGreaterEqual(min(dates), date(2021, 2, 8))
        self.assertLessEqual(max(dates), date(2024, 4, 30))
        self.assertEqual(
            self.model.cumulative_probability(date(2021, 2, 7)),
            0.0,
        )
        self.assertEqual(
            self.model.cumulative_probability(self.model.enrollment_close),
            1.0,
        )

    def test_enrollment_likelihood_is_joint_fixed_n_and_finite(self):
        value = enrollment_log_likelihood(self.history, self.model)
        self.assertTrue(isfinite(value))
        self.assertAlmostEqual(value, -0.6207741877195306)

    def test_planning_projection_does_not_leak_into_observation_likelihood(self):
        baseline = enrollment_log_likelihood(self.history, self.model)
        projection = next(
            item for item in self.history.enrollment_observations if item.is_projection
        )
        changed_projection = replace(
            projection,
            count=90,
            count_lower=85,
            count_upper=95,
        )
        observations = tuple(
            changed_projection if item is projection else item
            for item in self.history.enrollment_observations
        )
        changed_history = replace(
            self.history, enrollment_observations=observations
        )
        self.assertEqual(
            enrollment_log_likelihood(changed_history, self.model), baseline
        )

    def test_enrollment_reporting_lag_is_mixed_outside_count_likelihood(self):
        first = next(
            item
            for item in self.history.enrollment_observations
            if item.observation_id == "first_20_before_protocol_v3"
        )
        announcement = date(2022, 5, 2)
        threshold = replace(
            first,
            observation_date=None,
            announcement_date=announcement,
            observation_type=ObservationType.THRESHOLD_HIT,
            count_upper=20,
            reporting_lag=ReportingLag(
                "discrete_pmf", (0, 2), (0.25, 0.75)
            ),
            notes="Synthetic enrollment threshold with uncertain reporting lag.",
            accrual_anchor=False,
        )
        observations = tuple(
            threshold if item is first else item
            for item in self.history.enrollment_observations
        )
        changed_history = replace(
            self.history, enrollment_observations=observations
        )
        completion = next(
            item
            for item in observations
            if item.observation_id == "enrollment_complete"
        )

        expected = 0.0
        for constraint, weight in threshold.cutoff_choices():
            matrix = self.model.cumulative_probability_matrix(
                (constraint.cutoff_date, completion.observation_date)
            )
            expected += weight * joint_cumulative_count_probability(
                matrix,
                (20, 126),
                (20, 126),
            )

        actual = enrollment_log_likelihood(changed_history, self.model)
        self.assertAlmostEqual(exp(actual), expected, places=14)
        with self.assertRaisesRegex(ValueError, "lag mixture"):
            enrollment_log_likelihood(
                changed_history,
                self.model,
                max_lag_combinations=1,
            )

    def test_enrollment_model_validates_probability_and_date_contracts(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            PiecewiseEnrollmentModel(
                10,
                date(2021, 1, 1),
                (date(2021, 2, 1), date(2021, 2, 1)),
                (0.5, 0.5),
            )
        with self.assertRaisesRegex(ValueError, "sum to one"):
            PiecewiseEnrollmentModel(
                10,
                date(2021, 1, 1),
                (date(2021, 2, 1), date(2021, 3, 1)),
                (0.5, 0.4),
            )


def synthetic_history_with_lag_mixture():
    source = SourceRecord(
        "Synthetic source", "https://example.com/source", date(2026, 1, 1)
    )
    fixed = ReportingLag("fixed", (0,), (1.0,))
    lag = ReportingLag("discrete_pmf", (0, 1), (0.25, 0.75))
    start = date(2021, 1, 1)
    first = date(2022, 1, 1)
    second = date(2023, 1, 1)
    as_of = date(2024, 1, 2)
    enrollment = CountObservation(
        "complete",
        start + timedelta(days=1),
        start + timedelta(days=1),
        ObservationType.EXACT_AS_OF,
        3,
        3,
        3,
        fixed,
        source,
        "Synthetic exact enrollment.",
        accrual_anchor=True,
    )
    events = (
        CountObservation(
            "one_event",
            first,
            first,
            ObservationType.EXACT_AS_OF,
            1,
            1,
            1,
            fixed,
            source,
            "Synthetic first count.",
        ),
        CountObservation(
            "two_events",
            second,
            second,
            ObservationType.EXACT_AS_OF,
            2,
            2,
            2,
            fixed,
            source,
            "Synthetic second count.",
        ),
        CountObservation(
            "third_not_announced",
            as_of,
            None,
            ObservationType.THRESHOLD_NOT_ANNOUNCED,
            3,
            0,
            2,
            lag,
            source,
            "Synthetic right censor.",
        ),
    )
    history = PublicHistory(1, "TEST", start, 3, 1, 3, (enrollment,), events)
    return history, first, second, as_of


class EventHistoryLikelihoodTest(unittest.TestCase):
    def test_reporting_lag_is_mixed_outside_the_patient_likelihood(self):
        history, first, second, as_of = synthetic_history_with_lag_mixture()
        model = PiecewiseEnrollmentModel(
            3,
            history.study_start,
            (history.study_start + timedelta(days=1),),
            (1.0,),
        )
        probabilities = {
            first: np.array([0.20, 0.10, 0.30]),
            second: np.array([0.60, 0.50, 0.70]),
            as_of - timedelta(days=1): np.array([0.70, 0.60, 0.80]),
            as_of: np.array([0.80, 0.75, 0.90]),
        }
        likelihood = PublicHistoryLikelihood(history, model)
        actual_log = likelihood.event_log_likelihood(probabilities.__getitem__)
        lag_zero = joint_cumulative_count_probability(
            np.column_stack(
                (probabilities[first], probabilities[second], probabilities[as_of])
            ),
            (1, 2, 2),
            (1, 2, 2),
        )
        lag_one = joint_cumulative_count_probability(
            np.column_stack(
                (
                    probabilities[first],
                    probabilities[second],
                    probabilities[as_of - timedelta(days=1)],
                )
            ),
            (1, 2, 2),
            (1, 2, 2),
        )
        expected = 0.25 * lag_zero + 0.75 * lag_one
        self.assertAlmostEqual(exp(actual_log), expected, places=14)

    def test_actual_history_has_finite_likelihood_under_an_illustrative_curve(self):
        history = load_regal_public_history()
        model = default_regal_enrollment_model(history)
        entry_dates = model.sample_enrollment_dates(np.random.default_rng(20260823))
        provider = CalendarEventProbabilityProvider(
            entry_dates,
            lambda months: np.exp(-log(2.0) * months / 28.0),
        )
        likelihood = PublicHistoryLikelihood(history, model)
        event_log = likelihood.event_log_likelihood(provider)
        self.assertTrue(isfinite(event_log))
        self.assertTrue(isfinite(likelihood.enrollment_log_likelihood()))
        exact_uniform = ReportingLag(
            "discrete_uniform",
            tuple(range(15)),
            (1 / 15,) * 15,
        )
        exact_events = tuple(
            replace(observation, reporting_lag=exact_uniform)
            if observation.observation_id
            in {
                "interim_60_threshold",
                "event_80_not_announced",
            }
            else observation
            for observation in history.event_observations
        )
        exact_history = replace(history, event_observations=exact_events)
        exact_event_log = PublicHistoryLikelihood(
            exact_history, model
        ).event_log_likelihood(provider)
        self.assertLess(abs(event_log - exact_event_log), 1e-4)
        with self.assertRaisesRegex(ValueError, "lag mixture"):
            likelihood.event_log_likelihood(provider, max_lag_combinations=8)

    def test_calendar_provider_excludes_not_yet_randomized_patients(self):
        entries = (date(2024, 1, 1), date(2024, 2, 1))
        provider = CalendarEventProbabilityProvider(
            entries, lambda months: np.exp(-0.1 * months), days_per_time_unit=30.0
        )
        values = provider(date(2024, 1, 31))
        self.assertAlmostEqual(values[0], 1.0 - exp(-0.1))
        self.assertEqual(values[1], 0.0)
        with self.assertRaisesRegex(ValueError, "one value"):
            CalendarEventProbabilityProvider(
                entries, lambda months: np.array([0.5, 0.5, 0.5])
            )(date(2024, 3, 1))

    def test_unstated_threshold_date_can_be_backed_off_from_announcement(self):
        source = SourceRecord(
            "Synthetic source", "https://example.com/source", date(2026, 1, 1)
        )
        observation = CountObservation(
            "threshold",
            None,
            date(2026, 1, 10),
            ObservationType.THRESHOLD_HIT,
            2,
            2,
            2,
            ReportingLag("discrete_pmf", (0, 2), (0.4, 0.6)),
            source,
            "Synthetic threshold with unknown occurrence date.",
        )
        choices = observation.cutoff_choices()
        self.assertEqual(
            [item[0].cutoff_date for item in choices],
            [date(2026, 1, 10), date(2026, 1, 8)],
        )
        self.assertEqual([item[1] for item in choices], [0.4, 0.6])


if __name__ == "__main__":
    unittest.main()
