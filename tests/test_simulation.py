"""Tests for v2 event-driven branches and operating-characteristic validation."""

from dataclasses import replace
import os
import sys
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simulation import (  # noqa: E402
    EventDrivenTrialData,
    REGAL_V2_EFFICACY_DESIGN,
    evaluate_event_driven_trial,
    simulate_canonical_operating_characteristics,
    simulate_futility_sensitivity_grid,
    simulate_patient_level_exponential_null,
)
from trial_design import (  # noqa: E402
    FinalDecision,
    HazardRatioFutilityRule,
    InterimDecision,
    TrialDecisionDesign,
)


GPS = np.arange(63)
BAT = np.arange(63, 126)


def event_driven_fixture(event_groups):
    """Build a 126-patient trial from ``(time, subject_indices)`` groups."""

    followup = np.full(126, np.inf)
    event = np.zeros(126, dtype=bool)
    for event_time, indices in event_groups:
        indices = np.asarray(indices, dtype=int)
        if np.any(event[indices]):
            raise AssertionError("fixture assigns a subject more than once")
        followup[indices] = event_time
        event[indices] = True
    treatment = np.zeros(126, dtype=bool)
    treatment[GPS] = True
    # Four columns mirror the four public protocol stratification factors.  A
    # single factor combination keeps the branch fixtures analytically simple;
    # matrix handling itself is tested in test_trial_design.py.
    strata = np.zeros((126, 4), dtype=int)
    return EventDrivenTrialData(
        entry_time=np.zeros(126),
        followup_time=followup,
        event_observed=event,
        treatment=treatment,
        strata=strata,
    )


def paired_event_groups(pair_count, start_time=1.0):
    return [
        (start_time + pair, [GPS[pair], BAT[pair]])
        for pair in range(pair_count)
    ]


class EventDrivenDecisionTest(unittest.TestCase):
    def test_interim_efficacy_branch_stops_before_final(self):
        # BAT has only 63 subjects; the first 60 are sufficient and the
        # remaining events may come from GPS after the interim stop.
        groups = [(float(i + 1), [BAT[i]]) for i in range(63)]
        groups += [(100.0 + i, [GPS[i]]) for i in range(17)]
        result = evaluate_event_driven_trial(event_driven_fixture(groups))
        self.assertIs(result.interim_decision, InterimDecision.EFFICACY_STOP)
        self.assertIs(result.final_decision, FinalDecision.NOT_APPLICABLE)
        self.assertTrue(result.overall_success)
        self.assertIsNone(result.final)
        self.assertGreater(result.interim.primary.z, 2.339)

    def test_assumed_futility_branch_stops_before_final(self):
        groups = [(float(i + 1), [GPS[i]]) for i in range(63)]
        groups += [(100.0 + i, [BAT[i]]) for i in range(17)]
        design = replace(
            REGAL_V2_EFFICACY_DESIGN,
            futility_rule=HazardRatioFutilityRule(1.0),
        )
        result = evaluate_event_driven_trial(
            event_driven_fixture(groups), design
        )
        self.assertIs(result.interim_decision, InterimDecision.FUTILITY_STOP)
        self.assertIs(result.final_decision, FinalDecision.NOT_APPLICABLE)
        self.assertFalse(result.overall_success)
        self.assertGreater(result.interim.primary.one_step_hazard_ratio, 1.0)

    def test_continuation_can_await_or_reach_a_nonrejected_final(self):
        interim_only = event_driven_fixture(paired_event_groups(30))
        pending = evaluate_event_driven_trial(interim_only)
        self.assertIs(pending.interim_decision, InterimDecision.CONTINUE)
        self.assertIs(pending.final_decision, FinalDecision.NOT_REACHED)
        self.assertIsNone(pending.final)

        balanced_final = event_driven_fixture(paired_event_groups(40))
        complete = evaluate_event_driven_trial(balanced_final)
        self.assertIs(complete.interim_decision, InterimDecision.CONTINUE)
        self.assertIs(complete.final_decision, FinalDecision.DO_NOT_REJECT)
        self.assertFalse(complete.overall_success)
        self.assertEqual(complete.interim.observed_events, 60)
        self.assertEqual(complete.final.observed_events, 80)
        self.assertAlmostEqual(complete.final.primary.z, 0.0, places=12)

    def test_continuation_can_reject_at_final(self):
        groups = paired_event_groups(30)
        groups += [
            (40.0 + i, [BAT[30 + i]])
            for i in range(20)
        ]
        result = evaluate_event_driven_trial(event_driven_fixture(groups))
        self.assertIs(result.interim_decision, InterimDecision.CONTINUE)
        self.assertIs(result.final_decision, FinalDecision.REJECT)
        self.assertTrue(result.overall_success)
        self.assertLess(result.interim.primary.z, 2.339)
        self.assertGreater(result.final.primary.z, 2.011)

    def test_future_enrollment_is_excluded_from_an_event_cutoff(self):
        groups = paired_event_groups(40)
        base = event_driven_fixture(groups)
        entry = np.array(base.entry_time, copy=True)
        entry[~base.event_observed] = 1000.0
        shifted = EventDrivenTrialData(
            entry,
            base.followup_time,
            base.event_observed,
            base.treatment,
            base.strata,
        )
        result = evaluate_event_driven_trial(shifted)
        self.assertLess(result.final.cutoff_time, 1000.0)
        self.assertEqual(result.final.primary.events, 80)

    def test_data_and_design_contracts_are_validated_and_immutable(self):
        data = event_driven_fixture(paired_event_groups(30))
        with self.assertRaises(ValueError):
            data.entry_time[0] = 2.0
        with self.assertRaisesRegex(ValueError, "observed events"):
            EventDrivenTrialData(
                [0], [np.inf], [1], [0], [[0, 0, 0, 0]]
            )
        for interim, final in ((0, 80), (80, 80), (81, 80), (True, 80)):
            with self.subTest(interim=interim, final=final):
                with self.assertRaises(ValueError):
                    TrialDecisionDesign(
                        interim_events=interim, final_events=final
                    )
        with self.assertRaisesRegex(ValueError, "stops"):
            TrialDecisionDesign(futility_rule=object())

        class NeverStop:
            @staticmethod
            def stops(analysis):
                return False

        custom = TrialDecisionDesign(futility_rule=NeverStop())
        self.assertIs(
            evaluate_event_driven_trial(data, custom).interim_decision,
            InterimDecision.CONTINUE,
        )


class CanonicalValidationTest(unittest.TestCase):
    def test_null_simulation_realizes_type_one_error_and_conserves_branches(self):
        result = simulate_canonical_operating_characteristics(
            nsim=300000, seed=20260823
        )
        self.assertAlmostEqual(result.p_overall_success, 0.025, delta=0.001)
        self.assertAlmostEqual(result.p_interim_efficacy, 0.009649, delta=0.001)
        self.assertEqual(result.p_futility, 0.0)
        self.assertEqual(
            result.interim_efficacy_stops
            + result.futility_stops
            + result.continuations,
            result.n_simulations,
        )
        self.assertEqual(
            result.final_rejections + result.final_non_rejections,
            result.continuations,
        )

    def test_futility_grid_uses_paired_draws_and_is_monotone(self):
        rows = simulate_futility_sensitivity_grid(
            thresholds=(None, 0.8, 0.9, 1.0, 1.1, 1.2),
            nsim=100000,
            seed=99,
        )
        self.assertEqual(
            [row.futility_hr_threshold for row in rows],
            [None, 0.8, 0.9, 1.0, 1.1, 1.2],
        )
        self.assertEqual(len({row.interim_efficacy_stops for row in rows}), 1)
        self.assertEqual(rows[0].futility_stops, 0)
        numeric_futility = [row.futility_stops for row in rows[1:]]
        self.assertEqual(numeric_futility, sorted(numeric_futility, reverse=True))
        for row in rows[1:]:
            self.assertLessEqual(row.p_overall_success, rows[0].p_overall_success)

    def test_patient_level_event_path_preserves_null_behavior(self):
        result = simulate_patient_level_exponential_null(
            nsim=3000, seed=20260824
        )
        self.assertEqual(result.simulation_kind, "patient_level_exponential_null")
        self.assertAlmostEqual(result.p_overall_success, 0.025, delta=0.007)
        self.assertAlmostEqual(result.p_interim_efficacy, 0.009649, delta=0.004)
        self.assertEqual(result.futility_stops, 0)
        self.assertEqual(
            result.final_rejections + result.final_non_rejections,
            result.continuations,
        )

    def test_simulation_and_grid_inputs_are_validated(self):
        for nsim in (0, True, 1.5):
            with self.subTest(nsim=nsim):
                with self.assertRaises(ValueError):
                    simulate_canonical_operating_characteristics(nsim=nsim)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            simulate_futility_sensitivity_grid((None, None), nsim=10)
        with self.assertRaisesRegex(ValueError, "not be empty"):
            simulate_futility_sensitivity_grid((), nsim=10)

        class CustomRule:
            @staticmethod
            def stops(analysis):
                return False

        with self.assertRaisesRegex(ValueError, "patient-level"):
            simulate_canonical_operating_characteristics(
                design=TrialDecisionDesign(futility_rule=CustomRule()),
                nsim=10,
            )
        with self.assertRaisesRegex(ValueError, "at least final_events"):
            simulate_patient_level_exponential_null(
                nsim=1, patient_count=79
            )


if __name__ == "__main__":
    unittest.main()
