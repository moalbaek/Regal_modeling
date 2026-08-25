#!/usr/bin/env python3
"""Validate WP7 effect-family averaging on a compact synthetic trial.

The fixture exercises the complete WP7 -> WP6 path without reporting a REGAL
forecast.  Its dates, counts, and resulting probabilities are synthetic.
"""

import argparse
from datetime import date
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_likelihood import (  # noqa: E402
    CountObservation,
    ObservationType,
    PiecewiseEnrollmentModel,
    PublicHistory,
    ReportingLag,
    SourceRecord,
)
from posterior import (  # noqa: E402
    REQUIRED_EFFECT_FAMILIES,
    condition_effect_families_futility_sensitivity_grid,
    posterior_prior_sensitivity,
)
from trial_design import TrialDecisionDesign  # noqa: E402


SOURCE = SourceRecord(
    "Synthetic WP7 validation fixture",
    "https://example.com/wp7-validation",
    date(2026, 1, 1),
)
FIXED_LAG = ReportingLag("fixed", (0,), (1.0,))


def observation(identifier, cutoff, kind, count, lower, upper, *, accrual=False):
    return CountObservation(
        observation_id=identifier,
        observation_date=cutoff,
        announcement_date=(
            None if kind is ObservationType.THRESHOLD_NOT_ANNOUNCED else cutoff
        ),
        observation_type=kind,
        count=count,
        count_lower=lower,
        count_upper=upper,
        reporting_lag=FIXED_LAG,
        source=SOURCE,
        notes="Synthetic count used only to validate WP7 implementation mechanics.",
        accrual_anchor=accrual,
    )


def synthetic_history():
    return PublicHistory(
        schema_version=1,
        registry_id="SYNTHETIC-WP7",
        study_start=date(2021, 1, 1),
        target_enrollment=8,
        interim_event_threshold=2,
        final_event_threshold=4,
        enrollment_observations=(
            observation(
                "enrollment_first_2",
                date(2021, 1, 2),
                ObservationType.THRESHOLD_REACHED_BY,
                2,
                2,
                8,
                accrual=True,
            ),
            observation(
                "enrollment_complete",
                date(2021, 1, 8),
                ObservationType.EXACT_AS_OF,
                8,
                8,
                8,
                accrual=True,
            ),
        ),
        event_observations=(
            observation(
                "interim_2",
                date(2021, 1, 20),
                ObservationType.THRESHOLD_HIT,
                2,
                2,
                2,
            ),
            observation(
                "events_3",
                date(2021, 1, 30),
                ObservationType.EXACT_AS_OF,
                3,
                3,
                3,
            ),
            observation(
                "event_4_not_announced",
                date(2021, 2, 5),
                ObservationType.THRESHOLD_NOT_ANNOUNCED,
                4,
                0,
                3,
            ),
        ),
    )


def validate(nsim=100, seed=20260825):
    history = synthetic_history()
    enrollment = PiecewiseEnrollmentModel(
        total_enrollment=8,
        study_start=history.study_start,
        phase_end_dates=(date(2021, 1, 2), date(2021, 1, 8)),
        phase_probabilities=(0.25, 0.75),
    )
    futility_rows = condition_effect_families_futility_sensitivity_grid(
        thresholds=(None, 1.0),
        history=history,
        enrollment_model=enrollment,
        base_design=TrialDecisionDesign(interim_events=2, final_events=4),
        nsim=nsim,
        seed=seed,
        proposal_interim_z_targets=(0.0,),
    )
    projections = futility_rows[0]
    if len(futility_rows) != 2:
        raise AssertionError("WP7 futility sensitivity grid lost a requested row")
    for no_futility, assumed_futility in zip(*futility_rows):
        if no_futility.family is not assumed_futility.family:
            raise AssertionError("futility rows are misaligned across effect families")
        if no_futility.conditioning.log_p_public_history != (
            assumed_futility.conditioning.log_p_public_history
        ):
            raise AssertionError("paired futility rows did not reuse family histories")
        if no_futility.conditioning.p_continue_given_public_history < (
            assumed_futility.conditioning.p_continue_given_public_history
        ):
            raise AssertionError("adding a futility stop increased continuation")
    if tuple(item.family for item in projections) != REQUIRED_EFFECT_FAMILIES:
        raise AssertionError("WP7 did not run the complete required family set")
    for projection in projections:
        result = projection.conditioning
        if result.is_posterior_forecast:
            raise AssertionError("a within-family WP6 result claimed forecast status")
        if result.history_compatible_draws <= 0:
            raise AssertionError("a family retained no synthetic public history")
        if result.continuation_compatible_draws <= 0:
            raise AssertionError("a family retained no synthetic continuation history")
        if not np.isfinite(result.log_p_public_history):
            raise AssertionError("a family history likelihood is not finite")

    forecasts = posterior_prior_sensitivity(projections)
    for forecast in forecasts:
        if not forecast.is_posterior_forecast:
            raise AssertionError("complete model average did not claim forecast status")
        if abs(sum(forecast.model_posterior_weights.values()) - 1.0) > 1e-12:
            raise AssertionError("posterior model weights do not sum to one")
        probability = (
            forecast.p_final_rejection_given_public_history_and_continuation
        )
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise AssertionError("posterior rejection probability is invalid")

    print("WP7 effect-model averaging validation passed")
    print(f"  synthetic importance draws per family: {nsim:,}")
    print(f"  effect families: {len(projections)}")
    print("  paired futility assumptions: disabled, HR 1.0")
    print(
        "  model-weight sensitivities: "
        + ", ".join(item.sensitivity_name for item in forecasts)
    )
    print("  synthetic fixture only; none of these values is a REGAL forecast")
    return projections, forecasts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsim", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    if args.nsim < 1:
        parser.error("--nsim must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    validate(args.nsim, args.seed)


if __name__ == "__main__":
    main()
