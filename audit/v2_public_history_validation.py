#!/usr/bin/env python3
"""Validate the v2 public-history schema, accrual anchors, and likelihood.

The exponential curve in this audit is deliberately illustrative.  It exists to
exercise the complete calendar-CDF and reporting-lag path with a deterministic
seed; it is not calibrated, conditioned on interim continuation, or reported as
a REGAL forecast.
"""

import argparse
from dataclasses import replace
from math import isfinite, log
import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from event_likelihood import (  # noqa: E402
    CalendarEventProbabilityProvider,
    ObservationType,
    PublicHistoryLikelihood,
    ReportingLag,
    default_regal_enrollment_model,
    enrollment_anchor_checks,
    joint_cumulative_count_probability,
    load_regal_public_history,
)


def history_with_fixed_right_censor_lag(history, lag_days):
    events = []
    for observation in history.event_observations:
        if observation.observation_id == "event_80_not_announced":
            observation = replace(
                observation,
                reporting_lag=ReportingLag("fixed", (lag_days,), (1.0,)),
            )
        events.append(observation)
    return replace(history, event_observations=tuple(events))


def run(median_months=28.0, seed=20260823):
    history = load_regal_public_history()
    model = default_regal_enrollment_model(history)
    likelihood = PublicHistoryLikelihood(history, model)

    print("REGAL v2 public-history validation")
    print("=" * 36)
    print(f"Registry: {history.registry_id}")
    print(f"Study start: {history.study_start.isoformat()}")
    print(f"Fixed randomized total: {history.target_enrollment}")
    print()
    print("Enrollment anchors")
    for check in enrollment_anchor_checks(history, model):
        if check.observation_type is ObservationType.THRESHOLD_REACHED_BY:
            interval = f">={check.point_count}"
        elif check.count_lower == check.count_upper:
            interval = str(check.point_count)
        else:
            interval = (
                f"{check.count_lower}-{check.count_upper} "
                f"(center {check.point_count})"
            )
        print(
            f"  {check.cutoff_date}: public {interval}; "
            f"reference mean {check.expected_count:.1f}; "
            f"reachable={check.reachable}; centered={check.centered}"
        )
        if not check.reachable or not check.centered:
            raise SystemExit("default accrual model failed an anchor gate")
    print(
        "  joint enrollment log likelihood "
        "(self-consistency under the data-centered reference path; "
        "not independent evidence): "
        f"{likelihood.enrollment_log_likelihood():.6f}"
    )

    dates = model.sample_enrollment_dates(np.random.default_rng(seed))
    if min(dates) < history.study_start:
        raise SystemExit("sampled a pre-opening patient")
    provider = CalendarEventProbabilityProvider(
        dates,
        lambda months: np.exp(-log(2.0) * months / median_months),
    )
    mixed_log = likelihood.event_log_likelihood(provider)
    if not isfinite(mixed_log):
        raise SystemExit("illustrative event likelihood is not finite")
    print()
    print(
        f"Illustrative exponential pooled curve (median {median_months:g} months; "
        "validation only)"
    )
    print(f"  event-history log likelihood, lag mixture: {mixed_log:.6f}")
    for lag in (0, 7, 14):
        fixed_history = history_with_fixed_right_censor_lag(history, lag)
        fixed_log = PublicHistoryLikelihood(
            fixed_history, model
        ).event_log_likelihood(provider)
        print(f"  event-history log likelihood, event-80 lag {lag:2d}d: {fixed_log:.6f}")

    cumulative = np.tile(np.array([0.20, 0.50]), (2, 1))
    joint = joint_cumulative_count_probability(cumulative, (1, 1), (1, 1))
    independent = (2 * 0.20 * 0.80) * (2 * 0.50 * 0.50)
    print()
    print("Small-cohort correlation check")
    print(f"  correct joint probability:       {joint:.6f}")
    print(f"  product of marginal probabilities: {independent:.6f}")
    if abs(joint - 0.20) > 1e-12 or abs(independent - 0.16) > 1e-12:
        raise SystemExit("small-cohort likelihood check failed")
    print()
    print("PASS — schema, anchors, integer likelihood, and lag censoring validated.")
    print("This audit is not a conditional or posterior REGAL forecast.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--median-months", type=float, default=28.0)
    parser.add_argument("--seed", type=int, default=20260823)
    arguments = parser.parse_args()
    if not isfinite(arguments.median_months) or arguments.median_months <= 0.0:
        parser.error("--median-months must be finite and positive")
    run(arguments.median_months, arguments.seed)


if __name__ == "__main__":
    main()
