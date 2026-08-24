#!/usr/bin/env python3
"""Validate WP6 history/continuation conditioning on an illustrative scenario.

The deliberately strong treatment effect makes observed interim continuation
rare under the target scenario.  The audit compares the exact public-count
conditional base proposal with a continuation-centered mixture.  It is a
computational stress test, not a calibrated REGAL scenario or forecast.
"""

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_likelihood import load_regal_public_history  # noqa: E402
from posterior import (  # noqa: E402
    ScenarioPatients,
    WeibullEventTimeModel,
    condition_on_public_history,
    public_history_constraint_branches,
)


def illustrative_strong_effect_scenario(entry_dates, rng):
    """Uncalibrated exponential fixture used only to stress continuation IS."""

    patient_count = len(entry_dates)
    strata = rng.integers(0, 4, size=patient_count)
    treatment = np.zeros(patient_count, dtype=bool)
    treatment[rng.permutation(patient_count)[: patient_count // 2]] = True
    scale_days = np.where(treatment, 1800.0, 550.0)
    return ScenarioPatients(
        treatment=treatment,
        strata=strata,
        censoring_time=np.full(patient_count, np.inf),
        event_time_model=WeibullEventTimeModel(
            scale_days,
            np.ones(patient_count),
            np.ones(patient_count),
        ),
    )


def validate(nsim=300, seed=20260824):
    history = load_regal_public_history()
    branches = public_history_constraint_branches(history)
    if len(branches) != 9:
        raise AssertionError("REGAL history must retain all nine lag branches")
    if abs(sum(item.probability for item in branches) - 1.0) > 1e-12:
        raise AssertionError("REGAL lag-branch probabilities must sum to one")

    common = dict(
        scenario_sampler=illustrative_strong_effect_scenario,
        scenario_name="illustrative strong-effect continuation stress test",
        history=history,
        nsim=nsim,
        seed=seed,
    )
    base = condition_on_public_history(
        proposal_interim_z_targets=(),
        **common,
    )
    mixture = condition_on_public_history(
        proposal_interim_z_targets=(0.0,),
        **common,
    )

    for result in (base, mixture):
        if result.is_posterior_forecast:
            raise AssertionError("WP6 output must not claim posterior-forecast status")
        if result.history_compatible_draws <= 0:
            raise AssertionError("public-history proposal retained no enrollment path")
        if not np.isfinite(result.log_p_public_history):
            raise AssertionError("public-history compatibility must be finite")
        if result.continuation_compatible_draws != (
            result.final_rejection_draws
            + result.final_non_rejection_draws
            + result.final_not_reached_draws
        ):
            raise AssertionError("continued final branches do not conserve draws")

    if mixture.continuation_compatible_draws <= base.continuation_compatible_draws:
        raise AssertionError(
            "continuation-centered proposal did not improve rare-branch coverage"
        )
    if mixture.continuation_effective_sample_size <= 1.0:
        raise AssertionError("continuation-centered proposal has inadequate effective mass")
    if abs(mixture.log_p_public_history - base.log_p_public_history) > 0.75:
        raise AssertionError("proposal choices disagree on public-history compatibility")

    print("WP6 interim-conditioning validation passed")
    print(f"  importance draws per proposal: {nsim:,}")
    print(f"  lag branches: {len(branches)} (mass 1.0)")
    print(
        "  base continuation draws / ESS: "
        f"{base.continuation_compatible_draws} / "
        f"{base.continuation_effective_sample_size:.2f}"
    )
    print(
        "  centered continuation draws / ESS: "
        f"{mixture.continuation_compatible_draws} / "
        f"{mixture.continuation_effective_sample_size:.2f}"
    )
    print(
        "  log P(public history | illustrative scenario): "
        f"base {base.log_p_public_history:.4f}, "
        f"centered {mixture.log_p_public_history:.4f}"
    )
    print(
        "  P(continue | public history), centered proposal: "
        f"{mixture.p_continue_given_public_history:.6f}"
    )
    print("  illustrative stress test only; not a REGAL forecast")
    return base, mixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsim", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    if args.nsim < 1:
        parser.error("--nsim must be positive")
    validate(args.nsim, args.seed)


if __name__ == "__main__":
    main()
