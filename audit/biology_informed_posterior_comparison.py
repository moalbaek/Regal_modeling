"""Compare baseline and biology-informed REGAL posterior forecasts.

This audit runner reuses the same blinded public-history model and balanced model
family weights for every comparison. Only the responder/cure family's parameter
prior changes. The non-responder families are integrated once and reused; each
biology variant recomputes only the responder family on the same family-specific
seed and futility-threshold grid.

Examples
--------
    python audit/biology_informed_posterior_comparison.py
    python audit/biology_informed_posterior_comparison.py --nsim 20000 \
        --output data/biology_informed_posterior_comparison.json

The output is a sensitivity analysis, not an unblinded estimate of REGAL's arm
split or a claim that immune response causes survival benefit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biology_informed_posterior import (  # noqa: E402
    BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS,
    BIOLOGY_INFORMED_MECHANISM_FAVORING_SURVIVAL_PRIORS,
    BIOLOGY_INFORMED_SKEPTICAL_SURVIVAL_PRIORS,
    RESPONSE_EVIDENCE_ONLY_EFFECT_FAMILY_PRIORS,
)
from biology_priors import (  # noqa: E402
    POOLED_GPS_RESPONSE_POSTERIOR,
    WT1_RESPONDER_DURABLE_PRIOR_BALANCED,
    WT1_RESPONDER_DURABLE_PRIOR_MECHANISM_FAVORING,
    WT1_RESPONDER_DURABLE_PRIOR_SKEPTICAL,
)
from posterior import (  # noqa: E402
    BALANCED_MODEL_FAMILY_PRIOR,
    DEFAULT_EFFECT_FAMILY_PRIORS,
    GPSEffectFamily,
    condition_effect_families_futility_sensitivity_grid,
    condition_effect_family_futility_sensitivity,
    posterior_model_average,
)
from simulation import FUTILITY_HR_SENSITIVITY_GRID  # noqa: E402


def _responder_prior(priors):
    return next(
        prior for prior in priors if prior.family is GPSEffectFamily.RESPONDER_CURE
    )


def _replace_responder(projections, responder_projection):
    return tuple(
        responder_projection
        if projection.family is GPSEffectFamily.RESPONDER_CURE
        else projection
        for projection in projections
    )


def _threshold_label(value):
    return "disabled" if value is None else f"{float(value):.2f}"


def _forecast_summary(forecast):
    return {
        "p_final_rejection_given_history_and_continuation": (
            forecast.p_final_rejection_given_public_history_and_continuation
        ),
        "p_final_reached_given_history_and_continuation": (
            forecast.p_final_reached_given_public_history_and_continuation
        ),
        "responder_family_prior_weight": forecast.model_prior_weights[
            GPSEffectFamily.RESPONDER_CURE
        ],
        "responder_family_posterior_weight": forecast.model_posterior_weights[
            GPSEffectFamily.RESPONDER_CURE
        ],
        "family_posterior_weights": {
            family.value: weight
            for family, weight in forecast.model_posterior_weights.items()
        },
    }


def run_comparison(nsim=10_000, seed=20260825):
    thresholds = tuple(FUTILITY_HR_SENSITIVITY_GRID)
    baseline_rows = condition_effect_families_futility_sensitivity_grid(
        DEFAULT_EFFECT_FAMILY_PRIORS,
        thresholds=thresholds,
        nsim=nsim,
        seed=seed,
    )

    variants = {
        "baseline_wp7": DEFAULT_EFFECT_FAMILY_PRIORS,
        "response_evidence_only": RESPONSE_EVIDENCE_ONLY_EFFECT_FAMILY_PRIORS,
        "biology_skeptical_survival": BIOLOGY_INFORMED_SKEPTICAL_SURVIVAL_PRIORS,
        "biology_balanced_survival": BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS,
        "biology_mechanism_favoring_survival": (
            BIOLOGY_INFORMED_MECHANISM_FAVORING_SURVIVAL_PRIORS
        ),
    }

    responder_rows = {}
    for name, priors in variants.items():
        if name == "baseline_wp7":
            continue
        responder_rows[name] = condition_effect_family_futility_sensitivity(
            _responder_prior(priors),
            thresholds=thresholds,
            nsim=nsim,
            seed=seed,
        )

    results = {}
    for name in variants:
        threshold_results = {}
        for index, threshold in enumerate(thresholds):
            projections = baseline_rows[index]
            if name != "baseline_wp7":
                projections = _replace_responder(
                    projections, responder_rows[name][index]
                )
            forecast = posterior_model_average(
                projections, BALANCED_MODEL_FAMILY_PRIOR
            )
            threshold_results[_threshold_label(threshold)] = _forecast_summary(
                forecast
            )
        results[name] = threshold_results

    response_mean = POOLED_GPS_RESPONSE_POSTERIOR.mean
    return {
        "nsim_per_family": int(nsim),
        "seed": int(seed),
        "model_family_prior": BALANCED_MODEL_FAMILY_PRIOR.name,
        "futility_hr_thresholds": [
            None if value is None else float(value) for value in thresholds
        ],
        "biology_prior_summary": {
            "pooled_immune_response_mean": response_mean,
            "durable_probability_means": {
                "skeptical": WT1_RESPONDER_DURABLE_PRIOR_SKEPTICAL.mean,
                "balanced": WT1_RESPONDER_DURABLE_PRIOR_BALANCED.mean,
                "mechanism_favoring": (
                    WT1_RESPONDER_DURABLE_PRIOR_MECHANISM_FAVORING.mean
                ),
            },
            "implied_all_gps_durable_fraction_means": {
                "skeptical": (
                    response_mean * WT1_RESPONDER_DURABLE_PRIOR_SKEPTICAL.mean
                ),
                "balanced": (
                    response_mean * WT1_RESPONDER_DURABLE_PRIOR_BALANCED.mean
                ),
                "mechanism_favoring": (
                    response_mean
                    * WT1_RESPONDER_DURABLE_PRIOR_MECHANISM_FAVORING.mean
                ),
            },
        },
        "forecasts": results,
    }


def _print_table(result):
    thresholds = result["futility_hr_thresholds"]
    names = tuple(result["forecasts"])
    print("REGAL biology-informed posterior comparison")
    print(f"nsim/family={result['nsim_per_family']:,} seed={result['seed']}")
    print()
    header = ["variant"] + [
        f"futility={_threshold_label(value)}" for value in thresholds
    ]
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))
    for name in names:
        values = []
        for threshold in thresholds:
            item = result["forecasts"][name][_threshold_label(threshold)]
            values.append(
                f"{100 * item['p_final_rejection_given_history_and_continuation']:.2f}%"
            )
        print(" | ".join([name] + values))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsim", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.nsim <= 0:
        parser.error("--nsim must be positive")

    result = run_comparison(nsim=args.nsim, seed=args.seed)
    _print_table(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
