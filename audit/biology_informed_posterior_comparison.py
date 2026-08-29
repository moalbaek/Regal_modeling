"""Compare baseline and biology-informed REGAL posterior forecasts.

This audit runner reuses the same blinded public-history model and balanced model
family weights for every comparison. Only the responder/cure family's parameter
prior changes. The non-responder families are integrated once and reused; each
biology variant recomputes only the responder family on the same family-specific
seed and futility-threshold grid.

Examples
--------
    python audit/biology_informed_posterior_comparison.py
    python audit/biology_informed_posterior_comparison.py --nsim 150000 \
        --workers 7 \
        --output data/biology_informed_posterior_comparison.json

The output is a sensitivity analysis, not an unblinded estimate of REGAL's arm
split or a claim that immune response causes survival benefit. The CLI withholds
unqualified output when any posterior-forecast readiness gate fails; use
``--allow-diagnostic-output`` only to inspect explicitly labeled diagnostics.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from math import isfinite
from numbers import Integral
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biology_informed_posterior import (  # noqa: E402
    BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS,
    BIOLOGY_INFORMED_MECHANISM_FAVORING_SURVIVAL_PRIORS,
    BIOLOGY_INFORMED_SKEPTICAL_SURVIVAL_PRIORS,
    PHASE2_ONLY_EFFECT_FAMILY_PRIORS,
    REGAL_INTERIM_ONLY_EFFECT_FAMILY_PRIORS,
    RESPONSE_EVIDENCE_ONLY_EFFECT_FAMILY_PRIORS,
    effect_priors_with_biology,
)
from biology_priors import (  # noqa: E402
    POOLED_GPS_RESPONSE_POSTERIOR,
    POOLED_GPS_RESPONSE_POSTERIOR_SENSITIVITY,
    REGAL_INTERIM_DEFAULT_ASSUMED_EVALUABLE,
    REGAL_INTERIM_IMMUNE_SOURCE_URL,
    REGAL_INTERIM_REPORTED_RESPONSE_RATE,
    WT1_RESPONDER_DURABLE_PRIOR_BALANCED,
    WT1_RESPONDER_DURABLE_PRIOR_MECHANISM_FAVORING,
    WT1_RESPONDER_DURABLE_PRIOR_SKEPTICAL,
)
from posterior import (  # noqa: E402
    BALANCED_MODEL_FAMILY_PRIOR,
    DEFAULT_EFFECT_FAMILY_PRIORS,
    GPSEffectFamily,
    REQUIRED_EFFECT_FAMILIES,
    condition_effect_family_futility_sensitivity,
    posterior_model_average,
)
from simulation import FUTILITY_HR_SENSITIVITY_GRID  # noqa: E402


DEFAULT_AUDIT_IMPORTANCE_DRAWS = 150_000
DEFAULT_AUDIT_WORKERS = 7
# Match the production publisher's exact public-history-conditioned base
# proposal. This affects Monte Carlo efficiency, not the target estimand.
AUDIT_PROPOSAL_INTERIM_Z_TARGETS = ()


class AuditNotReadyError(RuntimeError):
    """The requested Monte Carlo run did not clear posterior-forecast gates."""


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
    conditioning = tuple(item.conditioning for item in forecast.family_results)

    def finite_extreme(values, operation):
        values = tuple(float(value) for value in values)
        if not values or any(not isfinite(value) for value in values):
            return None
        return operation(values)

    minimum_history_ess = finite_extreme(
        (item.history_effective_sample_size for item in conditioning), min
    )
    minimum_continuation_ess = finite_extreme(
        (item.continuation_effective_sample_size for item in conditioning), min
    )
    maximum_history_share = finite_extreme(
        (item.maximum_history_weight_share for item in conditioning), max
    )
    return {
        "is_posterior_forecast": forecast.is_posterior_forecast,
        "estimate_status": (
            "posterior_forecast"
            if forecast.is_posterior_forecast
            else "diagnostic_only"
        ),
        "readiness_issues": list(forecast.forecast_readiness_issues),
        "readiness_diagnostics": {
            "minimum_history_effective_sample_size": minimum_history_ess,
            "minimum_continuation_effective_sample_size": minimum_continuation_ess,
            "maximum_history_weight_share": maximum_history_share,
        },
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


def _comparison_readiness_issues(forecasts):
    issues = []
    for variant, threshold_results in forecasts.items():
        for threshold, summary in threshold_results.items():
            if summary["is_posterior_forecast"]:
                continue
            for issue in summary["readiness_issues"]:
                issues.append(f"{variant} futility={threshold}: {issue}")
    return tuple(issues)


def _require_ready_output(result):
    if result["is_posterior_forecast"]:
        return
    issues = tuple(result["readiness_issues"])
    preview = "; ".join(issues[:3])
    remaining = len(issues) - min(len(issues), 3)
    if remaining:
        preview += f"; and {remaining} more readiness failures"
    raise AuditNotReadyError(
        "comparison is diagnostic-only and cannot be published as a posterior "
        f"forecast: {preview}"
    )


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _comparison_variants():
    """Return ordered responder-prior comparisons used by the audit."""

    return {
        "baseline_wp7": DEFAULT_EFFECT_FAMILY_PRIORS,
        "response_evidence_only": RESPONSE_EVIDENCE_ONLY_EFFECT_FAMILY_PRIORS,
        "response_phase2_only": PHASE2_ONLY_EFFECT_FAMILY_PRIORS,
        "response_regal_interim_only": REGAL_INTERIM_ONLY_EFFECT_FAMILY_PRIORS,
        "biology_skeptical_survival": BIOLOGY_INFORMED_SKEPTICAL_SURVIVAL_PRIORS,
        "biology_balanced_survival": BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS,
        "biology_balanced_regal_assumed_n5": effect_priors_with_biology(
            response_beta_prior=POOLED_GPS_RESPONSE_POSTERIOR_SENSITIVITY[5],
            responder_cure_prior=WT1_RESPONDER_DURABLE_PRIOR_BALANCED,
        ),
        "biology_balanced_regal_assumed_n20": effect_priors_with_biology(
            response_beta_prior=POOLED_GPS_RESPONSE_POSTERIOR_SENSITIVITY[20],
            responder_cure_prior=WT1_RESPONDER_DURABLE_PRIOR_BALANCED,
        ),
        "biology_mechanism_favoring_survival": (
            BIOLOGY_INFORMED_MECHANISM_FAVORING_SURVIVAL_PRIORS
        ),
    }


def _family_worker(prior, thresholds, nsim, seed):
    return condition_effect_family_futility_sensitivity(
        prior,
        thresholds=thresholds,
        nsim=nsim,
        seed=seed,
        proposal_interim_z_targets=AUDIT_PROPOSAL_INTERIM_Z_TARGETS,
    )


def _integration_tasks(variants):
    tasks = [
        (f"baseline::{prior.family.value}", prior)
        for prior in DEFAULT_EFFECT_FAMILY_PRIORS
    ]
    tasks.extend(
        (f"variant::{name}", _responder_prior(priors))
        for name, priors in variants.items()
        if name != "baseline_wp7"
    )
    return tuple(tasks)


def _variant_prior_records(variants):
    records = {}
    for name, priors in variants.items():
        responder = _responder_prior(priors)
        records[name] = {
            "response_probability": dict(
                responder.response_probability.describe()
            ),
            "responder_durable_probability": dict(
                responder.responder_cure_probability.describe()
            ),
        }
    return records


def _integrate_tasks(tasks, *, thresholds, nsim, seed, workers, progress=False):
    """Integrate independent family priors, optionally across processes."""

    tasks = tuple(tasks)
    workers = _positive_integer(workers, "workers")
    results = {}
    if workers == 1:
        for name, prior in tasks:
            results[name] = _family_worker(prior, thresholds, nsim, seed)
            if progress:
                print(f"completed {name}", flush=True)
        return results

    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        futures = {
            executor.submit(_family_worker, prior, thresholds, nsim, seed): name
            for name, prior in tasks
        }
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()
            if progress:
                print(f"completed {name}", flush=True)
    return results


def _baseline_rows(integrations, thresholds):
    family_rows = tuple(
        integrations[f"baseline::{family.value}"]
        for family in REQUIRED_EFFECT_FAMILIES
    )
    if any(len(rows) != len(thresholds) for rows in family_rows):
        raise RuntimeError("effect-family futility grids are misaligned")
    return tuple(
        tuple(rows[index] for rows in family_rows)
        for index in range(len(thresholds))
    )


def run_comparison(
    nsim=DEFAULT_AUDIT_IMPORTANCE_DRAWS,
    seed=20260825,
    workers=DEFAULT_AUDIT_WORKERS,
    *,
    progress=False,
):
    nsim = _positive_integer(nsim, "nsim")
    workers = _positive_integer(workers, "workers")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    thresholds = tuple(FUTILITY_HR_SENSITIVITY_GRID)
    variants = _comparison_variants()
    integrations = _integrate_tasks(
        _integration_tasks(variants),
        thresholds=thresholds,
        nsim=nsim,
        seed=seed,
        workers=workers,
        progress=progress,
    )
    baseline_rows = _baseline_rows(integrations, thresholds)
    responder_rows = {
        name: integrations[f"variant::{name}"]
        for name in variants
        if name != "baseline_wp7"
    }

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

    readiness_issues = _comparison_readiness_issues(results)
    response_mean = POOLED_GPS_RESPONSE_POSTERIOR.mean
    return {
        "is_posterior_forecast": not readiness_issues,
        "estimate_status": (
            "posterior_forecast" if not readiness_issues else "diagnostic_only"
        ),
        "readiness_issues": list(readiness_issues),
        "nsim_per_family": int(nsim),
        "seed": int(seed),
        "workers": int(workers),
        "proposal_interim_z_targets": list(AUDIT_PROPOSAL_INTERIM_Z_TARGETS),
        "model_family_prior": BALANCED_MODEL_FAMILY_PRIOR.name,
        "futility_hr_thresholds": [
            None if value is None else float(value) for value in thresholds
        ],
        "forecast_variant_priors": _variant_prior_records(variants),
        "biology_prior_summary": {
            "pooled_immune_response_mean": response_mean,
            "regal_interim_response_evidence": {
                "reported_response_rate": REGAL_INTERIM_REPORTED_RESPONSE_RATE,
                "default_assumed_evaluable": (
                    REGAL_INTERIM_DEFAULT_ASSUMED_EVALUABLE
                ),
                "denominator_status": "working_assumption_not_publicly_disclosed",
                "source_url": REGAL_INTERIM_IMMUNE_SOURCE_URL,
            },
            "pooled_response_mean_by_assumed_regal_evaluable": {
                str(evaluable): prior.mean
                for evaluable, prior in (
                    POOLED_GPS_RESPONSE_POSTERIOR_SENSITIVITY.items()
                )
            },
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
    print(
        f"nsim/family={result['nsim_per_family']:,} seed={result['seed']} "
        f"workers={result['workers']}"
    )
    print(f"estimate status: {result['estimate_status']}")
    if not result["is_posterior_forecast"]:
        print(
            "WARNING: diagnostic-only output; one or more Monte Carlo readiness "
            "gates failed."
        )
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
    parser.add_argument(
        "--nsim", type=int, default=DEFAULT_AUDIT_IMPORTANCE_DRAWS
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--workers", type=int, default=DEFAULT_AUDIT_WORKERS)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-diagnostic-output",
        action="store_true",
        help=(
            "print or write results that failed forecast-readiness gates, with "
            "diagnostic-only labeling"
        ),
    )
    args = parser.parse_args(argv)
    if args.nsim <= 0:
        parser.error("--nsim must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    result = run_comparison(
        nsim=args.nsim,
        seed=args.seed,
        workers=args.workers,
        progress=True,
    )
    if not args.allow_diagnostic_output:
        try:
            _require_ready_output(result)
        except AuditNotReadyError as error:
            parser.error(
                f"{error}. Rerun with more --nsim, or pass "
                "--allow-diagnostic-output to inspect explicitly labeled diagnostics"
            )
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
