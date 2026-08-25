"""Versioned REGAL v2 result bundles and self-contained HTML publication.

Python is the canonical v2 engine.  This module converts completed
``PosteriorForecastResult`` objects into a strictly validated, JSON-safe bundle
and embeds that exact bundle in ``regal_explorer.html``.  The browser only
formats published values; it never recalculates the posterior.

The release invariant is enforced in both directions: a headline value exists
only when the primary complete model average reports
``is_posterior_forecast=True``, and a non-ready result retains its diagnostics
while the headline is set to ``null``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from math import isfinite
import os
from pathlib import Path
import re
import subprocess
import tempfile

from posterior import (
    BALANCED_MODEL_FAMILY_PRIOR,
    DEFAULT_EFFECT_FAMILY_PRIORS,
    DEFAULT_IMPORTANCE_DRAWS,
    DEFAULT_MODEL_FAMILY_PRIOR_SENSITIVITY,
    GPSEffectFamily,
    MAXIMUM_POSTERIOR_FORECAST_HISTORY_WEIGHT_SHARE,
    MINIMUM_POSTERIOR_FORECAST_ESS,
    PosteriorForecastResult,
    REQUIRED_EFFECT_FAMILIES,
    condition_effect_family_futility_sensitivity,
    posterior_model_average,
    posterior_prior_sensitivity,
)
from regal_data import RegalDataSnapshot, load_regal_data_snapshot
from simulation import FUTILITY_HR_SENSITIVITY_GRID
from trial_design import TrialDecisionDesign


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_BUNDLE_PATH = ROOT / "data" / "regal_v2_result_bundle.json"
DEFAULT_HTML_PATH = ROOT / "regal_explorer.html"

RESULT_BUNDLE_SCHEMA_VERSION = 1
RESULT_BUNDLE_TYPE = "regal_v2_posterior_forecast"
MODEL_VERSION = "v2"
PRIMARY_MODEL_WEIGHT_SENSITIVITY = BALANCED_MODEL_FAMILY_PRIOR.name
RESULT_BUNDLE_START = "<!-- REGAL_V2_RESULT_BUNDLE_START -->"
RESULT_BUNDLE_END = "<!-- REGAL_V2_RESULT_BUNDLE_END -->"

FAMILY_LABELS = {
    GPSEffectFamily.NO_EFFECT: "No effect",
    GPSEffectFamily.PROPORTIONAL_HAZARDS: "Proportional hazards",
    GPSEffectFamily.DELAYED_PROPORTIONAL_HAZARDS: "Delayed proportional hazards",
    GPSEffectFamily.CURE_FRACTION_DIFFERENCE: "Cure-fraction difference",
    GPSEffectFamily.DELAYED_CURE: "Delayed cure",
    GPSEffectFamily.WANING_PIECEWISE: "Waning / piecewise",
    GPSEffectFamily.RESPONDER_CURE: "Responder / cure exploratory",
}

ACTIVE_PRIOR_FIELDS = {
    GPSEffectFamily.NO_EFFECT: (),
    GPSEffectFamily.PROPORTIONAL_HAZARDS: ("hazard_ratio",),
    GPSEffectFamily.DELAYED_PROPORTIONAL_HAZARDS: (
        "hazard_ratio",
        "delay_months",
    ),
    GPSEffectFamily.CURE_FRACTION_DIFFERENCE: (
        "extra_cure_probability",
    ),
    GPSEffectFamily.DELAYED_CURE: (
        "extra_cure_probability",
        "delay_months",
    ),
    GPSEffectFamily.WANING_PIECEWISE: (
        "hazard_ratio",
        "delay_months",
        "late_hazard_ratio",
    ),
    GPSEffectFamily.RESPONDER_CURE: (
        "response_probability",
        "responder_cure_probability",
    ),
}


@dataclass(frozen=True)
class AnalysisRunMetadata:
    """Reproducibility metadata that is not already present in model results."""

    generated_at: datetime
    source_revision: str
    seed: int

    def __post_init__(self):
        generated_at = self.generated_at
        if not isinstance(generated_at, datetime) or generated_at.tzinfo is None:
            raise ValueError("generated_at must be a timezone-aware datetime")
        generated_at = generated_at.astimezone(timezone.utc)
        revision = str(self.source_revision).strip()
        if not revision:
            raise ValueError("source_revision must be non-empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        object.__setattr__(self, "generated_at", generated_at)
        object.__setattr__(self, "source_revision", revision)

    @property
    def generated_at_iso(self):
        return self.generated_at.isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_number(value):
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None


def _probability(value, name, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return number


def _prior_range_record(prior_range):
    if prior_range.is_point_mass:
        distribution = "point_mass"
    elif prior_range.log_scale:
        distribution = "log_uniform"
    else:
        distribution = "uniform"
    return {
        "distribution": distribution,
        "lower": prior_range.lower,
        "upper": prior_range.upper,
    }


def _effect_prior_record(prior):
    return {
        name: _prior_range_record(getattr(prior, name))
        for name in ACTIVE_PRIOR_FIELDS[prior.family]
    }


def _design_record(design):
    boundaries = design.efficacy_boundaries
    return {
        "interim_events": design.interim_events,
        "final_events": design.final_events,
        "alpha_one_sided": design.alpha,
        "spending_family": boundaries["spending_family"],
        "interim_information": boundaries["interim_information"],
        "interim_efficacy_z": boundaries["interim_z"],
        "final_efficacy_z": boundaries["final_z"],
        "futility_rule_disclosure": "The actual interim futility rule is unpublished.",
        "baseline_futility_assumption": "disabled",
    }


def _conditioning_diagnostics(conditioning):
    return {
        "importance_draws": conditioning.importance_draws,
        "history_compatible_draws": conditioning.history_compatible_draws,
        "continuation_compatible_draws": conditioning.continuation_compatible_draws,
        "interim_efficacy_draws": conditioning.interim_efficacy_draws,
        "interim_futility_draws": conditioning.interim_futility_draws,
        "non_estimable_interim_draws": conditioning.non_estimable_interim_draws,
        "final_rejection_draws": conditioning.final_rejection_draws,
        "final_non_rejection_draws": conditioning.final_non_rejection_draws,
        "final_not_reached_draws": conditioning.final_not_reached_draws,
        "history_effective_sample_size": _json_number(
            conditioning.history_effective_sample_size
        ),
        "continuation_effective_sample_size": _json_number(
            conditioning.continuation_effective_sample_size
        ),
        "maximum_history_weight_share": _json_number(
            conditioning.maximum_history_weight_share
        ),
        "valid_disclosure_lag_mass": _json_number(
            conditioning.valid_disclosure_lag_mass
        ),
        "proposal_interim_z_targets": list(
            conditioning.proposal_interim_z_targets
        ),
        "tilt_attempts": conditioning.tilt_attempts,
        "tilt_fallbacks": conditioning.tilt_fallbacks,
        "draws_with_tilt_fallback": conditioning.draws_with_tilt_fallback,
        "proposal_infeasible_draws": conditioning.proposal_infeasible_draws,
        "mean_tilt_iterations": _json_number(conditioning.mean_tilt_iterations),
        "maximum_tilt_error": _json_number(conditioning.maximum_tilt_error),
    }


def _family_record(item):
    conditioning = item.conditioning
    return {
        "family": item.family.value,
        "label": FAMILY_LABELS[item.family],
        "prior_weight": item.prior_weight,
        "posterior_weight": item.posterior_weight,
        "parameter_priors": _effect_prior_record(item.parameter_prior),
        "probabilities": {
            "public_history": _json_number(conditioning.p_public_history),
            "continuation_given_public_history": _json_number(
                conditioning.p_continue_given_public_history
            ),
            "public_history_and_continuation": _json_number(
                item.p_public_history_and_continuation
            ),
            "final_rejection_given_public_history_and_continuation": _json_number(
                conditioning.p_final_rejection_given_public_history_and_continuation
            ),
            "final_reached_given_public_history_and_continuation": _json_number(
                conditioning.p_final_reached_given_public_history_and_continuation
            ),
        },
        "diagnostics": _conditioning_diagnostics(conditioning),
    }


def _forecast_record(forecast, include_families=True):
    record = {
        "name": forecast.sensitivity_name,
        "assumed_futility_hr_threshold": forecast.assumed_futility_hr_threshold,
        "is_posterior_forecast": forecast.is_posterior_forecast,
        "estimate_status": (
            "posterior_forecast"
            if forecast.is_posterior_forecast
            else "diagnostic_only"
        ),
        "readiness_issues": list(forecast.forecast_readiness_issues),
        "probabilities": {
            "public_history_and_continuation": _json_number(
                forecast.p_public_history_and_continuation
            ),
            "final_rejection_given_public_history_and_continuation": _json_number(
                forecast.p_final_rejection_given_public_history_and_continuation
            ),
            "final_reached_given_public_history_and_continuation": _json_number(
                forecast.p_final_reached_given_public_history_and_continuation
            ),
        },
    }
    if include_families:
        record["families"] = [_family_record(item) for item in forecast.family_results]
    return record


def _forecast_conditioning_signature(forecast):
    return tuple(
        (
            item.family,
            item.parameter_prior,
            item.conditioning,
            item.log_p_public_history_and_continuation,
        )
        for item in forecast.family_results
    )


def _ordered_prior_forecasts(forecasts):
    forecasts = tuple(forecasts)
    if not forecasts or not all(
        isinstance(item, PosteriorForecastResult) for item in forecasts
    ):
        raise ValueError("prior forecasts must contain PosteriorForecastResult values")
    by_name = {item.sensitivity_name: item for item in forecasts}
    if len(by_name) != len(forecasts):
        raise ValueError("prior forecast sensitivity names must be unique")
    expected = tuple(item.name for item in DEFAULT_MODEL_FAMILY_PRIOR_SENSITIVITY)
    if set(by_name) != set(expected):
        raise ValueError(
            "prior forecasts must contain skeptical, balanced, and cure_favoring"
        )
    ordered = tuple(by_name[name] for name in expected)
    baseline_signature = _forecast_conditioning_signature(ordered[0])
    if any(
        _forecast_conditioning_signature(item) != baseline_signature
        for item in ordered[1:]
    ):
        raise ValueError("prior sensitivities must reuse identical family projections")
    if any(item.assumed_futility_hr_threshold is not None for item in ordered):
        raise ValueError("prior sensitivity baseline must leave futility disabled")
    return ordered


def _ordered_futility_forecasts(forecasts):
    forecasts = tuple(forecasts)
    if not forecasts or not all(
        isinstance(item, PosteriorForecastResult) for item in forecasts
    ):
        raise ValueError("futility forecasts must contain PosteriorForecastResult values")
    if any(item.sensitivity_name != PRIMARY_MODEL_WEIGHT_SENSITIVITY for item in forecasts):
        raise ValueError("futility sensitivity must use the balanced model-weight prior")
    by_threshold = {}
    for item in forecasts:
        threshold = item.assumed_futility_hr_threshold
        if threshold in by_threshold:
            raise ValueError("futility sensitivity thresholds must be unique")
        by_threshold[threshold] = item
    if set(by_threshold) != set(FUTILITY_HR_SENSITIVITY_GRID):
        raise ValueError("futility forecasts must contain the complete configured grid")
    return tuple(by_threshold[value] for value in FUTILITY_HR_SENSITIVITY_GRID)


def _importance_draws(forecasts):
    draws = {
        item.conditioning.importance_draws
        for forecast in forecasts
        for item in forecast.family_results
    }
    if len(draws) != 1:
        raise ValueError("every bundled family must use one importance-draw budget")
    value = draws.pop()
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("importance draw budget must be a positive integer")
    return value


def build_result_bundle(
    prior_forecasts,
    futility_forecasts,
    *,
    metadata,
    data_snapshot=None,
):
    """Build one strict bundle from a completed WP7 sensitivity analysis."""

    if not isinstance(metadata, AnalysisRunMetadata):
        raise ValueError("metadata must be AnalysisRunMetadata")
    if data_snapshot is None:
        data_snapshot = load_regal_data_snapshot()
    if not isinstance(data_snapshot, RegalDataSnapshot):
        raise ValueError("data_snapshot must be RegalDataSnapshot")
    prior_forecasts = _ordered_prior_forecasts(prior_forecasts)
    futility_forecasts = _ordered_futility_forecasts(futility_forecasts)
    primary = next(
        item
        for item in prior_forecasts
        if item.sensitivity_name == PRIMARY_MODEL_WEIGHT_SENSITIVITY
    )
    no_futility = futility_forecasts[0]
    if _forecast_conditioning_signature(primary) != _forecast_conditioning_signature(
        no_futility
    ):
        raise ValueError(
            "the primary prior row and no-futility row must reuse identical projections"
        )
    history = data_snapshot.history
    for forecast in prior_forecasts + futility_forecasts:
        for item in forecast.family_results:
            design = item.conditioning.design
            if design.interim_events != history.interim_event_threshold or (
                design.final_events != history.final_event_threshold
            ):
                raise ValueError("forecast design does not match the public-data thresholds")

    draw_budget = _importance_draws(prior_forecasts + futility_forecasts)
    ready = primary.is_posterior_forecast
    issues = list(primary.forecast_readiness_issues)
    if not ready and not issues:
        issues.append("The primary model average did not clear release readiness.")
    headline = None
    if ready:
        headline = {
            "metric": "final_rejection_given_public_history_and_continuation",
            "label": "P(final rejection | public history, interim continuation)",
            "value": primary.p_final_rejection_given_public_history_and_continuation,
            "model_weight_sensitivity": PRIMARY_MODEL_WEIGHT_SENSITIVITY,
            "assumed_futility_hr_threshold": None,
        }

    bundle = {
        "schema_version": RESULT_BUNDLE_SCHEMA_VERSION,
        "bundle_type": RESULT_BUNDLE_TYPE,
        "model_version": MODEL_VERSION,
        "generated_at": metadata.generated_at_iso,
        "source_revision": metadata.source_revision,
        "public_data": dict(data_snapshot.to_mapping()),
        "design": _design_record(primary.family_results[0].conditioning.design),
        "run": {
            "seed": metadata.seed,
            "importance_draws_per_family": draw_budget,
            "effect_family_count": len(REQUIRED_EFFECT_FAMILIES),
            "primary_model_weight_sensitivity": PRIMARY_MODEL_WEIGHT_SENSITIVITY,
            "primary_futility_assumption": "disabled; actual rule unpublished",
        },
        "release": {
            "status": "ready" if ready else "withheld",
            "is_posterior_forecast": ready,
            "headline": headline,
            "readiness_issues": issues,
            "disclosure": (
                "The headline is a continuation-conditioned posterior model average "
                "under explicit analyst priors and the displayed futility assumption."
                if ready
                else "No headline probability is published because the primary result "
                "did not clear every numerical readiness gate."
            ),
        },
        "prior_sensitivity": [
            _forecast_record(item, include_families=True)
            for item in prior_forecasts
        ],
        "futility_sensitivity": [
            _forecast_record(item, include_families=False)
            for item in futility_forecasts
        ],
    }
    validate_result_bundle(bundle)
    return bundle


def build_unpublished_result_bundle(
    *,
    generated_at,
    source_revision="unpublished",
    data_snapshot=None,
):
    """Build a schema-valid placeholder that cannot expose a headline value."""

    if not isinstance(generated_at, datetime) or generated_at.tzinfo is None:
        raise ValueError("generated_at must be a timezone-aware datetime")
    generated_at_iso = generated_at.astimezone(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    if data_snapshot is None:
        data_snapshot = load_regal_data_snapshot()
    bundle = {
        "schema_version": RESULT_BUNDLE_SCHEMA_VERSION,
        "bundle_type": RESULT_BUNDLE_TYPE,
        "model_version": MODEL_VERSION,
        "generated_at": generated_at_iso,
        "source_revision": str(source_revision).strip() or "unpublished",
        "public_data": dict(data_snapshot.to_mapping()),
        "design": _design_record(
            TrialDecisionDesign(
                interim_events=data_snapshot.history.interim_event_threshold,
                final_events=data_snapshot.history.final_event_threshold,
            )
        ),
        "run": {
            "seed": None,
            "importance_draws_per_family": None,
            "effect_family_count": len(REQUIRED_EFFECT_FAMILIES),
            "primary_model_weight_sensitivity": PRIMARY_MODEL_WEIGHT_SENSITIVITY,
            "primary_futility_assumption": "disabled; actual rule unpublished",
        },
        "release": {
            "status": "not_run",
            "is_posterior_forecast": False,
            "headline": None,
            "readiness_issues": [
                "No production REGAL posterior run has been bundled."
            ],
            "disclosure": (
                "The v2 reporting interface is present, but no headline probability "
                "is published without a production result that clears every gate."
            ),
        },
        "prior_sensitivity": [],
        "futility_sensitivity": [],
    }
    validate_result_bundle(bundle)
    return bundle


def _assert_json_finite(value, path="bundle"):
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_json_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{path} contains a non-finite JSON number")


def _validate_forecast_record(record, *, families_required):
    if not isinstance(record, dict):
        raise ValueError("forecast records must be objects")
    if not isinstance(record.get("name"), str) or not record["name"]:
        raise ValueError("forecast record name must be non-empty")
    if not isinstance(record.get("is_posterior_forecast"), bool):
        raise ValueError("forecast readiness flag must be boolean")
    expected_status = (
        "posterior_forecast"
        if record["is_posterior_forecast"]
        else "diagnostic_only"
    )
    if record.get("estimate_status") != expected_status:
        raise ValueError("forecast estimate status differs from its readiness flag")
    issues = record.get("readiness_issues")
    if not isinstance(issues, list) or not all(isinstance(item, str) for item in issues):
        raise ValueError("forecast readiness issues must be a string list")
    if record["is_posterior_forecast"] and issues:
        raise ValueError("a ready forecast cannot retain readiness issues")
    if not record["is_posterior_forecast"] and not issues:
        raise ValueError("a diagnostic-only forecast must state its readiness issues")
    probabilities = record.get("probabilities")
    if not isinstance(probabilities, dict):
        raise ValueError("forecast probabilities must be an object")
    for name, value in probabilities.items():
        if value is not None:
            _probability(value, f"forecast probability {name}")
        elif record["is_posterior_forecast"]:
            raise ValueError("a ready forecast cannot contain null probabilities")
    families = record.get("families")
    if families_required:
        if not isinstance(families, list):
            raise ValueError("prior sensitivity rows must include families")
        expected = {family.value for family in REQUIRED_EFFECT_FAMILIES}
        observed = {item.get("family") for item in families if isinstance(item, dict)}
        if len(families) != len(expected) or observed != expected:
            raise ValueError("forecast families must cover every required effect family")
        prior_total = sum(float(item["prior_weight"]) for item in families)
        posterior_total = sum(float(item["posterior_weight"]) for item in families)
        if abs(prior_total - 1.0) > 1e-12 or abs(posterior_total - 1.0) > 1e-12:
            raise ValueError("forecast family weights must sum to one")
        for item in families:
            try:
                family = GPSEffectFamily(item["family"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("family record has an unsupported family") from error
            if item.get("label") != FAMILY_LABELS[family]:
                raise ValueError("family record label differs from the canonical label")
            _probability(item["prior_weight"], "family prior weight")
            _probability(item["posterior_weight"], "family posterior weight")
            family_probabilities = item.get("probabilities")
            if not isinstance(family_probabilities, dict):
                raise ValueError("family probabilities must be an object")
            for name, value in family_probabilities.items():
                if value is not None:
                    _probability(value, f"family probability {name}")
                elif record["is_posterior_forecast"]:
                    raise ValueError("a ready family cannot contain null probabilities")
            if not isinstance(item.get("parameter_priors"), dict):
                raise ValueError("family parameter priors must be an object")
            diagnostics = item.get("diagnostics")
            if not isinstance(diagnostics, dict):
                raise ValueError("family diagnostics must be an object")
            for count_name in (
                "importance_draws",
                "history_compatible_draws",
                "continuation_compatible_draws",
                "interim_efficacy_draws",
                "interim_futility_draws",
                "non_estimable_interim_draws",
                "final_rejection_draws",
                "final_non_rejection_draws",
                "final_not_reached_draws",
                "tilt_attempts",
                "tilt_fallbacks",
                "draws_with_tilt_fallback",
                "proposal_infeasible_draws",
            ):
                count = diagnostics.get(count_name)
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError(f"family diagnostic {count_name} must be non-negative")
            if record["is_posterior_forecast"]:
                history_ess = diagnostics.get("history_effective_sample_size")
                continuation_ess = diagnostics.get(
                    "continuation_effective_sample_size"
                )
                maximum_share = diagnostics.get("maximum_history_weight_share")
                if history_ess is None or float(history_ess) < MINIMUM_POSTERIOR_FORECAST_ESS:
                    raise ValueError("a ready family must clear the history ESS gate")
                if continuation_ess is None or (
                    float(continuation_ess) < MINIMUM_POSTERIOR_FORECAST_ESS
                ):
                    raise ValueError("a ready family must clear the continuation ESS gate")
                if maximum_share is None or float(maximum_share) > (
                    MAXIMUM_POSTERIOR_FORECAST_HISTORY_WEIGHT_SHARE
                ):
                    raise ValueError(
                        "a ready family must clear the history-weight gate"
                    )
    elif families is not None:
        raise ValueError("futility sensitivity rows must not duplicate family records")


def validate_result_bundle(bundle):
    """Validate the versioned wire contract and release-label invariants."""

    if not isinstance(bundle, dict):
        raise ValueError("result bundle root must be an object")
    _assert_json_finite(bundle)
    if bundle.get("schema_version") != RESULT_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported result-bundle schema version")
    if bundle.get("bundle_type") != RESULT_BUNDLE_TYPE:
        raise ValueError("unsupported result-bundle type")
    if bundle.get("model_version") != MODEL_VERSION:
        raise ValueError("unsupported model version")
    generated_at = bundle.get("generated_at")
    if not isinstance(generated_at, str):
        raise ValueError("generated_at must be an ISO timestamp")
    try:
        parsed_time = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("generated_at must be an ISO timestamp") from error
    if parsed_time.tzinfo is None:
        raise ValueError("generated_at must include a timezone")
    if not isinstance(bundle.get("source_revision"), str) or not bundle[
        "source_revision"
    ].strip():
        raise ValueError("source_revision must be non-empty")

    public_data = bundle.get("public_data")
    if not isinstance(public_data, dict):
        raise ValueError("public_data must be an object")
    source = public_data.get("source")
    if not isinstance(source, dict) or not re.fullmatch(
        r"[0-9a-f]{64}", str(source.get("sha256", ""))
    ):
        raise ValueError("public-data source must contain a SHA-256 digest")
    if not isinstance(public_data.get("observations"), list) or not public_data[
        "observations"
    ]:
        raise ValueError("public_data must include observations")

    design = bundle.get("design")
    if not isinstance(design, dict):
        raise ValueError("design must be an object")
    if design.get("interim_events") != public_data.get("interim_event_threshold") or (
        design.get("final_events") != public_data.get("final_event_threshold")
    ):
        raise ValueError("bundle design and public-data thresholds differ")

    run = bundle.get("run")
    if not isinstance(run, dict):
        raise ValueError("run must be an object")
    if run.get("effect_family_count") != len(REQUIRED_EFFECT_FAMILIES):
        raise ValueError("run effect-family count is incorrect")
    if run.get("primary_model_weight_sensitivity") != PRIMARY_MODEL_WEIGHT_SENSITIVITY:
        raise ValueError("run primary model-weight sensitivity is incorrect")
    seed = run.get("seed")
    draws = run.get("importance_draws_per_family")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("run seed must be null or a non-negative integer")
    if draws is not None and (
        isinstance(draws, bool) or not isinstance(draws, int) or draws < 1
    ):
        raise ValueError("importance_draws_per_family must be null or positive")

    release = bundle.get("release")
    if not isinstance(release, dict):
        raise ValueError("release must be an object")
    status = release.get("status")
    if status not in {"ready", "withheld", "not_run"}:
        raise ValueError("release status is unsupported")
    ready = release.get("is_posterior_forecast")
    if not isinstance(ready, bool):
        raise ValueError("release readiness flag must be boolean")
    issues = release.get("readiness_issues")
    if not isinstance(issues, list) or not all(isinstance(item, str) for item in issues):
        raise ValueError("release readiness issues must be a string list")
    headline = release.get("headline")
    if ready:
        if status != "ready" or issues or not isinstance(headline, dict):
            raise ValueError("a ready release requires an issue-free headline")
        if headline.get("metric") != (
            "final_rejection_given_public_history_and_continuation"
        ) or headline.get("label") != (
            "P(final rejection | public history, interim continuation)"
        ):
            raise ValueError("release headline metric or label is not canonical")
        _probability(headline.get("value"), "release headline")
        if headline.get("model_weight_sensitivity") != PRIMARY_MODEL_WEIGHT_SENSITIVITY:
            raise ValueError("release headline uses the wrong model-weight sensitivity")
        if headline.get("assumed_futility_hr_threshold") is not None:
            raise ValueError("release headline must use the disclosed no-futility baseline")
    else:
        if status == "ready" or headline is not None or not issues:
            raise ValueError("a non-ready release must withhold its headline and state why")

    prior_rows = bundle.get("prior_sensitivity")
    futility_rows = bundle.get("futility_sensitivity")
    if not isinstance(prior_rows, list) or not isinstance(futility_rows, list):
        raise ValueError("sensitivity sections must be lists")
    if status == "not_run":
        if prior_rows or futility_rows or seed is not None or draws is not None:
            raise ValueError("a not-run bundle cannot contain analysis results")
        return bundle
    if seed is None or draws is None:
        raise ValueError("an analyzed bundle must record its seed and draw budget")

    expected_prior_names = [
        item.name for item in DEFAULT_MODEL_FAMILY_PRIOR_SENSITIVITY
    ]
    if [item.get("name") for item in prior_rows] != expected_prior_names:
        raise ValueError("prior sensitivity rows are incomplete or out of order")
    for row in prior_rows:
        _validate_forecast_record(row, families_required=True)
    observed_thresholds = [
        item.get("assumed_futility_hr_threshold") for item in futility_rows
    ]
    if observed_thresholds != list(FUTILITY_HR_SENSITIVITY_GRID):
        raise ValueError("futility sensitivity rows are incomplete or out of order")
    for row in futility_rows:
        _validate_forecast_record(row, families_required=False)
    primary = next(
        item for item in prior_rows if item["name"] == PRIMARY_MODEL_WEIGHT_SENSITIVITY
    )
    if any(item["is_posterior_forecast"] != ready for item in prior_rows):
        raise ValueError("model-prior rows disagree on shared numerical readiness")
    if primary["is_posterior_forecast"] != ready:
        raise ValueError("release readiness differs from the primary forecast row")
    if primary["readiness_issues"] != issues:
        raise ValueError("release issues differ from the primary forecast row")
    if ready and headline["value"] != primary["probabilities"][
        "final_rejection_given_public_history_and_continuation"
    ]:
        raise ValueError("release headline differs from the primary forecast value")
    return bundle


def canonical_result_json(bundle):
    validate_result_bundle(bundle)
    return json.dumps(
        bundle,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _load_json_strict(text):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant {value} is not allowed")

    return json.loads(text, parse_constant=reject_constant)


def load_result_bundle(path=DEFAULT_RESULT_BUNDLE_PATH):
    bundle = _load_json_strict(Path(path).read_text(encoding="utf-8"))
    validate_result_bundle(bundle)
    return bundle


def _embedded_json(bundle):
    # Escaping '<' prevents a source title or diagnostic from terminating the
    # application/json script element. The JSON parser restores the characters.
    return (
        canonical_result_json(bundle)
        .rstrip("\n")
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def embed_result_bundle(html_source, bundle):
    """Replace the one marked HTML bundle block and return updated source."""

    validate_result_bundle(bundle)
    if html_source.count(RESULT_BUNDLE_START) != 1 or html_source.count(
        RESULT_BUNDLE_END
    ) != 1:
        raise ValueError("HTML must contain exactly one result-bundle marker pair")
    pattern = re.compile(
        re.escape(RESULT_BUNDLE_START)
        + r".*?"
        + re.escape(RESULT_BUNDLE_END),
        re.DOTALL,
    )
    block = (
        RESULT_BUNDLE_START
        + '\n<script id="regal-v2-result" type="application/json">\n'
        + _embedded_json(bundle)
        + "\n</script>\n"
        + RESULT_BUNDLE_END
    )
    updated, count = pattern.subn(lambda _match: block, html_source)
    if count != 1:
        raise ValueError("HTML result-bundle block could not be replaced")
    return updated


def extract_embedded_result_bundle(html_source):
    """Parse and validate the JSON bundle embedded in the self-contained HTML."""

    start = html_source.find(RESULT_BUNDLE_START)
    end = html_source.find(RESULT_BUNDLE_END)
    if start < 0 or end < 0 or end <= start:
        raise ValueError("HTML result-bundle markers are missing or misordered")
    block = html_source[start:end]
    match = re.search(
        r'<script\s+id="regal-v2-result"\s+type="application/json">\s*'
        r"(?P<payload>.*?)\s*</script>",
        block,
        re.DOTALL,
    )
    if match is None:
        raise ValueError("HTML embedded result bundle was not found")
    bundle = _load_json_strict(match.group("payload"))
    validate_result_bundle(bundle)
    return bundle


def _atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def write_published_artifacts(
    bundle,
    *,
    json_path=DEFAULT_RESULT_BUNDLE_PATH,
    html_path=DEFAULT_HTML_PATH,
):
    """Write canonical JSON and embed the identical payload in the HTML."""

    validate_result_bundle(bundle)
    html_path = Path(html_path)
    updated_html = embed_result_bundle(
        html_path.read_text(encoding="utf-8"), bundle
    )
    _atomic_write(json_path, canonical_result_json(bundle))
    _atomic_write(html_path, updated_html)
    validate_published_artifacts(json_path=json_path, html_path=html_path)


def validate_published_artifacts(
    *,
    json_path=DEFAULT_RESULT_BUNDLE_PATH,
    html_path=DEFAULT_HTML_PATH,
):
    external = load_result_bundle(json_path)
    embedded = extract_embedded_result_bundle(
        Path(html_path).read_text(encoding="utf-8")
    )
    if embedded != external:
        raise ValueError("HTML and JSON result bundles differ")
    return external


def _family_worker(effect_prior, thresholds, nsim, seed):
    return condition_effect_family_futility_sensitivity(
        effect_prior,
        thresholds=thresholds,
        nsim=nsim,
        seed=seed,
    )


def run_regal_forecast_analysis(
    *,
    nsim=DEFAULT_IMPORTANCE_DRAWS,
    seed=20260825,
    workers=1,
    thresholds=FUTILITY_HR_SENSITIVITY_GRID,
):
    """Run every family once and return prior and futility model averages."""

    if isinstance(nsim, bool) or not isinstance(nsim, int) or nsim < 1:
        raise ValueError("nsim must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    thresholds = tuple(thresholds)
    by_family = {}
    if workers == 1:
        for prior in DEFAULT_EFFECT_FAMILY_PRIORS:
            by_family[prior.family] = _family_worker(
                prior, thresholds, nsim, seed
            )
            print(f"completed {prior.family.value}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_family_worker, prior, thresholds, nsim, seed): prior.family
                for prior in DEFAULT_EFFECT_FAMILY_PRIORS
            }
            for future in as_completed(futures):
                family = futures[future]
                by_family[family] = future.result()
                print(f"completed {family.value}", flush=True)
    family_rows = tuple(by_family[family] for family in REQUIRED_EFFECT_FAMILIES)
    if any(len(rows) != len(thresholds) for rows in family_rows):
        raise RuntimeError("effect-family futility grids are misaligned")
    projection_rows = tuple(
        tuple(family_rows[index][row] for index in range(len(family_rows)))
        for row in range(len(thresholds))
    )
    prior_forecasts = posterior_prior_sensitivity(projection_rows[0])
    futility_forecasts = tuple(
        posterior_model_average(row, BALANCED_MODEL_FAMILY_PRIOR)
        for row in projection_rows
    )
    return prior_forecasts, futility_forecasts


def _git_revision():
    environment = os.environ.get("GITHUB_SHA", "").strip()
    if environment:
        return environment
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _build_command(args):
    prior, futility = run_regal_forecast_analysis(
        nsim=args.nsim,
        seed=args.seed,
        workers=args.workers,
    )
    metadata = AnalysisRunMetadata(
        generated_at=datetime.now(timezone.utc),
        source_revision=args.source_revision or _git_revision(),
        seed=args.seed,
    )
    bundle = build_result_bundle(prior, futility, metadata=metadata)
    if args.require_ready and not bundle["release"]["is_posterior_forecast"]:
        raise RuntimeError(
            "production result did not clear release gates: "
            + "; ".join(bundle["release"]["readiness_issues"])
        )
    write_published_artifacts(
        bundle, json_path=args.output_json, html_path=args.html
    )
    print(
        "published result bundle with release status "
        + bundle["release"]["status"],
        flush=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="validate the committed JSON and embedded HTML bundles"
    )
    validate_parser.add_argument(
        "--input-json", type=Path, default=DEFAULT_RESULT_BUNDLE_PATH
    )
    validate_parser.add_argument("--html", type=Path, default=DEFAULT_HTML_PATH)

    build_parser = subparsers.add_parser(
        "build", help="run the canonical v2 analysis and publish its bundle"
    )
    build_parser.add_argument("--nsim", type=int, default=DEFAULT_IMPORTANCE_DRAWS)
    build_parser.add_argument("--seed", type=int, default=20260825)
    build_parser.add_argument(
        "--workers", type=int, default=max(1, min(os.cpu_count() or 1, 4))
    )
    build_parser.add_argument("--source-revision")
    build_parser.add_argument("--require-ready", action="store_true")
    build_parser.add_argument(
        "--output-json", type=Path, default=DEFAULT_RESULT_BUNDLE_PATH
    )
    build_parser.add_argument("--html", type=Path, default=DEFAULT_HTML_PATH)

    args = parser.parse_args(argv)
    if args.command == "validate":
        bundle = validate_published_artifacts(
            json_path=args.input_json, html_path=args.html
        )
        print(
            "result bundle validation passed (release status: "
            + bundle["release"]["status"]
            + ")"
        )
        return 0
    _build_command(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "AnalysisRunMetadata",
    "DEFAULT_HTML_PATH",
    "DEFAULT_RESULT_BUNDLE_PATH",
    "MODEL_VERSION",
    "RESULT_BUNDLE_SCHEMA_VERSION",
    "RESULT_BUNDLE_TYPE",
    "build_result_bundle",
    "build_unpublished_result_bundle",
    "canonical_result_json",
    "embed_result_bundle",
    "extract_embedded_result_bundle",
    "load_result_bundle",
    "run_regal_forecast_analysis",
    "validate_published_artifacts",
    "validate_result_bundle",
    "write_published_artifacts",
)
