"""REGAL v2 public-history data and joint count likelihoods.

The legacy explorer fits three cumulative event totals with weighted least
squares.  That treats cumulative observations as though they were independent
continuous measurements.  They are neither: enrollment and death counts are
integer-valued, and successive cumulative totals share all earlier patients and
events.

This module keeps the public disclosures typed and evaluates them jointly.  A
piecewise accrual distribution is conditional on the fixed randomized total, so
its interval counts are multinomial rather than independent Poisson variables.
For event history, each patient contributes a categorical probability of dying
in one of the disclosure intervals or surviving beyond them.  Dynamic
programming sums the resulting Poisson-multinomial probability over every
patient allocation compatible with the cumulative-count constraints.  This
preserves the correlation between increments and does not collapse the public
history to one fitted pooled survival curve.

The 80th-event disclosure is an announcement-process right censor: if the
threshold-to-announcement lag is ``L``, absence of an announcement by date ``d``
implies fewer than 80 events at ``d - L``.  The public-history JSON makes that
lag distribution explicit.  No quantity in this module conditions on the
interim treatment decision or constitutes a posterior REGAL forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from itertools import product
import json
from math import exp, isfinite, log, prod
from numbers import Integral
from pathlib import Path
from typing import Callable, Mapping, Optional, Tuple

import numpy as np


REGAL_PUBLIC_HISTORY_PATH = (
    Path(__file__).resolve().parent / "data" / "regal_public_history.json"
)
PROBABILITY_TOLERANCE = 1e-12
DEFAULT_MAX_DP_STATES = 4_000_000


def _parse_date(value, name, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, datetime):
        raise ValueError(f"{name} must be a date without a time component")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"{name} must be an ISO YYYY-MM-DD date") from error
    raise ValueError(f"{name} must be an ISO YYYY-MM-DD date")


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _nonempty_string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


class ObservationType(str, Enum):
    """Semantics of one cumulative-count disclosure."""

    EXACT_AS_OF = "exact_as_of"
    THRESHOLD_HIT = "threshold_hit"
    THRESHOLD_REACHED_BY = "threshold_reached_by"
    COUNT_INTERVAL = "count_interval"
    PROJECTED_COUNT_INTERVAL = "projected_count_interval"
    THRESHOLD_NOT_ANNOUNCED = "threshold_not_announced"


@dataclass(frozen=True)
class SourceRecord:
    """Primary or explanatory source attached to one public observation."""

    title: str
    url: str
    published_date: date

    def __post_init__(self):
        title = _nonempty_string(self.title, "source title")
        url = _nonempty_string(self.url, "source URL")
        if not url.startswith("https://"):
            raise ValueError("source URL must use https")
        published = _parse_date(self.published_date, "source published_date")
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "published_date", published)


@dataclass(frozen=True)
class ReportingLag:
    """Discrete threshold-to-announcement lag distribution in calendar days."""

    distribution: str
    days: Tuple[int, ...]
    probabilities: Tuple[float, ...]

    def __post_init__(self):
        distribution = _nonempty_string(self.distribution, "lag distribution")
        allowed_distributions = {
            "not_applicable",
            "fixed",
            "discrete_uniform",
            "discrete_pmf",
        }
        if distribution not in allowed_distributions:
            raise ValueError("unsupported lag distribution")
        days = tuple(_integer(value, "reporting lag day") for value in self.days)
        if any(isinstance(value, (bool, np.bool_)) for value in self.probabilities):
            raise ValueError("lag probabilities must be numeric, not boolean")
        try:
            probabilities = tuple(float(value) for value in self.probabilities)
        except (TypeError, ValueError) as error:
            raise ValueError("lag probabilities must be numeric") from error
        if not days or len(days) != len(probabilities):
            raise ValueError("lag days and probabilities must be non-empty and aligned")
        if len(set(days)) != len(days) or tuple(sorted(days)) != days:
            raise ValueError("lag days must be unique and increasing")
        if any(not isfinite(value) or value <= 0.0 for value in probabilities):
            raise ValueError("lag probabilities must be finite and positive")
        if distribution == "not_applicable" and days != (0,):
            raise ValueError("not_applicable lag must be zero days")
        if distribution == "fixed" and len(days) != 1:
            raise ValueError("fixed lag must contain exactly one day")
        total = sum(probabilities)
        if not abs(total - 1.0) <= PROBABILITY_TOLERANCE:
            raise ValueError("lag probabilities must sum to one")
        probabilities = tuple(value / total for value in probabilities)
        if distribution == "discrete_uniform":
            expected_days = tuple(range(days[0], days[-1] + 1))
            expected_probability = 1.0 / len(days)
            if days != expected_days or any(
                abs(value - expected_probability) > PROBABILITY_TOLERANCE
                for value in probabilities
            ):
                raise ValueError(
                    "discrete_uniform lag requires consecutive equally weighted days"
                )
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "days", days)
        object.__setattr__(self, "probabilities", probabilities)

    @classmethod
    def from_mapping(cls, payload):
        if not isinstance(payload, Mapping):
            raise ValueError("reporting_lag must be an object")
        distribution = payload.get("distribution")
        if distribution == "not_applicable":
            return cls(distribution, (0,), (1.0,))
        if distribution == "fixed":
            return cls(distribution, (_integer(payload.get("days"), "lag days"),), (1.0,))
        if distribution == "discrete_uniform":
            lower = _integer(payload.get("min_days"), "min_days")
            upper = _integer(payload.get("max_days"), "max_days")
            if upper < lower:
                raise ValueError("max_days must be at least min_days")
            days = tuple(range(lower, upper + 1))
            probability = 1.0 / len(days)
            return cls(distribution, days, (probability,) * len(days))
        if distribution == "discrete_pmf":
            values = payload.get("values")
            if not isinstance(values, list) or not values:
                raise ValueError("discrete_pmf requires a non-empty values list")
            pairs = sorted(
                (
                    _integer(item.get("days"), "lag days"),
                    float(item.get("probability")),
                )
                for item in values
            )
            return cls(
                distribution,
                tuple(item[0] for item in pairs),
                tuple(item[1] for item in pairs),
            )
        raise ValueError(
            "lag distribution must be not_applicable, fixed, "
            "discrete_uniform, or discrete_pmf"
        )

    @property
    def choices(self):
        return tuple(zip(self.days, self.probabilities))


@dataclass(frozen=True)
class CountConstraint:
    """Lower and upper cumulative-count bounds at one calendar cutoff."""

    cutoff_date: date
    lower: int
    upper: int
    label: str = ""

    def __post_init__(self):
        cutoff = _parse_date(self.cutoff_date, "cutoff_date")
        lower = _integer(self.lower, "lower")
        upper = _integer(self.upper, "upper")
        if upper < lower:
            raise ValueError("constraint upper must be at least lower")
        label = self.label
        if not isinstance(label, str):
            raise ValueError("constraint label must be a string")
        object.__setattr__(self, "cutoff_date", cutoff)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class CountObservation:
    """One typed enrollment or event-count disclosure."""

    observation_id: str
    observation_date: Optional[date]
    announcement_date: Optional[date]
    observation_type: ObservationType
    count: int
    count_lower: int
    count_upper: int
    reporting_lag: ReportingLag
    source: SourceRecord
    notes: str
    use_in_likelihood: bool = True
    accrual_anchor: bool = False

    def __post_init__(self):
        observation_id = _nonempty_string(self.observation_id, "observation id")
        observation_date = _parse_date(
            self.observation_date, "observation_date", allow_none=True
        )
        announcement_date = _parse_date(
            self.announcement_date, "announcement_date", allow_none=True
        )
        try:
            observation_type = ObservationType(self.observation_type)
        except ValueError as error:
            raise ValueError("unsupported observation_type") from error
        count = _integer(self.count, "count")
        lower = _integer(self.count_lower, "count_lower")
        upper = _integer(self.count_upper, "count_upper")
        if (
            observation_type is not ObservationType.THRESHOLD_NOT_ANNOUNCED
            and not lower <= count <= upper
        ):
            raise ValueError("count must lie inside count_lower/count_upper")
        if observation_type in (
            ObservationType.EXACT_AS_OF,
            ObservationType.THRESHOLD_HIT,
        ) and not (lower == count == upper):
            raise ValueError("exact and threshold-hit observations require exact bounds")
        if (
            observation_type is ObservationType.THRESHOLD_REACHED_BY
            and (lower != count or upper < count)
        ):
            raise ValueError(
                "threshold-reached-by observations require lower bound equal to count"
            )
        if observation_type is ObservationType.THRESHOLD_NOT_ANNOUNCED:
            if lower != 0 or upper != count - 1:
                raise ValueError(
                    "threshold-not-announced bounds must be zero through count minus one"
                )
            if observation_date is None or announcement_date is not None:
                raise ValueError(
                    "threshold-not-announced requires an as-of observation date "
                    "and no announcement date"
                )
        elif observation_type is ObservationType.THRESHOLD_HIT:
            if observation_date is None and announcement_date is None:
                raise ValueError(
                    "threshold-hit requires an observation or announcement date"
                )
        elif observation_date is None:
            raise ValueError("this observation type requires observation_date")
        if (
            observation_date is not None
            and announcement_date is not None
            and announcement_date < observation_date
            and observation_type is not ObservationType.PROJECTED_COUNT_INTERVAL
        ):
            raise ValueError(
                "announcement_date cannot precede a non-projected observation_date"
            )
        if not isinstance(self.reporting_lag, ReportingLag):
            raise ValueError("reporting_lag must be ReportingLag")
        if (
            observation_date is not None
            and announcement_date is not None
            and observation_type is not ObservationType.PROJECTED_COUNT_INTERVAL
            and self.reporting_lag.distribution == "fixed"
        ):
            observed_lag = (announcement_date - observation_date).days
            if self.reporting_lag.days != (observed_lag,):
                raise ValueError(
                    "fixed reporting lag must match announcement minus observation date"
                )
        if not isinstance(self.source, SourceRecord):
            raise ValueError("source must be SourceRecord")
        notes = _nonempty_string(self.notes, "notes")
        if not isinstance(self.use_in_likelihood, bool):
            raise ValueError("use_in_likelihood must be boolean")
        if not isinstance(self.accrual_anchor, bool):
            raise ValueError("accrual_anchor must be boolean")
        if (
            observation_type is ObservationType.PROJECTED_COUNT_INTERVAL
            and self.use_in_likelihood
        ):
            raise ValueError("planning projections cannot be likelihood evidence")
        object.__setattr__(self, "observation_id", observation_id)
        object.__setattr__(self, "observation_date", observation_date)
        object.__setattr__(self, "announcement_date", announcement_date)
        object.__setattr__(self, "observation_type", observation_type)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "count_lower", lower)
        object.__setattr__(self, "count_upper", upper)
        object.__setattr__(self, "notes", notes)

    @property
    def is_projection(self):
        return self.observation_type is ObservationType.PROJECTED_COUNT_INTERVAL

    def cutoff_choices(self):
        """Return constraint/weight alternatives induced by reporting lag."""

        if not self.use_in_likelihood or self.is_projection:
            return ()
        if self.observation_type is ObservationType.THRESHOLD_NOT_ANNOUNCED:
            return tuple(
                (
                    CountConstraint(
                        self.observation_date - timedelta(days=lag),
                        self.count_lower,
                        self.count_upper,
                        f"{self.observation_id}:lag={lag}",
                    ),
                    probability,
                )
                for lag, probability in self.reporting_lag.choices
            )
        if (
            self.observation_type is ObservationType.THRESHOLD_HIT
            and self.observation_date is None
        ):
            return tuple(
                (
                    CountConstraint(
                        self.announcement_date - timedelta(days=lag),
                        self.count,
                        self.count,
                        f"{self.observation_id}:lag={lag}",
                    ),
                    probability,
                )
                for lag, probability in self.reporting_lag.choices
            )
        return (
            (
                CountConstraint(
                    self.observation_date,
                    self.count_lower,
                    self.count_upper,
                    self.observation_id,
                ),
                1.0,
            ),
        )


def _event_calendar_validation_constraints(events, total):
    """Return count bounds that must hold regardless of unknown event lags."""

    constraints = []
    for observation in events:
        if observation.observation_type is ObservationType.THRESHOLD_NOT_ANNOUNCED:
            earliest_censor_cutoff = observation.observation_date - timedelta(
                days=max(observation.reporting_lag.days)
            )
            constraints.append(
                CountConstraint(
                    earliest_censor_cutoff,
                    observation.count_lower,
                    observation.count_upper,
                    f"{observation.observation_id}:definite-censor-bound",
                )
            )
            continue
        if observation.observation_date is not None:
            constraints.append(
                CountConstraint(
                    observation.observation_date,
                    observation.count_lower,
                    observation.count_upper,
                    observation.observation_id,
                )
            )
            continue
        if observation.observation_type is ObservationType.THRESHOLD_HIT:
            earliest_hit = observation.announcement_date - timedelta(
                days=max(observation.reporting_lag.days)
            )
            latest_hit = observation.announcement_date - timedelta(
                days=min(observation.reporting_lag.days)
            )
            constraints.extend(
                (
                    CountConstraint(
                        earliest_hit - timedelta(days=1),
                        0,
                        observation.count - 1,
                        f"{observation.observation_id}:before-earliest-hit",
                    ),
                    CountConstraint(
                        latest_hit,
                        observation.count,
                        total,
                        f"{observation.observation_id}:by-latest-hit",
                    ),
                )
            )
    return tuple(constraints)


@dataclass(frozen=True)
class PublicHistory:
    """Validated public inputs for one event-driven trial."""

    schema_version: int
    registry_id: str
    study_start: date
    target_enrollment: int
    interim_event_threshold: int
    final_event_threshold: int
    enrollment_observations: Tuple[CountObservation, ...]
    event_observations: Tuple[CountObservation, ...]

    def __post_init__(self):
        schema_version = _integer(self.schema_version, "schema_version", minimum=1)
        if schema_version != 1:
            raise ValueError(f"unsupported public-history schema version {schema_version}")
        registry_id = _nonempty_string(self.registry_id, "registry_id")
        study_start = _parse_date(self.study_start, "study_start")
        target = _integer(self.target_enrollment, "target_enrollment", minimum=1)
        interim = _integer(
            self.interim_event_threshold, "interim_event_threshold", minimum=1
        )
        final = _integer(
            self.final_event_threshold, "final_event_threshold", minimum=2
        )
        if not interim < final <= target:
            raise ValueError(
                "event thresholds must satisfy interim < final <= target enrollment"
            )
        enrollment = tuple(self.enrollment_observations)
        events = tuple(self.event_observations)
        if not enrollment or not events:
            raise ValueError("public history requires enrollment and event observations")
        if not all(isinstance(item, CountObservation) for item in enrollment + events):
            raise ValueError("public-history observations must be CountObservation")
        identifiers = [item.observation_id for item in enrollment + events]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("public-history observation ids must be unique")
        if any(item.count_upper > target for item in enrollment + events):
            raise ValueError("public counts cannot exceed target enrollment")
        if any(
            item.observation_date is not None
            and item.observation_date < study_start
            for item in enrollment + events
        ):
            raise ValueError("public observations cannot predate study start")
        dated_enrollment = sorted(
            (item.observation_date, item.count_lower, item.count_upper)
            for item in enrollment
            if item.observation_date is not None
        )
        for previous, current in zip(dated_enrollment, dated_enrollment[1:]):
            if current[1] < previous[1]:
                raise ValueError("enrollment counts must be non-decreasing over time")
        event_constraints = _event_calendar_validation_constraints(events, target)
        if merge_count_constraints(event_constraints, target) is None:
            raise ValueError("event count constraints are inconsistent over time")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "registry_id", registry_id)
        object.__setattr__(self, "study_start", study_start)
        object.__setattr__(self, "target_enrollment", target)
        object.__setattr__(self, "interim_event_threshold", interim)
        object.__setattr__(self, "final_event_threshold", final)
        object.__setattr__(self, "enrollment_observations", enrollment)
        object.__setattr__(self, "event_observations", events)


def _source_from_mapping(payload):
    if not isinstance(payload, Mapping):
        raise ValueError("source must be an object")
    return SourceRecord(
        title=payload.get("title"),
        url=payload.get("url"),
        published_date=payload.get("published_date"),
    )


def _observation_from_mapping(payload):
    if not isinstance(payload, Mapping):
        raise ValueError("observation must be an object")
    return CountObservation(
        observation_id=payload.get("id"),
        observation_date=payload.get("observation_date"),
        announcement_date=payload.get("announcement_date"),
        observation_type=payload.get("observation_type"),
        count=payload.get("count"),
        count_lower=payload.get("count_lower"),
        count_upper=payload.get("count_upper"),
        reporting_lag=ReportingLag.from_mapping(payload.get("reporting_lag")),
        source=_source_from_mapping(payload.get("source")),
        notes=payload.get("notes"),
        use_in_likelihood=payload.get("use_in_likelihood", True),
        accrual_anchor=payload.get("accrual_anchor", False),
    )


def load_regal_public_history(path=REGAL_PUBLIC_HISTORY_PATH):
    """Load and strictly validate the versioned REGAL public-history JSON."""

    source_path = Path(path)
    with source_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("public-history root must be an object")
    trial = payload.get("trial")
    if not isinstance(trial, Mapping):
        raise ValueError("public-history trial must be an object")
    return PublicHistory(
        schema_version=payload.get("schema_version"),
        registry_id=trial.get("registry_id"),
        study_start=trial.get("study_start"),
        target_enrollment=trial.get("target_enrollment"),
        interim_event_threshold=trial.get("interim_event_threshold"),
        final_event_threshold=trial.get("final_event_threshold"),
        enrollment_observations=tuple(
            _observation_from_mapping(item)
            for item in payload.get("enrollment_observations", ())
        ),
        event_observations=tuple(
            _observation_from_mapping(item)
            for item in payload.get("event_observations", ())
        ),
    )


def _normalize_numeric_bounds(lower_bounds, upper_bounds, total):
    if len(lower_bounds) != len(upper_bounds):
        raise ValueError("lower_bounds and upper_bounds must have the same length")
    if not lower_bounds:
        raise ValueError("at least one cumulative-count bound is required")
    lower = [_integer(value, "lower bound") for value in lower_bounds]
    upper = [_integer(value, "upper bound") for value in upper_bounds]
    if any(value > total for value in lower + upper):
        raise ValueError("count bounds cannot exceed the number of patients")
    for index in range(1, len(lower)):
        lower[index] = max(lower[index], lower[index - 1])
    for index in range(len(upper) - 2, -1, -1):
        upper[index] = min(upper[index], upper[index + 1])
    if any(lo > hi for lo, hi in zip(lower, upper)):
        return None
    return tuple(lower), tuple(upper)


def event_interval_probabilities(cumulative_probabilities):
    """Convert patient cumulative CDFs to mutually exclusive intervals.

    The final column is the probability of no event by the last cutoff.  WP6
    uses this public adapter to construct an importance proposal over the same
    interval representation used by the exact WP5 likelihood.
    """
    try:
        cumulative = np.asarray(cumulative_probabilities, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("cumulative probabilities must be numeric") from error
    if cumulative.ndim != 2 or cumulative.shape[0] < 1 or cumulative.shape[1] < 1:
        raise ValueError(
            "cumulative probabilities must be a non-empty patient-by-cutoff matrix"
        )
    if np.any(~np.isfinite(cumulative)):
        raise ValueError("cumulative probabilities must be finite")
    if np.any(cumulative < -PROBABILITY_TOLERANCE) or np.any(
        cumulative > 1.0 + PROBABILITY_TOLERANCE
    ):
        raise ValueError("cumulative probabilities must lie in [0, 1]")
    if np.any(np.diff(cumulative, axis=1) < -PROBABILITY_TOLERANCE):
        raise ValueError("cumulative probabilities must be non-decreasing by cutoff")
    cumulative = np.clip(cumulative, 0.0, 1.0)
    intervals = np.column_stack(
        (
            cumulative[:, 0],
            np.diff(cumulative, axis=1),
            1.0 - cumulative[:, -1],
        )
    )
    intervals = np.maximum(intervals, 0.0)
    row_totals = intervals.sum(axis=1)
    intervals /= row_totals[:, None]
    return intervals


def joint_cumulative_count_log_probability(
    cumulative_probabilities,
    lower_bounds,
    upper_bounds,
    *,
    max_states=DEFAULT_MAX_DP_STATES,
):
    """Log probability of correlated cumulative integer-count constraints.

    Each matrix row is one patient's cumulative event probability at ordered
    cutoffs.  Patients may have different enrollment dates, treatment profiles,
    and survival distributions.  Conditional independence is assumed across
    patients, but no independence is assumed across cutoffs.

    The dynamic program tracks event increments, pruning any increment that can
    no longer satisfy the cumulative upper bounds.  It then sums every retained
    increment vector satisfying all lower and upper bounds.
    """

    intervals = event_interval_probabilities(cumulative_probabilities)
    patient_count = intervals.shape[0]
    cutoff_count = intervals.shape[1] - 1
    if len(lower_bounds) != cutoff_count or len(upper_bounds) != cutoff_count:
        raise ValueError("one lower and upper bound is required per cutoff")
    bounds = _normalize_numeric_bounds(
        tuple(lower_bounds), tuple(upper_bounds), patient_count
    )
    if bounds is None:
        return float("-inf")
    lower, upper = bounds
    previous_lower = 0
    increment_upper = []
    for current_upper, current_lower in zip(upper, lower):
        increment_upper.append(current_upper - previous_lower)
        previous_lower = current_lower
    if any(value < 0 for value in increment_upper):
        return float("-inf")
    max_states = _integer(max_states, "max_states", minimum=1)
    shape = tuple(value + 1 for value in increment_upper)
    state_count = prod(shape)
    if state_count > max_states:
        raise ValueError(
            f"likelihood requires {state_count:,} DP states, above the "
            f"configured {max_states:,} limit"
        )

    dp = np.zeros(shape, dtype=float)
    dp[(0,) * cutoff_count] = 1.0
    log_scale = 0.0
    for patient_probabilities in intervals:
        updated = dp * patient_probabilities[-1]
        for axis, probability in enumerate(patient_probabilities[:-1]):
            if probability == 0.0 or shape[axis] == 1:
                continue
            source = [slice(None)] * cutoff_count
            target = [slice(None)] * cutoff_count
            source[axis] = slice(0, -1)
            target[axis] = slice(1, None)
            updated[tuple(target)] += dp[tuple(source)] * probability
        retained_mass = float(updated.sum())
        if retained_mass <= 0.0:
            return float("-inf")
        dp = updated / retained_mass
        log_scale += log(retained_mass)

    coordinates = np.indices(shape, sparse=True)
    cumulative = np.zeros(shape, dtype=np.int32)
    allowed = np.ones(shape, dtype=bool)
    for index, coordinate in enumerate(coordinates):
        cumulative = cumulative + coordinate
        allowed &= cumulative >= lower[index]
        allowed &= cumulative <= upper[index]
    allowed_mass = float(dp[allowed].sum())
    if allowed_mass <= 0.0:
        return float("-inf")
    return log_scale + log(allowed_mass)


def joint_cumulative_count_probability(*args, **kwargs):
    """Probability-scale wrapper for ``joint_cumulative_count_log_probability``."""

    result = joint_cumulative_count_log_probability(*args, **kwargs)
    return 0.0 if result == float("-inf") else exp(result)


def merge_count_constraints(constraints, total):
    """Merge same-date bounds and propagate monotonic cumulative limits.

    ``None`` denotes an inconsistent branch.  This is public because WP6 must
    sample the exact disclosure-lag branches represented by the WP5
    likelihood, rather than reimplementing subtly different count semantics.
    """
    by_date = {}
    labels = {}
    for constraint in constraints:
        if not isinstance(constraint, CountConstraint):
            raise ValueError("constraints must contain CountConstraint values")
        if constraint.upper > total:
            raise ValueError("constraint cannot exceed patient count")
        if constraint.cutoff_date not in by_date:
            by_date[constraint.cutoff_date] = [constraint.lower, constraint.upper]
            labels[constraint.cutoff_date] = [constraint.label]
        else:
            by_date[constraint.cutoff_date][0] = max(
                by_date[constraint.cutoff_date][0], constraint.lower
            )
            by_date[constraint.cutoff_date][1] = min(
                by_date[constraint.cutoff_date][1], constraint.upper
            )
            labels[constraint.cutoff_date].append(constraint.label)
    ordered = sorted(by_date)
    lower = [by_date[item][0] for item in ordered]
    upper = [by_date[item][1] for item in ordered]
    bounds = _normalize_numeric_bounds(lower, upper, total)
    if bounds is None:
        return None
    normalized_lower, normalized_upper = bounds
    return tuple(
        CountConstraint(
            cutoff,
            normalized_lower[index],
            normalized_upper[index],
            "+".join(filter(None, labels[cutoff])),
        )
        for index, cutoff in enumerate(ordered)
    )


@dataclass(frozen=True)
class PiecewiseEnrollmentModel:
    """Fixed-N piecewise-uniform enrollment-date distribution."""

    total_enrollment: int
    study_start: date
    phase_end_dates: Tuple[date, ...]
    phase_probabilities: Tuple[float, ...]

    def __post_init__(self):
        total = _integer(self.total_enrollment, "total_enrollment", minimum=1)
        start = _parse_date(self.study_start, "study_start")
        ends = tuple(
            _parse_date(value, "phase_end_date") for value in self.phase_end_dates
        )
        if any(
            isinstance(value, (bool, np.bool_)) for value in self.phase_probabilities
        ):
            raise ValueError("phase probabilities must be numeric, not boolean")
        try:
            probabilities = tuple(float(value) for value in self.phase_probabilities)
        except (TypeError, ValueError) as error:
            raise ValueError("phase probabilities must be numeric") from error
        if not ends or len(ends) != len(probabilities):
            raise ValueError("phase ends and probabilities must be non-empty and aligned")
        if any(current <= previous for previous, current in zip((start,) + ends, ends)):
            raise ValueError("phase end dates must be strictly increasing after study start")
        if any(not isfinite(value) or value <= 0.0 for value in probabilities):
            raise ValueError("phase probabilities must be finite and positive")
        probability_sum = sum(probabilities)
        if not abs(probability_sum - 1.0) <= PROBABILITY_TOLERANCE:
            raise ValueError("phase probabilities must sum to one")
        probabilities = tuple(value / probability_sum for value in probabilities)
        object.__setattr__(self, "total_enrollment", total)
        object.__setattr__(self, "study_start", start)
        object.__setattr__(self, "phase_end_dates", ends)
        object.__setattr__(self, "phase_probabilities", probabilities)

    @property
    def enrollment_close(self):
        return self.phase_end_dates[-1]

    @property
    def phase_start_dates(self):
        return (self.study_start,) + tuple(
            value + timedelta(days=1) for value in self.phase_end_dates[:-1]
        )

    def cumulative_probability(self, cutoff_date):
        cutoff = _parse_date(cutoff_date, "cutoff_date")
        if cutoff < self.study_start:
            return 0.0
        probability = 0.0
        for start, end, phase_probability in zip(
            self.phase_start_dates,
            self.phase_end_dates,
            self.phase_probabilities,
        ):
            if cutoff >= end:
                probability += phase_probability
                continue
            if cutoff >= start:
                elapsed_days = (cutoff - start).days + 1
                phase_days = (end - start).days + 1
                probability += phase_probability * elapsed_days / phase_days
            break
        return min(max(probability, 0.0), 1.0)

    def expected_cumulative_count(self, cutoff_date):
        return self.total_enrollment * self.cumulative_probability(cutoff_date)

    def cumulative_probability_matrix(self, cutoff_dates):
        probabilities = np.array(
            [self.cumulative_probability(value) for value in cutoff_dates],
            dtype=float,
        )
        return np.broadcast_to(
            probabilities, (self.total_enrollment, len(probabilities))
        ).copy()

    def sample_enrollment_dates(self, rng):
        """Draw exactly ``total_enrollment`` dates, never before study opening."""

        phase_draws = np.asarray(rng.random(self.total_enrollment), dtype=float)
        within_phase = np.asarray(rng.random(self.total_enrollment), dtype=float)
        if (
            phase_draws.shape != (self.total_enrollment,)
            or within_phase.shape != (self.total_enrollment,)
            or np.any(~np.isfinite(phase_draws))
            or np.any(~np.isfinite(within_phase))
            or np.any((phase_draws < 0.0) | (phase_draws >= 1.0))
            or np.any((within_phase < 0.0) | (within_phase >= 1.0))
        ):
            raise ValueError("rng.random must return finite draws in [0, 1)")
        cumulative_phase_probabilities = np.cumsum(self.phase_probabilities)
        cumulative_phase_probabilities[-1] = 1.0
        phase_indices = np.searchsorted(
            cumulative_phase_probabilities, phase_draws, side="right"
        )
        sampled = []
        starts = self.phase_start_dates
        for phase_index, draw in zip(phase_indices, within_phase):
            start = starts[int(phase_index)]
            end = self.phase_end_dates[int(phase_index)]
            duration = (end - start).days + 1
            sampled.append(start + timedelta(days=min(int(draw * duration), duration - 1)))
        return tuple(sampled)


def default_regal_enrollment_model(history=None):
    """Build the registry-anchored 20/104/126 reference accrual path.

    This is a convenient default parameterization, not an independent Bayesian
    prior. Later posterior work must avoid using likelihood evidence a second
    time when specifying an accrual-parameter prior.
    """

    if history is None:
        history = load_regal_public_history()
    if not isinstance(history, PublicHistory):
        raise ValueError("history must be PublicHistory")
    anchors = sorted(
        (item for item in history.enrollment_observations if item.accrual_anchor),
        key=lambda item: item.observation_date,
    )
    if not anchors or anchors[-1].count != history.target_enrollment:
        raise ValueError("accrual anchors must end at target enrollment")
    counts = [item.count for item in anchors]
    if any(current <= previous for previous, current in zip((0,) + tuple(counts), counts)):
        raise ValueError("accrual anchor point counts must be strictly increasing")
    increments = [counts[0]] + [
        current - previous for previous, current in zip(counts, counts[1:])
    ]
    return PiecewiseEnrollmentModel(
        total_enrollment=history.target_enrollment,
        study_start=history.study_start,
        phase_end_dates=tuple(item.observation_date for item in anchors),
        phase_probabilities=tuple(
            increment / history.target_enrollment for increment in increments
        ),
    )


@dataclass(frozen=True)
class EnrollmentAnchorCheck:
    observation_id: str
    observation_type: ObservationType
    cutoff_date: date
    count_lower: int
    point_count: int
    count_upper: int
    expected_count: float
    log_probability: float

    @property
    def reachable(self):
        return isfinite(self.log_probability)

    @property
    def centered(self):
        return abs(self.expected_count - self.point_count) <= 1e-9


def enrollment_anchor_checks(history=None, model=None):
    """Return reachability and reference-centering diagnostics for every anchor."""

    if history is None:
        history = load_regal_public_history()
    if model is None:
        model = default_regal_enrollment_model(history)
    checks = []
    for observation in history.enrollment_observations:
        if not observation.accrual_anchor:
            continue
        matrix = model.cumulative_probability_matrix((observation.observation_date,))
        log_probability = joint_cumulative_count_log_probability(
            matrix,
            (observation.count_lower,),
            (observation.count_upper,),
        )
        checks.append(
            EnrollmentAnchorCheck(
                observation_id=observation.observation_id,
                observation_type=observation.observation_type,
                cutoff_date=observation.observation_date,
                count_lower=observation.count_lower,
                point_count=observation.count,
                count_upper=observation.count_upper,
                expected_count=model.expected_cumulative_count(
                    observation.observation_date
                ),
                log_probability=log_probability,
            )
        )
    return tuple(checks)


def enrollment_log_likelihood(
    history,
    model,
    *,
    max_lag_combinations=4096,
    max_states=DEFAULT_MAX_DP_STATES,
):
    """Joint fixed-N likelihood, mixed over enrollment reporting lags."""

    if not isinstance(history, PublicHistory):
        raise ValueError("history must be PublicHistory")
    if not isinstance(model, PiecewiseEnrollmentModel):
        raise ValueError("model must be PiecewiseEnrollmentModel")
    if model.total_enrollment != history.target_enrollment:
        raise ValueError("enrollment model and history totals differ")
    observations = [
        item
        for item in history.enrollment_observations
        if item.use_in_likelihood and not item.is_projection
    ]
    choice_sets = [item.cutoff_choices() for item in observations]
    return _constraint_mixture_log_likelihood(
        choice_sets,
        lambda cutoff: np.full(
            history.target_enrollment,
            model.cumulative_probability(cutoff),
            dtype=float,
        ),
        history.target_enrollment,
        provider_label="enrollment probability provider",
        max_lag_combinations=max_lag_combinations,
        max_states=max_states,
    )


@dataclass(frozen=True, eq=False)
class CalendarEventProbabilityProvider:
    """Convert entry dates and a survival callable to calendar event CDFs.

    This adapter treats ``1 - S(t)`` as the probability of an observable death,
    so it assumes complete follow-up apart from administrative censoring. It
    does not add an independent loss-to-follow-up or withdrawal process.
    Callers that need attrition should supply an adjusted calendar event-CDF
    provider directly to ``PublicHistoryLikelihood.event_log_likelihood``.
    """

    entry_dates: Tuple[date, ...]
    survival_probability: Callable[[np.ndarray], np.ndarray]
    days_per_time_unit: float = 30.4375

    def __post_init__(self):
        entries = tuple(_parse_date(value, "entry_date") for value in self.entry_dates)
        if not entries:
            raise ValueError("entry_dates must be non-empty")
        if not callable(self.survival_probability):
            raise ValueError("survival_probability must be callable")
        if isinstance(self.days_per_time_unit, (bool, np.bool_)):
            raise ValueError("days_per_time_unit must be numeric, not boolean")
        try:
            days_per_time_unit = float(self.days_per_time_unit)
        except (TypeError, ValueError) as error:
            raise ValueError("days_per_time_unit must be numeric") from error
        if not isfinite(days_per_time_unit) or days_per_time_unit <= 0.0:
            raise ValueError("days_per_time_unit must be finite and positive")
        object.__setattr__(self, "entry_dates", entries)
        object.__setattr__(self, "days_per_time_unit", days_per_time_unit)

    def __call__(self, cutoff_date):
        cutoff = _parse_date(cutoff_date, "cutoff_date")
        active = np.array([cutoff >= value for value in self.entry_dates], dtype=bool)
        result = np.zeros(len(self.entry_dates), dtype=float)
        if not np.any(active):
            return result
        followup = np.array(
            [
                (cutoff - value).days / self.days_per_time_unit
                for value, is_active in zip(self.entry_dates, active)
                if is_active
            ],
            dtype=float,
        )
        try:
            survival = np.asarray(self.survival_probability(followup), dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("survival callable must return numeric probabilities") from error
        if survival.ndim == 0:
            survival = np.full(len(followup), float(survival))
        if survival.shape != followup.shape:
            raise ValueError("survival callable must return one value per follow-up time")
        if np.any(~np.isfinite(survival)) or np.any(
            (survival < -PROBABILITY_TOLERANCE)
            | (survival > 1.0 + PROBABILITY_TOLERANCE)
        ):
            raise ValueError("survival callable must return finite values in [0, 1]")
        result[active] = 1.0 - np.clip(survival, 0.0, 1.0)
        return result


@dataclass(frozen=True, eq=False)
class EventIncrementTrajectory:
    """One sampled latent allocation of patient events across cutoff intervals."""

    increments: np.ndarray
    cumulative_counts: np.ndarray

    def __post_init__(self):
        increments = np.asarray(self.increments, dtype=int)
        cumulative = np.asarray(self.cumulative_counts, dtype=int)
        if increments.ndim != 1 or cumulative.shape != increments.shape:
            raise ValueError("increments and cumulative_counts must be aligned vectors")
        if np.any(increments < 0) or not np.array_equal(
            np.cumsum(increments), cumulative
        ):
            raise ValueError("cumulative_counts must be the cumulative increments")
        increments = np.array(increments, copy=True)
        cumulative = np.array(cumulative, copy=True)
        increments.setflags(write=False)
        cumulative.setflags(write=False)
        object.__setattr__(self, "increments", increments)
        object.__setattr__(self, "cumulative_counts", cumulative)


def sample_event_increment_trajectory(cumulative_probabilities, rng):
    """Sample a latent integer event trajectory from patient-specific CDFs."""

    intervals = event_interval_probabilities(cumulative_probabilities)
    draws = np.asarray(rng.random(intervals.shape[0]), dtype=float)
    if draws.shape != (intervals.shape[0],) or np.any(~np.isfinite(draws)) or np.any(
        (draws < 0.0) | (draws >= 1.0)
    ):
        raise ValueError("rng.random must return finite draws in [0, 1)")
    categories = []
    for probabilities, draw in zip(intervals, draws):
        cumulative = np.cumsum(probabilities)
        cumulative[-1] = 1.0
        categories.append(np.searchsorted(cumulative, draw, side="right"))
    categories = np.asarray(categories, dtype=int)
    cutoff_count = intervals.shape[1] - 1
    increments = np.bincount(categories, minlength=cutoff_count + 1)[:cutoff_count]
    return EventIncrementTrajectory(increments, np.cumsum(increments))


def _logsumexp(values):
    finite_values = [value for value in values if isfinite(value)]
    if not finite_values:
        return float("-inf")
    maximum = max(finite_values)
    return maximum + log(sum(exp(value - maximum) for value in finite_values))


def _constraint_mixture_log_likelihood(
    choice_sets,
    cumulative_probability,
    patient_count,
    *,
    provider_label,
    max_lag_combinations,
    max_states,
):
    """Mix one joint integer-count likelihood over disclosure-lag choices."""

    if not callable(cumulative_probability):
        raise ValueError(f"{provider_label} must be callable")
    patient_count = _integer(patient_count, "patient_count", minimum=1)
    choice_sets = tuple(tuple(choices) for choices in choice_sets)
    if not choice_sets:
        return 0.0
    if any(not choices for choices in choice_sets):
        raise ValueError("reporting-lag choice sets must be non-empty")
    combination_count = prod(len(choices) for choices in choice_sets)
    max_lag_combinations = _integer(
        max_lag_combinations, "max_lag_combinations", minimum=1
    )
    if combination_count > max_lag_combinations:
        raise ValueError(
            f"reporting-lag mixture has {combination_count:,} combinations, "
            f"above the configured {max_lag_combinations:,} limit"
        )

    probability_cache = {}
    mixture_terms = []
    for choices in product(*choice_sets):
        constraints = [choice[0] for choice in choices]
        mixture_weight = prod(choice[1] for choice in choices)
        if mixture_weight <= 0.0:
            continue
        merged = merge_count_constraints(constraints, patient_count)
        if merged is None:
            continue
        columns = []
        for constraint in merged:
            cutoff = constraint.cutoff_date
            if cutoff not in probability_cache:
                values = np.asarray(cumulative_probability(cutoff), dtype=float)
                if values.ndim != 1 or len(values) != patient_count:
                    raise ValueError(
                        f"{provider_label} must return one value per randomized patient"
                    )
                probability_cache[cutoff] = values
            columns.append(probability_cache[cutoff])
        matrix = np.column_stack(columns)
        component = joint_cumulative_count_log_probability(
            matrix,
            tuple(item.lower for item in merged),
            tuple(item.upper for item in merged),
            max_states=max_states,
        )
        if isfinite(component):
            mixture_terms.append(log(mixture_weight) + component)
    return _logsumexp(mixture_terms)


@dataclass(frozen=True)
class PublicHistoryLikelihood:
    """Correlated enrollment- and event-count likelihood components.

    The two methods are intentionally separate. ``event_log_likelihood`` is
    conditional on the patient-level calendar CDFs supplied by the caller.
    Multiplying it by the marginal enrollment-anchor likelihood is valid only
    after the caller has made the enrollment conditioning/integration
    consistent. WP6 performs that latent-history step; this class does not hide
    a factorization assumption behind a combined score.
    """

    history: PublicHistory
    enrollment_model: Optional[PiecewiseEnrollmentModel] = None

    def __post_init__(self):
        if not isinstance(self.history, PublicHistory):
            raise ValueError("history must be PublicHistory")
        model = self.enrollment_model
        if model is None:
            model = default_regal_enrollment_model(self.history)
        if not isinstance(model, PiecewiseEnrollmentModel):
            raise ValueError("enrollment_model must be PiecewiseEnrollmentModel")
        if model.total_enrollment != self.history.target_enrollment:
            raise ValueError("enrollment model and history totals differ")
        object.__setattr__(self, "enrollment_model", model)

    def enrollment_log_likelihood(
        self,
        *,
        max_lag_combinations=4096,
        max_states=DEFAULT_MAX_DP_STATES,
    ):
        return enrollment_log_likelihood(
            self.history,
            self.enrollment_model,
            max_lag_combinations=max_lag_combinations,
            max_states=max_states,
        )

    def event_log_likelihood(
        self,
        cumulative_event_probability: Callable[[date], np.ndarray],
        *,
        max_lag_combinations=4096,
        max_states=DEFAULT_MAX_DP_STATES,
    ):
        """Integrate the joint event-count likelihood over reporting-lag choices.

        Unknown lags attached to different disclosures are treated as
        independent, so their branch probabilities are multiplied. Any
        alternative dependence model must be represented explicitly by the
        caller rather than inferred from the shared sponsor.
        """

        if not callable(cumulative_event_probability):
            raise ValueError("cumulative_event_probability must be callable")
        choice_sets = [
            observation.cutoff_choices()
            for observation in self.history.event_observations
            if observation.use_in_likelihood and not observation.is_projection
        ]
        return _constraint_mixture_log_likelihood(
            choice_sets,
            cumulative_event_probability,
            self.history.target_enrollment,
            provider_label="event probability provider",
            max_lag_combinations=max_lag_combinations,
            max_states=max_states,
        )
