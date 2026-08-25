"""REGAL v2 public-history and interim-continuation conditioning.

Work package 5 evaluates public count likelihoods conditional on a supplied
calendar event CDF.  This module performs the latent-history integration needed
by work package 6.  Each importance draw uses one internally consistent set of
enrollment dates, protocol strata, randomized treatment assignments, censoring
times, disclosure lags, and event times.  The same draw must satisfy the public
enrollment/event constraints, produce an estimable 60-event statistic in the
continuation region, and then supplies the unresolved outcomes used at the
80-event final analysis.

The proposal enumerates the allowed cumulative integer-count vectors, samples
one vector, and uses a backward dynamic program to draw patient event intervals
exactly conditional on its quotas.  A mixture of continuation-centered
exponential tilts can target otherwise rare interim score regions.  Exact
target-to-proposal density ratios remove both the count conditioning and the
score tilts. Event times within a selected interval are drawn from the original
conditional distribution, so their conditional density cancels from the
weight. This is importance sampling, not rejection sampling and not a product
of the marginal enrollment likelihood and an unrelated event realization.

Outputs are *conditional fixed-scenario projections*.  Work package 7 must
still average them over survival families and parameter uncertainty before any
quantity can be described as a posterior REGAL forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from itertools import product
from math import exp, isfinite, log, sqrt
from numbers import Integral
from typing import Optional, Protocol, Tuple

import numpy as np

from event_likelihood import (
    DEFAULT_MAX_DP_STATES,
    CountConstraint,
    PiecewiseEnrollmentModel,
    PublicHistory,
    default_regal_enrollment_model,
    event_interval_probabilities,
    load_regal_public_history,
    merge_count_constraints,
)
from simulation import (
    FUTILITY_HR_SENSITIVITY_GRID,
    EventDrivenTrialData,
    REGAL_V2_EFFICACY_DESIGN,
    evaluate_event_driven_trial,
)
from trial_design import (
    FinalDecision,
    HazardRatioFutilityRule,
    InterimDecision,
    TrialDecisionDesign,
    binary_indicator,
)


PROBABILITY_TOLERANCE = 1e-12
DEFAULT_IMPORTANCE_DRAWS = 10_000
DEFAULT_TILT_TOLERANCE = 1e-8
DEFAULT_MAX_TILT_ITERATIONS = 80
DEFAULT_MAX_COUNT_VECTORS = 4096
TARGET_CATEGORY_MARGIN = 0.25


class TiltProposalError(RuntimeError):
    """A draw-specific tilt failure for which the base proposal remains valid.

    Direct callers of :func:`exponential_tilt_event_intervals` may catch this
    error.  The high-level conditioning functions catch it per component,
    retain the exact base proposal, and expose the fallback in their returned
    diagnostics.
    """


class _QuotaProposalInfeasibleError(ValueError):
    """A sampled proposal component cannot realize the selected count vector."""


class _QuotaProbabilityUnderflowError(FloatingPointError):
    """A structurally positive quota probability underflowed numerically."""


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parse_date(value, name):
    if isinstance(value, datetime):
        raise ValueError(f"{name} must be a date without a time component")
    if isinstance(value, date):
        return value
    raise ValueError(f"{name} must be a date")


def _logsumexp(values):
    values = tuple(float(value) for value in values if isfinite(value))
    if not values:
        return float("-inf")
    maximum = max(values)
    return maximum + log(sum(exp(value - maximum) for value in values))


def _probability_from_log(value):
    if value == float("-inf"):
        return 0.0
    return exp(value)


class PatientEventTimeModel(Protocol):
    """Independent patient death-time distributions used by one scenario draw.

    ``cdf`` and ``ppf`` consume one value per patient.  A defective death-time
    distribution may return ``inf`` from ``ppf`` for its no-event mass.  Loss to
    follow-up is represented separately by ``ScenarioPatients.censoring_time``.
    """

    @property
    def patient_count(self) -> int:
        ...

    def cdf(self, followup_time: np.ndarray) -> np.ndarray:
        ...

    def ppf(self, probability: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True, eq=False)
class WeibullEventTimeModel:
    """Patient-specific Weibull or defective mixture-Weibull event times.

    ``eventual_event_probability`` is the finite-event mass.  It is one for an
    ordinary Weibull distribution and below one for a deliberately defective
    cure model.  Scale-aware clinical survival components can implement the
    :class:`PatientEventTimeModel` protocol directly; this compact class exists
    for validation fixtures and fixed parametric scenarios.
    """

    scale_time: np.ndarray
    shape: np.ndarray
    eventual_event_probability: np.ndarray

    def __post_init__(self):
        try:
            scale = np.asarray(self.scale_time, dtype=float)
            shape = np.asarray(self.shape, dtype=float)
            event_probability = np.asarray(
                self.eventual_event_probability, dtype=float
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Weibull event parameters must be numeric") from error
        if scale.ndim != 1 or not len(scale):
            raise ValueError("scale_time must be a non-empty vector")
        if shape.shape != scale.shape or event_probability.shape != scale.shape:
            raise ValueError("Weibull event parameters must have matching vectors")
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("scale_time must contain finite positive values")
        if np.any(~np.isfinite(shape)) or np.any(shape <= 0.0):
            raise ValueError("shape must contain finite positive values")
        if np.any(~np.isfinite(event_probability)) or np.any(
            (event_probability < 0.0) | (event_probability > 1.0)
        ):
            raise ValueError("eventual_event_probability must lie in [0, 1]")
        scale = np.array(scale, copy=True)
        shape = np.array(shape, copy=True)
        event_probability = np.array(event_probability, copy=True)
        for array in (scale, shape, event_probability):
            array.setflags(write=False)
        object.__setattr__(self, "scale_time", scale)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "eventual_event_probability", event_probability)

    @property
    def patient_count(self):
        return len(self.scale_time)

    def _vector(self, values, name, allow_infinity=False):
        try:
            result = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be numeric") from error
        if result.shape != (self.patient_count,):
            raise ValueError(f"{name} must contain one value per patient")
        invalid = np.isnan(result) if allow_infinity else ~np.isfinite(result)
        if np.any(invalid):
            qualifier = "non-NaN" if allow_infinity else "finite"
            raise ValueError(f"{name} must contain {qualifier} values")
        return result

    def cdf(self, followup_time):
        times = self._vector(followup_time, "followup_time", allow_infinity=True)
        if np.any(times < 0.0):
            raise ValueError("followup_time must be non-negative")
        cumulative_hazard = np.power(times / self.scale_time, self.shape)
        return self.eventual_event_probability * (-np.expm1(-cumulative_hazard))

    def ppf(self, probability):
        probabilities = self._vector(probability, "probability")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("probability must lie in [0, 1]")
        finite_event = probabilities < self.eventual_event_probability
        result = np.full(self.patient_count, np.inf, dtype=float)
        positive = finite_event & (probabilities > 0.0)
        if np.any(positive):
            conditional = probabilities[positive] / self.eventual_event_probability[
                positive
            ]
            result[positive] = self.scale_time[positive] * np.power(
                -np.log1p(-conditional), 1.0 / self.shape[positive]
            )
        result[finite_event & (probabilities == 0.0)] = 0.0
        return result


@dataclass(frozen=True, eq=False)
class ScenarioPatients:
    """Latent randomized assignments and outcome models for one trial draw.

    Censoring is sampled before event times and is measured from randomization.
    ``inf`` means complete lifetime ascertainment.  The supplied sampler must
    draw this object from the fixed target scenario; WP6 only changes the event
    interval proposal and accounts for that change exactly in its weight.
    """

    treatment: np.ndarray
    strata: np.ndarray
    censoring_time: np.ndarray
    event_time_model: PatientEventTimeModel

    def __post_init__(self):
        treatment = binary_indicator(self.treatment, "treatment")
        if treatment.ndim != 1 or not len(treatment):
            raise ValueError("treatment must be a non-empty vector")
        strata = np.asarray(self.strata, dtype=object)
        if strata.ndim not in (1, 2) or strata.shape[0] != len(treatment):
            raise ValueError("strata must have one row per patient")
        if strata.ndim == 2 and strata.shape[1] < 1:
            raise ValueError("strata must include at least one factor")
        try:
            censoring = np.asarray(self.censoring_time, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("censoring_time must be numeric") from error
        if censoring.shape != (len(treatment),):
            raise ValueError("censoring_time must contain one value per patient")
        if np.any(np.isnan(censoring)) or np.any(censoring < 0.0):
            raise ValueError("censoring_time must contain non-negative non-NaN values")
        model = self.event_time_model
        if not callable(getattr(model, "cdf", None)) or not callable(
            getattr(model, "ppf", None)
        ):
            raise ValueError("event_time_model must provide cdf and ppf methods")
        if getattr(model, "patient_count", None) != len(treatment):
            raise ValueError("event_time_model must contain one distribution per patient")
        treatment = np.array(treatment, copy=True)
        strata = np.array(strata, copy=True)
        censoring = np.array(censoring, copy=True)
        for array in (treatment, strata, censoring):
            array.setflags(write=False)
        object.__setattr__(self, "treatment", treatment)
        object.__setattr__(self, "strata", strata)
        object.__setattr__(self, "censoring_time", censoring)

    @property
    def patient_count(self):
        return len(self.treatment)


class ScenarioSampler(Protocol):
    """Callable target-scenario generator consumed by the WP6 engine."""

    def __call__(
        self, entry_dates: Tuple[date, ...], rng: np.random.Generator
    ) -> ScenarioPatients:
        ...


@dataclass(frozen=True)
class HistoryConstraintBranch:
    """One joint disclosure-lag branch and its target prior probability."""

    enrollment_constraints: Tuple[CountConstraint, ...]
    event_constraints: Tuple[CountConstraint, ...]
    probability: float

    def __post_init__(self):
        enrollment = tuple(self.enrollment_constraints)
        events = tuple(self.event_constraints)
        if not enrollment or not events:
            raise ValueError("history branches require enrollment and event constraints")
        if not all(isinstance(item, CountConstraint) for item in enrollment + events):
            raise ValueError("history branch constraints must be CountConstraint")
        probability = float(self.probability)
        if not isfinite(probability) or probability <= 0.0:
            raise ValueError("history branch probability must be finite and positive")
        object.__setattr__(self, "enrollment_constraints", enrollment)
        object.__setattr__(self, "event_constraints", events)
        object.__setattr__(self, "probability", probability)


def public_history_constraint_branches(
    history, *, max_lag_combinations=4096
):
    """Enumerate every internally consistent joint disclosure-lag branch.

    Unknown lags remain independent exactly as in WP5.  The returned branch
    probabilities are target prior masses and can sum to less than one if a
    lag combination is logically inconsistent; callers must retain that valid
    mass in an importance ratio instead of silently renormalizing it away.
    """

    if not isinstance(history, PublicHistory):
        raise ValueError("history must be PublicHistory")
    max_lag_combinations = _positive_integer(
        max_lag_combinations, "max_lag_combinations"
    )
    enrollment = tuple(
        item
        for item in history.enrollment_observations
        if item.use_in_likelihood and not item.is_projection
    )
    events = tuple(
        item
        for item in history.event_observations
        if item.use_in_likelihood and not item.is_projection
    )
    choice_sets = tuple(item.cutoff_choices() for item in enrollment + events)
    if any(not choices for choices in choice_sets):
        raise ValueError("likelihood observations must provide cutoff choices")
    combination_count = 1
    for choices in choice_sets:
        combination_count *= len(choices)
    if combination_count > max_lag_combinations:
        raise ValueError(
            f"reporting-lag mixture has {combination_count:,} combinations, "
            f"above the configured {max_lag_combinations:,} limit"
        )
    split = len(enrollment)
    branches = []
    for choices in product(*choice_sets):
        probability = float(np.prod([choice[1] for choice in choices]))
        enrollment_constraints = merge_count_constraints(
            [choice[0] for choice in choices[:split]], history.target_enrollment
        )
        event_constraints = merge_count_constraints(
            [choice[0] for choice in choices[split:]], history.target_enrollment
        )
        if enrollment_constraints is None or event_constraints is None:
            continue
        branches.append(
            HistoryConstraintBranch(
                enrollment_constraints,
                event_constraints,
                probability,
            )
        )
    if not branches:
        raise ValueError("public history has no internally consistent lag branch")
    valid_mass = sum(branch.probability for branch in branches)
    if valid_mass > 1.0 + PROBABILITY_TOLERANCE:
        raise ValueError("history branch probabilities exceed one")
    return tuple(branches)


def _sample_history_branch(branches, rng):
    probabilities = np.asarray([item.probability for item in branches], dtype=float)
    valid_mass = float(probabilities.sum())
    proposal = probabilities / valid_mass
    cumulative = np.cumsum(proposal)
    cumulative[-1] = 1.0
    draw = float(rng.random())
    if not isfinite(draw) or not 0.0 <= draw < 1.0:
        raise ValueError("rng.random must return a value in [0, 1)")
    index = int(np.searchsorted(cumulative, draw, side="right"))
    return branches[index], valid_mass


def _calendar_time(value, origin):
    return float((_parse_date(value, "cutoff date") - origin).days)


def _entry_times(entry_dates, origin):
    entries = tuple(_parse_date(value, "entry date") for value in entry_dates)
    return np.asarray([(value - origin).days for value in entries], dtype=float)


def _cumulative_event_probabilities(
    entry_times, patients, constraints, origin
):
    columns = []
    for constraint in constraints:
        cutoff = _calendar_time(constraint.cutoff_date, origin)
        available = cutoff - entry_times
        active = available >= 0.0
        followup = np.zeros(len(entry_times), dtype=float)
        followup[active] = np.minimum(
            available[active], patients.censoring_time[active]
        )
        values = np.asarray(patients.event_time_model.cdf(followup), dtype=float)
        if values.shape != (len(entry_times),):
            raise ValueError("event-time cdf must return one value per patient")
        values = np.where(active, values, 0.0)
        columns.append(values)
    matrix = np.column_stack(columns)
    # Reuse WP5's strict probability and monotonicity validation.
    event_interval_probabilities(matrix)
    return matrix


def _target_cumulative_counts(
    cumulative_probabilities,
    constraints,
    *,
    max_count_vectors=DEFAULT_MAX_COUNT_VECTORS,
):
    expected = np.sum(cumulative_probabilities, axis=0)
    target = np.empty(len(constraints), dtype=float)
    for index, constraint in enumerate(constraints):
        lower = float(constraint.lower)
        upper = float(constraint.upper)
        if lower == upper:
            target[index] = lower
        else:
            # Keep a range target away from either integer edge.  This preserves
            # support for every allowed increment while centering on the target
            # model whenever its expectation already lies inside the interval.
            interior_lower = lower + 0.5
            interior_upper = upper - 0.5
            if interior_lower > interior_upper:
                interior_lower = interior_upper = 0.5 * (lower + upper)
            target[index] = min(
                max(float(expected[index]), interior_lower), interior_upper
            )
    if np.any(np.diff(target) < -PROBABILITY_TOLERANCE):
        raise TiltProposalError(
            "could not construct monotone proposal count targets"
        )
    lower = np.asarray([item.lower for item in constraints], dtype=float)
    upper = np.asarray([item.upper for item in constraints], dtype=float)
    if np.any(target < lower) or np.any(target > upper):
        raise TiltProposalError(
            "proposal count targets fall outside public constraints"
        )

    # Independent clipping can put adjacent range targets on the same value,
    # even when a positive count increment is allowed.  Such a zero target
    # would remove compatible assignments from the exponential-tilt proposal.
    # The uniform mean of every allowed integer trajectory is a point in the
    # same monotone constraint polytope and has positive mass in every category
    # that is positive in at least one compatible trajectory.  Move only as far
    # toward that support-preserving reference as needed to give each possible
    # category a modest numerical margin.
    allowed = np.asarray(
        _allowed_cumulative_count_vectors(
            constraints,
            max_count_vectors=max_count_vectors,
        ),
        dtype=float,
    )
    patient_count = cumulative_probabilities.shape[0]
    allowed_categories = np.diff(
        np.column_stack(
            (
                np.zeros(len(allowed), dtype=float),
                allowed,
                np.full(len(allowed), patient_count, dtype=float),
            )
        ),
        axis=1,
    )
    possible_positive = np.any(allowed_categories > 0.0, axis=0)
    reference = np.mean(allowed, axis=0)
    reference_categories = np.mean(allowed_categories, axis=0)
    target_categories = np.diff(
        np.concatenate(([0.0], target, [float(patient_count)]))
    )
    blend = 0.0
    for category in np.flatnonzero(possible_positive):
        desired = min(
            TARGET_CATEGORY_MARGIN,
            float(reference_categories[category]),
        )
        current = float(target_categories[category])
        if current + PROBABILITY_TOLERANCE >= desired:
            continue
        reference_value = float(reference_categories[category])
        if reference_value <= current:
            raise TiltProposalError(
                "could not preserve proposal support for every count interval"
            )
        blend = max(blend, (desired - current) / (reference_value - current))
    if blend > 0.0:
        target = (1.0 - blend) * target + blend * reference

    final_categories = np.diff(
        np.concatenate(([0.0], target, [float(patient_count)]))
    )
    if np.any(final_categories[possible_positive] <= PROBABILITY_TOLERANCE):
        raise TiltProposalError(
            "proposal count targets do not preserve compatible support"
        )
    if np.any(np.abs(final_categories[~possible_positive]) > PROBABILITY_TOLERANCE):
        raise TiltProposalError(
            "proposal count targets assign mass to an impossible interval"
        )
    return target


@dataclass(frozen=True, eq=False)
class ExponentialTilt:
    probabilities: np.ndarray
    iterations: int
    maximum_moment_error: float
    target_cumulative_counts: np.ndarray
    target_interim_z: float

    def __post_init__(self):
        probabilities = np.asarray(self.probabilities, dtype=float)
        targets = np.asarray(self.target_cumulative_counts, dtype=float)
        if probabilities.ndim != 2 or not len(probabilities):
            raise ValueError("tilt probabilities must be a non-empty matrix")
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("tilt probabilities must be finite and non-negative")
        if np.any(np.abs(probabilities.sum(axis=1) - 1.0) > 1e-10):
            raise ValueError("tilt probability rows must sum to one")
        iterations = _positive_integer(self.iterations, "tilt iterations")
        error = float(self.maximum_moment_error)
        if not isfinite(error) or error < 0.0:
            raise ValueError("maximum_moment_error must be finite and non-negative")
        target_z = float(self.target_interim_z)
        if not isfinite(target_z):
            raise ValueError("target_interim_z must be finite")
        probabilities = np.array(probabilities, copy=True)
        targets = np.array(targets, copy=True)
        probabilities.setflags(write=False)
        targets.setflags(write=False)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "maximum_moment_error", error)
        object.__setattr__(self, "target_cumulative_counts", targets)
        object.__setattr__(self, "target_interim_z", target_z)


def _interim_constraint_index(history, constraints):
    threshold = history.interim_event_threshold
    exact = [
        index
        for index, constraint in enumerate(constraints)
        if constraint.lower == threshold == constraint.upper
    ]
    if not exact:
        raise ValueError(
            "public event constraints must identify the exact interim threshold"
        )
    return exact[0]


def _feature_tensor(
    interval_probabilities,
    target_cumulative,
    constraints,
    treatment,
    interim_index,
    target_interim_z,
):
    patient_count, category_count = interval_probabilities.shape
    interval_targets = np.diff(np.concatenate(([0.0], target_cumulative)))
    category_targets = np.concatenate(
        (interval_targets, [patient_count - target_cumulative[-1]])
    )
    if np.any(category_targets < -PROBABILITY_TOLERANCE):
        raise TiltProposalError("proposal count targets are not monotone")
    category_targets = np.maximum(category_targets, 0.0)
    positive_categories = np.flatnonzero(category_targets > 0.0)
    zero_categories = np.flatnonzero(category_targets == 0.0)
    for category in zero_categories:
        if np.any(interval_probabilities[:, category] > 0.0):
            if category < category_count - 1:
                previous_lower = 0 if category == 0 else constraints[category - 1].lower
                forced_zero = constraints[category].upper <= previous_lower
                # A zero target is safe only when every compatible cumulative
                # vector forces this increment to zero. Structural zero
                # probabilities are safe regardless.
                if not forced_zero:
                    raise TiltProposalError(
                        "zero proposal target would remove compatible support"
                    )
            elif constraints[-1].lower < patient_count:
                raise TiltProposalError(
                    "zero survivor target would remove compatible support"
                )
    if not len(positive_categories):
        raise TiltProposalError(
            "importance proposal has no positive category target"
        )
    if np.any(
        np.sum(interval_probabilities[:, positive_categories], axis=1)
        <= 0.0
    ):
        raise TiltProposalError(
            "public count constraints are impossible for at least one patient"
        )

    # Category totals are identified with the last positive category as the
    # reference.  The optional final feature targets the treated share of
    # events through the interim cutoff, a close score proxy that makes rare
    # continuation regions accessible while retaining exact likelihood ratios.
    modeled_categories = positive_categories[:-1]
    use_continuation_feature = bool(
        np.any(treatment) and np.any(~treatment) and target_cumulative[interim_index] > 0
    )
    feature_count = len(modeled_categories) + int(use_continuation_feature)
    features = np.zeros(
        (patient_count, category_count, feature_count), dtype=float
    )
    targets = []
    for feature_index, category in enumerate(modeled_categories):
        features[:, category, feature_index] = 1.0
        targets.append(category_targets[category])
    if use_continuation_feature:
        early = np.arange(category_count) <= interim_index
        features[:, early, -1] = treatment[:, None]
        interim_events = target_cumulative[interim_index]
        treated_fraction = float(np.mean(treatment))
        null_mean = interim_events * treated_fraction
        null_sd = sqrt(
            max(
                interim_events * treated_fraction * (1.0 - treated_fraction),
                np.finfo(float).eps,
            )
        )
        target_treated_events = null_mean - target_interim_z * null_sd
        target_treated_events = min(
            max(target_treated_events, 0.25), interim_events - 0.25
        )
        targets.append(target_treated_events)
    return (
        features,
        np.asarray(targets, dtype=float),
        positive_categories,
        category_targets,
    )


def _probabilities_from_parameters(base, features, parameters, positive_categories):
    patient_count, category_count = base.shape
    logits = np.full((patient_count, category_count), float("-inf"), dtype=float)
    positive_base = base[:, positive_categories]
    with np.errstate(divide="ignore"):
        logits[:, positive_categories] = np.log(positive_base)
    if features.shape[2]:
        logits += np.tensordot(features, parameters, axes=([2], [0]))
    row_max = np.max(logits, axis=1)
    if np.any(~np.isfinite(row_max)):
        raise TiltProposalError(
            "importance tilt has a patient with no supported category"
        )
    unnormalized = np.exp(logits - row_max[:, None])
    totals = unnormalized.sum(axis=1)
    return unnormalized / totals[:, None]


def _feature_moments(probabilities, features):
    feature_count = features.shape[2]
    if not feature_count:
        return np.empty(0), np.empty((0, 0))
    means_by_patient = np.einsum("pc,pcf->pf", probabilities, features)
    means = means_by_patient.sum(axis=0)
    second = np.einsum("pc,pcf,pcg->fg", probabilities, features, features)
    covariance = second - np.einsum(
        "pf,pg->fg", means_by_patient, means_by_patient
    )
    return means, covariance


def exponential_tilt_event_intervals(
    cumulative_probabilities,
    constraints,
    treatment,
    *,
    history,
    target_interim_z=0.0,
    tolerance=DEFAULT_TILT_TOLERANCE,
    max_iterations=DEFAULT_MAX_TILT_ITERATIONS,
    max_count_vectors=DEFAULT_MAX_COUNT_VECTORS,
):
    """Fit one maximum-entropy event-interval importance proposal.

    Category moments target the public cumulative count bounds.  A treated
    early-event moment targets an approximate interim z value; it changes only
    proposal efficiency, never the estimand.
    """

    if not isinstance(history, PublicHistory):
        raise ValueError("history must be PublicHistory")
    constraints = tuple(constraints)
    if not constraints or not all(
        isinstance(item, CountConstraint) for item in constraints
    ):
        raise ValueError("constraints must contain CountConstraint values")
    treatment = binary_indicator(treatment, "treatment")
    intervals = event_interval_probabilities(cumulative_probabilities)
    if len(treatment) != intervals.shape[0]:
        raise ValueError("treatment must contain one value per patient")
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError) as error:
        raise ValueError("tolerance must be numeric") from error
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    max_iterations = _positive_integer(max_iterations, "max_iterations")
    target_z = float(target_interim_z)
    if not isfinite(target_z):
        raise ValueError("target_interim_z must be finite")
    target_cumulative = _target_cumulative_counts(
        cumulative_probabilities,
        constraints,
        max_count_vectors=max_count_vectors,
    )
    interim_index = _interim_constraint_index(history, constraints)
    features, targets, positive_categories, category_targets = _feature_tensor(
        intervals,
        target_cumulative,
        constraints,
        treatment,
        interim_index,
        target_z,
    )
    parameters = np.zeros(features.shape[2], dtype=float)
    probabilities = _probabilities_from_parameters(
        intervals, features, parameters, positive_categories
    )
    maximum_error = float("inf")
    for iteration in range(1, max_iterations + 1):
        means, covariance = _feature_moments(probabilities, features)
        residual = means - targets
        maximum_error = float(np.max(np.abs(residual))) if len(residual) else 0.0
        category_means = probabilities.sum(axis=0)
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(category_means - category_targets))),
        )
        if maximum_error <= tolerance:
            return ExponentialTilt(
                probabilities,
                iteration,
                maximum_error,
                target_cumulative,
                target_z,
            )
        if not len(parameters):
            break
        try:
            step = np.linalg.solve(covariance, residual)
        except np.linalg.LinAlgError:
            try:
                step = np.linalg.lstsq(covariance, residual, rcond=None)[0]
            except np.linalg.LinAlgError as error:
                raise TiltProposalError(
                    "importance-tilt linear solve did not converge"
                ) from error
        if np.any(~np.isfinite(step)):
            raise TiltProposalError(
                "importance-tilt Newton step is not finite"
            )
        baseline_norm = float(np.linalg.norm(residual))
        accepted = False
        step_scale = 1.0
        for _ in range(30):
            candidate_parameters = parameters - step_scale * step
            candidate = _probabilities_from_parameters(
                intervals, features, candidate_parameters, positive_categories
            )
            candidate_means, _ = _feature_moments(candidate, features)
            candidate_norm = float(np.linalg.norm(candidate_means - targets))
            if candidate_norm < baseline_norm:
                parameters = candidate_parameters
                probabilities = candidate
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            break
    raise TiltProposalError(
        "importance tilt did not converge: maximum moment error "
        f"{maximum_error:.6g} after {max_iterations} iterations"
    )


def _allowed_cumulative_count_vectors(constraints, *, max_count_vectors):
    """Enumerate monotone integer totals inside one merged constraint branch."""

    max_count_vectors = _positive_integer(max_count_vectors, "max_count_vectors")
    vectors = []

    def visit(index, previous, values):
        if index == len(constraints):
            vectors.append(tuple(values))
            if len(vectors) > max_count_vectors:
                raise ValueError(
                    "public count constraints contain more than "
                    f"{max_count_vectors:,} allowed integer trajectories"
                )
            return
        constraint = constraints[index]
        lower = max(previous, constraint.lower)
        for value in range(lower, constraint.upper + 1):
            visit(index + 1, value, values + [value])

    visit(0, 0, [])
    if not vectors:
        raise ValueError("public event constraints allow no integer trajectory")
    return tuple(vectors)


def _category_quotas(cumulative_counts, patient_count):
    cumulative = np.asarray(cumulative_counts, dtype=int)
    increments = np.diff(np.concatenate(([0], cumulative)))
    quotas = np.concatenate((increments, [patient_count - cumulative[-1]]))
    if np.any(quotas < 0) or int(quotas.sum()) != patient_count:
        raise ValueError("cumulative event counts do not define valid category quotas")
    return quotas.astype(int)


def _quota_assignment_feasible(support, quotas):
    """Hall-condition check for a patient-to-category capacitated matching."""

    support = np.asarray(support, dtype=bool)
    quotas = np.asarray(quotas, dtype=int)
    if support.ndim != 2 or quotas.shape != (support.shape[1],):
        raise ValueError("support and quotas must be aligned")
    if np.any(quotas < 0) or int(quotas.sum()) != support.shape[0]:
        return False
    supported = support & (quotas > 0)[None, :]
    if np.any(~np.any(supported, axis=1)):
        return False
    category_count = support.shape[1]
    # For a bipartite b-matching, Hall's condition can be checked over the
    # category subsets because there are only five natural REGAL intervals.
    for mask in range(1, 1 << category_count):
        subset = np.asarray(
            [bool(mask & (1 << category)) for category in range(category_count)]
        )
        forced_into_subset = np.all(~supported | subset[None, :], axis=1)
        if int(np.count_nonzero(forced_into_subset)) > int(quotas[subset].sum()):
            return False
    return True


def _quota_dp_dimensions(patient_count, quotas):
    """Return one quota layer and the full logical patient-by-state cell count."""

    active_categories = np.flatnonzero(quotas[:-1] > 0)
    shape = tuple(int(quotas[category]) + 1 for category in active_categories)
    layer_shape = shape or (1,)
    layer_cells = int(np.prod(layer_shape, dtype=np.int64))
    total_cells = (int(patient_count) + 1) * layer_cells
    return active_categories, shape, layer_shape, total_cells


def _quota_log_probability(probabilities, quotas, *, max_states):
    """Exact log probability of one category-count vector."""

    probabilities = np.asarray(probabilities, dtype=float)
    quotas = np.asarray(quotas, dtype=int)
    if probabilities.ndim != 2 or quotas.shape != (probabilities.shape[1],):
        raise ValueError("probabilities and quotas must be aligned")
    if not _quota_assignment_feasible(probabilities > 0.0, quotas):
        return float("-inf")
    active_categories, shape, layer_shape, total_cells = _quota_dp_dimensions(
        probabilities.shape[0], quotas
    )
    max_states = _positive_integer(max_states, "max_states")
    if total_cells > max_states:
        raise ValueError(
            f"quota likelihood requires {total_cells:,} logical DP cells, above the "
            f"configured {max_states:,} limit"
        )
    dp = np.zeros(layer_shape, dtype=float)
    dp[(0,) * len(shape) if shape else (0,)] = 1.0
    log_scale = 0.0
    terminal = probabilities.shape[1] - 1
    for patient_probabilities in probabilities:
        updated = dp * patient_probabilities[terminal]
        for axis, category in enumerate(active_categories):
            source = [slice(None)] * len(shape)
            target = [slice(None)] * len(shape)
            source[axis] = slice(0, -1)
            target[axis] = slice(1, None)
            updated[tuple(target)] += (
                dp[tuple(source)] * patient_probabilities[category]
            )
        scale = float(np.max(updated))
        if scale <= 0.0:
            raise _QuotaProbabilityUnderflowError(
                "quota likelihood underflowed despite positive structural support"
            )
        dp = updated / scale
        log_scale += log(scale)
    target_index = (
        tuple(int(quotas[category]) for category in active_categories)
        if len(shape)
        else (0,)
    )
    mass = float(dp[target_index])
    if mass <= 0.0:
        raise _QuotaProbabilityUnderflowError(
            "quota likelihood underflowed despite positive structural support"
        )
    return log_scale + log(mass)


def _quota_suffix_table(probabilities, quotas, *, max_states):
    """Backward dynamic program for an exact conditional category draw."""

    probabilities = np.asarray(probabilities, dtype=float)
    quotas = np.asarray(quotas, dtype=int)
    if not _quota_assignment_feasible(probabilities > 0.0, quotas):
        raise _QuotaProposalInfeasibleError(
            "selected count vector is structurally impossible"
        )
    active_categories, shape, layer_shape, total_cells = _quota_dp_dimensions(
        probabilities.shape[0], quotas
    )
    max_states = _positive_integer(max_states, "max_states")
    if total_cells > max_states:
        raise ValueError(
            f"conditional quota sampler requires {total_cells:,} logical DP cells, "
            f"above the configured {max_states:,} limit"
        )
    suffix = np.zeros((probabilities.shape[0] + 1,) + layer_shape, dtype=float)
    zero_index = (probabilities.shape[0],) + (
        (0,) * len(shape) if shape else (0,)
    )
    suffix[zero_index] = 1.0
    log_scales = np.zeros(probabilities.shape[0] + 1, dtype=float)
    terminal = probabilities.shape[1] - 1
    for patient in range(probabilities.shape[0] - 1, -1, -1):
        following = suffix[patient + 1]
        current = following * probabilities[patient, terminal]
        for axis, category in enumerate(active_categories):
            source = [slice(None)] * len(shape)
            target = [slice(None)] * len(shape)
            source[axis] = slice(0, -1)
            target[axis] = slice(1, None)
            current[tuple(target)] += (
                following[tuple(source)] * probabilities[patient, category]
            )
        scale = float(np.max(current))
        if scale <= 0.0:
            raise _QuotaProbabilityUnderflowError(
                "conditional quota sampler underflowed despite positive "
                "structural support"
            )
        suffix[patient] = current / scale
        log_scales[patient] = log_scales[patient + 1] + log(scale)
    target_index = (0,) + (
        tuple(int(quotas[category]) for category in active_categories)
        if len(shape)
        else (0,)
    )
    target_mass = float(suffix[target_index])
    if target_mass <= 0.0:
        raise _QuotaProbabilityUnderflowError(
            "conditional quota sampler underflowed despite positive "
            "structural support"
        )
    log_probability = log_scales[0] + log(target_mass)
    return suffix, active_categories, log_probability


def _sample_exact_quota_assignment(
    probabilities, quotas, suffix, active_categories, rng
):
    patient_count, category_count = probabilities.shape
    terminal = category_count - 1
    remaining = np.asarray(
        [quotas[category] for category in active_categories], dtype=int
    )
    categories = np.full(patient_count, terminal, dtype=int)
    for patient in range(patient_count):
        following = suffix[patient + 1]
        state_index = tuple(remaining) if len(remaining) else (0,)
        candidates = [terminal]
        weights = [float(probabilities[patient, terminal] * following[state_index])]
        for axis, category in enumerate(active_categories):
            if remaining[axis] <= 0:
                continue
            next_remaining = np.array(remaining, copy=True)
            next_remaining[axis] -= 1
            candidates.append(int(category))
            weights.append(
                float(
                    probabilities[patient, category]
                    * following[tuple(next_remaining)]
                )
            )
        weights = np.asarray(weights, dtype=float)
        total = float(weights.sum())
        if total <= 0.0:
            raise RuntimeError("conditional quota sampler reached zero suffix mass")
        cumulative = np.cumsum(weights / total)
        cumulative[-1] = 1.0
        draw = float(rng.random())
        if not isfinite(draw) or not 0.0 <= draw < 1.0:
            raise ValueError("rng.random must return a conditional draw in [0, 1)")
        selected = int(candidates[np.searchsorted(cumulative, draw, side="right")])
        categories[patient] = selected
        matches = np.flatnonzero(active_categories == selected)
        if len(matches):
            remaining[int(matches[0])] -= 1
    if np.any(remaining != 0):
        raise RuntimeError("conditional quota sampler did not fill every event quota")
    return categories


def _conditional_uniforms(cumulative_probabilities, categories, rng):
    patient_count, cutoff_count = cumulative_probabilities.shape
    draws = np.asarray(rng.random(patient_count), dtype=float)
    if draws.shape != (patient_count,) or np.any(~np.isfinite(draws)) or np.any(
        (draws < 0.0) | (draws >= 1.0)
    ):
        raise ValueError("rng.random must return finite conditional draws in [0, 1)")
    lower = np.zeros(patient_count, dtype=float)
    upper = np.ones(patient_count, dtype=float)
    before = categories > 0
    lower[before] = cumulative_probabilities[
        np.arange(patient_count)[before], categories[before] - 1
    ]
    event_category = categories < cutoff_count
    upper[event_category] = cumulative_probabilities[
        np.arange(patient_count)[event_category], categories[event_category]
    ]
    width = upper - lower
    if np.any(width <= 0.0):
        raise RuntimeError("sampled event interval has zero target probability")
    probabilities = lower + draws * width
    at_lower = probabilities <= lower
    probabilities[at_lower] = np.nextafter(lower[at_lower], upper[at_lower])
    at_upper = probabilities >= upper
    probabilities[at_upper] = np.nextafter(upper[at_upper], lower[at_upper])
    return probabilities


def _counts_at_constraints(calendar_times, constraints, origin):
    return np.asarray(
        [
            np.count_nonzero(
                calendar_times <= _calendar_time(constraint.cutoff_date, origin)
            )
            for constraint in constraints
        ],
        dtype=int,
    )


def _constraints_satisfied(counts, constraints):
    return all(
        constraint.lower <= int(count) <= constraint.upper
        for count, constraint in zip(counts, constraints)
    )


@dataclass(frozen=True, eq=False)
class HistoryImportanceDraw:
    """One complete proposed trial and its exact target/proposal log ratio."""

    trial_data: EventDrivenTrialData
    entry_dates: Tuple[date, ...]
    log_importance_weight: float
    enrollment_compatible: bool
    event_compatible: bool
    enrollment_counts: Tuple[int, ...]
    event_counts: Tuple[int, ...]
    proposal_component: int
    proposal_infeasible: bool
    tilt_attempts: int
    tilt_fallbacks: int
    tilt_iterations: Tuple[int, ...]
    maximum_tilt_error: Optional[float]

    def __post_init__(self):
        if not isinstance(self.trial_data, EventDrivenTrialData):
            raise ValueError("trial_data must be EventDrivenTrialData")
        entries = tuple(_parse_date(value, "entry date") for value in self.entry_dates)
        if len(entries) != self.trial_data.size:
            raise ValueError("entry_dates must contain one date per patient")
        weight = float(self.log_importance_weight)
        if np.isnan(weight) or weight == float("inf"):
            raise ValueError("log_importance_weight must be finite or negative infinity")
        if not isinstance(self.enrollment_compatible, bool) or not isinstance(
            self.event_compatible, bool
        ):
            raise ValueError("compatibility flags must be boolean")
        if not isinstance(self.proposal_infeasible, bool):
            raise ValueError("proposal_infeasible must be boolean")
        if self.proposal_infeasible and (
            weight != float("-inf") or self.event_compatible
        ):
            raise ValueError(
                "proposal-infeasible draws must be event-incompatible with zero weight"
            )
        component = int(self.proposal_component)
        attempts = int(self.tilt_attempts)
        fallbacks = int(self.tilt_fallbacks)
        iterations = tuple(int(value) for value in self.tilt_iterations)
        if (
            component < 0
            or attempts < 0
            or fallbacks < 0
            or fallbacks > attempts
            or len(iterations) != attempts - fallbacks
            or any(value < 1 for value in iterations)
        ):
            raise ValueError("proposal diagnostics are invalid")
        error = self.maximum_tilt_error
        if error is None:
            if iterations:
                raise ValueError(
                    "maximum_tilt_error is required when a tilt converged"
                )
        else:
            error = float(error)
            if not isfinite(error) or error < 0.0:
                raise ValueError(
                    "maximum_tilt_error must be finite and non-negative"
                )
            if not iterations:
                raise ValueError(
                    "maximum_tilt_error must be None without a converged tilt"
                )
        object.__setattr__(self, "entry_dates", entries)
        object.__setattr__(self, "log_importance_weight", weight)
        object.__setattr__(self, "enrollment_counts", tuple(self.enrollment_counts))
        object.__setattr__(self, "event_counts", tuple(self.event_counts))
        object.__setattr__(self, "proposal_component", component)
        object.__setattr__(self, "tilt_attempts", attempts)
        object.__setattr__(self, "tilt_fallbacks", fallbacks)
        object.__setattr__(self, "tilt_iterations", iterations)
        object.__setattr__(self, "maximum_tilt_error", error)

    @property
    def public_history_compatible(self):
        return self.enrollment_compatible and self.event_compatible


def _log_assignment_probability(probabilities, categories):
    selected = probabilities[np.arange(len(categories)), categories]
    if np.any(selected <= 0.0):
        return float("-inf")
    return float(np.log(selected).sum())


def _zero_weight_history_draw(
    history,
    branch,
    entry_dates,
    entries,
    patients,
    rng,
    *,
    proposal_component=0,
    tilt_attempts=0,
    tilt_fallbacks=0,
    tilt_iterations=(),
    maximum_tilt_error=None,
    proposal_infeasible=False,
):
    """Return a harmless target draw for a zero-weight proposal outcome."""

    uniforms = np.asarray(rng.random(patients.patient_count), dtype=float)
    if uniforms.shape != (patients.patient_count,) or np.any(~np.isfinite(uniforms)):
        raise ValueError("rng.random must return one finite draw per patient")
    if np.any((uniforms < 0.0) | (uniforms >= 1.0)):
        raise ValueError("rng.random draws must lie in [0, 1)")
    death_time = np.asarray(patients.event_time_model.ppf(uniforms), dtype=float)
    if death_time.shape != (patients.patient_count,) or np.any(np.isnan(death_time)):
        raise ValueError("event-time ppf must return one non-NaN value per patient")
    observed_event = np.isfinite(death_time) & (
        death_time <= patients.censoring_time
    )
    followup = np.minimum(death_time, patients.censoring_time)
    trial_data = EventDrivenTrialData(
        entry_time=entries,
        followup_time=followup,
        event_observed=observed_event,
        treatment=patients.treatment,
        strata=patients.strata,
    )
    origin = history.study_start
    enrollment_counts = _counts_at_constraints(
        entries, branch.enrollment_constraints, origin
    )
    event_calendar = entries[observed_event] + followup[observed_event]
    event_counts = _counts_at_constraints(
        event_calendar, branch.event_constraints, origin
    )
    return HistoryImportanceDraw(
        trial_data=trial_data,
        entry_dates=entry_dates,
        log_importance_weight=float("-inf"),
        enrollment_compatible=_constraints_satisfied(
            enrollment_counts, branch.enrollment_constraints
        ),
        event_compatible=False,
        enrollment_counts=tuple(int(value) for value in enrollment_counts),
        event_counts=tuple(int(value) for value in event_counts),
        proposal_component=proposal_component,
        proposal_infeasible=proposal_infeasible,
        tilt_attempts=tilt_attempts,
        tilt_fallbacks=tilt_fallbacks,
        tilt_iterations=tilt_iterations,
        maximum_tilt_error=maximum_tilt_error,
    )


def draw_history_importance_sample(
    history,
    enrollment_model,
    scenario_sampler,
    branches,
    proposal_z_targets,
    rng,
    *,
    tilt_tolerance=DEFAULT_TILT_TOLERANCE,
    max_tilt_iterations=DEFAULT_MAX_TILT_ITERATIONS,
    max_count_vectors=DEFAULT_MAX_COUNT_VECTORS,
    max_quota_states=DEFAULT_MAX_DP_STATES,
):
    """Draw one complete WP6 importance sample.

    This lower-level function is public for audits.  Most callers should use
    :func:`condition_on_public_history` or its paired futility grid wrapper.
    """

    entry_dates = enrollment_model.sample_enrollment_dates(rng)
    patients = scenario_sampler(entry_dates, rng)
    if not isinstance(patients, ScenarioPatients):
        raise ValueError("scenario_sampler must return ScenarioPatients")
    if patients.patient_count != history.target_enrollment:
        raise ValueError("scenario draw and public-history patient totals differ")
    branch, valid_branch_mass = _sample_history_branch(branches, rng)
    origin = history.study_start
    entries = _entry_times(entry_dates, origin)
    cumulative = _cumulative_event_probabilities(
        entries, patients, branch.event_constraints, origin
    )
    target_intervals = event_interval_probabilities(cumulative)
    allowed_counts = _allowed_cumulative_count_vectors(
        branch.event_constraints,
        max_count_vectors=max_count_vectors,
    )
    structurally_possible = []
    target_support = target_intervals > 0.0
    for counts in allowed_counts:
        quotas = _category_quotas(counts, patients.patient_count)
        if _quota_assignment_feasible(target_support, quotas):
            structurally_possible.append((counts, quotas))
    if not structurally_possible:
        return _zero_weight_history_draw(
            history,
            branch,
            entry_dates,
            entries,
            patients,
            rng,
        )
    tilts = []
    tilt_fallbacks = 0
    for target_z in proposal_z_targets:
        try:
            tilt = exponential_tilt_event_intervals(
                cumulative,
                branch.event_constraints,
                patients.treatment,
                history=history,
                target_interim_z=target_z,
                tolerance=tilt_tolerance,
                max_iterations=max_tilt_iterations,
                max_count_vectors=max_count_vectors,
            )
        except TiltProposalError:
            # The base component always remains a valid proposal for a
            # structurally feasible draw.  Omitting only the failed component
            # and evaluating the realized mixture density below leaves the
            # importance ratio exact while keeping one numerical failure from
            # aborting the full run.
            tilt_fallbacks += 1
            continue
        tilts.append(tilt)
    tilts = tuple(tilts)
    tilt_iterations = tuple(item.iterations for item in tilts)
    maximum_tilt_error = (
        max(item.maximum_moment_error for item in tilts) if tilts else None
    )
    proposal_probabilities = (target_intervals,) + tuple(
        tilt.probabilities for tilt in tilts
    )
    mixture_count = len(proposal_probabilities)
    component_draw = float(rng.random())
    if not isfinite(component_draw) or not 0.0 <= component_draw < 1.0:
        raise ValueError("rng.random must return a proposal draw in [0, 1)")
    proposal_component = min(int(component_draw * mixture_count), mixture_count - 1)
    count_draw = float(rng.random())
    if not isfinite(count_draw) or not 0.0 <= count_draw < 1.0:
        raise ValueError("rng.random must return a count-vector draw in [0, 1)")
    count_index = min(
        int(count_draw * len(structurally_possible)),
        len(structurally_possible) - 1,
    )
    selected_counts, quotas = structurally_possible[count_index]
    selected_probabilities = proposal_probabilities[proposal_component]
    try:
        suffix, active_categories, suffix_log_probability = _quota_suffix_table(
            selected_probabilities,
            quotas,
            max_states=max_quota_states,
        )
    except _QuotaProposalInfeasibleError:
        # Component and count-vector selection are independent.  An infeasible
        # pair is therefore a genuine atom of the implemented proposal, not a
        # reason to condition the Monte Carlo sample on proposal success.  Its
        # zero weight remains in the nsim denominator and preserves unbiasedness.
        return _zero_weight_history_draw(
            history,
            branch,
            entry_dates,
            entries,
            patients,
            rng,
            proposal_component=proposal_component,
            tilt_attempts=len(proposal_z_targets),
            tilt_fallbacks=tilt_fallbacks,
            tilt_iterations=tilt_iterations,
            maximum_tilt_error=maximum_tilt_error,
            proposal_infeasible=True,
        )
    categories = _sample_exact_quota_assignment(
        selected_probabilities,
        quotas,
        suffix,
        active_categories,
        rng,
    )

    target_log_probability = _log_assignment_probability(
        target_intervals, categories
    )
    vector_log_probabilities = []
    for component, probabilities in enumerate(proposal_probabilities):
        if component == proposal_component:
            vector_log_probabilities.append(suffix_log_probability)
        else:
            vector_log_probabilities.append(
                _quota_log_probability(
                    probabilities,
                    quotas,
                    max_states=max_quota_states,
                )
            )
    proposal_terms = []
    for probabilities, vector_log_probability in zip(
        proposal_probabilities,
        vector_log_probabilities,
    ):
        if not isfinite(vector_log_probability):
            # This component cannot generate the selected count vector and
            # therefore contributes zero density to the realized mixture.  An
            # infeasible selected component has already returned a counted
            # zero-weight draw before conditional assignment.
            proposal_terms.append(float("-inf"))
            continue
        proposal_terms.append(
            -log(mixture_count)
            - log(len(structurally_possible))
            + _log_assignment_probability(probabilities, categories)
            - vector_log_probability
        )
    proposal_log_probability = _logsumexp(proposal_terms)
    if not isfinite(target_log_probability) or not isfinite(
        proposal_log_probability
    ):
        raise RuntimeError("importance proposal sampled an unsupported assignment")
    log_weight = (
        log(valid_branch_mass)
        + target_log_probability
        - proposal_log_probability
    )

    event_probabilities = _conditional_uniforms(cumulative, categories, rng)
    death_time = np.asarray(
        patients.event_time_model.ppf(event_probabilities), dtype=float
    )
    if death_time.shape != (patients.patient_count,):
        raise ValueError("event-time ppf must return one value per patient")
    if np.any(np.isnan(death_time)) or np.any(death_time < 0.0):
        raise ValueError("event-time ppf must return non-negative non-NaN values")
    observed_event = np.isfinite(death_time) & (
        death_time <= patients.censoring_time
    )
    followup = np.minimum(death_time, patients.censoring_time)
    trial_data = EventDrivenTrialData(
        entry_time=entries,
        followup_time=followup,
        event_observed=observed_event,
        treatment=patients.treatment,
        strata=patients.strata,
    )

    enrollment_counts = _counts_at_constraints(
        entries, branch.enrollment_constraints, origin
    )
    event_calendar = entries[observed_event] + followup[observed_event]
    event_counts = _counts_at_constraints(
        event_calendar, branch.event_constraints, origin
    )
    category_counts = np.cumsum(
        np.bincount(
            categories,
            minlength=len(branch.event_constraints) + 1,
        )[: len(branch.event_constraints)]
    )
    if tuple(int(value) for value in category_counts) != tuple(selected_counts):
        raise RuntimeError("quota proposal did not preserve its selected count vector")
    if not np.array_equal(event_counts, category_counts):
        raise RuntimeError(
            "conditional event-time inversion did not preserve sampled intervals"
        )
    return HistoryImportanceDraw(
        trial_data=trial_data,
        entry_dates=entry_dates,
        log_importance_weight=log_weight,
        enrollment_compatible=_constraints_satisfied(
            enrollment_counts, branch.enrollment_constraints
        ),
        event_compatible=_constraints_satisfied(
            event_counts, branch.event_constraints
        ),
        enrollment_counts=tuple(int(value) for value in enrollment_counts),
        event_counts=tuple(int(value) for value in event_counts),
        proposal_component=proposal_component,
        proposal_infeasible=False,
        tilt_attempts=len(proposal_z_targets),
        tilt_fallbacks=tilt_fallbacks,
        tilt_iterations=tilt_iterations,
        maximum_tilt_error=maximum_tilt_error,
    )


def _effective_sample_size(log_weights):
    if not log_weights:
        return 0.0
    numerator = 2.0 * _logsumexp(log_weights)
    denominator = _logsumexp(2.0 * value for value in log_weights)
    return exp(numerator - denominator)


def _max_weight_share(log_weights):
    if not log_weights:
        return float("nan")
    return exp(max(log_weights) - _logsumexp(log_weights))


def _conditional_probability(numerator, denominator):
    if not denominator:
        return float("nan")
    return _probability_from_log(
        _logsumexp(numerator) - _logsumexp(denominator)
    )


@dataclass(frozen=True)
class ConditioningResult:
    """Importance-weighted fixed-scenario projection after public continuation."""

    scenario_name: str
    design: TrialDecisionDesign
    importance_draws: int
    history_compatible_draws: int
    continuation_compatible_draws: int
    interim_efficacy_draws: int
    interim_futility_draws: int
    non_estimable_interim_draws: int
    final_rejection_draws: int
    final_non_rejection_draws: int
    final_not_reached_draws: int
    log_p_public_history: float
    p_continue_given_public_history: float
    p_final_rejection_given_public_history_and_continuation: float
    p_final_reached_given_public_history_and_continuation: float
    history_effective_sample_size: float
    continuation_effective_sample_size: float
    maximum_history_weight_share: float
    valid_disclosure_lag_mass: float
    proposal_interim_z_targets: Tuple[float, ...]
    tilt_attempts: int
    tilt_fallbacks: int
    draws_with_tilt_fallback: int
    proposal_infeasible_draws: int
    mean_tilt_iterations: Optional[float]
    maximum_tilt_error: Optional[float]

    @property
    def p_public_history(self):
        return _probability_from_log(self.log_p_public_history)

    @property
    def is_posterior_forecast(self):
        return False

    @property
    def assumed_futility_hr_threshold(self):
        rule = self.design.futility_rule
        return rule.threshold if isinstance(rule, HazardRatioFutilityRule) else None

    @property
    def tilt_fallback_rate(self):
        return self.tilt_fallbacks / self.tilt_attempts if self.tilt_attempts else 0.0


class _ConditioningAccumulator:
    def __init__(self, design):
        self.design = design
        self.history = []
        self.continuation = []
        self.interim_efficacy = []
        self.interim_futility = []
        self.non_estimable = []
        self.final_rejection = []
        self.final_non_rejection = []
        self.final_not_reached = []
        self.history_count = 0
        self.continuation_count = 0

    def update(self, draw):
        if not draw.public_history_compatible:
            return
        weight = draw.log_importance_weight
        self.history.append(weight)
        self.history_count += 1
        decision = evaluate_event_driven_trial(draw.trial_data, self.design)
        if decision.interim is None or not isfinite(decision.interim.primary.z):
            self.non_estimable.append(weight)
            return
        if decision.interim_decision is InterimDecision.EFFICACY_STOP:
            self.interim_efficacy.append(weight)
            return
        if decision.interim_decision is InterimDecision.FUTILITY_STOP:
            self.interim_futility.append(weight)
            return
        if decision.interim_decision is not InterimDecision.CONTINUE:
            raise RuntimeError(
                "a public-history-compatible draw did not reach the interim look"
            )
        self.continuation.append(weight)
        self.continuation_count += 1
        if decision.final_decision is FinalDecision.REJECT:
            self.final_rejection.append(weight)
        elif decision.final_decision is FinalDecision.DO_NOT_REJECT:
            self.final_non_rejection.append(weight)
        elif decision.final_decision is FinalDecision.NOT_REACHED:
            self.final_not_reached.append(weight)
        else:
            raise RuntimeError("continued trial returned an inapplicable final decision")

    def result(
        self,
        scenario_name,
        nsim,
        valid_lag_mass,
        proposal_z_targets,
        tilt_attempts,
        tilt_fallbacks,
        draws_with_tilt_fallback,
        proposal_infeasible_draws,
        mean_tilt_iterations,
        maximum_tilt_error,
    ):
        log_p_history = (
            _logsumexp(self.history) - log(nsim)
            if self.history
            else float("-inf")
        )
        final_reached = self.final_rejection + self.final_non_rejection
        return ConditioningResult(
            scenario_name=scenario_name,
            design=self.design,
            importance_draws=nsim,
            history_compatible_draws=self.history_count,
            continuation_compatible_draws=self.continuation_count,
            interim_efficacy_draws=len(self.interim_efficacy),
            interim_futility_draws=len(self.interim_futility),
            non_estimable_interim_draws=len(self.non_estimable),
            final_rejection_draws=len(self.final_rejection),
            final_non_rejection_draws=len(self.final_non_rejection),
            final_not_reached_draws=len(self.final_not_reached),
            log_p_public_history=log_p_history,
            p_continue_given_public_history=_conditional_probability(
                self.continuation, self.history
            ),
            p_final_rejection_given_public_history_and_continuation=(
                _conditional_probability(self.final_rejection, self.continuation)
            ),
            p_final_reached_given_public_history_and_continuation=(
                _conditional_probability(final_reached, self.continuation)
            ),
            history_effective_sample_size=_effective_sample_size(self.history),
            continuation_effective_sample_size=_effective_sample_size(
                self.continuation
            ),
            maximum_history_weight_share=_max_weight_share(self.history),
            valid_disclosure_lag_mass=valid_lag_mass,
            proposal_interim_z_targets=tuple(proposal_z_targets),
            tilt_attempts=tilt_attempts,
            tilt_fallbacks=tilt_fallbacks,
            draws_with_tilt_fallback=draws_with_tilt_fallback,
            proposal_infeasible_draws=proposal_infeasible_draws,
            mean_tilt_iterations=mean_tilt_iterations,
            maximum_tilt_error=maximum_tilt_error,
        )


def _proposal_target_for_design(design):
    efficacy = design.efficacy_boundaries["interim_z"]
    rule = design.futility_rule
    if rule is None:
        return 0.0
    if not isinstance(rule, HazardRatioFutilityRule):
        return 0.0
    approximate_lower_z = -log(rule.threshold) * sqrt(design.interim_events) / 2.0
    if approximate_lower_z >= efficacy:
        return efficacy - 0.15
    return 0.5 * (approximate_lower_z + efficacy)


def _normalize_proposal_targets(designs, proposal_interim_z_targets):
    if proposal_interim_z_targets is None:
        values = [0.0] + [_proposal_target_for_design(design) for design in designs]
    else:
        try:
            values = [float(value) for value in proposal_interim_z_targets]
        except (TypeError, ValueError) as error:
            raise ValueError("proposal_interim_z_targets must be numeric") from error
    if any(not isfinite(value) for value in values):
        raise ValueError("proposal_interim_z_targets must be finite")
    unique = []
    for value in values:
        if not any(abs(value - prior) <= 1e-12 for prior in unique):
            unique.append(value)
    return tuple(unique)


def _condition_designs_on_public_history(
    scenario_sampler,
    designs,
    *,
    scenario_name,
    history=None,
    enrollment_model=None,
    nsim=DEFAULT_IMPORTANCE_DRAWS,
    seed=20260824,
    proposal_interim_z_targets=None,
    max_lag_combinations=4096,
    max_count_vectors=DEFAULT_MAX_COUNT_VECTORS,
    max_quota_states=DEFAULT_MAX_DP_STATES,
    tilt_tolerance=DEFAULT_TILT_TOLERANCE,
    max_tilt_iterations=DEFAULT_MAX_TILT_ITERATIONS,
):
    if not callable(scenario_sampler):
        raise ValueError("scenario_sampler must be callable")
    if not isinstance(scenario_name, str) or not scenario_name.strip():
        raise ValueError("scenario_name must be a non-empty string")
    designs = tuple(designs)
    if not designs or not all(isinstance(item, TrialDecisionDesign) for item in designs):
        raise ValueError("designs must contain TrialDecisionDesign values")
    if history is None:
        history = load_regal_public_history()
    if not isinstance(history, PublicHistory):
        raise ValueError("history must be PublicHistory")
    if enrollment_model is None:
        enrollment_model = default_regal_enrollment_model(history)
    if not isinstance(enrollment_model, PiecewiseEnrollmentModel):
        raise ValueError("enrollment_model must be PiecewiseEnrollmentModel")
    if enrollment_model.total_enrollment != history.target_enrollment:
        raise ValueError("enrollment model and history totals differ")
    for design in designs:
        if (
            design.interim_events != history.interim_event_threshold
            or design.final_events != history.final_event_threshold
        ):
            raise ValueError("trial design and public-history event thresholds differ")
    nsim = _positive_integer(nsim, "nsim")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    targets = _normalize_proposal_targets(designs, proposal_interim_z_targets)
    branches = public_history_constraint_branches(
        history, max_lag_combinations=max_lag_combinations
    )
    valid_lag_mass = sum(item.probability for item in branches)
    rng = np.random.default_rng(int(seed))
    accumulators = [_ConditioningAccumulator(design) for design in designs]
    iteration_sum = 0
    iteration_count = 0
    tilt_attempts = 0
    tilt_fallbacks = 0
    draws_with_tilt_fallback = 0
    proposal_infeasible_draws = 0
    maximum_tilt_error = None
    for _ in range(nsim):
        draw = draw_history_importance_sample(
            history,
            enrollment_model,
            scenario_sampler,
            branches,
            targets,
            rng,
            tilt_tolerance=tilt_tolerance,
            max_tilt_iterations=max_tilt_iterations,
            max_count_vectors=max_count_vectors,
            max_quota_states=max_quota_states,
        )
        iteration_sum += sum(draw.tilt_iterations)
        iteration_count += len(draw.tilt_iterations)
        tilt_attempts += draw.tilt_attempts
        tilt_fallbacks += draw.tilt_fallbacks
        draws_with_tilt_fallback += int(draw.tilt_fallbacks > 0)
        proposal_infeasible_draws += int(draw.proposal_infeasible)
        if draw.tilt_iterations:
            maximum_tilt_error = (
                draw.maximum_tilt_error
                if maximum_tilt_error is None
                else max(maximum_tilt_error, draw.maximum_tilt_error)
            )
        for accumulator in accumulators:
            accumulator.update(draw)
    mean_iterations = iteration_sum / iteration_count if iteration_count else None
    return tuple(
        accumulator.result(
            scenario_name.strip(),
            nsim,
            valid_lag_mass,
            targets,
            tilt_attempts,
            tilt_fallbacks,
            draws_with_tilt_fallback,
            proposal_infeasible_draws,
            mean_iterations,
            maximum_tilt_error,
        )
        for accumulator in accumulators
    )


def condition_on_public_history(
    scenario_sampler,
    *,
    scenario_name,
    design=REGAL_V2_EFFICACY_DESIGN,
    history=None,
    enrollment_model=None,
    nsim=DEFAULT_IMPORTANCE_DRAWS,
    seed=20260824,
    proposal_interim_z_targets=None,
    max_lag_combinations=4096,
    max_count_vectors=DEFAULT_MAX_COUNT_VECTORS,
    max_quota_states=DEFAULT_MAX_DP_STATES,
    tilt_tolerance=DEFAULT_TILT_TOLERANCE,
    max_tilt_iterations=DEFAULT_MAX_TILT_ITERATIONS,
):
    """Project one fixed scenario after public history and interim continuation."""

    return _condition_designs_on_public_history(
        scenario_sampler,
        (design,),
        scenario_name=scenario_name,
        history=history,
        enrollment_model=enrollment_model,
        nsim=nsim,
        seed=seed,
        proposal_interim_z_targets=proposal_interim_z_targets,
        max_lag_combinations=max_lag_combinations,
        max_count_vectors=max_count_vectors,
        max_quota_states=max_quota_states,
        tilt_tolerance=tilt_tolerance,
        max_tilt_iterations=max_tilt_iterations,
    )[0]


def condition_futility_sensitivity_grid(
    scenario_sampler,
    thresholds=FUTILITY_HR_SENSITIVITY_GRID,
    *,
    scenario_name,
    base_design=REGAL_V2_EFFICACY_DESIGN,
    history=None,
    enrollment_model=None,
    nsim=DEFAULT_IMPORTANCE_DRAWS,
    seed=20260824,
    proposal_interim_z_targets=None,
    max_lag_combinations=4096,
    max_count_vectors=DEFAULT_MAX_COUNT_VECTORS,
    max_quota_states=DEFAULT_MAX_DP_STATES,
    tilt_tolerance=DEFAULT_TILT_TOLERANCE,
    max_tilt_iterations=DEFAULT_MAX_TILT_ITERATIONS,
):
    """Run paired continuation conditioning across assumed futility rules."""

    if not isinstance(base_design, TrialDecisionDesign):
        raise ValueError("base_design must be TrialDecisionDesign")
    if base_design.futility_rule is not None:
        raise ValueError("base_design must not already contain a futility rule")
    thresholds = tuple(thresholds)
    if not thresholds:
        raise ValueError("thresholds must not be empty")
    normalized = []
    for threshold in thresholds:
        if threshold is None:
            normalized.append(None)
        else:
            normalized.append(HazardRatioFutilityRule(threshold).threshold)
    if len(set(normalized)) != len(normalized):
        raise ValueError("thresholds must not contain duplicates")
    designs = tuple(
        replace(
            base_design,
            futility_rule=(
                None
                if threshold is None
                else HazardRatioFutilityRule(threshold)
            ),
        )
        for threshold in normalized
    )
    return _condition_designs_on_public_history(
        scenario_sampler,
        designs,
        scenario_name=scenario_name,
        history=history,
        enrollment_model=enrollment_model,
        nsim=nsim,
        seed=seed,
        proposal_interim_z_targets=proposal_interim_z_targets,
        max_lag_combinations=max_lag_combinations,
        max_count_vectors=max_count_vectors,
        max_quota_states=max_quota_states,
        tilt_tolerance=tilt_tolerance,
        max_tilt_iterations=max_tilt_iterations,
    )


__all__ = (
    "ConditioningResult",
    "DEFAULT_IMPORTANCE_DRAWS",
    "DEFAULT_MAX_COUNT_VECTORS",
    "DEFAULT_MAX_TILT_ITERATIONS",
    "DEFAULT_TILT_TOLERANCE",
    "ExponentialTilt",
    "HistoryConstraintBranch",
    "HistoryImportanceDraw",
    "PatientEventTimeModel",
    "ScenarioPatients",
    "ScenarioSampler",
    "TiltProposalError",
    "WeibullEventTimeModel",
    "condition_futility_sensitivity_grid",
    "condition_on_public_history",
    "draw_history_importance_sample",
    "exponential_tilt_event_intervals",
    "public_history_constraint_branches",
)
