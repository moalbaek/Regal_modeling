"""REGAL v2 public-history conditioning and posterior model averaging.

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

The low-level ``ConditioningResult`` remains a *conditional fixed-scenario (or
within-family prior-predictive) projection*. Work package 7 adds explicit GPS
effect-family priors and combines a complete family set only after weighting by
``P(public history, continuation | family)``. Only the resulting
``PosteriorForecastResult`` advertises posterior-forecast status.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from enum import Enum
from itertools import product
from math import exp, expm1, isfinite, log, sqrt
from numbers import Integral
from types import MappingProxyType
from typing import Optional, Protocol, Tuple

import numpy as np

from bat_regimens import (
    BATComponent,
    BATDesign,
    DEFAULT_COMPONENT_LIBRARY,
    PRIMARY_EQUAL_STRATA,
    component_for,
)

from event_likelihood import (
    DEFAULT_MAX_DP_STATES,
    CountConstraint,
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
from survival_models import (
    CureMixtureComponent,
    ExponentialBackgroundMortality,
    SurvivalScale,
)


PROBABILITY_TOLERANCE = 1e-12
DEFAULT_IMPORTANCE_DRAWS = 10_000
DEFAULT_TILT_TOLERANCE = 1e-8
DEFAULT_MAX_TILT_ITERATIONS = 80
DEFAULT_MAX_COUNT_VECTORS = 4096
TARGET_CATEGORY_MARGIN = 0.25
MONTH_DAYS = 365.25 / 12.0
MINIMUM_POSTERIOR_FORECAST_ESS = 100.0
MAXIMUM_POSTERIOR_FORECAST_HISTORY_WEIGHT_SHARE = 0.05


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


def _unit_probability(value, name):
    """Validate and clamp only floating-point noise outside ``[0, 1]``."""

    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if (
        not isfinite(probability)
        or probability < -PROBABILITY_TOLERANCE
        or probability > 1.0 + PROBABILITY_TOLERANCE
    ):
        raise ValueError(f"{name} must lie in [0, 1]")
    return min(max(probability, 0.0), 1.0)


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
class PiecewiseWeibullEventTimeModel:
    """Patient event times under piecewise multipliers of a Weibull hazard.

    ``constant_hazard`` is an additive hazard that is never modified by the
    piecewise multiplier. It is zero for an uncured overall-survival input,
    the population hazard for an uncured net-survival input, and the only
    active hazard for an explicitly cured patient. ``weibull_weight`` is one
    for an uncured component and zero for a cured patient. WP7 uses this
    latent-state representation for the cure-difference and responder/cure
    families; its statistical marginal-PH families use
    :class:`PiecewiseMixtureHazardEventTimeModel` instead.

    ``breakpoints`` has shape ``(patients, segments - 1)`` and
    ``hazard_multipliers`` has shape ``(patients, segments)``.  A delayed PH
    model, for example, uses one breakpoint and multipliers ``(1, HR)``.
    """

    scale_time: np.ndarray
    shape: np.ndarray
    constant_hazard: np.ndarray
    weibull_weight: np.ndarray
    breakpoints: np.ndarray
    hazard_multipliers: np.ndarray

    def __post_init__(self):
        try:
            scale = np.asarray(self.scale_time, dtype=float)
            shape = np.asarray(self.shape, dtype=float)
            constant = np.asarray(self.constant_hazard, dtype=float)
            weight = np.asarray(self.weibull_weight, dtype=float)
            breakpoints = np.asarray(self.breakpoints, dtype=float)
            multipliers = np.asarray(self.hazard_multipliers, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("piecewise event parameters must be numeric") from error
        if scale.ndim != 1 or not len(scale):
            raise ValueError("scale_time must be a non-empty vector")
        patient_count = len(scale)
        if shape.shape != (patient_count,):
            raise ValueError("shape must contain one value per patient")
        if constant.shape != (patient_count,) or weight.shape != (patient_count,):
            raise ValueError(
                "constant_hazard and weibull_weight must contain one value per patient"
            )
        if breakpoints.ndim != 2 or breakpoints.shape[0] != patient_count:
            raise ValueError("breakpoints must have one row per patient")
        if multipliers.shape != (patient_count, breakpoints.shape[1] + 1):
            raise ValueError(
                "hazard_multipliers must have one more column than breakpoints"
            )
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("scale_time must contain finite positive values")
        if np.any(~np.isfinite(shape)) or np.any(shape <= 0.0):
            raise ValueError("shape must contain finite positive values")
        if np.any(~np.isfinite(constant)) or np.any(constant < 0.0):
            raise ValueError("constant_hazard must contain finite non-negative values")
        if np.any(~np.isfinite(weight)) or np.any(weight < 0.0):
            raise ValueError("weibull_weight must contain finite non-negative values")
        if np.any(~np.isfinite(breakpoints)) or np.any(breakpoints < 0.0):
            raise ValueError("breakpoints must contain finite non-negative values")
        if breakpoints.shape[1] > 1 and np.any(
            np.diff(breakpoints, axis=1) <= 0.0
        ):
            raise ValueError("breakpoints must be strictly increasing within each patient")
        if np.any(~np.isfinite(multipliers)) or np.any(multipliers <= 0.0):
            raise ValueError("hazard_multipliers must contain finite positive values")
        copied = []
        for array in (scale, shape, constant, weight, breakpoints, multipliers):
            value = np.array(array, copy=True)
            value.setflags(write=False)
            copied.append(value)
        (
            scale,
            shape,
            constant,
            weight,
            breakpoints,
            multipliers,
        ) = copied
        object.__setattr__(self, "scale_time", scale)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "constant_hazard", constant)
        object.__setattr__(self, "weibull_weight", weight)
        object.__setattr__(self, "breakpoints", breakpoints)
        object.__setattr__(self, "hazard_multipliers", multipliers)

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

    def _disease_cumulative_hazard(self, times):
        baseline = np.power(times / self.scale_time, self.shape)
        disease = np.zeros(self.patient_count, dtype=float)
        start_time = np.zeros(self.patient_count, dtype=float)
        start_hazard = np.zeros(self.patient_count, dtype=float)
        for index in range(self.breakpoints.shape[1]):
            end_time = self.breakpoints[:, index]
            end_hazard = np.power(end_time / self.scale_time, self.shape)
            active = times > start_time
            interval_end = np.minimum(times, end_time)
            interval_hazard = np.power(
                interval_end / self.scale_time, self.shape
            )
            increment = np.where(
                active,
                np.maximum(interval_hazard - start_hazard, 0.0),
                0.0,
            )
            disease += self.hazard_multipliers[:, index] * increment
            start_time = end_time
            start_hazard = end_hazard
        active = times > start_time
        increment = np.where(
            active,
            np.maximum(baseline - start_hazard, 0.0),
            0.0,
        )
        disease += self.hazard_multipliers[:, -1] * increment
        return disease

    def _cumulative_hazard(self, times):
        result = np.zeros(self.patient_count, dtype=float)
        constant = self.constant_hazard > 0.0
        result[constant] = self.constant_hazard[constant] * times[constant]
        disease = self.weibull_weight > 0.0
        if np.any(disease):
            cumulative = self._disease_cumulative_hazard(times)
            result[disease] += self.weibull_weight[disease] * cumulative[disease]
        return result

    def cdf(self, followup_time):
        times = self._vector(followup_time, "followup_time", allow_infinity=True)
        if np.any(times < 0.0):
            raise ValueError("followup_time must be non-negative")
        return -np.expm1(-self._cumulative_hazard(times))

    def _inverse_disease_only(self, target, selected):
        indices = np.flatnonzero(selected)
        result = np.full(len(indices), np.inf, dtype=float)
        required = target[indices] / self.weibull_weight[indices]
        accumulated = np.zeros(len(indices), dtype=float)
        start_hazard = np.zeros(len(indices), dtype=float)
        unresolved = np.ones(len(indices), dtype=bool)
        for column in range(self.breakpoints.shape[1]):
            end_time = self.breakpoints[indices, column]
            end_hazard = np.power(
                end_time / self.scale_time[indices], self.shape[indices]
            )
            multiplier = self.hazard_multipliers[indices, column]
            boundary = accumulated + multiplier * (end_hazard - start_hazard)
            within = unresolved & (required <= boundary)
            if np.any(within):
                baseline_target = start_hazard[within] + (
                    required[within] - accumulated[within]
                ) / multiplier[within]
                result[within] = self.scale_time[indices[within]] * np.power(
                    baseline_target, 1.0 / self.shape[indices[within]]
                )
                unresolved[within] = False
            accumulated = boundary
            start_hazard = end_hazard
        if np.any(unresolved):
            multiplier = self.hazard_multipliers[indices, -1]
            baseline_target = start_hazard[unresolved] + (
                required[unresolved] - accumulated[unresolved]
            ) / multiplier[unresolved]
            result[unresolved] = self.scale_time[indices[unresolved]] * np.power(
                baseline_target, 1.0 / self.shape[indices[unresolved]]
            )
        return indices, result

    def ppf(self, probability):
        probabilities = self._vector(probability, "probability")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("probability must lie in [0, 1]")
        result = np.full(self.patient_count, np.inf, dtype=float)
        result[probabilities == 0.0] = 0.0
        finite = (probabilities > 0.0) & (probabilities < 1.0)
        target = np.zeros(self.patient_count, dtype=float)
        target[finite] = -np.log1p(-probabilities[finite])
        constant_only = finite & (self.constant_hazard > 0.0) & (
            self.weibull_weight == 0.0
        )
        result[constant_only] = (
            target[constant_only] / self.constant_hazard[constant_only]
        )
        disease_only = finite & (self.constant_hazard == 0.0) & (
            self.weibull_weight > 0.0
        )
        if np.any(disease_only):
            indices, values = self._inverse_disease_only(target, disease_only)
            result[indices] = values
        mixed = finite & (self.constant_hazard > 0.0) & (
            self.weibull_weight > 0.0
        )
        if np.any(mixed):
            indices = np.flatnonzero(mixed)
            lower = np.zeros(len(indices), dtype=float)
            last_break = (
                self.breakpoints[indices, -1]
                if self.breakpoints.shape[1]
                else np.zeros(len(indices), dtype=float)
            )
            upper = np.maximum(
                self.scale_time[indices] + last_break,
                target[indices] / self.constant_hazard[indices],
            )
            probe = np.zeros(self.patient_count, dtype=float)
            for _ in range(64):
                probe.fill(0.0)
                probe[indices] = upper
                too_low = self._cumulative_hazard(probe)[indices] < target[indices]
                if not np.any(too_low):
                    break
                upper[too_low] *= 2.0
            else:
                raise FloatingPointError("could not bracket mixed event-time quantile")
            for _ in range(80):
                midpoint = 0.5 * (lower + upper)
                probe.fill(0.0)
                probe[indices] = midpoint
                below = self._cumulative_hazard(probe)[indices] < target[indices]
                lower[below] = midpoint[below]
                upper[~below] = midpoint[~below]
            result[indices] = 0.5 * (lower + upper)
        return result


@dataclass(frozen=True, eq=False)
class PiecewiseMixtureHazardEventTimeModel:
    """Piecewise transforms of a scale-aware marginal cure-mixture hazard.

    This model supplies the statistical no-effect, PH, delayed-PH, and waning
    families.  The baseline curve is exactly the component-level marginal
    survival from WP2.  Multipliers act on its all-cause cumulative hazard, so
    the PH family is genuinely proportional on that marginal curve rather
    than only within a sampled latent cure class.  That statistical all-cause
    construction does not isolate a biological disease hazard.
    """

    scale_time: np.ndarray
    shape: np.ndarray
    cure_fraction: np.ndarray
    background_hazard: np.ndarray
    net_scale: np.ndarray
    breakpoints: np.ndarray
    hazard_multipliers: np.ndarray

    def __post_init__(self):
        try:
            scale = np.asarray(self.scale_time, dtype=float)
            shape = np.asarray(self.shape, dtype=float)
            cure = np.asarray(self.cure_fraction, dtype=float)
            background = np.asarray(self.background_hazard, dtype=float)
            raw_net = np.asarray(self.net_scale, dtype=object)
            breakpoints = np.asarray(self.breakpoints, dtype=float)
            multipliers = np.asarray(self.hazard_multipliers, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("mixture-hazard event parameters must be numeric") from error
        if scale.ndim != 1 or not len(scale):
            raise ValueError("scale_time must be a non-empty vector")
        patient_count = len(scale)
        if any(
            array.shape != (patient_count,)
            for array in (shape, cure, background, raw_net)
        ):
            raise ValueError("mixture-hazard parameters must contain one value per patient")
        if any(not isinstance(value, (bool, np.bool_)) for value in raw_net.flat):
            raise ValueError("net_scale must contain boolean values")
        net = np.asarray(raw_net, dtype=bool)
        if breakpoints.ndim != 2 or breakpoints.shape[0] != patient_count:
            raise ValueError("breakpoints must have one row per patient")
        if multipliers.shape != (patient_count, breakpoints.shape[1] + 1):
            raise ValueError(
                "hazard_multipliers must have one more column than breakpoints"
            )
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("scale_time must contain finite positive values")
        if np.any(~np.isfinite(shape)) or np.any(shape <= 0.0):
            raise ValueError("shape must contain finite positive values")
        if np.any(~np.isfinite(cure)) or np.any((cure < 0.0) | (cure > 1.0)):
            raise ValueError("cure_fraction must lie in [0, 1]")
        if np.any(~np.isfinite(background)) or np.any(background < 0.0):
            raise ValueError("background_hazard must contain finite non-negative values")
        if np.any(~np.isfinite(breakpoints)) or np.any(breakpoints < 0.0):
            raise ValueError("breakpoints must contain finite non-negative values")
        if breakpoints.shape[1] > 1 and np.any(
            np.diff(breakpoints, axis=1) <= 0.0
        ):
            raise ValueError("breakpoints must be strictly increasing within each patient")
        if np.any(~np.isfinite(multipliers)) or np.any(multipliers <= 0.0):
            raise ValueError("hazard_multipliers must contain finite positive values")
        copied = []
        for array in (
            scale,
            shape,
            cure,
            background,
            net,
            breakpoints,
            multipliers,
        ):
            value = np.array(array, copy=True)
            value.setflags(write=False)
            copied.append(value)
        (
            scale,
            shape,
            cure,
            background,
            net,
            breakpoints,
            multipliers,
        ) = copied
        object.__setattr__(self, "scale_time", scale)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "cure_fraction", cure)
        object.__setattr__(self, "background_hazard", background)
        object.__setattr__(self, "net_scale", net)
        object.__setattr__(self, "breakpoints", breakpoints)
        object.__setattr__(self, "hazard_multipliers", multipliers)

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

    def _baseline_cumulative_hazard(self, times):
        disease_hazard = np.power(times / self.scale_time, self.shape)
        uncured_survival = np.exp(-disease_hazard)
        background_hazard = np.zeros(self.patient_count, dtype=float)
        positive = self.background_hazard > 0.0
        background_hazard[positive] = (
            self.background_hazard[positive] * times[positive]
        )
        background_survival = np.exp(-background_hazard)
        overall = (
            self.cure_fraction * background_survival
            + (1.0 - self.cure_fraction) * uncured_survival
        )
        net = background_survival * (
            self.cure_fraction
            + (1.0 - self.cure_fraction) * uncured_survival
        )
        survival = np.where(self.net_scale, net, overall)
        result = np.full(self.patient_count, np.inf, dtype=float)
        positive_survival = survival > 0.0
        result[positive_survival] = -np.log(survival[positive_survival])
        return result

    def _cumulative_hazard(self, times):
        baseline = self._baseline_cumulative_hazard(times)
        result = np.zeros(self.patient_count, dtype=float)
        start_time = np.zeros(self.patient_count, dtype=float)
        start_hazard = np.zeros(self.patient_count, dtype=float)
        for index in range(self.breakpoints.shape[1]):
            end_time = self.breakpoints[:, index]
            end_hazard = self._baseline_cumulative_hazard(end_time)
            active = times > start_time
            interval_end = np.minimum(times, end_time)
            interval_hazard = self._baseline_cumulative_hazard(interval_end)
            increment = np.where(
                active,
                np.maximum(interval_hazard - start_hazard, 0.0),
                0.0,
            )
            result += self.hazard_multipliers[:, index] * increment
            start_time = end_time
            start_hazard = end_hazard
        active = times > start_time
        increment = np.where(
            active,
            np.maximum(baseline - start_hazard, 0.0),
            0.0,
        )
        result += self.hazard_multipliers[:, -1] * increment
        return result

    def cdf(self, followup_time):
        times = self._vector(followup_time, "followup_time", allow_infinity=True)
        if np.any(times < 0.0):
            raise ValueError("followup_time must be non-negative")
        return -np.expm1(-self._cumulative_hazard(times))

    def ppf(self, probability):
        probabilities = self._vector(probability, "probability")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("probability must lie in [0, 1]")
        result = np.full(self.patient_count, np.inf, dtype=float)
        result[probabilities == 0.0] = 0.0
        target = np.zeros(self.patient_count, dtype=float)
        interior = (probabilities > 0.0) & (probabilities < 1.0)
        target[interior] = -np.log1p(-probabilities[interior])
        limits = self._cumulative_hazard(
            np.full(self.patient_count, np.inf, dtype=float)
        )
        feasible = interior & ((target < limits) | np.isinf(limits))
        if not np.any(feasible):
            return result
        indices = np.flatnonzero(feasible)
        lower = np.zeros(len(indices), dtype=float)
        last_break = (
            self.breakpoints[indices, -1]
            if self.breakpoints.shape[1]
            else np.zeros(len(indices), dtype=float)
        )
        upper = self.scale_time[indices] + last_break
        positive_background = self.background_hazard[indices] > 0.0
        if np.any(positive_background):
            upper[positive_background] = np.maximum(
                upper[positive_background],
                target[indices[positive_background]]
                / self.background_hazard[indices[positive_background]],
            )
        probe = np.zeros(self.patient_count, dtype=float)
        for _ in range(128):
            probe.fill(0.0)
            probe[indices] = upper
            too_low = self._cumulative_hazard(probe)[indices] < target[indices]
            if not np.any(too_low):
                break
            upper[too_low] *= 2.0
        else:
            raise FloatingPointError("could not bracket mixture event-time quantile")
        for _ in range(80):
            midpoint = 0.5 * (lower + upper)
            probe.fill(0.0)
            probe[indices] = midpoint
            below = self._cumulative_hazard(probe)[indices] < target[indices]
            lower[below] = midpoint[below]
            upper[~below] = midpoint[~below]
        result[indices] = 0.5 * (lower + upper)
        return result


@dataclass(frozen=True, eq=False)
class DelayedCureEventTimeModel:
    """Baseline hazard until a landmark, then population hazard only.

    Patients with ``switch_to_background=True`` retain their original event
    distribution through ``switch_time``.  If alive at that landmark, their
    Weibull disease hazard stops while the unmodified constant population
    hazard continues.  Other patients stay on the baseline curve.  This is a
    continuous delayed-cure construction rather than a time-zero cure label.
    """

    scale_time: np.ndarray
    shape: np.ndarray
    constant_hazard: np.ndarray
    weibull_weight: np.ndarray
    post_switch_hazard: np.ndarray
    switch_time: np.ndarray
    switch_to_background: np.ndarray
    _base_model: PiecewiseWeibullEventTimeModel = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self):
        try:
            scale = np.asarray(self.scale_time, dtype=float)
            shape = np.asarray(self.shape, dtype=float)
            constant = np.asarray(self.constant_hazard, dtype=float)
            weight = np.asarray(self.weibull_weight, dtype=float)
            post_switch = np.asarray(self.post_switch_hazard, dtype=float)
            switch_time = np.asarray(self.switch_time, dtype=float)
            raw_switches = np.asarray(self.switch_to_background, dtype=object)
        except (TypeError, ValueError) as error:
            raise ValueError("delayed-cure event parameters must be numeric") from error
        if any(
            not isinstance(value, (bool, np.bool_))
            for value in raw_switches.flat
        ):
            raise ValueError("switch_to_background must contain boolean values")
        switches = np.asarray(raw_switches, dtype=bool)
        if scale.ndim != 1 or not len(scale):
            raise ValueError("scale_time must be a non-empty vector")
        patient_count = len(scale)
        if any(
            array.shape != (patient_count,)
            for array in (
                shape,
                constant,
                weight,
                post_switch,
                switch_time,
                switches,
            )
        ):
            raise ValueError("delayed-cure parameters must contain one value per patient")
        if np.any(~np.isfinite(post_switch)) or np.any(post_switch < 0.0):
            raise ValueError(
                "post_switch_hazard must contain finite non-negative values"
            )
        if np.any(~np.isfinite(switch_time)) or np.any(switch_time < 0.0):
            raise ValueError("switch_time must contain finite non-negative values")
        base = PiecewiseWeibullEventTimeModel(
            scale,
            shape,
            constant,
            weight,
            np.empty((patient_count, 0), dtype=float),
            np.ones((patient_count, 1), dtype=float),
        )
        copied_switch = np.array(switch_time, copy=True)
        copied_switches = np.array(switches, copy=True)
        copied_post_switch = np.array(post_switch, copy=True)
        copied_switch.setflags(write=False)
        copied_switches.setflags(write=False)
        copied_post_switch.setflags(write=False)
        object.__setattr__(self, "scale_time", base.scale_time)
        object.__setattr__(self, "shape", base.shape)
        object.__setattr__(self, "constant_hazard", base.constant_hazard)
        object.__setattr__(self, "weibull_weight", base.weibull_weight)
        object.__setattr__(self, "post_switch_hazard", copied_post_switch)
        object.__setattr__(self, "switch_time", copied_switch)
        object.__setattr__(self, "switch_to_background", copied_switches)
        object.__setattr__(self, "_base_model", base)

    @property
    def patient_count(self):
        return len(self.scale_time)

    def cdf(self, followup_time):
        times = self._base_model._vector(
            followup_time, "followup_time", allow_infinity=True
        )
        if np.any(times < 0.0):
            raise ValueError("followup_time must be non-negative")
        cumulative = self._base_model._cumulative_hazard(times)
        selected = self.switch_to_background
        if np.any(selected):
            before_time = np.minimum(times[selected], self.switch_time[selected])
            disease = self.weibull_weight[selected] * np.power(
                before_time / self.scale_time[selected], self.shape[selected]
            )
            before_constant = self.constant_hazard[selected] * before_time
            after_time = np.maximum(
                times[selected] - self.switch_time[selected], 0.0
            )
            after_constant = np.zeros(np.count_nonzero(selected), dtype=float)
            positive = self.post_switch_hazard[selected] > 0.0
            after_constant[positive] = (
                self.post_switch_hazard[selected][positive] * after_time[positive]
            )
            cumulative[selected] = before_constant + disease + after_constant
        return -np.expm1(-cumulative)

    def ppf(self, probability):
        probabilities = self._base_model._vector(probability, "probability")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("probability must lie in [0, 1]")
        result = self._base_model.ppf(probabilities)
        selected = self.switch_to_background & (probabilities > 0.0) & (
            probabilities < 1.0
        )
        if not np.any(selected):
            return result
        indices = np.flatnonzero(selected)
        target = -np.log1p(-probabilities[indices])
        delay = self.switch_time[indices]
        landmark_hazard = (
            self.constant_hazard[indices] * delay
            + self.weibull_weight[indices]
            * np.power(delay / self.scale_time[indices], self.shape[indices])
        )
        after = target > landmark_hazard
        if np.any(after):
            after_indices = indices[after]
            population = self.post_switch_hazard[after_indices]
            result[after_indices] = np.inf
            positive = population > 0.0
            if np.any(positive):
                result[after_indices[positive]] = delay[after][positive] + (
                    target[after][positive] - landmark_hazard[after][positive]
                ) / population[positive]
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


class EnrollmentDateSampler(Protocol):
    """Prior-predictive enrollment interface consumed by the WP6 engine."""

    @property
    def total_enrollment(self) -> int:
        ...

    def sample_enrollment_dates(
        self, rng: np.random.Generator
    ) -> Tuple[date, ...]:
        ...


@dataclass(frozen=True)
class LogLinearEnrollmentPrior:
    """Accrual prior that does not center itself on enrollment-count evidence.

    Each trial draws one log-linear calendar slope uniformly from the supplied
    range and then draws exactly ``N`` enrollment dates between registry
    opening and the known enrollment close.  The intermediate 20- and
    104-patient quantities never enter this prior; they remain likelihood
    evidence in :func:`public_history_constraint_branches`.
    """

    total_enrollment: int
    study_start: date
    enrollment_close: date
    log_rate_slope_lower: float = -2.0
    log_rate_slope_upper: float = 2.0

    def __post_init__(self):
        total = _positive_integer(self.total_enrollment, "total_enrollment")
        start = _parse_date(self.study_start, "study_start")
        close = _parse_date(self.enrollment_close, "enrollment_close")
        if close < start:
            raise ValueError("enrollment_close must not precede study_start")
        if isinstance(self.log_rate_slope_lower, (bool, np.bool_)) or isinstance(
            self.log_rate_slope_upper, (bool, np.bool_)
        ):
            raise ValueError("log-rate slope bounds must be numeric, not boolean")
        lower = float(self.log_rate_slope_lower)
        upper = float(self.log_rate_slope_upper)
        if not isfinite(lower) or not isfinite(upper) or lower > upper:
            raise ValueError("log-rate slope bounds must be finite and ordered")
        object.__setattr__(self, "total_enrollment", total)
        object.__setattr__(self, "study_start", start)
        object.__setattr__(self, "enrollment_close", close)
        object.__setattr__(self, "log_rate_slope_lower", lower)
        object.__setattr__(self, "log_rate_slope_upper", upper)

    def sample_enrollment_dates(self, rng):
        slope_draw = float(rng.random())
        if not isfinite(slope_draw) or not 0.0 <= slope_draw < 1.0:
            raise ValueError("rng.random must return a slope draw in [0, 1)")
        slope = self.log_rate_slope_lower + slope_draw * (
            self.log_rate_slope_upper - self.log_rate_slope_lower
        )
        uniforms = np.asarray(rng.random(self.total_enrollment), dtype=float)
        if uniforms.shape != (self.total_enrollment,) or np.any(
            ~np.isfinite(uniforms)
        ):
            raise ValueError("rng.random must return one finite draw per patient")
        if np.any((uniforms < 0.0) | (uniforms >= 1.0)):
            raise ValueError("rng.random draws must lie in [0, 1)")
        if abs(slope) <= 1e-10:
            positions = uniforms
        else:
            positions = np.log1p(uniforms * expm1(slope)) / slope
        duration = (self.enrollment_close - self.study_start).days + 1
        offsets = np.minimum((positions * duration).astype(int), duration - 1)
        return tuple(
            self.study_start + timedelta(days=int(offset)) for offset in offsets
        )


def default_regal_enrollment_prior(history=None):
    """Return the WP7 accrual prior, using no intermediate count as a center."""

    if history is None:
        history = load_regal_public_history()
    if not isinstance(history, PublicHistory):
        raise ValueError("history must be PublicHistory")
    completion_dates = [
        item.observation_date
        for item in history.enrollment_observations
        if item.accrual_anchor and item.count == history.target_enrollment
    ]
    if not completion_dates:
        raise ValueError("public history must identify the enrollment-close boundary")
    return LogLinearEnrollmentPrior(
        total_enrollment=history.target_enrollment,
        study_start=history.study_start,
        enrollment_close=max(completion_dates),
    )


class GPSEffectFamily(str, Enum):
    """Mutually explicit GPS effect structures averaged by work package 7."""

    NO_EFFECT = "no_effect"
    PROPORTIONAL_HAZARDS = "proportional_hazards"
    DELAYED_PROPORTIONAL_HAZARDS = "delayed_proportional_hazards"
    CURE_FRACTION_DIFFERENCE = "cure_fraction_difference"
    DELAYED_CURE = "delayed_cure"
    WANING_PIECEWISE = "waning_piecewise"
    RESPONDER_CURE = "responder_cure_exploratory"


REQUIRED_EFFECT_FAMILIES = tuple(GPSEffectFamily)


@dataclass(frozen=True)
class UniformPriorRange:
    """A transparent linear- or log-uniform scalar prior range."""

    lower: float
    upper: float
    log_scale: bool = False

    def __post_init__(self):
        if isinstance(self.lower, (bool, np.bool_)) or isinstance(
            self.upper, (bool, np.bool_)
        ):
            raise ValueError("prior bounds must be numeric, not boolean")
        lower = float(self.lower)
        upper = float(self.upper)
        if not isfinite(lower) or not isfinite(upper) or lower > upper:
            raise ValueError("prior bounds must be finite and ordered")
        if not isinstance(self.log_scale, bool):
            raise ValueError("log_scale must be boolean")
        if self.log_scale and lower <= 0.0:
            raise ValueError("log-uniform prior bounds must be positive")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def is_point_mass(self):
        return self.lower == self.upper

    def is_point_at(self, value):
        return self.is_point_mass and self.lower == float(value)

    def sample(self, rng):
        if self.is_point_mass:
            return self.lower
        draw = float(rng.random())
        if not isfinite(draw) or not 0.0 <= draw < 1.0:
            raise ValueError("rng.random must return a prior draw in [0, 1)")
        if self.log_scale:
            return exp(log(self.lower) + draw * (log(self.upper) - log(self.lower)))
        return self.lower + draw * (self.upper - self.lower)


POINT_ZERO_PRIOR = UniformPriorRange(0.0, 0.0)
POINT_ONE_PRIOR = UniformPriorRange(1.0, 1.0, log_scale=True)


@dataclass(frozen=True)
class EffectParameters:
    """One parameter draw from a named GPS effect-family prior."""

    family: GPSEffectFamily
    hazard_ratio: float = 1.0
    delay_months: float = 0.0
    late_hazard_ratio: float = 1.0
    extra_cure_probability: float = 0.0
    response_probability: float = 0.0
    responder_cure_probability: float = 0.0

    def __post_init__(self):
        try:
            family = GPSEffectFamily(self.family)
        except (TypeError, ValueError) as error:
            raise ValueError("family must be a GPSEffectFamily") from error
        raw_values = {
            "hazard_ratio": self.hazard_ratio,
            "delay_months": self.delay_months,
            "late_hazard_ratio": self.late_hazard_ratio,
            "extra_cure_probability": self.extra_cure_probability,
            "response_probability": self.response_probability,
            "responder_cure_probability": self.responder_cure_probability,
        }
        if any(isinstance(value, (bool, np.bool_)) for value in raw_values.values()):
            raise ValueError("effect parameters must be numeric, not boolean")
        values = {
            "hazard_ratio": float(self.hazard_ratio),
            "delay_months": float(self.delay_months),
            "late_hazard_ratio": float(self.late_hazard_ratio),
            "extra_cure_probability": float(self.extra_cure_probability),
            "response_probability": float(self.response_probability),
            "responder_cure_probability": float(
                self.responder_cure_probability
            ),
        }
        if any(not isfinite(value) for value in values.values()):
            raise ValueError("effect parameters must be finite")
        if values["hazard_ratio"] <= 0.0 or values["late_hazard_ratio"] <= 0.0:
            raise ValueError("hazard ratios must be positive")
        if values["delay_months"] < 0.0:
            raise ValueError("delay_months must be non-negative")
        for name in (
            "extra_cure_probability",
            "response_probability",
            "responder_cure_probability",
        ):
            if not 0.0 <= values[name] <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        active = {
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
                "late_hazard_ratio",
                "delay_months",
            ),
            GPSEffectFamily.RESPONDER_CURE: (
                "response_probability",
                "responder_cure_probability",
            ),
        }[family]
        neutral = {
            "hazard_ratio": 1.0,
            "delay_months": 0.0,
            "late_hazard_ratio": 1.0,
            "extra_cure_probability": 0.0,
            "response_probability": 0.0,
            "responder_cure_probability": 0.0,
        }
        ignored = set(values) - set(active)
        if any(values[name] != neutral[name] for name in ignored):
            raise ValueError(f"{family.value} received parameters it does not use")
        object.__setattr__(self, "family", family)
        for name, value in values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class EffectFamilyPrior:
    """Within-family parameter prior, kept separate from model weights."""

    family: GPSEffectFamily
    hazard_ratio: UniformPriorRange = POINT_ONE_PRIOR
    delay_months: UniformPriorRange = POINT_ZERO_PRIOR
    late_hazard_ratio: UniformPriorRange = POINT_ONE_PRIOR
    extra_cure_probability: UniformPriorRange = POINT_ZERO_PRIOR
    response_probability: UniformPriorRange = POINT_ZERO_PRIOR
    responder_cure_probability: UniformPriorRange = POINT_ZERO_PRIOR

    def __post_init__(self):
        try:
            family = GPSEffectFamily(self.family)
        except (TypeError, ValueError) as error:
            raise ValueError("family must be a GPSEffectFamily") from error
        ranges = {
            "hazard_ratio": self.hazard_ratio,
            "delay_months": self.delay_months,
            "late_hazard_ratio": self.late_hazard_ratio,
            "extra_cure_probability": self.extra_cure_probability,
            "response_probability": self.response_probability,
            "responder_cure_probability": self.responder_cure_probability,
        }
        if not all(isinstance(value, UniformPriorRange) for value in ranges.values()):
            raise ValueError("effect parameter priors must be UniformPriorRange values")
        if ranges["hazard_ratio"].lower <= 0.0 or ranges[
            "late_hazard_ratio"
        ].lower <= 0.0:
            raise ValueError("hazard-ratio prior bounds must be positive")
        if ranges["delay_months"].lower < 0.0:
            raise ValueError("delay prior bounds must be non-negative")
        for name in (
            "extra_cure_probability",
            "response_probability",
            "responder_cure_probability",
        ):
            if ranges[name].lower < 0.0 or ranges[name].upper > 1.0:
                raise ValueError(f"{name} prior bounds must lie in [0, 1]")
        sample = EffectParameters(
            family=family,
            hazard_ratio=ranges["hazard_ratio"].lower,
            delay_months=ranges["delay_months"].lower,
            late_hazard_ratio=ranges["late_hazard_ratio"].lower,
            extra_cure_probability=ranges["extra_cure_probability"].lower,
            response_probability=ranges["response_probability"].lower,
            responder_cure_probability=ranges[
                "responder_cure_probability"
            ].lower,
        )
        # Validate the opposite corner too, catching a non-neutral ignored range.
        EffectParameters(
            family=family,
            hazard_ratio=ranges["hazard_ratio"].upper,
            delay_months=ranges["delay_months"].upper,
            late_hazard_ratio=ranges["late_hazard_ratio"].upper,
            extra_cure_probability=ranges["extra_cure_probability"].upper,
            response_probability=ranges["response_probability"].upper,
            responder_cure_probability=ranges[
                "responder_cure_probability"
            ].upper,
        )
        object.__setattr__(self, "family", sample.family)

    def sample(self, rng):
        family = self.family
        values = {}
        if family in (
            GPSEffectFamily.PROPORTIONAL_HAZARDS,
            GPSEffectFamily.DELAYED_PROPORTIONAL_HAZARDS,
            GPSEffectFamily.WANING_PIECEWISE,
        ):
            values["hazard_ratio"] = self.hazard_ratio.sample(rng)
        if family in (
            GPSEffectFamily.DELAYED_PROPORTIONAL_HAZARDS,
            GPSEffectFamily.DELAYED_CURE,
            GPSEffectFamily.WANING_PIECEWISE,
        ):
            values["delay_months"] = self.delay_months.sample(rng)
        if family is GPSEffectFamily.WANING_PIECEWISE:
            values["late_hazard_ratio"] = self.late_hazard_ratio.sample(rng)
        if family in (
            GPSEffectFamily.CURE_FRACTION_DIFFERENCE,
            GPSEffectFamily.DELAYED_CURE,
        ):
            values["extra_cure_probability"] = (
                self.extra_cure_probability.sample(rng)
            )
        if family is GPSEffectFamily.RESPONDER_CURE:
            values["response_probability"] = self.response_probability.sample(rng)
            values["responder_cure_probability"] = (
                self.responder_cure_probability.sample(rng)
            )
        return EffectParameters(family=family, **values)


DEFAULT_EFFECT_FAMILY_PRIORS = (
    EffectFamilyPrior(GPSEffectFamily.NO_EFFECT),
    EffectFamilyPrior(
        GPSEffectFamily.PROPORTIONAL_HAZARDS,
        hazard_ratio=UniformPriorRange(0.50, 1.10, log_scale=True),
    ),
    EffectFamilyPrior(
        GPSEffectFamily.DELAYED_PROPORTIONAL_HAZARDS,
        hazard_ratio=UniformPriorRange(0.45, 1.05, log_scale=True),
        delay_months=UniformPriorRange(3.0, 18.0),
    ),
    EffectFamilyPrior(
        GPSEffectFamily.CURE_FRACTION_DIFFERENCE,
        extra_cure_probability=UniformPriorRange(0.0, 0.60),
    ),
    EffectFamilyPrior(
        GPSEffectFamily.DELAYED_CURE,
        delay_months=UniformPriorRange(3.0, 18.0),
        extra_cure_probability=UniformPriorRange(0.0, 0.60),
    ),
    EffectFamilyPrior(
        GPSEffectFamily.WANING_PIECEWISE,
        hazard_ratio=UniformPriorRange(0.40, 0.90, log_scale=True),
        delay_months=UniformPriorRange(6.0, 24.0),
        late_hazard_ratio=UniformPriorRange(0.85, 1.15, log_scale=True),
    ),
    EffectFamilyPrior(
        GPSEffectFamily.RESPONDER_CURE,
        response_probability=UniformPriorRange(0.60, 0.95),
        responder_cure_probability=UniformPriorRange(0.20, 0.85),
    ),
)


@dataclass(frozen=True)
class GPSEffectScenarioSampler:
    """Prior-predictive patient generator for one GPS effect family.

    BAT assignments are drawn before randomization.  Four binary protocol
    factors are then drawn and treatment is balanced independently inside each
    realized combined stratum.  This is a prior over an unknown patient-level
    factor distribution, not a claim about REGAL's confidential covariates.
    """

    effect_prior: EffectFamilyPrior
    bat_design: BATDesign = PRIMARY_EQUAL_STRATA
    component_library: Mapping = field(
        default_factory=lambda: DEFAULT_COMPONENT_LIBRARY
    )
    background_mortality: ExponentialBackgroundMortality = (
        ExponentialBackgroundMortality(0.02)
    )
    censoring_annual_probability: float = 0.0
    protocol_factor_probabilities: Tuple[float, ...] = (0.5, 0.5, 0.5, 0.5)

    def __post_init__(self):
        if not isinstance(self.effect_prior, EffectFamilyPrior):
            raise ValueError("effect_prior must be EffectFamilyPrior")
        if not isinstance(self.bat_design, BATDesign):
            raise ValueError("bat_design must be BATDesign")
        if not isinstance(self.component_library, Mapping):
            raise ValueError("component_library must be a mapping")
        library = dict(self.component_library)
        self.bat_design.validate_library(library)
        observation = library.get(BATComponent.OBSERVATION)
        if not isinstance(observation, CureMixtureComponent):
            raise ValueError(
                "component library must include a valid observation profile"
            )
        if not isinstance(self.background_mortality, ExponentialBackgroundMortality):
            raise ValueError(
                "background_mortality must be ExponentialBackgroundMortality"
            )
        if isinstance(self.censoring_annual_probability, (bool, np.bool_)):
            raise ValueError(
                "censoring_annual_probability must be numeric, not boolean"
            )
        censoring = float(self.censoring_annual_probability)
        if not isfinite(censoring) or not 0.0 <= censoring < 1.0:
            raise ValueError("censoring_annual_probability must lie in [0, 1)")
        try:
            raw_factors = tuple(self.protocol_factor_probabilities)
        except TypeError as error:
            raise ValueError(
                "protocol factor probabilities must be an iterable"
            ) from error
        if any(isinstance(value, (bool, np.bool_)) for value in raw_factors):
            raise ValueError("protocol factor probabilities must be numeric, not boolean")
        try:
            factors = tuple(float(value) for value in raw_factors)
        except (TypeError, ValueError) as error:
            raise ValueError("protocol factor probabilities must be numeric") from error
        if len(factors) != 4 or any(
            not isfinite(value) or not 0.0 <= value <= 1.0 for value in factors
        ):
            raise ValueError(
                "protocol_factor_probabilities must contain four values in [0, 1]"
            )
        object.__setattr__(self, "component_library", MappingProxyType(library))
        object.__setattr__(self, "censoring_annual_probability", censoring)
        object.__setattr__(self, "protocol_factor_probabilities", factors)

    def _sample_protocol_factors(self, patient_count, rng):
        draws = np.asarray(rng.random((patient_count, 4)), dtype=float)
        if draws.shape != (patient_count, 4) or np.any(~np.isfinite(draws)):
            raise ValueError("rng.random must return four factor draws per patient")
        if np.any((draws < 0.0) | (draws >= 1.0)):
            raise ValueError("rng.random factor draws must lie in [0, 1)")
        return draws < np.asarray(self.protocol_factor_probabilities, dtype=float)

    @staticmethod
    def _uniform_vector(rng, patient_count, label):
        draws = np.asarray(rng.random(patient_count), dtype=float)
        if draws.shape != (patient_count,) or np.any(~np.isfinite(draws)):
            raise ValueError(f"rng.random must return one finite {label} draw per patient")
        if np.any((draws < 0.0) | (draws >= 1.0)):
            raise ValueError(f"rng.random {label} draws must lie in [0, 1)")
        return draws

    @staticmethod
    def _stratified_randomize(factors, rng):
        patient_count = len(factors)
        _, cells = np.unique(factors, axis=0, return_inverse=True)
        treatment = np.zeros(patient_count, dtype=bool)
        for cell in range(int(cells.max()) + 1):
            indices = np.flatnonzero(cells == cell)
            count = len(indices) // 2
            if len(indices) % 2:
                extra = float(rng.random())
                if not isfinite(extra) or not 0.0 <= extra < 1.0:
                    raise ValueError("rng.random must return a randomization draw in [0, 1)")
                count += int(extra < 0.5)
            order = np.asarray(rng.permutation(indices), dtype=int)
            if order.shape != indices.shape:
                raise ValueError("rng.permutation returned an invalid stratum order")
            treatment[order[:count]] = True
        return treatment

    def _component_vectors(self, assignments):
        components = tuple(
            component_for(assignment, self.component_library)
            for assignment in assignments
        )
        scale = np.asarray(
            [item.uncured.scale_months * MONTH_DAYS for item in components],
            dtype=float,
        )
        shape = np.asarray([item.uncured.shape for item in components], dtype=float)
        cure = np.asarray([item.cure_fraction for item in components], dtype=float)
        background = self.background_mortality.monthly_hazard / MONTH_DAYS
        net_scale = np.asarray(
            [item.survival_scale is SurvivalScale.NET for item in components],
            dtype=bool,
        )
        constant = np.asarray(
            [
                background if is_net else 0.0 for is_net in net_scale
            ],
            dtype=float,
        )
        return scale, shape, cure, constant, net_scale

    def _apply_cure(self, cured, constant, weight):
        background = self.background_mortality.monthly_hazard / MONTH_DAYS
        constant[cured] = background
        weight[cured] = 0.0

    def _sample_uncured_bat_components(self, patient_count, rng):
        positive_pathways = tuple(
            pathway
            for pathway in self.bat_design.pathways
            if pathway.probability > 0.0
        )
        components = tuple(
            self.component_library[pathway.regimen.survival_component]
            for pathway in positive_pathways
        )
        masses = np.asarray(
            [
                pathway.probability * (1.0 - component.cure_fraction)
                for pathway, component in zip(positive_pathways, components)
            ],
            dtype=float,
        )
        total = float(masses.sum())
        if not isfinite(total) or total <= 0.0:
            raise ValueError("BAT prior has no uncured responder-profile mass")
        cumulative = np.cumsum(masses / total)
        cumulative[-1] = 1.0
        draws = self._uniform_vector(rng, patient_count, "BAT-profile")
        indices = np.searchsorted(cumulative, draws, side="right")
        return tuple(components[index] for index in indices)

    def _responder_vectors(
        self, treatment, scale, shape, cure, constant, weight, parameters, rng
    ):
        patient_count = len(treatment)
        baseline_cured = self._uniform_vector(
            rng, patient_count, "baseline-cure"
        ) < cure
        response = self._uniform_vector(rng, patient_count, "response") < (
            parameters.response_probability
        )
        responder_cured = self._uniform_vector(
            rng, patient_count, "responder-cure"
        ) < (
            parameters.responder_cure_probability
        )
        treated_responder = treatment & response
        treated_nonresponder = treatment & ~response
        responder_components = self._sample_uncured_bat_components(
            patient_count, rng
        )
        if np.any(treated_responder):
            responder_indices = np.flatnonzero(treated_responder)
            for index in responder_indices:
                component = responder_components[index]
                scale[index] = component.uncured.scale_months * MONTH_DAYS
                shape[index] = component.uncured.shape
                constant[index] = (
                    self.background_mortality.monthly_hazard / MONTH_DAYS
                    if component.survival_scale is SurvivalScale.NET
                    else 0.0
                )
        observation = self.component_library[BATComponent.OBSERVATION]
        if np.any(treated_nonresponder):
            scale[treated_nonresponder] = observation.uncured.scale_months * MONTH_DAYS
            shape[treated_nonresponder] = observation.uncured.shape
            constant[treated_nonresponder] = (
                self.background_mortality.monthly_hazard / MONTH_DAYS
                if observation.survival_scale is SurvivalScale.NET
                else 0.0
            )
        nonresponder_cured = self._uniform_vector(
            rng, patient_count, "nonresponder-cure"
        ) < observation.cure_fraction
        cured = (~treatment & baseline_cured) | (
            treated_responder & responder_cured
        ) | (treated_nonresponder & nonresponder_cured)
        self._apply_cure(cured, constant, weight)

    def __call__(self, entry_dates, rng):
        entries = tuple(_parse_date(value, "entry date") for value in entry_dates)
        if not entries:
            raise ValueError("entry_dates must contain at least one patient")
        patient_count = len(entries)
        bat = self.bat_design.sample(patient_count, rng)
        factors = self._sample_protocol_factors(patient_count, rng)
        treatment = self._stratified_randomize(factors, rng)
        parameters = self.effect_prior.sample(rng)
        scale, shape, cure, constant, net_scale = self._component_vectors(
            bat.assignments
        )
        weight = np.ones(patient_count, dtype=float)
        family = parameters.family
        delayed_switch = None
        marginal_hazard_family = family in (
            GPSEffectFamily.NO_EFFECT,
            GPSEffectFamily.PROPORTIONAL_HAZARDS,
            GPSEffectFamily.DELAYED_PROPORTIONAL_HAZARDS,
            GPSEffectFamily.WANING_PIECEWISE,
        )

        if family is GPSEffectFamily.RESPONDER_CURE:
            self._responder_vectors(
                treatment,
                scale,
                shape,
                cure,
                constant,
                weight,
                parameters,
                rng,
            )
        elif not marginal_hazard_family:
            base_cured = self._uniform_vector(
                rng, patient_count, "baseline-cure"
            ) < cure
            extra_cured = np.zeros(patient_count, dtype=bool)
            if family in (
                GPSEffectFamily.CURE_FRACTION_DIFFERENCE,
                GPSEffectFamily.DELAYED_CURE,
            ):
                extra_draws = self._uniform_vector(
                    rng, patient_count, "extra-cure"
                )
                extra_cured = treatment & ~base_cured & (
                    extra_draws < parameters.extra_cure_probability
                )
            self._apply_cure(base_cured, constant, weight)
            if family is GPSEffectFamily.CURE_FRACTION_DIFFERENCE:
                self._apply_cure(extra_cured, constant, weight)
            elif family is GPSEffectFamily.DELAYED_CURE:
                delayed_switch = extra_cured

        censoring_model = ExponentialBackgroundMortality(
            self.censoring_annual_probability
        )
        censoring = np.asarray(
            censoring_model.sample_event_times(rng, (patient_count,)), dtype=float
        ) * MONTH_DAYS

        if family is GPSEffectFamily.DELAYED_CURE:
            event_model = DelayedCureEventTimeModel(
                scale,
                shape,
                constant,
                weight,
                np.full(
                    patient_count,
                    self.background_mortality.monthly_hazard / MONTH_DAYS,
                ),
                np.full(patient_count, parameters.delay_months * MONTH_DAYS),
                delayed_switch,
            )
        else:
            if family in (
                GPSEffectFamily.DELAYED_PROPORTIONAL_HAZARDS,
                GPSEffectFamily.WANING_PIECEWISE,
            ):
                breakpoints = np.full(
                    (patient_count, 1), parameters.delay_months * MONTH_DAYS
                )
                multipliers = np.ones((patient_count, 2), dtype=float)
                affected = treatment & (weight > 0.0)
                if family is GPSEffectFamily.DELAYED_PROPORTIONAL_HAZARDS:
                    multipliers[affected, 1] = parameters.hazard_ratio
                else:
                    multipliers[affected, 0] = parameters.hazard_ratio
                    multipliers[affected, 1] = parameters.late_hazard_ratio
            else:
                breakpoints = np.empty((patient_count, 0), dtype=float)
                multipliers = np.ones((patient_count, 1), dtype=float)
                if family is GPSEffectFamily.PROPORTIONAL_HAZARDS:
                    multipliers[treatment & (weight > 0.0), 0] = (
                        parameters.hazard_ratio
                    )
            if marginal_hazard_family:
                event_model = PiecewiseMixtureHazardEventTimeModel(
                    scale,
                    shape,
                    cure,
                    np.full(
                        patient_count,
                        self.background_mortality.monthly_hazard / MONTH_DAYS,
                    ),
                    net_scale,
                    breakpoints,
                    multipliers,
                )
            else:
                event_model = PiecewiseWeibullEventTimeModel(
                    scale,
                    shape,
                    constant,
                    weight,
                    breakpoints,
                    multipliers,
                )
        return ScenarioPatients(
            treatment=treatment,
            strata=factors.astype(int),
            censoring_time=censoring,
            event_time_model=event_model,
        )


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
    log_denominator = _logsumexp(denominator)
    if not isfinite(log_denominator):
        return float("nan")
    return _unit_probability(
        _probability_from_log(
            _logsumexp(numerator) - log_denominator
        ),
        "conditional probability",
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


@dataclass(frozen=True)
class EffectFamilyProjection:
    """One within-family prior-predictive WP6 projection."""

    family: GPSEffectFamily
    parameter_prior: EffectFamilyPrior
    conditioning: ConditioningResult

    def __post_init__(self):
        try:
            family = GPSEffectFamily(self.family)
        except (TypeError, ValueError) as error:
            raise ValueError("family must be a GPSEffectFamily") from error
        if not isinstance(self.parameter_prior, EffectFamilyPrior):
            raise ValueError("parameter_prior must be EffectFamilyPrior")
        if self.parameter_prior.family is not family:
            raise ValueError("projection family and parameter prior differ")
        if not isinstance(self.conditioning, ConditioningResult):
            raise ValueError("conditioning must be ConditioningResult")
        object.__setattr__(self, "family", family)


@dataclass(frozen=True)
class ModelFamilyWeightPrior:
    """Prior masses over the complete required effect-family set."""

    name: str
    weights: Tuple[Tuple[GPSEffectFamily, float], ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("model-weight prior name must be non-empty")
        normalized = []
        seen = set()
        try:
            supplied = tuple(self.weights)
        except TypeError as error:
            raise ValueError("weights must contain (family, probability) pairs") from error
        for item in supplied:
            try:
                family_value, probability_value = item
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "weights must contain (family, probability) pairs"
                ) from error
            try:
                family = GPSEffectFamily(family_value)
            except (TypeError, ValueError) as error:
                raise ValueError("model-weight keys must be GPSEffectFamily values") from error
            if family in seen:
                raise ValueError("model-weight prior must not repeat a family")
            if isinstance(probability_value, (bool, np.bool_)):
                raise ValueError("model-family prior weights must be numeric, not boolean")
            probability = float(probability_value)
            if not isfinite(probability) or probability <= 0.0:
                raise ValueError("every required family must have positive prior mass")
            seen.add(family)
            normalized.append((family, probability))
        if seen != set(REQUIRED_EFFECT_FAMILIES):
            raise ValueError("model-weight prior must cover every required effect family")
        total = sum(value for _, value in normalized)
        if abs(total - 1.0) > PROBABILITY_TOLERANCE:
            raise ValueError("model-family prior weights must sum to one")
        normalized = tuple(
            (family, value / total) for family, value in normalized
        )
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "weights", normalized)

    @property
    def as_mapping(self):
        return MappingProxyType(dict(self.weights))


BALANCED_MODEL_FAMILY_PRIOR = ModelFamilyWeightPrior(
    "balanced",
    (
        (GPSEffectFamily.NO_EFFECT, 0.20),
        (GPSEffectFamily.PROPORTIONAL_HAZARDS, 0.15),
        (GPSEffectFamily.DELAYED_PROPORTIONAL_HAZARDS, 0.15),
        (GPSEffectFamily.CURE_FRACTION_DIFFERENCE, 0.15),
        (GPSEffectFamily.DELAYED_CURE, 0.15),
        (GPSEffectFamily.WANING_PIECEWISE, 0.10),
        (GPSEffectFamily.RESPONDER_CURE, 0.10),
    ),
)

SKEPTICAL_MODEL_FAMILY_PRIOR = ModelFamilyWeightPrior(
    "skeptical",
    (
        (GPSEffectFamily.NO_EFFECT, 0.40),
        (GPSEffectFamily.PROPORTIONAL_HAZARDS, 0.15),
        (GPSEffectFamily.DELAYED_PROPORTIONAL_HAZARDS, 0.15),
        (GPSEffectFamily.CURE_FRACTION_DIFFERENCE, 0.08),
        (GPSEffectFamily.DELAYED_CURE, 0.08),
        (GPSEffectFamily.WANING_PIECEWISE, 0.10),
        (GPSEffectFamily.RESPONDER_CURE, 0.04),
    ),
)

CURE_FAVORING_MODEL_FAMILY_PRIOR = ModelFamilyWeightPrior(
    "cure_favoring",
    (
        (GPSEffectFamily.NO_EFFECT, 0.10),
        (GPSEffectFamily.PROPORTIONAL_HAZARDS, 0.10),
        (GPSEffectFamily.DELAYED_PROPORTIONAL_HAZARDS, 0.10),
        (GPSEffectFamily.CURE_FRACTION_DIFFERENCE, 0.20),
        (GPSEffectFamily.DELAYED_CURE, 0.20),
        (GPSEffectFamily.WANING_PIECEWISE, 0.10),
        (GPSEffectFamily.RESPONDER_CURE, 0.20),
    ),
)

DEFAULT_MODEL_FAMILY_PRIOR_SENSITIVITY = (
    SKEPTICAL_MODEL_FAMILY_PRIOR,
    BALANCED_MODEL_FAMILY_PRIOR,
    CURE_FAVORING_MODEL_FAMILY_PRIOR,
)


@dataclass(frozen=True)
class PosteriorFamilyResult:
    """Prior and posterior contribution from one GPS effect family."""

    family: GPSEffectFamily
    prior_weight: float
    posterior_weight: float
    log_p_public_history_and_continuation: float
    parameter_prior: EffectFamilyPrior
    conditioning: ConditioningResult

    @property
    def p_public_history_and_continuation(self):
        return _probability_from_log(self.log_p_public_history_and_continuation)


@dataclass(frozen=True)
class PosteriorForecastResult:
    """Continuation-conditioned model average with explicit readiness checks."""

    sensitivity_name: str
    family_results: Tuple[PosteriorFamilyResult, ...]
    log_p_public_history_and_continuation: float
    p_final_rejection_given_public_history_and_continuation: float
    p_final_reached_given_public_history_and_continuation: float

    @property
    def p_public_history_and_continuation(self):
        return _probability_from_log(self.log_p_public_history_and_continuation)

    @property
    def forecast_readiness_issues(self):
        """Reasons this complete average is not ready for a forecast label.

        Completeness is necessary but not sufficient. Every family likelihood
        and continuation-conditioned projection must also clear the minimum
        effective-sample-size and maximum history-weight-concentration gates.
        These are release safeguards, not a proof of Monte Carlo convergence.
        """

        issues = []
        families = tuple(item.family for item in self.family_results)
        if len(families) != len(REQUIRED_EFFECT_FAMILIES) or set(families) != set(
            REQUIRED_EFFECT_FAMILIES
        ):
            issues.append("model average does not contain every required family")
        for item in self.family_results:
            result = item.conditioning
            if not isfinite(result.log_p_public_history):
                issues.append(
                    f"{item.family.value} public-history log evidence is not finite"
                )
            conditional_probabilities = (
                (
                    "continuation",
                    result.p_continue_given_public_history,
                ),
                (
                    "final rejection",
                    result.p_final_rejection_given_public_history_and_continuation,
                ),
                (
                    "final reach",
                    result.p_final_reached_given_public_history_and_continuation,
                ),
            )
            nonfinite_conditionals = tuple(
                name
                for name, probability in conditional_probabilities
                if not isfinite(probability)
            )
            if nonfinite_conditionals:
                issues.append(
                    f"{item.family.value} has non-finite conditional probabilities: "
                    + ", ".join(nonfinite_conditionals)
                )
            if not isfinite(result.history_effective_sample_size):
                issues.append(f"{item.family.value} history ESS is not finite")
            elif (
                result.history_effective_sample_size
                < MINIMUM_POSTERIOR_FORECAST_ESS
            ):
                issues.append(
                    f"{item.family.value} history ESS is below "
                    f"{MINIMUM_POSTERIOR_FORECAST_ESS:g}"
                )
            if not isfinite(result.continuation_effective_sample_size):
                issues.append(
                    f"{item.family.value} continuation ESS is not finite"
                )
            elif (
                result.continuation_effective_sample_size
                < MINIMUM_POSTERIOR_FORECAST_ESS
            ):
                issues.append(
                    f"{item.family.value} continuation ESS is below "
                    f"{MINIMUM_POSTERIOR_FORECAST_ESS:g}"
                )
            if not isfinite(result.maximum_history_weight_share):
                issues.append(
                    f"{item.family.value} maximum history weight share is not finite"
                )
            elif (
                result.maximum_history_weight_share
                > MAXIMUM_POSTERIOR_FORECAST_HISTORY_WEIGHT_SHARE
            ):
                issues.append(
                    f"{item.family.value} maximum history weight share exceeds "
                    f"{MAXIMUM_POSTERIOR_FORECAST_HISTORY_WEIGHT_SHARE:g}"
                )
        return tuple(issues)

    @property
    def is_posterior_forecast(self):
        return not self.forecast_readiness_issues

    @property
    def model_prior_weights(self):
        return MappingProxyType(
            {item.family: item.prior_weight for item in self.family_results}
        )

    @property
    def model_posterior_weights(self):
        return MappingProxyType(
            {item.family: item.posterior_weight for item in self.family_results}
        )

    @property
    def assumed_futility_hr_threshold(self):
        return self.family_results[0].conditioning.assumed_futility_hr_threshold


def posterior_model_average(
    projections,
    model_weight_prior=BALANCED_MODEL_FAMILY_PRIOR,
):
    """Average family projections using ``P(history, continue | family)``.

    Within-family parameter uncertainty is already integrated by each
    ``scenario_sampler`` draw.  Bayes' rule then gives

    ``w_j | H,C proportional to w_j P(H | j) P(C | H,j)``.

    The final rejection forecast is the posterior-weighted average of the WP6
    family-specific conditional projections.  A result is returned only when
    all required families are present, preventing a partial sensitivity run
    from being mislabeled as the v2 forecast.
    """

    projections = tuple(projections)
    if not projections or not all(
        isinstance(item, EffectFamilyProjection) for item in projections
    ):
        raise ValueError("projections must contain EffectFamilyProjection values")
    if not isinstance(model_weight_prior, ModelFamilyWeightPrior):
        raise ValueError("model_weight_prior must be ModelFamilyWeightPrior")
    by_family = {}
    for projection in projections:
        if projection.family in by_family:
            raise ValueError("projections must not repeat an effect family")
        by_family[projection.family] = projection
    if set(by_family) != set(REQUIRED_EFFECT_FAMILIES):
        raise ValueError("projections must cover every required effect family")
    ordered = tuple(by_family[family] for family in REQUIRED_EFFECT_FAMILIES)
    first_design = ordered[0].conditioning.design
    if any(item.conditioning.design != first_design for item in ordered[1:]):
        raise ValueError("all effect-family projections must use the same trial design")

    prior_weights = model_weight_prior.as_mapping
    joint_logs = []
    for projection in ordered:
        result = projection.conditioning
        continuation = result.p_continue_given_public_history
        log_history = float(result.log_p_public_history)
        if np.isnan(log_history) or log_history == float("inf"):
            raise ValueError("family public-history log probability is invalid")
        if log_history == float("-inf"):
            joint = float("-inf")
        else:
            continuation = _unit_probability(
                continuation, "family continuation probability"
            )
            joint = (
                float("-inf")
                if continuation == 0.0
                else log_history + log(continuation)
            )
        joint_logs.append(joint)
    evidence_terms = [
        log(prior_weights[projection.family]) + joint
        for projection, joint in zip(ordered, joint_logs)
    ]
    log_evidence = _logsumexp(evidence_terms)
    if not isfinite(log_evidence):
        raise ValueError("all effect families have zero history/continuation evidence")
    posterior_weights = [
        0.0 if not isfinite(term) else exp(term - log_evidence)
        for term in evidence_terms
    ]
    family_results = []
    rejection = 0.0
    reached = 0.0
    for projection, joint, posterior_weight in zip(
        ordered, joint_logs, posterior_weights
    ):
        conditioning = projection.conditioning
        if posterior_weight > 0.0:
            family_rejection = _unit_probability(
                conditioning.p_final_rejection_given_public_history_and_continuation,
                "family final-rejection probability",
            )
            family_reached = _unit_probability(
                conditioning.p_final_reached_given_public_history_and_continuation,
                "family final-reach probability",
            )
            if family_rejection > family_reached + PROBABILITY_TOLERANCE:
                raise ValueError(
                    "family final-rejection probability cannot exceed final reach"
                )
            rejection += posterior_weight * family_rejection
            reached += posterior_weight * family_reached
        family_results.append(
            PosteriorFamilyResult(
                family=projection.family,
                prior_weight=prior_weights[projection.family],
                posterior_weight=posterior_weight,
                log_p_public_history_and_continuation=joint,
                parameter_prior=projection.parameter_prior,
                conditioning=conditioning,
            )
        )
    rejection = _unit_probability(
        rejection, "posterior final-rejection probability"
    )
    reached = _unit_probability(reached, "posterior final-reach probability")
    if rejection > reached + PROBABILITY_TOLERANCE:
        raise ValueError(
            "posterior final-rejection probability cannot exceed final reach"
        )
    rejection = min(rejection, reached)
    return PosteriorForecastResult(
        sensitivity_name=model_weight_prior.name,
        family_results=tuple(family_results),
        log_p_public_history_and_continuation=log_evidence,
        p_final_rejection_given_public_history_and_continuation=rejection,
        p_final_reached_given_public_history_and_continuation=reached,
    )


def posterior_prior_sensitivity(
    projections,
    model_weight_priors=DEFAULT_MODEL_FAMILY_PRIOR_SENSITIVITY,
):
    """Reweight identical family likelihoods across named model priors."""

    priors = tuple(model_weight_priors)
    if not priors or not all(
        isinstance(item, ModelFamilyWeightPrior) for item in priors
    ):
        raise ValueError("model_weight_priors must contain named model priors")
    if len({item.name for item in priors}) != len(priors):
        raise ValueError("model-weight sensitivity names must be unique")
    projections = tuple(projections)
    return tuple(posterior_model_average(projections, prior) for prior in priors)


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
    if not callable(getattr(enrollment_model, "sample_enrollment_dates", None)):
        raise ValueError("enrollment_model must provide sample_enrollment_dates")
    total_enrollment = getattr(enrollment_model, "total_enrollment", None)
    if (
        isinstance(total_enrollment, (bool, np.bool_))
        or not isinstance(total_enrollment, Integral)
        or int(total_enrollment) < 1
    ):
        raise ValueError("enrollment_model must expose a positive patient total")
    if int(total_enrollment) != history.target_enrollment:
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


def _normalize_effect_family_priors(effect_priors):
    priors = tuple(effect_priors)
    if not priors or not all(isinstance(item, EffectFamilyPrior) for item in priors):
        raise ValueError("effect_priors must contain EffectFamilyPrior values")
    if len({item.family for item in priors}) != len(priors):
        raise ValueError("effect_priors must not repeat an effect family")
    if {item.family for item in priors} != set(REQUIRED_EFFECT_FAMILIES):
        raise ValueError("effect_priors must cover every required effect family")
    return {item.family: item for item in priors}


def _effect_family_seed(seed, index):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    return int(
        np.random.SeedSequence((int(seed), index)).generate_state(
            1, dtype=np.uint64
        )[0]
    )


def condition_effect_families_on_public_history(
    effect_priors=DEFAULT_EFFECT_FAMILY_PRIORS,
    *,
    bat_design=PRIMARY_EQUAL_STRATA,
    component_library=DEFAULT_COMPONENT_LIBRARY,
    background_mortality=ExponentialBackgroundMortality(0.02),
    censoring_annual_probability=0.0,
    protocol_factor_probabilities=(0.5, 0.5, 0.5, 0.5),
    design=REGAL_V2_EFFICACY_DESIGN,
    history=None,
    enrollment_model=None,
    nsim=DEFAULT_IMPORTANCE_DRAWS,
    seed=20260825,
    proposal_interim_z_targets=None,
    max_lag_combinations=4096,
    max_count_vectors=DEFAULT_MAX_COUNT_VECTORS,
    max_quota_states=DEFAULT_MAX_DP_STATES,
    tilt_tolerance=DEFAULT_TILT_TOLERANCE,
    max_tilt_iterations=DEFAULT_MAX_TILT_ITERATIONS,
):
    """Integrate parameter uncertainty separately inside every effect family.

    The default enrollment model here is deliberately the WP7 log-linear
    prior, not WP5's data-centered reference curve.  Public enrollment counts
    are applied exactly once by the shared history-conditioning likelihood.
    """

    ordered = _normalize_effect_family_priors(effect_priors)
    if history is None:
        history = load_regal_public_history()
    if not isinstance(history, PublicHistory):
        raise ValueError("history must be PublicHistory")
    if enrollment_model is None:
        enrollment_model = default_regal_enrollment_prior(history)
    projections = []
    for index, family in enumerate(REQUIRED_EFFECT_FAMILIES):
        prior = ordered[family]
        sampler = GPSEffectScenarioSampler(
            effect_prior=prior,
            bat_design=bat_design,
            component_library=component_library,
            background_mortality=background_mortality,
            censoring_annual_probability=censoring_annual_probability,
            protocol_factor_probabilities=protocol_factor_probabilities,
        )
        family_seed = _effect_family_seed(seed, index)
        result = condition_on_public_history(
            sampler,
            scenario_name=f"WP7 {family.value} prior predictive",
            design=design,
            history=history,
            enrollment_model=enrollment_model,
            nsim=nsim,
            seed=family_seed,
            proposal_interim_z_targets=proposal_interim_z_targets,
            max_lag_combinations=max_lag_combinations,
            max_count_vectors=max_count_vectors,
            max_quota_states=max_quota_states,
            tilt_tolerance=tilt_tolerance,
            max_tilt_iterations=max_tilt_iterations,
        )
        projections.append(
            EffectFamilyProjection(
                family=family,
                parameter_prior=prior,
                conditioning=result,
            )
        )
    return tuple(projections)


def condition_effect_families_futility_sensitivity_grid(
    effect_priors=DEFAULT_EFFECT_FAMILY_PRIORS,
    thresholds=FUTILITY_HR_SENSITIVITY_GRID,
    *,
    bat_design=PRIMARY_EQUAL_STRATA,
    component_library=DEFAULT_COMPONENT_LIBRARY,
    background_mortality=ExponentialBackgroundMortality(0.02),
    censoring_annual_probability=0.0,
    protocol_factor_probabilities=(0.5, 0.5, 0.5, 0.5),
    base_design=REGAL_V2_EFFICACY_DESIGN,
    history=None,
    enrollment_model=None,
    nsim=DEFAULT_IMPORTANCE_DRAWS,
    seed=20260825,
    proposal_interim_z_targets=None,
    max_lag_combinations=4096,
    max_count_vectors=DEFAULT_MAX_COUNT_VECTORS,
    max_quota_states=DEFAULT_MAX_DP_STATES,
    tilt_tolerance=DEFAULT_TILT_TOLERANCE,
    max_tilt_iterations=DEFAULT_MAX_TILT_ITERATIONS,
):
    """Return complete family sets for paired futility-rule assumptions.

    Each family calls WP6's paired grid once, so all thresholds for that family
    reuse identical latent histories and importance weights.  The returned
    outer rows align with ``thresholds`` and are ready for
    :func:`posterior_model_average`.
    """

    ordered = _normalize_effect_family_priors(effect_priors)
    thresholds = tuple(thresholds)
    if history is None:
        history = load_regal_public_history()
    if not isinstance(history, PublicHistory):
        raise ValueError("history must be PublicHistory")
    if enrollment_model is None:
        enrollment_model = default_regal_enrollment_prior(history)
    projection_rows = None
    for index, family in enumerate(REQUIRED_EFFECT_FAMILIES):
        prior = ordered[family]
        sampler = GPSEffectScenarioSampler(
            effect_prior=prior,
            bat_design=bat_design,
            component_library=component_library,
            background_mortality=background_mortality,
            censoring_annual_probability=censoring_annual_probability,
            protocol_factor_probabilities=protocol_factor_probabilities,
        )
        results = condition_futility_sensitivity_grid(
            sampler,
            thresholds=thresholds,
            scenario_name=f"WP7 {family.value} prior predictive",
            base_design=base_design,
            history=history,
            enrollment_model=enrollment_model,
            nsim=nsim,
            seed=_effect_family_seed(seed, index),
            proposal_interim_z_targets=proposal_interim_z_targets,
            max_lag_combinations=max_lag_combinations,
            max_count_vectors=max_count_vectors,
            max_quota_states=max_quota_states,
            tilt_tolerance=tilt_tolerance,
            max_tilt_iterations=max_tilt_iterations,
        )
        if projection_rows is None:
            projection_rows = [[] for _ in results]
        elif len(results) != len(projection_rows):
            raise RuntimeError("futility sensitivity rows changed across families")
        for row, result in zip(projection_rows, results):
            row.append(
                EffectFamilyProjection(
                    family=family,
                    parameter_prior=prior,
                    conditioning=result,
                )
            )
    return tuple(tuple(row) for row in projection_rows)


def posterior_forecast_futility_sensitivity_grid(
    effect_priors=DEFAULT_EFFECT_FAMILY_PRIORS,
    thresholds=FUTILITY_HR_SENSITIVITY_GRID,
    *,
    model_weight_prior=BALANCED_MODEL_FAMILY_PRIOR,
    **conditioning_options,
):
    """Average the complete paired family grid under one model-weight prior."""

    projection_rows = condition_effect_families_futility_sensitivity_grid(
        effect_priors,
        thresholds,
        **conditioning_options,
    )
    return tuple(
        posterior_model_average(row, model_weight_prior)
        for row in projection_rows
    )


def posterior_forecast_prior_sensitivity(
    effect_priors=DEFAULT_EFFECT_FAMILY_PRIORS,
    model_weight_priors=DEFAULT_MODEL_FAMILY_PRIOR_SENSITIVITY,
    **conditioning_options,
):
    """Run all family projections once and report model-weight sensitivity."""

    projections = condition_effect_families_on_public_history(
        effect_priors,
        **conditioning_options,
    )
    return posterior_prior_sensitivity(projections, model_weight_priors)


__all__ = (
    "BALANCED_MODEL_FAMILY_PRIOR",
    "CURE_FAVORING_MODEL_FAMILY_PRIOR",
    "ConditioningResult",
    "DEFAULT_EFFECT_FAMILY_PRIORS",
    "DEFAULT_IMPORTANCE_DRAWS",
    "DEFAULT_MAX_COUNT_VECTORS",
    "DEFAULT_MAX_TILT_ITERATIONS",
    "DEFAULT_MODEL_FAMILY_PRIOR_SENSITIVITY",
    "DEFAULT_TILT_TOLERANCE",
    "DelayedCureEventTimeModel",
    "EffectFamilyPrior",
    "EffectFamilyProjection",
    "EffectParameters",
    "EnrollmentDateSampler",
    "ExponentialTilt",
    "GPSEffectFamily",
    "GPSEffectScenarioSampler",
    "HistoryConstraintBranch",
    "HistoryImportanceDraw",
    "LogLinearEnrollmentPrior",
    "MAXIMUM_POSTERIOR_FORECAST_HISTORY_WEIGHT_SHARE",
    "MINIMUM_POSTERIOR_FORECAST_ESS",
    "ModelFamilyWeightPrior",
    "PatientEventTimeModel",
    "PiecewiseMixtureHazardEventTimeModel",
    "PiecewiseWeibullEventTimeModel",
    "PosteriorFamilyResult",
    "PosteriorForecastResult",
    "REQUIRED_EFFECT_FAMILIES",
    "SKEPTICAL_MODEL_FAMILY_PRIOR",
    "ScenarioPatients",
    "ScenarioSampler",
    "TiltProposalError",
    "UniformPriorRange",
    "WeibullEventTimeModel",
    "condition_effect_families_futility_sensitivity_grid",
    "condition_effect_families_on_public_history",
    "condition_futility_sensitivity_grid",
    "condition_on_public_history",
    "default_regal_enrollment_prior",
    "draw_history_importance_sample",
    "exponential_tilt_event_intervals",
    "posterior_forecast_futility_sensitivity_grid",
    "posterior_forecast_prior_sensitivity",
    "posterior_model_average",
    "posterior_prior_sensitivity",
    "public_history_constraint_branches",
)
