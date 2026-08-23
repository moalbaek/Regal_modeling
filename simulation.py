"""Isolated v2 event-driven trial mechanics and design validation.

This module applies the protocol-compatible stratified score test at the 60th
and 80th observed deaths.  It preserves all mutually exclusive interim branches:
efficacy stop, assumed futility stop, and continuation.  The actual REGAL
futility rule is unpublished, so the committed efficacy design has no futility
default and ``simulate_futility_sensitivity_grid`` reports explicit assumed
hazard-ratio thresholds instead.

All deaths tied at an event-calendar cutoff remain in the analysis. The
patient-level path uses the realized event count as the information proxy and
recalculates the Lan-DeMets boundaries; the canonical-normal simulator validates
alpha spending and branch conservation. It is an operating-characteristic
diagnostic, not a patient-level REGAL forecast and not conditioning on the
observed interim continuation. Patient-level data use
``evaluate_event_driven_trial`` and the stratified analysis in
``trial_design.py``; the unstratified score and one-step HR are diagnostics only.
"""

from dataclasses import dataclass, replace
from math import isfinite, sqrt
from numbers import Integral
from statistics import NormalDist
from typing import Optional

import numpy as np

from trial_design import (
    FinalDecision,
    HazardRatioFutilityRule,
    InterimDecision,
    StratifiedLogRankResult,
    TrialDecisionDesign,
    binary_indicator,
    classify_interim,
    lan_demets_obrien_fleming_spending,
    stratified_logrank,
    unstratified_logrank_diagnostic,
)


REGAL_V2_EFFICACY_DESIGN = TrialDecisionDesign()
FUTILITY_HR_SENSITIVITY_GRID = (None, 0.80, 0.90, 1.00, 1.10, 1.20)


@dataclass(frozen=True, eq=False)
class EventDrivenTrialData:
    """Patient-level entry, follow-up, outcome, arm, and protocol strata.

    ``followup_time`` is measured from randomization to an observed death or
    censoring.  Infinite follow-up is permitted only for a censored subject so
    later event-driven cutoffs can administratively censor that subject.
    ``strata`` may contain one combined label per patient or one column per
    protocol factor.
    """

    entry_time: np.ndarray
    followup_time: np.ndarray
    event_observed: np.ndarray
    treatment: np.ndarray
    strata: np.ndarray

    def __post_init__(self):
        try:
            entry = np.asarray(self.entry_time, dtype=float)
            followup = np.asarray(self.followup_time, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("entry_time and followup_time must be numeric") from error
        if entry.ndim != 1 or len(entry) < 1:
            raise ValueError("entry_time must be a non-empty one-dimensional array")
        if followup.ndim != 1 or len(followup) != len(entry):
            raise ValueError(
                "followup_time must be one-dimensional with one value per subject"
            )
        if np.any(~np.isfinite(entry)) or np.any(entry < 0.0):
            raise ValueError("entry_time must contain finite, non-negative values")
        if np.any(np.isnan(followup)) or np.any(followup < 0.0):
            raise ValueError("followup_time must contain non-negative, non-NaN values")

        event = binary_indicator(self.event_observed, "event_observed")
        treatment = binary_indicator(self.treatment, "treatment")
        if len(event) != len(entry) or len(treatment) != len(entry):
            raise ValueError(
                "event_observed and treatment must have one value per subject"
            )
        if np.any(event & ~np.isfinite(followup)):
            raise ValueError("observed events must have finite follow-up times")
        event_calendar = entry[event] + followup[event]
        if np.any(~np.isfinite(event_calendar)):
            raise ValueError("observed event calendar times must be finite")

        strata = np.asarray(self.strata, dtype=object)
        if strata.ndim not in (1, 2) or strata.shape[0] != len(entry):
            raise ValueError("strata must have one row per subject")
        if strata.ndim == 2 and strata.shape[1] < 1:
            raise ValueError("strata must include at least one factor")

        entry = np.array(entry, copy=True)
        followup = np.array(followup, copy=True)
        event = np.array(event, copy=True)
        treatment = np.array(treatment, copy=True)
        strata = np.array(strata, copy=True)
        for array in (entry, followup, event, treatment, strata):
            array.setflags(write=False)
        object.__setattr__(self, "entry_time", entry)
        object.__setattr__(self, "followup_time", followup)
        object.__setattr__(self, "event_observed", event)
        object.__setattr__(self, "treatment", treatment)
        object.__setattr__(self, "strata", strata)

    @property
    def size(self):
        return len(self.entry_time)


@dataclass(frozen=True)
class AnalysisSnapshot:
    """Primary and diagnostic analyses at one event-driven calendar cutoff."""

    planned_events: int
    observed_events: int
    information_fraction: float
    cutoff_time: float
    primary: StratifiedLogRankResult
    unstratified_diagnostic: StratifiedLogRankResult
    efficacy_boundary: Optional[float] = None


@dataclass(frozen=True)
class TrialDecisionResult:
    """Complete result of applying the two-look decision process to one trial."""

    design: TrialDecisionDesign
    interim_decision: InterimDecision
    final_decision: FinalDecision
    interim: Optional[AnalysisSnapshot] = None
    final: Optional[AnalysisSnapshot] = None

    @property
    def overall_success(self):
        return (
            self.interim_decision is InterimDecision.EFFICACY_STOP
            or self.final_decision is FinalDecision.REJECT
        )


def _analysis_at_event_count(data, planned_events, planned_final_events):
    event_calendar = (
        data.entry_time[data.event_observed]
        + data.followup_time[data.event_observed]
    )
    if len(event_calendar) < planned_events:
        return None
    cutoff = float(np.partition(event_calendar, planned_events - 1)[planned_events - 1])
    enrolled = data.entry_time <= cutoff
    available_followup = cutoff - data.entry_time[enrolled]
    analysis_time = np.minimum(data.followup_time[enrolled], available_followup)
    event = data.event_observed[enrolled] & (
        data.entry_time[enrolled] + data.followup_time[enrolled] <= cutoff
    )
    treatment = data.treatment[enrolled]
    strata = data.strata[enrolled]
    primary = stratified_logrank(analysis_time, event, treatment, strata)
    diagnostic = unstratified_logrank_diagnostic(
        analysis_time, event, treatment
    )
    observed_events = int(np.count_nonzero(event))
    return AnalysisSnapshot(
        planned_events=planned_events,
        observed_events=observed_events,
        information_fraction=observed_events / planned_final_events,
        cutoff_time=cutoff,
        primary=primary,
        unstratified_diagnostic=diagnostic,
    )


def evaluate_event_driven_trial(data, design=REGAL_V2_EFFICACY_DESIGN):
    """Apply realized-information decisions to one patient-level trial.

    Calendar ties are not broken arbitrarily: every death at a trigger time is
    included. Event counts proxy information relative to the planned final
    count. If a tie reaches the final target at the first operational analysis,
    the duplicate interim look is skipped and the cumulative final alpha is
    applied once.
    """

    if not isinstance(data, EventDrivenTrialData):
        raise ValueError("data must be EventDrivenTrialData")
    if not isinstance(design, TrialDecisionDesign):
        raise ValueError("design must be TrialDecisionDesign")
    interim = _analysis_at_event_count(
        data, design.interim_events, design.final_events
    )
    if interim is None:
        return TrialDecisionResult(
            design=design,
            interim_decision=InterimDecision.NOT_REACHED,
            final_decision=FinalDecision.NOT_APPLICABLE,
        )

    if interim.observed_events >= design.final_events:
        final = _analysis_at_event_count(
            data, design.final_events, design.final_events
        )
        final_alpha_spent = lan_demets_obrien_fleming_spending(
            1.0, design.alpha
        )
        final_boundary = -NormalDist().inv_cdf(final_alpha_spent)
        final = replace(final, efficacy_boundary=final_boundary)
        final_decision = (
            FinalDecision.REJECT
            if isfinite(final.primary.z) and final.primary.z >= final_boundary
            else FinalDecision.DO_NOT_REJECT
        )
        return TrialDecisionResult(
            design=design,
            interim_decision=InterimDecision.CONTINUE,
            final_decision=final_decision,
            interim=interim,
            final=final,
        )

    interim_boundaries = design.efficacy_boundaries_for_event_counts(
        interim.observed_events
    )
    interim = replace(
        interim, efficacy_boundary=interim_boundaries["interim_z"]
    )
    interim_decision = classify_interim(
        interim.primary, interim.efficacy_boundary, design.futility_rule
    )
    if interim_decision is not InterimDecision.CONTINUE:
        return TrialDecisionResult(
            design=design,
            interim_decision=interim_decision,
            final_decision=FinalDecision.NOT_APPLICABLE,
            interim=interim,
        )

    final = _analysis_at_event_count(
        data, design.final_events, design.final_events
    )
    if final is None:
        return TrialDecisionResult(
            design=design,
            interim_decision=interim_decision,
            final_decision=FinalDecision.NOT_REACHED,
            interim=interim,
        )
    realized_boundaries = design.efficacy_boundaries_for_event_counts(
        interim.observed_events, final.observed_events
    )
    final = replace(final, efficacy_boundary=realized_boundaries["final_z"])
    final_decision = (
        FinalDecision.REJECT
        if isfinite(final.primary.z)
        and final.primary.z >= final.efficacy_boundary
        else FinalDecision.DO_NOT_REJECT
    )
    return TrialDecisionResult(
        design=design,
        interim_decision=interim_decision,
        final_decision=final_decision,
        interim=interim,
        final=final,
    )


@dataclass(frozen=True)
class BranchOperatingCharacteristics:
    """Conserved branch counts from one operating-characteristic simulation."""

    simulation_kind: str
    n_simulations: int
    final_z_mean: Optional[float]
    futility_hr_threshold: Optional[float]
    interim_efficacy_stops: int
    futility_stops: int
    continuations: int
    final_rejections: int
    final_non_rejections: int

    def __post_init__(self):
        if not isinstance(self.simulation_kind, str) or not self.simulation_kind.strip():
            raise ValueError("simulation_kind must be a non-empty string")
        object.__setattr__(self, "simulation_kind", self.simulation_kind.strip())
        count_fields = (
            "n_simulations",
            "interim_efficacy_stops",
            "futility_stops",
            "continuations",
            "final_rejections",
            "final_non_rejections",
        )
        for name in count_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            value = int(value)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.n_simulations < 1:
            raise ValueError("n_simulations must be positive")
        if self.final_z_mean is not None:
            if isinstance(self.final_z_mean, bool):
                raise ValueError("final_z_mean must be numeric or None")
            try:
                final_z_mean = float(self.final_z_mean)
            except (TypeError, ValueError) as error:
                raise ValueError("final_z_mean must be numeric or None") from error
            if not isfinite(final_z_mean):
                raise ValueError("final_z_mean must be finite or None")
            object.__setattr__(self, "final_z_mean", final_z_mean)
        if self.futility_hr_threshold is not None:
            if isinstance(self.futility_hr_threshold, bool):
                raise ValueError(
                    "futility_hr_threshold must be numeric or None"
                )
            try:
                threshold = float(self.futility_hr_threshold)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "futility_hr_threshold must be numeric or None"
                ) from error
            if not isfinite(threshold) or threshold <= 0.0:
                raise ValueError(
                    "futility_hr_threshold must be finite and positive or None"
                )
            object.__setattr__(self, "futility_hr_threshold", threshold)
        if (
            self.interim_efficacy_stops
            + self.futility_stops
            + self.continuations
            != self.n_simulations
        ):
            raise ValueError("interim branch counts do not conserve simulations")
        if self.final_rejections + self.final_non_rejections != self.continuations:
            raise ValueError("final branch counts do not conserve continuations")

    @property
    def p_interim_efficacy(self):
        return self.interim_efficacy_stops / self.n_simulations

    @property
    def p_futility(self):
        return self.futility_stops / self.n_simulations

    @property
    def p_continue(self):
        return self.continuations / self.n_simulations

    @property
    def p_final_rejection_given_continue(self):
        if not self.continuations:
            return 0.0
        return self.final_rejections / self.continuations

    @property
    def p_overall_success(self):
        return (
            self.interim_efficacy_stops + self.final_rejections
        ) / self.n_simulations

    def as_dict(self):
        return {
            "simulation_kind": self.simulation_kind,
            "n_simulations": self.n_simulations,
            "final_z_mean": self.final_z_mean,
            "futility_hr_threshold": self.futility_hr_threshold,
            "interim_efficacy_stops": self.interim_efficacy_stops,
            "futility_stops": self.futility_stops,
            "continuations": self.continuations,
            "final_rejections": self.final_rejections,
            "final_non_rejections": self.final_non_rejections,
            "p_interim_efficacy": self.p_interim_efficacy,
            "p_futility": self.p_futility,
            "p_continue": self.p_continue,
            "p_final_rejection_given_continue": (
                self.p_final_rejection_given_continue
            ),
            "p_overall_success": self.p_overall_success,
        }


def _validate_simulation_inputs(nsim, final_z_mean):
    if isinstance(nsim, bool) or not isinstance(nsim, Integral) or nsim < 1:
        raise ValueError("nsim must be a positive integer")
    try:
        final_z_mean = float(final_z_mean)
    except (TypeError, ValueError) as error:
        raise ValueError("final_z_mean must be numeric") from error
    if not isfinite(final_z_mean):
        raise ValueError("final_z_mean must be finite")
    return int(nsim), final_z_mean


def _canonical_z_draws(design, nsim, seed, final_z_mean):
    nsim, final_z_mean = _validate_simulation_inputs(nsim, final_z_mean)
    if not isinstance(design, TrialDecisionDesign):
        raise ValueError("design must be TrialDecisionDesign")
    rng = np.random.default_rng(seed)
    first_noise = rng.standard_normal(nsim)
    second_noise = rng.standard_normal(nsim)
    information = design.interim_information
    rho = sqrt(information)
    interim_z = final_z_mean * rho + first_noise
    final_z = (
        final_z_mean
        + rho * first_noise
        + sqrt(1.0 - information) * second_noise
    )
    return interim_z, final_z, final_z_mean


def _diagnostic_hr_from_z(z_value, event_count):
    """Balanced 1:1 one-step HR mapping used only by the canonical diagnostic."""

    log_hr = -2.0 * np.asarray(z_value, dtype=float) / sqrt(event_count)
    return np.exp(np.clip(log_hr, -745.0, 709.0))


def _summarize_canonical(interim_z, final_z, final_z_mean, design):
    boundaries = design.efficacy_boundaries
    interim_efficacy = interim_z >= boundaries["interim_z"]
    eligible_for_futility = ~interim_efficacy
    if design.futility_rule is None:
        futility = np.zeros(len(interim_z), dtype=bool)
        threshold = None
    else:
        if not isinstance(design.futility_rule, HazardRatioFutilityRule):
            raise ValueError(
                "canonical simulation supports HazardRatioFutilityRule or None; "
                "apply custom rules to patient-level analyses"
            )
        interim_hr = _diagnostic_hr_from_z(
            interim_z, design.interim_events
        )
        threshold = design.futility_rule.threshold
        futility = eligible_for_futility & (interim_hr >= threshold)
    continuation = eligible_for_futility & ~futility
    final_rejection = continuation & (final_z >= boundaries["final_z"])
    final_non_rejection = continuation & ~final_rejection
    return BranchOperatingCharacteristics(
        simulation_kind="canonical_correlated_normal",
        n_simulations=len(interim_z),
        final_z_mean=final_z_mean,
        futility_hr_threshold=threshold,
        interim_efficacy_stops=int(np.count_nonzero(interim_efficacy)),
        futility_stops=int(np.count_nonzero(futility)),
        continuations=int(np.count_nonzero(continuation)),
        final_rejections=int(np.count_nonzero(final_rejection)),
        final_non_rejections=int(np.count_nonzero(final_non_rejection)),
    )


def simulate_canonical_operating_characteristics(
    design=REGAL_V2_EFFICACY_DESIGN,
    nsim=200000,
    seed=20260823,
    final_z_mean=0.0,
):
    """Simulate correlated z statistics for design and branch validation.

    ``final_z_mean=0`` is the null.  Under a canonical proportional-hazards
    alternative, the interim mean is ``final_z_mean * sqrt(60/80)``.  This
    diagnostic does not replace patient-level stratified simulation.
    """

    interim_z, final_z, normalized_mean = _canonical_z_draws(
        design, nsim, seed, final_z_mean
    )
    return _summarize_canonical(
        interim_z, final_z, normalized_mean, design
    )


def simulate_futility_sensitivity_grid(
    thresholds=FUTILITY_HR_SENSITIVITY_GRID,
    design=REGAL_V2_EFFICACY_DESIGN,
    nsim=200000,
    seed=20260823,
    final_z_mean=0.0,
):
    """Evaluate assumed futility thresholds on identical paired z draws.

    ``None`` disables futility.  Every numeric threshold means stop when the
    interim one-step HR is at least that value.  The shared draws isolate the
    rule's effect from Monte-Carlo noise across rows.
    """

    if not isinstance(design, TrialDecisionDesign):
        raise ValueError("design must be TrialDecisionDesign")
    try:
        supplied = tuple(thresholds)
    except TypeError as error:
        raise ValueError("thresholds must be an iterable") from error
    if not supplied:
        raise ValueError("thresholds must not be empty")
    normalized = []
    for threshold in supplied:
        if threshold is None:
            value = None
        else:
            value = HazardRatioFutilityRule(threshold).threshold
        if value in normalized:
            raise ValueError("thresholds must not contain duplicates")
        normalized.append(value)

    draw_design = replace(design, futility_rule=None)
    interim_z, final_z, normalized_mean = _canonical_z_draws(
        draw_design, nsim, seed, final_z_mean
    )
    rows = []
    for threshold in normalized:
        rule = None if threshold is None else HazardRatioFutilityRule(threshold)
        row_design = replace(design, futility_rule=rule)
        rows.append(
            _summarize_canonical(
                interim_z, final_z, normalized_mean, row_design
            )
        )
    return tuple(rows)


def _synthetic_protocol_factors(patient_count):
    """Four public factor columns arranged in four balanced combinations."""

    patterns = np.asarray(
        (
            (0, 0, 0, 0),
            (0, 1, 1, 0),
            (1, 0, 0, 1),
            (1, 1, 1, 1),
        ),
        dtype=np.int8,
    )
    return patterns[np.arange(patient_count) % len(patterns)]


def _stratum_balanced_treatment(strata):
    """Deterministic near-1:1 allocation within every synthetic stratum."""

    treatment = np.zeros(len(strata), dtype=bool)
    odd_round_up = False
    for pattern in np.unique(strata, axis=0):
        indices = np.flatnonzero(np.all(strata == pattern, axis=1))
        treatment_count = len(indices) // 2
        if len(indices) % 2:
            treatment_count += int(odd_round_up)
            odd_round_up = not odd_round_up
        treatment[indices[:treatment_count]] = True
    return treatment


def simulate_patient_level_exponential_null(
    design=REGAL_V2_EFFICACY_DESIGN,
    nsim=5000,
    seed=20260824,
    patient_count=126,
    enrollment_window=24.0,
    median_survival=12.0,
):
    """Validate the complete event-driven stratified path under a simple null.

    GPS and BAT receive identical exponential survival.  Enrollment is uniform,
    treatment is balanced inside four synthetic combinations of the four public
    protocol factors, and every patient is followed until death.  The model is
    deliberately not a REGAL survival scenario; it checks that calendar event
    triggers plus the patient-level primary analysis preserve null behavior.
    """

    nsim, _ = _validate_simulation_inputs(nsim, 0.0)
    if not isinstance(design, TrialDecisionDesign):
        raise ValueError("design must be TrialDecisionDesign")
    if (
        isinstance(patient_count, bool)
        or not isinstance(patient_count, Integral)
        or patient_count < design.final_events
    ):
        raise ValueError("patient_count must be an integer at least final_events")
    try:
        enrollment_window = float(enrollment_window)
        median_survival = float(median_survival)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "enrollment_window and median_survival must be numeric"
        ) from error
    if not isfinite(enrollment_window) or enrollment_window < 0.0:
        raise ValueError("enrollment_window must be finite and non-negative")
    if not isfinite(median_survival) or median_survival <= 0.0:
        raise ValueError("median_survival must be finite and positive")

    patient_count = int(patient_count)
    strata = _synthetic_protocol_factors(patient_count)
    treatment = _stratum_balanced_treatment(strata)
    rng = np.random.default_rng(seed)
    interim_efficacy_stops = 0
    futility_stops = 0
    continuations = 0
    final_rejections = 0
    final_non_rejections = 0
    scale = median_survival / np.log(2.0)

    for _ in range(nsim):
        data = EventDrivenTrialData(
            entry_time=rng.uniform(0.0, enrollment_window, patient_count),
            followup_time=rng.exponential(scale, patient_count),
            event_observed=np.ones(patient_count, dtype=bool),
            treatment=treatment,
            strata=strata,
        )
        result = evaluate_event_driven_trial(data, design)
        if result.interim_decision is InterimDecision.EFFICACY_STOP:
            interim_efficacy_stops += 1
        elif result.interim_decision is InterimDecision.FUTILITY_STOP:
            futility_stops += 1
        else:
            if result.interim_decision is not InterimDecision.CONTINUE:
                raise RuntimeError("patient-level null trial failed to reach interim")
            continuations += 1
            if result.final_decision is FinalDecision.REJECT:
                final_rejections += 1
            elif result.final_decision is FinalDecision.DO_NOT_REJECT:
                final_non_rejections += 1
            else:
                raise RuntimeError("patient-level null trial failed to reach final")

    threshold = (
        design.futility_rule.threshold
        if isinstance(design.futility_rule, HazardRatioFutilityRule)
        else None
    )
    return BranchOperatingCharacteristics(
        simulation_kind="patient_level_exponential_null",
        n_simulations=nsim,
        final_z_mean=None,
        futility_hr_threshold=threshold,
        interim_efficacy_stops=interim_efficacy_stops,
        futility_stops=futility_stops,
        continuations=continuations,
        final_rejections=final_rejections,
        final_non_rejections=final_non_rejections,
    )


__all__ = (
    "AnalysisSnapshot",
    "BranchOperatingCharacteristics",
    "EventDrivenTrialData",
    "FUTILITY_HR_SENSITIVITY_GRID",
    "REGAL_V2_EFFICACY_DESIGN",
    "TrialDecisionResult",
    "evaluate_event_driven_trial",
    "simulate_canonical_operating_characteristics",
    "simulate_futility_sensitivity_grid",
    "simulate_patient_level_exponential_null",
)
