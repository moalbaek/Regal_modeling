"""V1-audit and isolated v2 trial-decision primitives.

``obrien_fleming_two_look`` preserves the classical discrete-look ``c / sqrt(t)``
boundary used by the v1 reproducibility audit.  V2 uses the separate
``lan_demets_obrien_fleming_two_look`` implementation, which spends one-sided
alpha with the Lan-DeMets O'Brien-Fleming spending function.

The v2 primary test is represented by ``stratified_logrank``.  It is the score
test of a treatment-only Cox model stratified over the supplied factor
combinations.  Its score statistic is suitable for the efficacy decision.  The
associated one-step hazard-ratio estimate is retained only as a diagnostic and
as an explicit sensitivity input for the unpublished interim futility rule.

None of the v2 objects in this module are imported by the v1 explorer.
"""

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from math import erfc, exp, isfinite, pi, sqrt
from numbers import Integral
from statistics import NormalDist
from typing import Optional, Protocol

import numpy as np


def _bivariate_normal_cdf(x, y, rho, quadrature_order=96):
    """P(X <= x, Y <= y) for standard normals with correlation ``rho``.

    Conditional-normal integration with Gauss-Legendre quadrature avoids adding a
    SciPy dependency. The lower integration limit of -9 is negligible at the
    precision needed for clinical-trial boundaries.
    """

    if not -1.0 < rho < 1.0:
        raise ValueError("rho must be strictly between -1 and 1")
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    lower, upper = -9.0, float(x)
    points = 0.5 * (nodes + 1.0) * (upper - lower) + lower
    normal = NormalDist()
    denom = sqrt(1.0 - rho * rho)
    values = np.fromiter(
        (
            exp(-point * point / 2.0)
            / sqrt(2.0 * pi)
            * normal.cdf((y - rho * point) / denom)
            for point in points
        ),
        dtype=float,
        count=len(points),
    )
    return float(0.5 * (upper - lower) * np.dot(weights, values))


@lru_cache(maxsize=None)
def _solve_obrien_fleming_two_look(alpha, interim_information):
    """Solve and cache the expensive scalar boundary calculation."""

    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5")
    if not 0.0 < interim_information < 1.0:
        raise ValueError("interim_information must be between 0 and 1")

    rho = sqrt(interim_information)

    def crossing_probability(constant):
        interim = constant / sqrt(interim_information)
        no_cross = _bivariate_normal_cdf(interim, constant, rho)
        return 1.0 - no_cross

    # At c=0 the chance of crossing either one-sided boundary is at least 0.5,
    # so it brackets every supported alpha from below. Expand the upper end
    # until its crossing probability is below alpha instead of assuming the
    # conventional [1, 4] range, which fails for large (and very small) alpha.
    lower, upper = 0.0, 1.0
    while crossing_probability(upper) > alpha:
        upper *= 2.0
    for _ in range(60):
        midpoint = 0.5 * (lower + upper)
        if crossing_probability(midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    final = 0.5 * (lower + upper)
    return final / sqrt(interim_information), final


def obrien_fleming_two_look(alpha=0.025, interim_information=0.75):
    """Return classical one-sided two-look O'Brien-Fleming efficacy boundaries.

    Boundaries have the form ``c / sqrt(t)`` at information fraction ``t`` and
    ``c`` at the final look. ``c`` is calibrated so the probability of crossing
    either correlated normal boundary under the null equals ``alpha``. This
    classical construction differs slightly from Lan-DeMets O'Brien-Fleming
    alpha spending; see the module docstring.

    The expensive solve is cached, while each call returns a fresh dict so a
    caller cannot mutate cached state.
    """

    alpha = float(alpha)
    interim_information = float(interim_information)
    interim, final = _solve_obrien_fleming_two_look(alpha, interim_information)
    return {
        "interim_z": interim,
        "final_z": final,
        "alpha": alpha,
        "interim_information": interim_information,
    }


def lan_demets_obrien_fleming_spending(information_fraction, alpha=0.025):
    """Return cumulative one-sided alpha spent by information fraction ``t``.

    REGAL describes a one-sided 0.025 Lan-DeMets O'Brien-Fleming design.  The
    convention matching the published design values uses

    ``alpha(t) = 2 * {1 - Phi(z_(1-alpha/2) / sqrt(t))}``.

    ``erfc`` evaluates the normal upper tail without the cancellation incurred
    by subtracting a CDF close to one.
    """

    if isinstance(alpha, bool) or isinstance(information_fraction, bool):
        raise ValueError("alpha and information_fraction must be numeric")
    try:
        alpha = float(alpha)
        information_fraction = float(information_fraction)
    except (TypeError, ValueError) as error:
        raise ValueError("alpha and information_fraction must be numeric") from error
    if not isfinite(alpha) or not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be finite and between 0 and 0.5")
    if (
        not isfinite(information_fraction)
        or not 0.0 < information_fraction <= 1.0
    ):
        raise ValueError("information_fraction must be finite and in (0, 1]")
    reference = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    return erfc(reference / sqrt(2.0 * information_fraction))


@lru_cache(maxsize=None)
def _solve_lan_demets_obrien_fleming_two_look(
    alpha, interim_information, final_information
):
    """Solve sequential z boundaries at the realized information fractions."""

    interim_spend = lan_demets_obrien_fleming_spending(
        interim_information, alpha
    )
    if interim_spend <= 0.0:
        raise ValueError(
            "interim alpha spend is below floating-point resolution; "
            "use a later information fraction"
        )
    normal = NormalDist()
    interim_z = -normal.inv_cdf(interim_spend)
    final_spending_information = min(final_information, 1.0)
    final_spend = lan_demets_obrien_fleming_spending(
        final_spending_information, alpha
    )
    rho = sqrt(interim_information / final_information)

    def crossing_probability(final_z):
        no_cross = _bivariate_normal_cdf(interim_z, final_z, rho)
        return 1.0 - no_cross

    lower, upper = 0.0, 1.0
    while crossing_probability(upper) > final_spend:
        upper *= 2.0
    for _ in range(70):
        midpoint = 0.5 * (lower + upper)
        if crossing_probability(midpoint) > final_spend:
            lower = midpoint
        else:
            upper = midpoint
    final_z = 0.5 * (lower + upper)
    return (
        interim_z,
        final_z,
        interim_spend,
        final_spend,
        final_spending_information,
        rho,
    )


def lan_demets_obrien_fleming_two_look(
    alpha=0.025, interim_information=0.75, final_information=1.0
):
    """Return v2 Lan-DeMets O'Brien-Fleming boundaries for two looks.

    At REGAL's 60/80 information fraction this returns approximately 2.340 at
    interim and 2.012 at final.  The first boundary spends ``alpha(t_1)``; the
    second is solved so the correlated probability of crossing either boundary
    equals the cumulative spend at the final look. Information fractions are
    relative to the planned final information. ``final_information`` may exceed
    one after an event-count overshoot; spending is then capped at one while the
    canonical correlation continues to use the realized information ratio.
    """

    if isinstance(alpha, bool) or isinstance(interim_information, bool):
        raise ValueError("alpha and interim_information must be numeric")
    if isinstance(final_information, bool):
        raise ValueError("final_information must be numeric")
    try:
        alpha = float(alpha)
        interim_information = float(interim_information)
        final_information = float(final_information)
    except (TypeError, ValueError) as error:
        raise ValueError("information fractions and alpha must be numeric") from error
    # The spending function performs the finite/range checks.  Call it before
    # the cached solve so invalid NaNs never become cache keys.
    lan_demets_obrien_fleming_spending(interim_information, alpha)
    if interim_information >= 1.0:
        raise ValueError("interim_information must be strictly below 1")
    if not isfinite(final_information) or final_information <= interim_information:
        raise ValueError(
            "final_information must be finite and exceed interim_information"
        )
    (
        interim,
        final,
        interim_spend,
        final_spend,
        final_spending_information,
        rho,
    ) = _solve_lan_demets_obrien_fleming_two_look(
        alpha, interim_information, final_information
    )
    return {
        "interim_z": interim,
        "final_z": final,
        "alpha": alpha,
        "interim_information": interim_information,
        "final_information": final_information,
        "final_spending_information": final_spending_information,
        "canonical_correlation": rho,
        "interim_alpha_spent": interim_spend,
        "final_alpha_spent": final_spend,
        "spending_family": "Lan-DeMets O'Brien-Fleming",
    }


@dataclass(frozen=True)
class StratifiedLogRankResult:
    """Treatment score at beta=0 and its explicitly diagnostic HR estimate.

    Treatment must be coded one for GPS and zero for BAT.  Positive ``z`` favors
    GPS.  ``one_step_hazard_ratio`` is ``exp(score / variance)`` and is not a
    fully iterated Cox estimate.
    """

    score: float
    variance: float
    z: float
    one_step_hazard_ratio: float
    events: int
    strata: int
    informative_strata: int

    @property
    def estimable(self):
        return self.variance > 0.0


def binary_indicator(values, name):
    """Validate a one-dimensional zero/one vector and return booleans."""

    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only zero/one values") from error
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.any(~np.isfinite(array)) or np.any((array != 0.0) & (array != 1.0)):
        raise ValueError(f"{name} must contain only zero/one values")
    return array.astype(bool)


def _stratum_groups(strata, size):
    values = np.asarray(strata, dtype=object)
    if values.ndim == 1:
        if len(values) != size:
            raise ValueError("strata must have one row per subject")
        keys = list(values)
    elif values.ndim == 2:
        if values.shape[0] != size or values.shape[1] < 1:
            raise ValueError("strata must have one non-empty row per subject")
        keys = [tuple(row) for row in values]
    else:
        raise ValueError("strata must be a one- or two-dimensional array")

    groups = {}
    for index, raw_key in enumerate(keys):
        parts = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        normalized = []
        for part in parts:
            if isinstance(part, np.generic):
                part = part.item()
            if part is None or (
                isinstance(part, float) and not isfinite(part)
            ):
                raise ValueError("strata must not contain missing values")
            try:
                hash(part)
            except TypeError as error:
                raise ValueError("strata values must be hashable") from error
            normalized.append(part)
        key = tuple(normalized) if isinstance(raw_key, tuple) else normalized[0]
        groups.setdefault(key, []).append(index)
    return tuple(np.asarray(indices, dtype=int) for indices in groups.values())


def stratified_logrank(time, event, treatment, strata):
    """Calculate the stratified log-rank/Cox score test at beta zero.

    ``strata`` may be a one-dimensional vector of already-combined stratum
    labels or a two-dimensional matrix whose columns are separate protocol
    factors.  Risk sets and tied-event hypergeometric variances are calculated
    independently inside every observed factor combination and then summed.
    """

    try:
        times = np.asarray(time, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("time must contain numeric values") from error
    if times.ndim != 1 or len(times) < 1:
        raise ValueError("time must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("time must contain finite, non-negative values")
    events = binary_indicator(event, "event")
    treatments = binary_indicator(treatment, "treatment")
    if len(events) != len(times) or len(treatments) != len(times):
        raise ValueError("time, event, and treatment must have the same length")
    groups = _stratum_groups(strata, len(times))

    score = 0.0
    variance = 0.0
    informative_strata = 0
    for indices in groups:
        stratum_time = times[indices]
        stratum_event = events[indices]
        stratum_treatment = treatments[indices]
        stratum_variance = 0.0
        for event_time in np.unique(stratum_time[stratum_event]):
            at_risk = stratum_time >= event_time
            at_event = stratum_event & (stratum_time == event_time)
            risk_count = int(np.count_nonzero(at_risk))
            event_count = int(np.count_nonzero(at_event))
            treated_risk = int(np.count_nonzero(stratum_treatment & at_risk))
            treated_events = int(np.count_nonzero(stratum_treatment & at_event))
            expected = event_count * treated_risk / risk_count
            score += treated_events - expected
            if risk_count > 1:
                control_risk = risk_count - treated_risk
                contribution = (
                    event_count
                    * (risk_count - event_count)
                    * treated_risk
                    * control_risk
                    / (risk_count * risk_count * (risk_count - 1.0))
                )
                variance += contribution
                stratum_variance += contribution
        informative_strata += int(stratum_variance > 0.0)

    if variance > 0.0:
        z_value = -score / sqrt(variance)
        try:
            hazard_ratio = exp(score / variance)
        except OverflowError:
            hazard_ratio = float("inf")
    else:
        z_value = float("nan")
        hazard_ratio = float("nan")
    return StratifiedLogRankResult(
        score=float(score),
        variance=float(variance),
        z=float(z_value),
        one_step_hazard_ratio=float(hazard_ratio),
        events=int(np.count_nonzero(events)),
        strata=len(groups),
        informative_strata=informative_strata,
    )


def unstratified_logrank_diagnostic(time, event, treatment):
    """Return the v1-style unstratified score as a diagnostic only."""

    size = np.asarray(time).size
    return stratified_logrank(
        time, event, treatment, np.zeros(size, dtype=np.int8)
    )


@dataclass(frozen=True)
class HazardRatioFutilityRule:
    """Assumed one-step-HR futility rule for sensitivity analysis.

    The rule stops when the diagnostic interim hazard ratio is greater than or
    equal to ``threshold``.  REGAL's actual futility rule is not public, so no
    instance is embedded in the default design.
    """

    threshold: float

    def __post_init__(self):
        if isinstance(self.threshold, bool):
            raise ValueError("futility threshold must be numeric")
        try:
            threshold = float(self.threshold)
        except (TypeError, ValueError) as error:
            raise ValueError("futility threshold must be numeric") from error
        if not isfinite(threshold) or threshold <= 0.0:
            raise ValueError("futility threshold must be finite and positive")
        object.__setattr__(self, "threshold", threshold)

    def stops(self, analysis):
        estimate = analysis.one_step_hazard_ratio
        return isfinite(estimate) and estimate >= self.threshold


class FutilityRule(Protocol):
    """Extensible patient-level futility decision interface."""

    def stops(self, analysis: StratifiedLogRankResult) -> bool:
        ...


class InterimDecision(str, Enum):
    """Mutually exclusive result at the interim look."""

    NOT_REACHED = "not_reached"
    EFFICACY_STOP = "efficacy_stop"
    FUTILITY_STOP = "futility_stop"
    CONTINUE = "continue"


class FinalDecision(str, Enum):
    """Result at the final look after interim continuation."""

    NOT_APPLICABLE = "not_applicable"
    NOT_REACHED = "not_reached"
    REJECT = "reject"
    DO_NOT_REJECT = "do_not_reject"


@dataclass(frozen=True)
class TrialDecisionDesign:
    """Two-look REGAL decision design, with no assumed futility default."""

    interim_events: int = 60
    final_events: int = 80
    alpha: float = 0.025
    futility_rule: Optional[FutilityRule] = None

    def __post_init__(self):
        for name in ("interim_events", "final_events"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.interim_events < 1:
            raise ValueError("interim_events must be positive")
        if self.final_events <= self.interim_events:
            raise ValueError("final_events must exceed interim_events")
        if isinstance(self.alpha, bool):
            raise ValueError("alpha must be numeric")
        try:
            alpha = float(self.alpha)
        except (TypeError, ValueError) as error:
            raise ValueError("alpha must be numeric") from error
        if not isfinite(alpha) or not 0.0 < alpha < 0.5:
            raise ValueError("alpha must be finite and between 0 and 0.5")
        object.__setattr__(self, "alpha", alpha)
        if self.futility_rule is not None and not callable(
            getattr(self.futility_rule, "stops", None)
        ):
            raise ValueError("futility_rule must provide a stops(analysis) method")

    @property
    def interim_information(self):
        return self.interim_events / self.final_events

    @property
    def efficacy_boundaries(self):
        return lan_demets_obrien_fleming_two_look(
            self.alpha, self.interim_information
        )

    def efficacy_boundaries_for_event_counts(
        self, interim_observed_events, final_observed_events=None
    ):
        """Return boundaries using observed event counts as information.

        Every death tied at a calendar cutoff remains in the analysis. Event
        count is the public design's available information proxy, so realized
        counts are divided by the planned final count. A final-look overshoot
        spends no more than the design alpha but still changes the canonical
        correlation through its additional information.
        """

        counts = (interim_observed_events,)
        if final_observed_events is not None:
            counts += (final_observed_events,)
        for value in counts:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError("observed event counts must be integers")
            if value < 1:
                raise ValueError("observed event counts must be positive")
        interim_observed_events = int(interim_observed_events)
        if final_observed_events is None:
            final_observed_events = self.final_events
        else:
            final_observed_events = int(final_observed_events)
        if interim_observed_events >= self.final_events:
            raise ValueError(
                "interim observed events must be below planned final events"
            )
        if final_observed_events <= interim_observed_events:
            raise ValueError(
                "final observed events must exceed interim observed events"
            )
        return lan_demets_obrien_fleming_two_look(
            self.alpha,
            interim_observed_events / self.final_events,
            final_observed_events / self.final_events,
        )


def classify_interim(analysis, efficacy_boundary, futility_rule=None):
    """Apply efficacy first, then assumed futility, otherwise continue."""

    if not isinstance(analysis, StratifiedLogRankResult):
        raise ValueError("analysis must be a StratifiedLogRankResult")
    try:
        efficacy_boundary = float(efficacy_boundary)
    except (TypeError, ValueError) as error:
        raise ValueError("efficacy_boundary must be numeric") from error
    if not isfinite(efficacy_boundary):
        raise ValueError("efficacy_boundary must be finite")
    if isfinite(analysis.z) and analysis.z >= efficacy_boundary:
        return InterimDecision.EFFICACY_STOP
    if futility_rule is not None:
        if not callable(getattr(futility_rule, "stops", None)):
            raise ValueError("futility_rule must provide a stops(analysis) method")
        if futility_rule.stops(analysis):
            return InterimDecision.FUTILITY_STOP
    return InterimDecision.CONTINUE


__all__ = (
    "FinalDecision",
    "FutilityRule",
    "HazardRatioFutilityRule",
    "InterimDecision",
    "StratifiedLogRankResult",
    "TrialDecisionDesign",
    "binary_indicator",
    "classify_interim",
    "lan_demets_obrien_fleming_spending",
    "lan_demets_obrien_fleming_two_look",
    "obrien_fleming_two_look",
    "stratified_logrank",
    "unstratified_logrank_diagnostic",
)
