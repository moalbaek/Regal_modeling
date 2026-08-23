"""Survival primitives for the REGAL v2 model.

The legacy explorer conditions each treatment-specific survival distribution on
belonging to its longest-surviving fraction.  That mechanism is intentionally not
reproduced here: trial eligibility is decided before randomization and therefore
cannot depend on a patient's future realized survival under the assigned therapy.

V2 instead represents case mix with a baseline frailty drawn before eligibility and
randomization.  Eligibility may depend on that prognostic quantity, and the same
selected frailty distribution is then randomized across both arms.  Conditional
event times retain support immediately after enrollment, so selection creates no
guaranteed-survival interval and does not mechanically inflate the cure fraction.

The module is not imported by ``regal_explorer.py``.  That separation preserves the
v1 legacy scenario outputs while later v2 work builds on these corrected primitives.
"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite, log
from typing import Protocol, Tuple, Union

import numpy as np


Size = Union[int, Tuple[int, ...]]


def _scalar_or_array(value):
    array = np.asarray(value, dtype=float)
    return float(array) if array.ndim == 0 else array


def _nonnegative_times(months):
    values = np.asarray(months, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("months must contain finite, non-negative values")
    return values


def _positive_frailty(frailty):
    values = np.asarray(frailty, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("frailty must contain finite, positive values")
    return values


class SurvivalScale(str, Enum):
    """Scale on which an uncured survival input was estimated."""

    OVERALL = "overall"
    NET = "net"


class BackgroundMortality(Protocol):
    """Population-mortality interface consumed by a cure mixture."""

    def survival(self, months):
        ...

    def sample_event_times(self, rng, size: Size):
        ...


@dataclass(frozen=True)
class ExponentialBackgroundMortality:
    """Constant population mortality expressed as an annual death probability."""

    annual_death_probability: float

    def __post_init__(self):
        probability = float(self.annual_death_probability)
        if not isfinite(probability) or not 0.0 <= probability < 1.0:
            raise ValueError("annual_death_probability must be in [0, 1)")
        object.__setattr__(self, "annual_death_probability", probability)

    @property
    def monthly_hazard(self):
        if self.annual_death_probability == 0.0:
            return 0.0
        return -log(1.0 - self.annual_death_probability) / 12.0

    def survival(self, months):
        times = _nonnegative_times(months)
        return _scalar_or_array(np.exp(-self.monthly_hazard * times))

    def sample_event_times(self, rng, size: Size):
        if self.monthly_hazard == 0.0:
            return np.full(size, np.inf, dtype=float)
        uniforms = np.maximum(rng.random(size=size), np.finfo(float).tiny)
        return -np.log(uniforms) / self.monthly_hazard


@dataclass(frozen=True)
class WeibullSurvival:
    """Uncured Weibull survival with a median defined at frailty one.

    ``frailty`` is a proportional-hazards multiplier: values above one indicate
    worse baseline prognosis and values below one indicate better prognosis.
    """

    median_months: float
    shape: float

    def __post_init__(self):
        median = float(self.median_months)
        shape = float(self.shape)
        if not isfinite(median) or median <= 0.0:
            raise ValueError("median_months must be finite and positive")
        if not isfinite(shape) or shape <= 0.0:
            raise ValueError("shape must be finite and positive")
        object.__setattr__(self, "median_months", median)
        object.__setattr__(self, "shape", shape)

    @property
    def scale_months(self):
        return self.median_months / log(2.0) ** (1.0 / self.shape)

    def survival(self, months, frailty=1.0):
        times = _nonnegative_times(months)
        risks = _positive_frailty(frailty)
        cumulative_hazard = risks * (times / self.scale_months) ** self.shape
        return _scalar_or_array(np.exp(-cumulative_hazard))

    def sample_event_times(self, rng, frailty):
        risks = _positive_frailty(frailty)
        uniforms = np.maximum(rng.random(size=risks.shape), np.finfo(float).tiny)
        times = self.scale_months * (-np.log(uniforms) / risks) ** (1.0 / self.shape)
        return _scalar_or_array(times)


@dataclass(frozen=True)
class CureMixtureComponent:
    """Scale-aware all-cause survival for one treatment component.

    Existing literature inputs default to ``overall``: their uncured curve already
    contains population mortality, so background mortality is added only to the
    cured fraction.  A ``net`` input excludes population mortality, so the background
    curve multiplies the complete cure mixture.
    """

    name: str
    uncured: WeibullSurvival
    cure_fraction: float
    survival_scale: SurvivalScale = SurvivalScale.OVERALL

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        cure = float(self.cure_fraction)
        if not isfinite(cure) or not 0.0 <= cure <= 1.0:
            raise ValueError("cure_fraction must be in [0, 1]")
        try:
            scale = SurvivalScale(self.survival_scale)
        except ValueError as error:
            raise ValueError("survival_scale must be 'overall' or 'net'") from error
        object.__setattr__(self, "cure_fraction", cure)
        object.__setattr__(self, "survival_scale", scale)

    def survival(self, months, background: BackgroundMortality, frailty=1.0):
        uncured = np.asarray(self.uncured.survival(months, frailty), dtype=float)
        population = np.asarray(background.survival(months), dtype=float)
        cure = self.cure_fraction
        if self.survival_scale is SurvivalScale.OVERALL:
            result = cure * population + (1.0 - cure) * uncured
        else:
            result = population * (cure + (1.0 - cure) * uncured)
        return _scalar_or_array(result)

    def sample_event_times(self, rng, background: BackgroundMortality, frailty):
        risks = _positive_frailty(frailty)
        cured = rng.random(size=risks.shape) < self.cure_fraction
        uncured_times = np.asarray(
            self.uncured.sample_event_times(rng, risks), dtype=float
        )
        population_times = np.asarray(
            background.sample_event_times(rng, risks.shape), dtype=float
        )
        if self.survival_scale is SurvivalScale.OVERALL:
            times = np.where(cured, population_times, uncured_times)
        else:
            times = np.where(cured, population_times, np.minimum(uncured_times, population_times))
        return _scalar_or_array(times)


@dataclass(frozen=True)
class RandomizedCohort:
    """Baseline frailty and assignments for an already eligible cohort."""

    frailty: np.ndarray
    treatment: np.ndarray

    def __post_init__(self):
        frailty = _positive_frailty(self.frailty)
        treatment = np.asarray(self.treatment, dtype=bool)
        if frailty.ndim != 1 or treatment.ndim != 1:
            raise ValueError("frailty and treatment must be one-dimensional")
        if len(frailty) != len(treatment):
            raise ValueError("frailty and treatment must have the same length")
        frailty = np.array(frailty, copy=True)
        treatment = np.array(treatment, copy=True)
        frailty.setflags(write=False)
        treatment.setflags(write=False)
        object.__setattr__(self, "frailty", frailty)
        object.__setattr__(self, "treatment", treatment)


@dataclass(frozen=True)
class FrailtyCaseMix:
    """Pre-outcome eligibility model followed by randomized arm assignment.

    Population frailty is log-normal with mean one.  Eligibility follows

    ``logit P(eligible | z) = intercept - health_gradient * log(z)``.

    Positive ``health_gradient`` values preferentially enroll lower-risk patients.
    The default frailty variance and gradient are both zero, so v2 reproduces the
    component curve without hidden heterogeneity or enrichment until a case-mix
    prior is supplied and justified.
    """

    population_log_sd: float = 0.0
    eligibility_logit_intercept: float = 0.0
    eligibility_health_gradient: float = 0.0
    max_draw_multiplier: int = 1000

    def __post_init__(self):
        log_sd = float(self.population_log_sd)
        intercept = float(self.eligibility_logit_intercept)
        gradient = float(self.eligibility_health_gradient)
        if not isfinite(log_sd) or log_sd < 0.0:
            raise ValueError("population_log_sd must be finite and non-negative")
        if not isfinite(intercept):
            raise ValueError("eligibility_logit_intercept must be finite")
        if not isfinite(gradient) or gradient < 0.0:
            raise ValueError("eligibility_health_gradient must be finite and non-negative")
        if not isinstance(self.max_draw_multiplier, int) or self.max_draw_multiplier < 1:
            raise ValueError("max_draw_multiplier must be a positive integer")
        object.__setattr__(self, "population_log_sd", log_sd)
        object.__setattr__(self, "eligibility_logit_intercept", intercept)
        object.__setattr__(self, "eligibility_health_gradient", gradient)

    def draw_population(self, size, rng):
        if not isinstance(size, int) or size < 0:
            raise ValueError("size must be a non-negative integer")
        sigma = self.population_log_sd
        return rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma, size=size)

    def eligibility_probability(self, frailty):
        risks = _positive_frailty(frailty)
        logits = (
            self.eligibility_logit_intercept
            - self.eligibility_health_gradient * np.log(risks)
        )
        probabilities = np.empty_like(logits, dtype=float)
        positive = logits >= 0.0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_logits = np.exp(logits[~positive])
        probabilities[~positive] = exp_logits / (1.0 + exp_logits)
        return _scalar_or_array(probabilities)

    def sample_enrolled(self, size, rng):
        """Draw exactly ``size`` eligible frailties without drawing any outcomes."""

        if not isinstance(size, int) or size < 1:
            raise ValueError("size must be a positive integer")
        accepted = []
        accepted_count = 0
        draw_count = 0
        max_draws = size * self.max_draw_multiplier
        while accepted_count < size and draw_count < max_draws:
            remaining_capacity = max_draws - draw_count
            batch_size = min(max(256, 2 * (size - accepted_count)), remaining_capacity)
            candidates = self.draw_population(batch_size, rng)
            probabilities = self.eligibility_probability(candidates)
            keep = rng.random(batch_size) < probabilities
            selected = candidates[keep]
            if selected.size:
                accepted.append(selected)
                accepted_count += selected.size
            draw_count += batch_size
        if accepted_count < size:
            raise RuntimeError(
                "eligibility model accepted too few patients; increase the intercept "
                "or max_draw_multiplier"
            )
        return np.concatenate(accepted)[:size]

    @staticmethod
    def randomize(frailty, rng, treatment_fraction=0.5):
        """Randomize an eligible cohort after selection, with exact arm counts."""

        risks = _positive_frailty(frailty)
        if risks.ndim != 1 or len(risks) < 1:
            raise ValueError("frailty must be a non-empty one-dimensional array")
        fraction = float(treatment_fraction)
        if not isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("treatment_fraction must be in [0, 1]")
        treatment_count = int(np.floor(len(risks) * fraction + 0.5))
        treatment = np.zeros(len(risks), dtype=bool)
        treatment[rng.permutation(len(risks))[:treatment_count]] = True
        return RandomizedCohort(frailty=risks, treatment=treatment)

    def sample_randomized(self, size, rng, treatment_fraction=0.5):
        frailty = self.sample_enrolled(size, rng)
        return self.randomize(frailty, rng, treatment_fraction)


def marginal_survival(component, months, background, frailty):
    """Average a component's survival over an enrolled frailty distribution."""

    risks = _positive_frailty(frailty)
    if risks.ndim != 1 or not len(risks):
        raise ValueError("frailty must be a non-empty one-dimensional array")
    times = _nonnegative_times(months)
    values = component.survival(times[..., None], background, risks)
    return _scalar_or_array(np.mean(values, axis=-1))
