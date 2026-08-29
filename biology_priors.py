"""Biology-informed priors for REGAL exploratory responder models.

This module keeps external biological evidence separate from the blinded REGAL
public-history likelihood.  It intentionally does not infer treatment efficacy
from immune response.  Its first use is to replace an arbitrary flat prior on
the probability that a GPS-treated patient mounts a measurable WT1-specific
T-cell response with an explicit beta-binomial update.

Evidence currently encoded
--------------------------
* GPS phase 2 AML: 9 immune responders among 14 evaluable patients.
* REGAL interim immune substudy: 8 responders among 10 randomly selected
  GPS-treated patients reported by SELLAS.

With a Beta(1, 1) reference prior, pooling those observations yields
Beta(18, 8): posterior mean 18 / 26 = 0.6923.  The model should sample from the
full distribution, not hard-code the observed 8/10 = 80% point estimate.

The phase-2 and REGAL observations are exposed separately so sensitivity
analyses can down-weight or omit the historical cohort if exchangeability is
questioned.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral


@dataclass(frozen=True)
class BinomialEvidence:
    """One auditable binomial evidence contribution."""

    label: str
    responders: int
    evaluable: int

    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        for name, value in (("responders", self.responders), ("evaluable", self.evaluable)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if self.evaluable <= 0:
            raise ValueError("evaluable must be positive")
        if not 0 <= self.responders <= self.evaluable:
            raise ValueError("responders must lie in [0, evaluable]")

    @property
    def nonresponders(self) -> int:
        return int(self.evaluable - self.responders)


@dataclass(frozen=True)
class BetaPrior:
    """Beta distribution usable as a sampling prior for a probability."""

    alpha: float
    beta: float
    label: str = ""

    def __post_init__(self):
        alpha = float(self.alpha)
        beta = float(self.beta)
        if not isfinite(alpha) or not isfinite(beta) or alpha <= 0.0 or beta <= 0.0:
            raise ValueError("beta prior parameters must be finite and positive")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self, rng) -> float:
        value = float(rng.beta(self.alpha, self.beta))
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("rng.beta must return a finite draw in [0, 1]")
        return value

    def update(self, *evidence: BinomialEvidence, label: str | None = None) -> "BetaPrior":
        alpha = self.alpha + sum(item.responders for item in evidence)
        beta = self.beta + sum(item.nonresponders for item in evidence)
        return BetaPrior(alpha, beta, self.label if label is None else label)


REFERENCE_RESPONSE_PRIOR = BetaPrior(1.0, 1.0, "reference_uniform")

GPS_PHASE2_IMMUNE_EVIDENCE = BinomialEvidence(
    "GPS phase 2 AML immune evaluable cohort",
    responders=9,
    evaluable=14,
)

REGAL_INTERIM_IMMUNE_EVIDENCE = BinomialEvidence(
    "REGAL interim randomly selected GPS immune cohort",
    responders=8,
    evaluable=10,
)

GPS_PHASE2_RESPONSE_POSTERIOR = REFERENCE_RESPONSE_PRIOR.update(
    GPS_PHASE2_IMMUNE_EVIDENCE,
    label="gps_phase2_only_beta_10_6",
)

REGAL_INTERIM_RESPONSE_POSTERIOR = REFERENCE_RESPONSE_PRIOR.update(
    REGAL_INTERIM_IMMUNE_EVIDENCE,
    label="regal_interim_only_beta_9_3",
)

POOLED_GPS_RESPONSE_POSTERIOR = REFERENCE_RESPONSE_PRIOR.update(
    GPS_PHASE2_IMMUNE_EVIDENCE,
    REGAL_INTERIM_IMMUNE_EVIDENCE,
    label="pooled_gps_beta_18_8",
)


__all__ = [
    "BetaPrior",
    "BinomialEvidence",
    "GPS_PHASE2_IMMUNE_EVIDENCE",
    "GPS_PHASE2_RESPONSE_POSTERIOR",
    "POOLED_GPS_RESPONSE_POSTERIOR",
    "REFERENCE_RESPONSE_PRIOR",
    "REGAL_INTERIM_IMMUNE_EVIDENCE",
    "REGAL_INTERIM_RESPONSE_POSTERIOR",
]
