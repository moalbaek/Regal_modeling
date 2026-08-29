"""Biology-informed priors for REGAL exploratory responder models.

This module keeps immunogenicity evidence as an explicit endpoint-level update
separate from the blinded REGAL event-count likelihood. The evidence includes a
public REGAL interim immunogenicity disclosure, but that distinct endpoint is
not reused as survival-event evidence. Two questions are represented:

1. How often does GPS generate a measurable WT1-specific immune response?
2. Conditional on being an immune responder, how plausible is a durable
   remission / low-hazard tail attributable to that response?

The first question is updated directly with beta-binomial data. The second is
*not* treated as a formal meta-analysis because the available WT1 studies differ
substantially in vaccine, disease setting, endpoint, age, transplant exposure,
and responder definition. Instead, it is encoded as an explicit conservative
mixture prior whose skeptical component retains substantial probability mass.

Immune-response evidence currently encoded
------------------------------------------
* GPS phase 2 AML: 9 immune responders among 14 evaluable patients.
* REGAL interim immune substudy: SELLAS reported an 80% response rate in a
  randomly selected GPS-treated sample but did not publicly disclose its size.
  The default 8/10 translation is therefore an explicit working assumption,
  accompanied by denominator sensitivity.

With a Beta(1, 1) reference prior and the default assumed REGAL denominator of
10, pooling those observations yields Beta(18, 8): posterior mean 18 / 26 =
0.6923. The model samples from the full posterior and exposes alternative
denominator assumptions rather than treating the unreported sample size as fact.

Responder-survival evidence map
-------------------------------
The durable-benefit prior deliberately does not plug small-study responder hazard
ratios directly into REGAL. Its three components are interpretations of the
literature, not fitted likelihoods:

* skeptical: mean durable-remission probability 0.15. This receives the largest
  or near-largest weight because randomized OCV-501 did not improve DFS or OS
  overall despite post-hoc responder associations.
* moderate: mean 0.40. This represents the recurring association between
  WT1-specific immune response and longer remission/survival in GPS phase 2 and
  independent WT1 vaccine platforms after severe shrinkage for confounding.
* strong: mean 0.70. This represents a genuine durable-tail mechanism, but gets
  little weight because the most dramatic evidence comes from very small,
  non-randomized or post-transplant cohorts.

The balanced mixture weights are 45% skeptical, 45% moderate, and 10% strong,
for a prior mean of 0.3175. Skeptical and mechanism-favoring alternatives are
also exposed for sensitivity analysis. The parameter remains the probability
that an immune responder enters the responder/cure model's durable-remission
state; it is not a literal clinical cure-rate estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from types import MappingProxyType


REGAL_INTERIM_IMMUNE_SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/1390478/"
    "000110465925005648/tm254291d1_ex99-1.htm"
)
REGAL_INTERIM_REPORTED_RESPONSE_RATE = 0.80
REGAL_INTERIM_DEFAULT_ASSUMED_EVALUABLE = 10
REGAL_INTERIM_DENOMINATOR_SENSITIVITY = (5, 10, 15, 20)


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
        if not isinstance(self.label, str):
            raise ValueError("beta prior label must be a string")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def lower(self) -> float:
        return 0.0

    @property
    def upper(self) -> float:
        return 1.0

    def sample(self, rng) -> float:
        value = float(rng.beta(self.alpha, self.beta))
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("rng.beta must return a finite draw in [0, 1]")
        return value

    def describe(self):
        """Return an auditable structural description of this prior."""

        return {
            "distribution": "beta",
            "alpha": self.alpha,
            "beta": self.beta,
            "label": self.label,
            "mean": self.mean,
        }

    def update(self, *evidence: BinomialEvidence, label: str | None = None) -> "BetaPrior":
        alpha = self.alpha + sum(item.responders for item in evidence)
        beta = self.beta + sum(item.nonresponders for item in evidence)
        return BetaPrior(alpha, beta, self.label if label is None else label)


@dataclass(frozen=True)
class BetaMixturePrior:
    """Finite mixture of beta distributions for a probability.

    This is used for the responder durable-remission parameter because the
    literature is heterogeneous enough that a single pseudo-count update would
    imply false precision. Component weights must sum to one and remain visible
    to callers for sensitivity/audit reporting.
    """

    label: str
    components: tuple[tuple[float, BetaPrior], ...]

    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        try:
            components = tuple(self.components)
        except TypeError as error:
            raise ValueError("components must contain (weight, BetaPrior) pairs") from error
        if not components:
            raise ValueError("components must not be empty")
        normalized = []
        total = 0.0
        for item in components:
            try:
                weight, prior = item
            except (TypeError, ValueError) as error:
                raise ValueError("components must contain (weight, BetaPrior) pairs") from error
            if isinstance(weight, bool):
                raise ValueError("mixture weights must be numeric")
            weight = float(weight)
            if not isfinite(weight) or weight <= 0.0:
                raise ValueError("mixture weights must be finite and positive")
            if not isinstance(prior, BetaPrior):
                raise ValueError("mixture components must be BetaPrior values")
            total += weight
            normalized.append((weight, prior))
        if abs(total - 1.0) > 1e-12:
            raise ValueError("mixture weights must sum to one")
        object.__setattr__(self, "components", tuple(normalized))

    @property
    def mean(self) -> float:
        return sum(weight * prior.mean for weight, prior in self.components)

    @property
    def lower(self) -> float:
        return 0.0

    @property
    def upper(self) -> float:
        return 1.0

    def sample(self, rng) -> float:
        selector = float(rng.random())
        if not isfinite(selector) or not 0.0 <= selector < 1.0:
            raise ValueError("rng.random must return a finite draw in [0, 1)")
        cumulative = 0.0
        for index, (weight, prior) in enumerate(self.components):
            cumulative += weight
            if selector < cumulative or index == len(self.components) - 1:
                return prior.sample(rng)
        raise RuntimeError("beta-mixture component selection failed")

    def describe(self):
        """Return an auditable structural description of this prior."""

        return {
            "distribution": "beta_mixture",
            "label": self.label,
            "mean": self.mean,
            "components": [
                {"weight": weight, "prior": prior.describe()}
                for weight, prior in self.components
            ],
        }


@dataclass(frozen=True)
class SurvivalEvidenceAnchor:
    """Human-readable evidence used to elicit the responder survival mixture."""

    label: str
    design: str
    finding: str
    interpretation: str

    def __post_init__(self):
        for name in ("label", "design", "finding", "interpretation"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


REFERENCE_RESPONSE_PRIOR = BetaPrior(1.0, 1.0, "reference_uniform")

GPS_PHASE2_IMMUNE_EVIDENCE = BinomialEvidence(
    "GPS phase 2 AML immune evaluable cohort",
    responders=9,
    evaluable=14,
)


def regal_interim_assumed_evidence(evaluable):
    """Translate the reported 80% rate under an explicit denominator assumption.

    SELLAS publicly reported the response percentage and random sampling, but
    not the sample size. Only assumed denominators that imply an integer number
    of responders are accepted, keeping the unverifiable precision assumption
    visible and easy to vary.
    """

    if isinstance(evaluable, bool) or not isinstance(evaluable, Integral):
        raise ValueError("assumed REGAL evaluable count must be an integer")
    evaluable = int(evaluable)
    if evaluable <= 0:
        raise ValueError("assumed REGAL evaluable count must be positive")
    expected = REGAL_INTERIM_REPORTED_RESPONSE_RATE * evaluable
    responders = int(round(expected))
    if abs(expected - responders) > 1e-12:
        raise ValueError(
            "assumed REGAL evaluable count must make the reported 80% an "
            "integer responder count"
        )
    return BinomialEvidence(
        (
            "REGAL interim randomly selected GPS immune cohort "
            f"(80% reported; n={evaluable} working assumption)"
        ),
        responders=responders,
        evaluable=evaluable,
    )


REGAL_INTERIM_IMMUNE_EVIDENCE = regal_interim_assumed_evidence(
    REGAL_INTERIM_DEFAULT_ASSUMED_EVALUABLE
)

GPS_PHASE2_RESPONSE_POSTERIOR = REFERENCE_RESPONSE_PRIOR.update(
    GPS_PHASE2_IMMUNE_EVIDENCE,
    label="gps_phase2_only_beta_10_6",
)

REGAL_INTERIM_RESPONSE_POSTERIOR_SENSITIVITY = MappingProxyType(
    {
        evaluable: REFERENCE_RESPONSE_PRIOR.update(
            regal_interim_assumed_evidence(evaluable),
            label=f"regal_interim_only_assumed_n_{evaluable}",
        )
        for evaluable in REGAL_INTERIM_DENOMINATOR_SENSITIVITY
    }
)

POOLED_GPS_RESPONSE_POSTERIOR_SENSITIVITY = MappingProxyType(
    {
        evaluable: REFERENCE_RESPONSE_PRIOR.update(
            GPS_PHASE2_IMMUNE_EVIDENCE,
            regal_interim_assumed_evidence(evaluable),
            label=f"pooled_gps_regal_assumed_n_{evaluable}",
        )
        for evaluable in REGAL_INTERIM_DENOMINATOR_SENSITIVITY
    }
)

REGAL_INTERIM_RESPONSE_POSTERIOR = REGAL_INTERIM_RESPONSE_POSTERIOR_SENSITIVITY[
    REGAL_INTERIM_DEFAULT_ASSUMED_EVALUABLE
]

POOLED_GPS_RESPONSE_POSTERIOR = POOLED_GPS_RESPONSE_POSTERIOR_SENSITIVITY[
    REGAL_INTERIM_DEFAULT_ASSUMED_EVALUABLE
]


# The following anchors are deliberately descriptive. They explain the elicited
# mixture but are not converted into pseudo-counts or treated as exchangeable
# observations.
WT1_RESPONDER_SURVIVAL_EVIDENCE = (
    SurvivalEvidenceAnchor(
        "OCV-501 randomized AML CR1 trial",
        "randomized double-blind placebo-controlled phase 2",
        "DFS HR 0.933 and OS HR 0.956 overall; immune responders did better post hoc",
        "strong anchor against assuming that WT1 vaccination automatically creates survival benefit",
    ),
    SurvivalEvidenceAnchor(
        "GPS phase 2 AML CR1 correlative cohort",
        "single-arm phase 2 responder analysis",
        "immune-responder DFS and OS medians not reached versus 15.6 and 35.8 months in nonresponders",
        "supports a responder-survival association but receives strong shrinkage for small n and prognostic confounding",
    ),
    SurvivalEvidenceAnchor(
        "WT1 mRNA dendritic-cell vaccine AML",
        "single-arm phase 2 responder analysis",
        "5-year OS 53.8% in responders versus 25.0% in nonresponders; CR1 5-year RFS 50% versus 7.7%",
        "independent-platform support for a durable responder subset, discounted for non-randomized responder classification",
    ),
    SurvivalEvidenceAnchor(
        "2026 pediatric post-allo-HSCT WT1 peptide study",
        "single-arm phase 2 with week-6 immune landmark analysis",
        "3-year OS 90.9% in immune responders versus 40.0% in nonresponders",
        "strong mechanistic tail evidence but heavily discounted for pediatric post-transplant setting and very small n",
    ),
)


# These component means are intentionally much less aggressive than the extreme
# responder hazard ratios reported by some small studies. They describe the
# probability of entering the responder/cure model's durable-remission state.
WT1_DURABLE_SKEPTICAL_COMPONENT = BetaPrior(
    1.5, 8.5, "wt1_durable_skeptical_mean_0.15"
)
WT1_DURABLE_MODERATE_COMPONENT = BetaPrior(
    4.0, 6.0, "wt1_durable_moderate_mean_0.40"
)
WT1_DURABLE_STRONG_COMPONENT = BetaPrior(
    7.0, 3.0, "wt1_durable_strong_mean_0.70"
)

WT1_RESPONDER_DURABLE_PRIOR_SKEPTICAL = BetaMixturePrior(
    "wt1_responder_durable_skeptical",
    (
        (0.60, WT1_DURABLE_SKEPTICAL_COMPONENT),
        (0.35, WT1_DURABLE_MODERATE_COMPONENT),
        (0.05, WT1_DURABLE_STRONG_COMPONENT),
    ),
)

WT1_RESPONDER_DURABLE_PRIOR_BALANCED = BetaMixturePrior(
    "wt1_responder_durable_balanced",
    (
        (0.45, WT1_DURABLE_SKEPTICAL_COMPONENT),
        (0.45, WT1_DURABLE_MODERATE_COMPONENT),
        (0.10, WT1_DURABLE_STRONG_COMPONENT),
    ),
)

WT1_RESPONDER_DURABLE_PRIOR_MECHANISM_FAVORING = BetaMixturePrior(
    "wt1_responder_durable_mechanism_favoring",
    (
        (0.30, WT1_DURABLE_SKEPTICAL_COMPONENT),
        (0.50, WT1_DURABLE_MODERATE_COMPONENT),
        (0.20, WT1_DURABLE_STRONG_COMPONENT),
    ),
)


__all__ = [
    "BetaMixturePrior",
    "BetaPrior",
    "BinomialEvidence",
    "GPS_PHASE2_IMMUNE_EVIDENCE",
    "GPS_PHASE2_RESPONSE_POSTERIOR",
    "POOLED_GPS_RESPONSE_POSTERIOR",
    "POOLED_GPS_RESPONSE_POSTERIOR_SENSITIVITY",
    "REFERENCE_RESPONSE_PRIOR",
    "REGAL_INTERIM_DEFAULT_ASSUMED_EVALUABLE",
    "REGAL_INTERIM_DENOMINATOR_SENSITIVITY",
    "REGAL_INTERIM_IMMUNE_EVIDENCE",
    "REGAL_INTERIM_IMMUNE_SOURCE_URL",
    "REGAL_INTERIM_REPORTED_RESPONSE_RATE",
    "REGAL_INTERIM_RESPONSE_POSTERIOR",
    "REGAL_INTERIM_RESPONSE_POSTERIOR_SENSITIVITY",
    "SurvivalEvidenceAnchor",
    "WT1_DURABLE_MODERATE_COMPONENT",
    "WT1_DURABLE_SKEPTICAL_COMPONENT",
    "WT1_DURABLE_STRONG_COMPONENT",
    "WT1_RESPONDER_DURABLE_PRIOR_BALANCED",
    "WT1_RESPONDER_DURABLE_PRIOR_MECHANISM_FAVORING",
    "WT1_RESPONDER_DURABLE_PRIOR_SKEPTICAL",
    "WT1_RESPONDER_SURVIVAL_EVIDENCE",
    "regal_interim_assumed_evidence",
]
