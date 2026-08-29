"""Biology-informed REGAL responder-family priors.

This module is deliberately additive: it leaves the validated WP7 core defaults
unchanged and exposes alternative effect-prior tuples that callers can pass to
``condition_effect_families_on_public_history`` or the posterior forecast helpers.
That makes biology-informed runs directly comparable with the existing balanced
analysis and prevents external immunology evidence from silently contaminating
the blinded public-history likelihood.

Two independent external-evidence updates are available:

* response probability: beta-binomial evidence from GPS phase 2 plus the REGAL
  interim immune substudy;
* responder durable benefit: a deliberately conservative beta-mixture elicited
  from the WT1 responder-survival literature, with substantial skeptical mass
  because randomized OCV-501 was negative overall.

The durable-benefit prior maps to the existing responder/cure family's
``responder_cure_probability`` parameter. It therefore represents the chance
that a measurable immune responder enters the model's durable-remission state;
it is not a literal clinical cure-rate estimate and does not import the extreme
hazard ratios from small responder analyses.
"""

from __future__ import annotations

from biology_priors import (
    BetaPrior,
    GPS_PHASE2_RESPONSE_POSTERIOR,
    POOLED_GPS_RESPONSE_POSTERIOR,
    REGAL_INTERIM_RESPONSE_POSTERIOR,
    WT1_RESPONDER_DURABLE_PRIOR_BALANCED,
    WT1_RESPONDER_DURABLE_PRIOR_MECHANISM_FAVORING,
    WT1_RESPONDER_DURABLE_PRIOR_SKEPTICAL,
)
from posterior import (
    DEFAULT_EFFECT_FAMILY_PRIORS,
    EffectFamilyPrior,
    EffectParameters,
    GPSEffectFamily,
    UniformPriorRange,
)


DEFAULT_RESPONSE_ONLY_CURE_PRIOR = UniformPriorRange(0.20, 0.85)


class BiologyInformedResponderEffectPrior(EffectFamilyPrior):
    """Responder/cure family with externally informed probability priors.

    ``response_beta_prior`` controls the probability that a GPS-treated patient
    mounts a measurable WT1-specific immune response.

    ``responder_cure_prior`` controls the conditional probability that such an
    immune responder enters the responder/cure model's durable-remission state.
    It may be the literature-elicited beta mixture or the original broad uniform
    range for response-evidence-only sensitivity runs.
    """

    def __init__(
        self,
        response_beta_prior: BetaPrior = POOLED_GPS_RESPONSE_POSTERIOR,
        responder_cure_prior=WT1_RESPONDER_DURABLE_PRIOR_BALANCED,
    ):
        if not isinstance(response_beta_prior, BetaPrior):
            raise ValueError("response_beta_prior must be a BetaPrior")
        if not callable(getattr(responder_cure_prior, "sample", None)):
            raise ValueError("responder_cure_prior must provide sample(rng)")
        super().__init__(
            GPSEffectFamily.RESPONDER_CURE,
            response_probability=UniformPriorRange(0.0, 1.0),
            responder_cure_probability=UniformPriorRange(0.0, 1.0),
        )
        object.__setattr__(self, "response_beta_prior", response_beta_prior)
        object.__setattr__(self, "responder_cure_prior", responder_cure_prior)

    def sample(self, rng):
        cure_probability = float(self.responder_cure_prior.sample(rng))
        if not 0.0 <= cure_probability <= 1.0:
            raise ValueError("responder_cure_prior must sample in [0, 1]")
        return EffectParameters(
            family=GPSEffectFamily.RESPONDER_CURE,
            response_probability=self.response_beta_prior.sample(rng),
            responder_cure_probability=cure_probability,
        )


def _replace_responder_prior(responder):
    return tuple(
        responder if prior.family is GPSEffectFamily.RESPONDER_CURE else prior
        for prior in DEFAULT_EFFECT_FAMILY_PRIORS
    )


def effect_priors_with_response_evidence(
    response_beta_prior: BetaPrior = POOLED_GPS_RESPONSE_POSTERIOR,
):
    """Change only responder-rate evidence; preserve the original cure range."""

    return _replace_responder_prior(
        BiologyInformedResponderEffectPrior(
            response_beta_prior=response_beta_prior,
            responder_cure_prior=DEFAULT_RESPONSE_ONLY_CURE_PRIOR,
        )
    )


def effect_priors_with_biology(
    response_beta_prior: BetaPrior = POOLED_GPS_RESPONSE_POSTERIOR,
    responder_cure_prior=WT1_RESPONDER_DURABLE_PRIOR_BALANCED,
):
    """Apply both immune-response and WT1 responder-survival evidence."""

    return _replace_responder_prior(
        BiologyInformedResponderEffectPrior(
            response_beta_prior=response_beta_prior,
            responder_cure_prior=responder_cure_prior,
        )
    )


# Isolate the immunogenicity update from the responder-survival update.
RESPONSE_EVIDENCE_ONLY_EFFECT_FAMILY_PRIORS = effect_priors_with_response_evidence()

# Full biology-informed default: pooled response evidence + balanced survival mixture.
BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS = effect_priors_with_biology()

# Survival-prior sensitivity while holding the pooled response evidence fixed.
BIOLOGY_INFORMED_SKEPTICAL_SURVIVAL_PRIORS = effect_priors_with_biology(
    responder_cure_prior=WT1_RESPONDER_DURABLE_PRIOR_SKEPTICAL
)
BIOLOGY_INFORMED_MECHANISM_FAVORING_SURVIVAL_PRIORS = effect_priors_with_biology(
    responder_cure_prior=WT1_RESPONDER_DURABLE_PRIOR_MECHANISM_FAVORING
)

# Exchangeability sensitivity for the immune-response evidence. These retain the
# original responder-cure range so the response-rate source can be varied alone.
PHASE2_ONLY_EFFECT_FAMILY_PRIORS = effect_priors_with_response_evidence(
    GPS_PHASE2_RESPONSE_POSTERIOR
)
REGAL_INTERIM_ONLY_EFFECT_FAMILY_PRIORS = effect_priors_with_response_evidence(
    REGAL_INTERIM_RESPONSE_POSTERIOR
)


__all__ = [
    "BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS",
    "BIOLOGY_INFORMED_MECHANISM_FAVORING_SURVIVAL_PRIORS",
    "BIOLOGY_INFORMED_SKEPTICAL_SURVIVAL_PRIORS",
    "BiologyInformedResponderEffectPrior",
    "DEFAULT_RESPONSE_ONLY_CURE_PRIOR",
    "PHASE2_ONLY_EFFECT_FAMILY_PRIORS",
    "REGAL_INTERIM_ONLY_EFFECT_FAMILY_PRIORS",
    "RESPONSE_EVIDENCE_ONLY_EFFECT_FAMILY_PRIORS",
    "effect_priors_with_biology",
    "effect_priors_with_response_evidence",
]
