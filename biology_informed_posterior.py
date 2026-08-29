"""Biology-informed REGAL responder-family priors.

This module is deliberately additive: it leaves the validated WP7 core defaults
unchanged and exposes an alternative effect-prior tuple that callers can pass to
``condition_effect_families_on_public_history`` or the posterior forecast helpers.
That makes biology-informed runs directly comparable with the existing balanced
analysis and prevents external immunology evidence from silently contaminating
the blinded public-history likelihood.
"""

from __future__ import annotations

from biology_priors import (
    BetaPrior,
    GPS_PHASE2_RESPONSE_POSTERIOR,
    POOLED_GPS_RESPONSE_POSTERIOR,
    REGAL_INTERIM_RESPONSE_POSTERIOR,
)
from posterior import (
    DEFAULT_EFFECT_FAMILY_PRIORS,
    EffectFamilyPrior,
    EffectParameters,
    GPSEffectFamily,
    UniformPriorRange,
)


class BiologyInformedResponderEffectPrior(EffectFamilyPrior):
    """Responder/cure effect prior with beta-distributed response probability.

    ``responder_cure_probability`` intentionally retains the existing broad
    Uniform(0.20, 0.85) prior.  The current evidence update therefore changes
    only the probability of mounting a measurable WT1-specific immune response;
    it does *not* infer the magnitude of survival benefit from responder-status
    associations in small or non-randomized WT1 studies.
    """

    def __init__(
        self,
        response_beta_prior: BetaPrior = POOLED_GPS_RESPONSE_POSTERIOR,
        responder_cure_probability: UniformPriorRange = UniformPriorRange(0.20, 0.85),
    ):
        if not isinstance(response_beta_prior, BetaPrior):
            raise ValueError("response_beta_prior must be a BetaPrior")
        super().__init__(
            GPSEffectFamily.RESPONDER_CURE,
            response_probability=UniformPriorRange(0.0, 1.0),
            responder_cure_probability=responder_cure_probability,
        )
        object.__setattr__(self, "response_beta_prior", response_beta_prior)

    def sample(self, rng):
        return EffectParameters(
            family=GPSEffectFamily.RESPONDER_CURE,
            response_probability=self.response_beta_prior.sample(rng),
            responder_cure_probability=self.responder_cure_probability.sample(rng),
        )


def effect_priors_with_response_evidence(
    response_beta_prior: BetaPrior = POOLED_GPS_RESPONSE_POSTERIOR,
):
    """Return the validated default family set with only responder-rate changed."""

    responder = BiologyInformedResponderEffectPrior(response_beta_prior)
    return tuple(
        responder if prior.family is GPSEffectFamily.RESPONDER_CURE else prior
        for prior in DEFAULT_EFFECT_FAMILY_PRIORS
    )


BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS = effect_priors_with_response_evidence()
PHASE2_ONLY_EFFECT_FAMILY_PRIORS = effect_priors_with_response_evidence(
    GPS_PHASE2_RESPONSE_POSTERIOR
)
REGAL_INTERIM_ONLY_EFFECT_FAMILY_PRIORS = effect_priors_with_response_evidence(
    REGAL_INTERIM_RESPONSE_POSTERIOR
)


__all__ = [
    "BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS",
    "BiologyInformedResponderEffectPrior",
    "PHASE2_ONLY_EFFECT_FAMILY_PRIORS",
    "REGAL_INTERIM_ONLY_EFFECT_FAMILY_PRIORS",
    "effect_priors_with_response_evidence",
]
