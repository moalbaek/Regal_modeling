import numpy as np

from biology_informed_posterior import (
    BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS,
    BiologyInformedResponderEffectPrior,
    PHASE2_ONLY_EFFECT_FAMILY_PRIORS,
    REGAL_INTERIM_ONLY_EFFECT_FAMILY_PRIORS,
)
from biology_priors import (
    GPS_PHASE2_RESPONSE_POSTERIOR,
    POOLED_GPS_RESPONSE_POSTERIOR,
    REGAL_INTERIM_RESPONSE_POSTERIOR,
)
from posterior import DEFAULT_EFFECT_FAMILY_PRIORS, GPSEffectFamily


def _responder_prior(priors):
    return next(item for item in priors if item.family is GPSEffectFamily.RESPONDER_CURE)


def test_beta_binomial_response_posteriors_match_encoded_evidence():
    assert (GPS_PHASE2_RESPONSE_POSTERIOR.alpha, GPS_PHASE2_RESPONSE_POSTERIOR.beta) == (10.0, 6.0)
    assert (REGAL_INTERIM_RESPONSE_POSTERIOR.alpha, REGAL_INTERIM_RESPONSE_POSTERIOR.beta) == (9.0, 3.0)
    assert (POOLED_GPS_RESPONSE_POSTERIOR.alpha, POOLED_GPS_RESPONSE_POSTERIOR.beta) == (18.0, 8.0)
    assert np.isclose(POOLED_GPS_RESPONSE_POSTERIOR.mean, 18.0 / 26.0)


def test_biology_informed_tuple_changes_only_responder_family():
    assert len(BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS) == len(DEFAULT_EFFECT_FAMILY_PRIORS)
    for baseline, informed in zip(DEFAULT_EFFECT_FAMILY_PRIORS, BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS):
        assert baseline.family is informed.family
        if baseline.family is GPSEffectFamily.RESPONDER_CURE:
            assert isinstance(informed, BiologyInformedResponderEffectPrior)
        else:
            assert informed is baseline


def test_pooled_responder_prior_samples_beta_distribution_and_preserves_cure_range():
    rng = np.random.default_rng(20260829)
    prior = _responder_prior(BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS)
    draws = [prior.sample(rng) for _ in range(30_000)]
    response = np.array([draw.response_probability for draw in draws])
    cure = np.array([draw.responder_cure_probability for draw in draws])

    assert abs(response.mean() - 18.0 / 26.0) < 0.01
    assert 0.20 <= cure.min() < cure.max() <= 0.85
    assert np.all((response >= 0.0) & (response <= 1.0))


def test_sensitivity_tuples_use_requested_evidence_source():
    assert np.isclose(
        _responder_prior(PHASE2_ONLY_EFFECT_FAMILY_PRIORS).response_beta_prior.mean,
        10.0 / 16.0,
    )
    assert np.isclose(
        _responder_prior(REGAL_INTERIM_ONLY_EFFECT_FAMILY_PRIORS).response_beta_prior.mean,
        9.0 / 12.0,
    )
