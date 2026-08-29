from datetime import date

import numpy as np

from biology_informed_posterior import (
    BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS,
    BIOLOGY_INFORMED_MECHANISM_FAVORING_SURVIVAL_PRIORS,
    BIOLOGY_INFORMED_SKEPTICAL_SURVIVAL_PRIORS,
    BiologyInformedResponderEffectPrior,
    PHASE2_ONLY_EFFECT_FAMILY_PRIORS,
    REGAL_INTERIM_ONLY_EFFECT_FAMILY_PRIORS,
    RESPONSE_EVIDENCE_ONLY_EFFECT_FAMILY_PRIORS,
)
from biology_priors import (
    BetaMixturePrior,
    BetaPrior,
    GPS_PHASE2_RESPONSE_POSTERIOR,
    POOLED_GPS_RESPONSE_POSTERIOR,
    REGAL_INTERIM_RESPONSE_POSTERIOR,
    WT1_RESPONDER_DURABLE_PRIOR_BALANCED,
    WT1_RESPONDER_DURABLE_PRIOR_MECHANISM_FAVORING,
    WT1_RESPONDER_DURABLE_PRIOR_SKEPTICAL,
    WT1_RESPONDER_SURVIVAL_EVIDENCE,
)
from posterior import (
    DEFAULT_EFFECT_FAMILY_PRIORS,
    GPSEffectFamily,
    GPSEffectScenarioSampler,
    UniformPriorRange,
)


def _responder_prior(priors):
    return next(item for item in priors if item.family is GPSEffectFamily.RESPONDER_CURE)


def test_beta_binomial_response_posteriors_match_encoded_evidence():
    assert (GPS_PHASE2_RESPONSE_POSTERIOR.alpha, GPS_PHASE2_RESPONSE_POSTERIOR.beta) == (10.0, 6.0)
    assert (REGAL_INTERIM_RESPONSE_POSTERIOR.alpha, REGAL_INTERIM_RESPONSE_POSTERIOR.beta) == (9.0, 3.0)
    assert (POOLED_GPS_RESPONSE_POSTERIOR.alpha, POOLED_GPS_RESPONSE_POSTERIOR.beta) == (18.0, 8.0)
    assert np.isclose(POOLED_GPS_RESPONSE_POSTERIOR.mean, 18.0 / 26.0)


def test_beta_mixture_validates_weights_and_samples_declared_mean():
    with np.testing.assert_raises_regex(ValueError, "sum to one"):
        BetaMixturePrior(
            "invalid",
            ((0.6, BetaPrior(2, 2)), (0.5, BetaPrior(3, 3))),
        )

    prior = BetaMixturePrior(
        "test",
        ((0.25, BetaPrior(2, 8)), (0.75, BetaPrior(6, 4))),
    )
    expected = 0.25 * 0.2 + 0.75 * 0.6
    assert np.isclose(prior.mean, expected)
    rng = np.random.default_rng(20260829)
    draws = np.array([prior.sample(rng) for _ in range(50_000)])
    assert abs(draws.mean() - expected) < 0.01
    assert np.all((draws >= 0.0) & (draws <= 1.0))


def test_wt1_survival_mixtures_are_conservative_and_ordered():
    assert len(WT1_RESPONDER_SURVIVAL_EVIDENCE) == 4
    assert np.isclose(WT1_RESPONDER_DURABLE_PRIOR_SKEPTICAL.mean, 0.265)
    assert np.isclose(WT1_RESPONDER_DURABLE_PRIOR_BALANCED.mean, 0.3175)
    assert np.isclose(WT1_RESPONDER_DURABLE_PRIOR_MECHANISM_FAVORING.mean, 0.385)
    assert (
        WT1_RESPONDER_DURABLE_PRIOR_SKEPTICAL.mean
        < WT1_RESPONDER_DURABLE_PRIOR_BALANCED.mean
        < WT1_RESPONDER_DURABLE_PRIOR_MECHANISM_FAVORING.mean
    )
    # All evidence-informed means are below the legacy Uniform(0.20, 0.85)
    # midpoint of 0.525; this is intentionally a shrinkage update, not an
    # optimistic translation of small-study responder hazard ratios.
    assert WT1_RESPONDER_DURABLE_PRIOR_MECHANISM_FAVORING.mean < 0.525


def test_biology_informed_tuple_changes_only_responder_family():
    assert len(BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS) == len(DEFAULT_EFFECT_FAMILY_PRIORS)
    for baseline, informed in zip(DEFAULT_EFFECT_FAMILY_PRIORS, BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS):
        assert baseline.family is informed.family
        if baseline.family is GPSEffectFamily.RESPONDER_CURE:
            assert isinstance(informed, BiologyInformedResponderEffectPrior)
            assert informed.response_beta_prior is POOLED_GPS_RESPONSE_POSTERIOR
            assert informed.responder_cure_prior is WT1_RESPONDER_DURABLE_PRIOR_BALANCED
        else:
            assert informed is baseline


def test_response_only_prior_preserves_legacy_cure_range():
    prior = _responder_prior(RESPONSE_EVIDENCE_ONLY_EFFECT_FAMILY_PRIORS)
    assert isinstance(prior.responder_cure_prior, UniformPriorRange)
    assert prior.responder_cure_prior.lower == 0.20
    assert prior.responder_cure_prior.upper == 0.85

    rng = np.random.default_rng(20260830)
    draws = [prior.sample(rng) for _ in range(30_000)]
    response = np.array([draw.response_probability for draw in draws])
    cure = np.array([draw.responder_cure_probability for draw in draws])
    assert abs(response.mean() - 18.0 / 26.0) < 0.01
    assert abs(cure.mean() - 0.525) < 0.01


def test_full_biology_prior_samples_response_and_survival_distributions():
    rng = np.random.default_rng(20260831)
    prior = _responder_prior(BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS)
    draws = [prior.sample(rng) for _ in range(50_000)]
    response = np.array([draw.response_probability for draw in draws])
    cure = np.array([draw.responder_cure_probability for draw in draws])

    assert abs(response.mean() - 18.0 / 26.0) < 0.01
    assert abs(cure.mean() - WT1_RESPONDER_DURABLE_PRIOR_BALANCED.mean) < 0.01
    assert np.all((response >= 0.0) & (response <= 1.0))
    assert np.all((cure >= 0.0) & (cure <= 1.0))


def test_full_biology_prior_runs_through_existing_responder_scenario_sampler():
    prior = _responder_prior(BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS)
    patients = GPSEffectScenarioSampler(prior)(
        (date(2022, 1, 1),) * 126,
        np.random.default_rng(20260901),
    )
    assert patients.patient_count == 126
    assert patients.treatment.sum() > 0
    probabilities = np.linspace(0.05, 0.75, 126)
    times = patients.event_time_model.ppf(probabilities)
    np.testing.assert_allclose(
        patients.event_time_model.cdf(times),
        probabilities,
        rtol=0.0,
        atol=3e-12,
    )


def test_survival_sensitivity_tuples_hold_response_evidence_fixed():
    skeptical = _responder_prior(BIOLOGY_INFORMED_SKEPTICAL_SURVIVAL_PRIORS)
    balanced = _responder_prior(BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS)
    favorable = _responder_prior(BIOLOGY_INFORMED_MECHANISM_FAVORING_SURVIVAL_PRIORS)

    assert skeptical.response_beta_prior is POOLED_GPS_RESPONSE_POSTERIOR
    assert balanced.response_beta_prior is POOLED_GPS_RESPONSE_POSTERIOR
    assert favorable.response_beta_prior is POOLED_GPS_RESPONSE_POSTERIOR
    assert skeptical.responder_cure_prior is WT1_RESPONDER_DURABLE_PRIOR_SKEPTICAL
    assert balanced.responder_cure_prior is WT1_RESPONDER_DURABLE_PRIOR_BALANCED
    assert favorable.responder_cure_prior is WT1_RESPONDER_DURABLE_PRIOR_MECHANISM_FAVORING


def test_response_evidence_source_sensitivity_remains_isolated():
    phase2 = _responder_prior(PHASE2_ONLY_EFFECT_FAMILY_PRIORS)
    regal = _responder_prior(REGAL_INTERIM_ONLY_EFFECT_FAMILY_PRIORS)
    assert np.isclose(phase2.response_beta_prior.mean, 10.0 / 16.0)
    assert np.isclose(regal.response_beta_prior.mean, 9.0 / 12.0)
    assert isinstance(phase2.responder_cure_prior, UniformPriorRange)
    assert isinstance(regal.responder_cure_prior, UniformPriorRange)
