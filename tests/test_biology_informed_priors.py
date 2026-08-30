from datetime import date
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from audit.biology_informed_posterior_comparison import (
    AUDIT_PROPOSAL_INTERIM_Z_TARGETS,
    AuditNotReadyError,
    DEFAULT_AUDIT_IMPORTANCE_DRAWS,
    DEFAULT_AUDIT_WORKERS,
    _comparison_variants,
    _comparison_readiness_issues,
    _family_worker,
    _forecast_summary,
    _require_ready_output,
    _variant_prior_records,
    main as audit_main,
)
from biology_informed_posterior import (
    BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS,
    BIOLOGY_INFORMED_MECHANISM_FAVORING_SURVIVAL_PRIORS,
    BIOLOGY_INFORMED_SKEPTICAL_SURVIVAL_PRIORS,
    BiologyInformedResponderEffectPrior,
    DEFAULT_RESPONSE_ONLY_CURE_PRIOR,
    PHASE2_ONLY_EFFECT_FAMILY_PRIORS,
    REGAL_INTERIM_ONLY_EFFECT_FAMILY_PRIORS,
    RESPONSE_EVIDENCE_ONLY_EFFECT_FAMILY_PRIORS,
)
from biology_priors import (
    BetaMixturePrior,
    BetaPrior,
    GPS_PHASE2_RESPONSE_POSTERIOR,
    POOLED_GPS_RESPONSE_POSTERIOR,
    POOLED_GPS_RESPONSE_POSTERIOR_SENSITIVITY,
    REGAL_INTERIM_DEFAULT_ASSUMED_EVALUABLE,
    REGAL_INTERIM_IMMUNE_EVIDENCE,
    REGAL_INTERIM_IMMUNE_SOURCE_URL,
    REGAL_INTERIM_RESPONSE_POSTERIOR,
    REGAL_INTERIM_RESPONSE_POSTERIOR_SENSITIVITY,
    WT1_RESPONDER_DURABLE_PRIOR_BALANCED,
    WT1_RESPONDER_DURABLE_PRIOR_MECHANISM_FAVORING,
    WT1_RESPONDER_DURABLE_PRIOR_SKEPTICAL,
    WT1_RESPONDER_SURVIVAL_EVIDENCE,
    regal_interim_assumed_evidence,
)
from posterior import (
    DEFAULT_EFFECT_FAMILY_PRIORS,
    GPSEffectFamily,
    GPSEffectScenarioSampler,
    UniformPriorRange,
)
from report import _effect_prior_record


def _responder_prior(priors):
    return next(
        item for item in priors if item.family is GPSEffectFamily.RESPONDER_CURE
    )


class BiologyInformedPriorTest(unittest.TestCase):
    def test_beta_binomial_response_posteriors_match_encoded_evidence(self):
        self.assertEqual(
            (
                GPS_PHASE2_RESPONSE_POSTERIOR.alpha,
                GPS_PHASE2_RESPONSE_POSTERIOR.beta,
            ),
            (10.0, 6.0),
        )
        self.assertEqual(
            (
                REGAL_INTERIM_RESPONSE_POSTERIOR.alpha,
                REGAL_INTERIM_RESPONSE_POSTERIOR.beta,
            ),
            (9.0, 3.0),
        )
        self.assertEqual(
            (
                POOLED_GPS_RESPONSE_POSTERIOR.alpha,
                POOLED_GPS_RESPONSE_POSTERIOR.beta,
            ),
            (18.0, 8.0),
        )
        self.assertTrue(
            np.isclose(POOLED_GPS_RESPONSE_POSTERIOR.mean, 18.0 / 26.0)
        )

    def test_regal_denominator_is_explicit_and_has_sensitivity(self):
        self.assertEqual(REGAL_INTERIM_DEFAULT_ASSUMED_EVALUABLE, 10)
        self.assertEqual(REGAL_INTERIM_IMMUNE_EVIDENCE.responders, 8)
        self.assertIn("working assumption", REGAL_INTERIM_IMMUNE_EVIDENCE.label)
        self.assertTrue(REGAL_INTERIM_IMMUNE_SOURCE_URL.startswith("https://"))
        self.assertEqual(regal_interim_assumed_evidence(5).responders, 4)
        with self.assertRaisesRegex(ValueError, "integer responder count"):
            regal_interim_assumed_evidence(6)
        self.assertEqual(
            tuple(REGAL_INTERIM_RESPONSE_POSTERIOR_SENSITIVITY),
            (5, 10, 15, 20),
        )
        pooled_means = [
            prior.mean
            for prior in POOLED_GPS_RESPONSE_POSTERIOR_SENSITIVITY.values()
        ]
        self.assertEqual(pooled_means, sorted(pooled_means))

    def test_beta_mixture_validates_weights_and_samples_declared_mean(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            BetaMixturePrior(
                "invalid",
                ((0.6, BetaPrior(2, 2)), (0.5, BetaPrior(3, 3))),
            )

        prior = BetaMixturePrior(
            "test",
            ((0.25, BetaPrior(2, 8)), (0.75, BetaPrior(6, 4))),
        )
        expected = 0.25 * 0.2 + 0.75 * 0.6
        self.assertTrue(np.isclose(prior.mean, expected))
        rng = np.random.default_rng(20260829)
        draws = np.array([prior.sample(rng) for _ in range(50_000)])
        self.assertLess(abs(draws.mean() - expected), 0.01)
        self.assertTrue(np.all((draws >= 0.0) & (draws <= 1.0)))

    def test_wt1_survival_mixtures_are_conservative_and_ordered(self):
        self.assertEqual(len(WT1_RESPONDER_SURVIVAL_EVIDENCE), 4)
        self.assertTrue(
            np.isclose(WT1_RESPONDER_DURABLE_PRIOR_SKEPTICAL.mean, 0.265)
        )
        self.assertTrue(
            np.isclose(WT1_RESPONDER_DURABLE_PRIOR_BALANCED.mean, 0.3175)
        )
        self.assertTrue(
            np.isclose(
                WT1_RESPONDER_DURABLE_PRIOR_MECHANISM_FAVORING.mean, 0.385
            )
        )
        self.assertLess(
            WT1_RESPONDER_DURABLE_PRIOR_SKEPTICAL.mean,
            WT1_RESPONDER_DURABLE_PRIOR_BALANCED.mean,
        )
        self.assertLess(
            WT1_RESPONDER_DURABLE_PRIOR_BALANCED.mean,
            WT1_RESPONDER_DURABLE_PRIOR_MECHANISM_FAVORING.mean,
        )
        # All evidence-informed means are below the legacy Uniform(0.20, 0.85)
        # midpoint of 0.525; this is intentionally shrinkage rather than a
        # literal translation of small-study responder hazard ratios.
        self.assertLess(
            WT1_RESPONDER_DURABLE_PRIOR_MECHANISM_FAVORING.mean, 0.525
        )

    def test_biology_informed_tuple_changes_only_responder_family(self):
        self.assertEqual(
            len(BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS),
            len(DEFAULT_EFFECT_FAMILY_PRIORS),
        )
        for baseline, informed in zip(
            DEFAULT_EFFECT_FAMILY_PRIORS,
            BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS,
        ):
            self.assertIs(baseline.family, informed.family)
            if baseline.family is GPSEffectFamily.RESPONDER_CURE:
                self.assertIsInstance(informed, BiologyInformedResponderEffectPrior)
                self.assertIs(
                    informed.response_probability,
                    POOLED_GPS_RESPONSE_POSTERIOR,
                )
                self.assertIs(
                    informed.responder_cure_probability,
                    WT1_RESPONDER_DURABLE_PRIOR_BALANCED,
                )
            else:
                self.assertIs(informed, baseline)

    def test_biology_priors_have_truthful_identity_and_serialization(self):
        balanced = _responder_prior(BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS)
        skeptical = _responder_prior(
            BIOLOGY_INFORMED_SKEPTICAL_SURVIVAL_PRIORS
        )
        self.assertNotEqual(balanced, skeptical)
        self.assertEqual(len({balanced, skeptical}), 2)
        self.assertIn("BetaPrior", repr(balanced))
        self.assertIn("BetaMixturePrior", repr(balanced))

        record = _effect_prior_record(balanced)
        response = record["response_probability"]
        durable = record["responder_cure_probability"]
        self.assertEqual(response["distribution"], "beta")
        self.assertEqual((response["alpha"], response["beta"]), (18.0, 8.0))
        self.assertEqual(durable["distribution"], "beta_mixture")
        self.assertEqual(len(durable["components"]), 3)
        self.assertTrue(
            np.isclose(durable["mean"], WT1_RESPONDER_DURABLE_PRIOR_BALANCED.mean)
        )

    def test_response_only_prior_preserves_legacy_cure_range(self):
        prior = _responder_prior(RESPONSE_EVIDENCE_ONLY_EFFECT_FAMILY_PRIORS)
        canonical = _responder_prior(DEFAULT_EFFECT_FAMILY_PRIORS)
        self.assertIsInstance(prior.responder_cure_prior, UniformPriorRange)
        self.assertIs(DEFAULT_RESPONSE_ONLY_CURE_PRIOR, canonical.responder_cure_probability)
        self.assertIs(prior.responder_cure_prior, canonical.responder_cure_probability)

        rng = np.random.default_rng(20260830)
        draws = [prior.sample(rng) for _ in range(30_000)]
        response = np.array([draw.response_probability for draw in draws])
        cure = np.array([draw.responder_cure_probability for draw in draws])
        self.assertLess(abs(response.mean() - 18.0 / 26.0), 0.01)
        self.assertLess(abs(cure.mean() - 0.525), 0.01)

    def test_full_biology_prior_samples_response_and_survival_distributions(self):
        rng = np.random.default_rng(20260831)
        prior = _responder_prior(BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS)
        draws = [prior.sample(rng) for _ in range(50_000)]
        response = np.array([draw.response_probability for draw in draws])
        cure = np.array([draw.responder_cure_probability for draw in draws])

        self.assertLess(abs(response.mean() - 18.0 / 26.0), 0.01)
        self.assertLess(
            abs(cure.mean() - WT1_RESPONDER_DURABLE_PRIOR_BALANCED.mean),
            0.01,
        )
        self.assertTrue(np.all((response >= 0.0) & (response <= 1.0)))
        self.assertTrue(np.all((cure >= 0.0) & (cure <= 1.0)))

    def test_full_biology_prior_runs_through_existing_scenario_sampler(self):
        prior = _responder_prior(BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS)
        patients = GPSEffectScenarioSampler(prior)(
            (date(2022, 1, 1),) * 126,
            np.random.default_rng(20260901),
        )
        self.assertEqual(patients.patient_count, 126)
        self.assertGreater(patients.treatment.sum(), 0)
        probabilities = np.linspace(0.05, 0.75, 126)
        times = patients.event_time_model.ppf(probabilities)
        np.testing.assert_allclose(
            patients.event_time_model.cdf(times),
            probabilities,
            rtol=0.0,
            atol=3e-12,
        )

    def test_survival_sensitivity_tuples_hold_response_evidence_fixed(self):
        skeptical = _responder_prior(
            BIOLOGY_INFORMED_SKEPTICAL_SURVIVAL_PRIORS
        )
        balanced = _responder_prior(BIOLOGY_INFORMED_EFFECT_FAMILY_PRIORS)
        favorable = _responder_prior(
            BIOLOGY_INFORMED_MECHANISM_FAVORING_SURVIVAL_PRIORS
        )

        self.assertIs(skeptical.response_beta_prior, POOLED_GPS_RESPONSE_POSTERIOR)
        self.assertIs(balanced.response_beta_prior, POOLED_GPS_RESPONSE_POSTERIOR)
        self.assertIs(favorable.response_beta_prior, POOLED_GPS_RESPONSE_POSTERIOR)
        self.assertIs(
            skeptical.responder_cure_prior,
            WT1_RESPONDER_DURABLE_PRIOR_SKEPTICAL,
        )
        self.assertIs(
            balanced.responder_cure_prior,
            WT1_RESPONDER_DURABLE_PRIOR_BALANCED,
        )
        self.assertIs(
            favorable.responder_cure_prior,
            WT1_RESPONDER_DURABLE_PRIOR_MECHANISM_FAVORING,
        )

    def test_response_evidence_source_sensitivity_remains_isolated(self):
        phase2 = _responder_prior(PHASE2_ONLY_EFFECT_FAMILY_PRIORS)
        regal = _responder_prior(REGAL_INTERIM_ONLY_EFFECT_FAMILY_PRIORS)
        self.assertTrue(np.isclose(phase2.response_beta_prior.mean, 10.0 / 16.0))
        self.assertTrue(np.isclose(regal.response_beta_prior.mean, 9.0 / 12.0))
        self.assertIsInstance(phase2.responder_cure_prior, UniformPriorRange)
        self.assertIsInstance(regal.responder_cure_prior, UniformPriorRange)

    def test_audit_runs_source_and_denominator_sensitivity_forecast_variants(self):
        variants = _comparison_variants()
        self.assertEqual(
            tuple(variants),
            (
                "baseline_wp7",
                "response_evidence_only",
                "response_phase2_only",
                "response_regal_interim_only",
                "biology_skeptical_survival",
                "biology_balanced_survival",
                "biology_balanced_regal_assumed_n5",
                "biology_balanced_regal_assumed_n20",
                "biology_mechanism_favoring_survival",
            ),
        )
        n5 = _responder_prior(variants["biology_balanced_regal_assumed_n5"])
        n20 = _responder_prior(variants["biology_balanced_regal_assumed_n20"])
        self.assertIs(
            n5.response_probability,
            POOLED_GPS_RESPONSE_POSTERIOR_SENSITIVITY[5],
        )
        self.assertIs(
            n20.response_probability,
            POOLED_GPS_RESPONSE_POSTERIOR_SENSITIVITY[20],
        )
        self.assertLess(n5.response_probability.mean, n20.response_probability.mean)
        records = _variant_prior_records(variants)
        for name, record in records.items():
            with self.subTest(variant=name):
                self.assertEqual(
                    set(record),
                    {"response_probability", "responder_cure_probability"},
                )
        self.assertEqual(
            (
                records["biology_balanced_regal_assumed_n5"][
                    "response_probability"
                ]["alpha"],
                records["biology_balanced_regal_assumed_n5"][
                    "response_probability"
                ]["beta"],
            ),
            (14.0, 7.0),
        )
        self.assertEqual(
            (
                records["biology_balanced_regal_assumed_n20"][
                    "response_probability"
                ]["alpha"],
                records["biology_balanced_regal_assumed_n20"][
                    "response_probability"
                ]["beta"],
            ),
            (26.0, 10.0),
        )

    def test_audit_worker_uses_production_base_proposal(self):
        prior = _responder_prior(DEFAULT_EFFECT_FAMILY_PRIORS)
        with mock.patch(
            "audit.biology_informed_posterior_comparison."
            "condition_effect_family_futility_sensitivity",
            return_value=("projection",),
        ) as condition:
            result = _family_worker(prior, (None,), 123, 20260825)
        self.assertEqual(result, ("projection",))
        self.assertEqual(AUDIT_PROPOSAL_INTERIM_Z_TARGETS, ())
        condition.assert_called_once_with(
            prior,
            thresholds=(None,),
            nsim=123,
            seed=20260825,
            proposal_interim_z_targets=(),
        )

    def test_audit_cli_defaults_can_clear_the_enforced_readiness_budget(self):
        ready = {"is_posterior_forecast": True, "readiness_issues": []}
        with mock.patch(
            "audit.biology_informed_posterior_comparison.run_comparison",
            return_value=ready,
        ) as run, mock.patch(
            "audit.biology_informed_posterior_comparison._print_table"
        ):
            self.assertIsNone(audit_main([]))
        run.assert_called_once_with(
            nsim=DEFAULT_AUDIT_IMPORTANCE_DRAWS,
            seed=20260825,
            workers=DEFAULT_AUDIT_WORKERS,
            progress=True,
        )
        self.assertEqual(DEFAULT_AUDIT_IMPORTANCE_DRAWS, 150_000)
        self.assertEqual(DEFAULT_AUDIT_WORKERS, 7)

    def test_audit_summary_preserves_and_enforces_readiness_status(self):
        conditioning = SimpleNamespace(
            history_effective_sample_size=10.0,
            continuation_effective_sample_size=8.0,
            maximum_history_weight_share=0.20,
        )
        forecast = SimpleNamespace(
            family_results=(SimpleNamespace(conditioning=conditioning),),
            is_posterior_forecast=False,
            forecast_readiness_issues=("history ESS is below 100",),
            p_final_rejection_given_public_history_and_continuation=0.75,
            p_final_reached_given_public_history_and_continuation=0.90,
            model_prior_weights={GPSEffectFamily.RESPONDER_CURE: 0.10},
            model_posterior_weights={GPSEffectFamily.RESPONDER_CURE: 0.20},
        )
        summary = _forecast_summary(forecast)
        self.assertFalse(summary["is_posterior_forecast"])
        self.assertEqual(summary["estimate_status"], "diagnostic_only")
        self.assertEqual(
            summary["readiness_diagnostics"][
                "minimum_continuation_effective_sample_size"
            ],
            8.0,
        )
        forecasts = {"biology": {"disabled": summary}}
        issues = _comparison_readiness_issues(forecasts)
        self.assertEqual(len(issues), 1)
        result = {
            "is_posterior_forecast": False,
            "readiness_issues": list(issues),
        }
        with self.assertRaisesRegex(AuditNotReadyError, "diagnostic-only"):
            _require_ready_output(result)


if __name__ == "__main__":
    unittest.main()
