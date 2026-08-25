"""WP8 versioned result-bundle and HTML publication tests."""

from dataclasses import replace
from datetime import datetime, timezone
import json
from math import log
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from posterior import (  # noqa: E402
    BALANCED_MODEL_FAMILY_PRIOR,
    DEFAULT_EFFECT_FAMILY_PRIORS,
    ConditioningResult,
    EffectFamilyProjection,
    GPSEffectFamily,
    posterior_model_average,
    posterior_prior_sensitivity,
)
from regal_data import load_regal_data_snapshot  # noqa: E402
from report import (  # noqa: E402
    AnalysisRunMetadata,
    RESULT_BUNDLE_END,
    RESULT_BUNDLE_START,
    build_result_bundle,
    build_unpublished_result_bundle,
    canonical_result_json,
    embed_result_bundle,
    extract_embedded_result_bundle,
    validate_published_artifacts,
    validate_result_bundle,
)
from simulation import FUTILITY_HR_SENSITIVITY_GRID  # noqa: E402
from trial_design import HazardRatioFutilityRule, TrialDecisionDesign  # noqa: E402


def conditioning_result(family, *, design=None, rejection=0.5, history_ess=400.0):
    if design is None:
        design = TrialDecisionDesign()
    return ConditioningResult(
        scenario_name=family.value,
        design=design,
        importance_draws=1000,
        history_compatible_draws=700,
        continuation_compatible_draws=400,
        interim_efficacy_draws=150,
        interim_futility_draws=0,
        non_estimable_interim_draws=150,
        final_rejection_draws=200,
        final_non_rejection_draws=150,
        final_not_reached_draws=50,
        log_p_public_history=log(0.4),
        p_continue_given_public_history=0.5,
        p_final_rejection_given_public_history_and_continuation=rejection,
        p_final_reached_given_public_history_and_continuation=0.9,
        history_effective_sample_size=history_ess,
        continuation_effective_sample_size=250.0,
        maximum_history_weight_share=0.01,
        valid_disclosure_lag_mass=1.0,
        proposal_interim_z_targets=(0.0,),
        tilt_attempts=1000,
        tilt_fallbacks=2,
        draws_with_tilt_fallback=2,
        proposal_infeasible_draws=0,
        mean_tilt_iterations=3.0,
        maximum_tilt_error=1e-10,
    )


def projections(*, history_ess=400.0, design=None):
    rejection = {
        family: 0.15 + 0.1 * index
        for index, family in enumerate(GPSEffectFamily)
    }
    return tuple(
        EffectFamilyProjection(
            prior.family,
            prior,
            conditioning_result(
                prior.family,
                design=design,
                rejection=rejection[prior.family],
                history_ess=history_ess,
            ),
        )
        for prior in DEFAULT_EFFECT_FAMILY_PRIORS
    )


def sensitivity_results(*, history_ess=400.0):
    baseline = projections(history_ess=history_ess)
    prior_rows = posterior_prior_sensitivity(baseline)
    futility_rows = []
    for threshold in FUTILITY_HR_SENSITIVITY_GRID:
        if threshold is None:
            row = baseline
        else:
            design = TrialDecisionDesign(
                futility_rule=HazardRatioFutilityRule(threshold)
            )
            row = tuple(
                replace(
                    item,
                    conditioning=replace(item.conditioning, design=design),
                )
                for item in baseline
            )
        futility_rows.append(posterior_model_average(row, BALANCED_MODEL_FAMILY_PRIOR))
    return prior_rows, tuple(futility_rows)


class RegalDataSnapshotTest(unittest.TestCase):
    def test_snapshot_fingerprints_the_validated_public_history(self):
        snapshot = load_regal_data_snapshot()
        self.assertEqual(snapshot.as_of_date.isoformat(), "2026-08-11")
        self.assertEqual(len(snapshot.source_sha256), 64)
        payload = dict(snapshot.to_mapping())
        self.assertEqual(payload["registry_id"], "NCT04229979")
        self.assertEqual(payload["source"]["path"], "data/regal_public_history.json")
        self.assertEqual(payload["source"]["sha256"], snapshot.source_sha256)
        self.assertEqual(len(payload["observations"]), 7)


class ResultBundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = load_regal_data_snapshot()
        cls.metadata = AnalysisRunMetadata(
            generated_at=datetime(2026, 8, 25, 12, 30, tzinfo=timezone.utc),
            source_revision="abc1234",
            seed=20260825,
        )

    def ready_bundle(self):
        prior_rows, futility_rows = sensitivity_results()
        return build_result_bundle(
            prior_rows,
            futility_rows,
            metadata=self.metadata,
            data_snapshot=self.snapshot,
        )

    def test_ready_bundle_exposes_only_the_primary_release_headline(self):
        bundle = self.ready_bundle()
        self.assertTrue(bundle["release"]["is_posterior_forecast"])
        self.assertEqual(bundle["release"]["status"], "ready")
        primary = next(
            item for item in bundle["prior_sensitivity"] if item["name"] == "balanced"
        )
        self.assertEqual(
            bundle["release"]["headline"]["value"],
            primary["probabilities"][
                "final_rejection_given_public_history_and_continuation"
            ],
        )
        self.assertEqual(
            [item["name"] for item in bundle["prior_sensitivity"]],
            ["skeptical", "balanced", "cure_favoring"],
        )
        self.assertEqual(
            [item["assumed_futility_hr_threshold"] for item in bundle["futility_sensitivity"]],
            list(FUTILITY_HR_SENSITIVITY_GRID),
        )
        serialized = canonical_result_json(bundle)
        self.assertEqual(json.loads(serialized), bundle)

    def test_failed_readiness_gate_withholds_the_headline_but_keeps_diagnostics(self):
        prior_rows, futility_rows = sensitivity_results(history_ess=99.0)
        bundle = build_result_bundle(
            prior_rows,
            futility_rows,
            metadata=self.metadata,
            data_snapshot=self.snapshot,
        )
        self.assertFalse(bundle["release"]["is_posterior_forecast"])
        self.assertEqual(bundle["release"]["status"], "withheld")
        self.assertIsNone(bundle["release"]["headline"])
        self.assertTrue(
            any("history ESS" in issue for issue in bundle["release"]["readiness_issues"])
        )
        primary = next(
            item for item in bundle["prior_sensitivity"] if item["name"] == "balanced"
        )
        self.assertEqual(
            primary["families"][0]["diagnostics"]["history_effective_sample_size"],
            99.0,
        )

    def test_wire_validator_rejects_nonfinite_numbers_and_false_release_labels(self):
        bundle = json.loads(canonical_result_json(self.ready_bundle()))
        bundle["release"]["headline"]["value"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_result_bundle(bundle)

        bundle = json.loads(canonical_result_json(self.ready_bundle()))
        bundle["release"]["is_posterior_forecast"] = False
        with self.assertRaisesRegex(ValueError, "withhold"):
            validate_result_bundle(bundle)

    def test_unpublished_bundle_has_no_analysis_values(self):
        bundle = build_unpublished_result_bundle(
            generated_at=self.metadata.generated_at,
            source_revision=self.metadata.source_revision,
            data_snapshot=self.snapshot,
        )
        self.assertEqual(bundle["release"]["status"], "not_run")
        self.assertIsNone(bundle["release"]["headline"])
        self.assertEqual(bundle["prior_sensitivity"], [])
        self.assertEqual(bundle["futility_sensitivity"], [])

    def test_html_embedding_round_trips_and_requires_unique_markers(self):
        bundle = self.ready_bundle()
        template = (
            "<html><body>\n"
            + RESULT_BUNDLE_START
            + '\n<script id="regal-v2-result" type="application/json">{}</script>\n'
            + RESULT_BUNDLE_END
            + "\n</body></html>"
        )
        updated = embed_result_bundle(template, bundle)
        self.assertEqual(extract_embedded_result_bundle(updated), bundle)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            embed_result_bundle("<html></html>", bundle)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            embed_result_bundle(template + template, bundle)

    def test_committed_json_and_self_contained_html_are_identical(self):
        bundle = validate_published_artifacts()
        html = (ROOT / "regal_explorer.html").read_text(encoding="utf-8")
        self.assertIn("renderV2Forecast()", html)
        self.assertIn("release.is_posterior_forecast===true", html)
        self.assertEqual(extract_embedded_result_bundle(html), bundle)


if __name__ == "__main__":
    unittest.main()
