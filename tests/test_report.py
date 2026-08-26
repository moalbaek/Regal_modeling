"""WP8 versioned result-bundle and HTML publication tests."""

from contextlib import redirect_stderr
from dataclasses import replace
from datetime import datetime, timezone
from io import StringIO
import json
from math import log
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock


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
    PRODUCTION_PROPOSAL_INTERIM_Z_TARGETS,
    RESULT_BUNDLE_END,
    RESULT_BUNDLE_START,
    _family_worker,
    _git_revision,
    build_result_bundle,
    build_unpublished_result_bundle,
    _build_command,
    canonical_result_json,
    embed_result_bundle,
    extract_embedded_result_bundle,
    main as report_main,
    run_regal_forecast_analysis,
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


def result_html_template():
    return (
        "<html><body>\n"
        + RESULT_BUNDLE_START
        + '\n<script id="regal-v2-result" type="application/json">{}</script>\n'
        + RESULT_BUNDLE_END
        + "\n</body></html>"
    )


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

    def test_production_worker_uses_the_auditable_base_proposal(self):
        prior = DEFAULT_EFFECT_FAMILY_PRIORS[0]
        with mock.patch(
            "report.condition_effect_family_futility_sensitivity",
            return_value=("projection",),
        ) as run_family:
            result = _family_worker(
                prior,
                FUTILITY_HR_SENSITIVITY_GRID,
                123,
                20260825,
            )
        self.assertEqual(result, ("projection",))
        self.assertEqual(PRODUCTION_PROPOSAL_INTERIM_Z_TARGETS, ())
        run_family.assert_called_once_with(
            prior,
            thresholds=FUTILITY_HR_SENSITIVITY_GRID,
            nsim=123,
            seed=20260825,
            proposal_interim_z_targets=(),
        )

    def test_production_worker_accepts_an_explicit_proposal_cross_check(self):
        prior = DEFAULT_EFFECT_FAMILY_PRIORS[0]
        with mock.patch(
            "report.condition_effect_family_futility_sensitivity",
            return_value=("projection",),
        ) as run_family:
            result = _family_worker(
                prior,
                FUTILITY_HR_SENSITIVITY_GRID,
                123,
                20260825,
                (0.0, 1.25),
            )
        self.assertEqual(result, ("projection",))
        run_family.assert_called_once_with(
            prior,
            thresholds=FUTILITY_HR_SENSITIVITY_GRID,
            nsim=123,
            seed=20260825,
            proposal_interim_z_targets=(0.0, 1.25),
        )

    def test_build_cli_parses_automatic_and_explicit_proposal_targets(self):
        cases = (
            ([], ()),
            (["--proposal-interim-z-targets", "auto"], None),
            (["--proposal-interim-z-targets", "0", "1.25"], (0.0, 1.25)),
        )
        for extra, expected in cases:
            with self.subTest(extra=extra), mock.patch(
                "report._build_command", return_value=0
            ) as build:
                self.assertEqual(report_main(["build", "--nsim", "1", *extra]), 0)
                self.assertEqual(
                    build.call_args.args[0].proposal_interim_z_targets,
                    expected,
                )

    def test_build_cli_rejects_mixed_or_nonfinite_proposal_targets(self):
        for values in (("auto", "0"), ("nan",), ("not-a-number",)):
            with self.subTest(values=values), mock.patch(
                "report._build_command"
            ) as build:
                stderr = StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    report_main(
                        ["build", "--proposal-interim-z-targets", *values]
                    )
                self.assertEqual(raised.exception.code, 2)
                self.assertNotIn("Traceback", stderr.getvalue())
                build.assert_not_called()

    def test_git_revision_marks_dirty_or_unknown_worktree_state(self):
        with mock.patch.dict("report.os.environ", {"GITHUB_SHA": ""}), mock.patch(
            "report.subprocess.check_output",
            side_effect=["abc123\n", " M report.py\n"],
        ):
            self.assertEqual(_git_revision(), "abc123-dirty")
        with mock.patch.dict("report.os.environ", {"GITHUB_SHA": ""}), mock.patch(
            "report.subprocess.check_output",
            side_effect=["abc123\n", subprocess.CalledProcessError(1, "git")],
        ):
            self.assertEqual(_git_revision(), "abc123-state-unknown")

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

    def test_every_sensitivity_row_serializes_its_numerical_gate_summary(self):
        bundle = self.ready_bundle()
        expected = {
            "minimum_history_effective_sample_size": 400.0,
            "minimum_continuation_effective_sample_size": 250.0,
            "maximum_history_weight_share": 0.01,
        }
        rows = bundle["prior_sensitivity"] + bundle["futility_sensitivity"]
        for row in rows:
            with self.subTest(
                row=row["name"],
                threshold=row["assumed_futility_hr_threshold"],
            ):
                self.assertEqual(row["readiness_diagnostics"], expected)
        for row in bundle["futility_sensitivity"]:
            self.assertNotIn("families", row)

    def test_wire_validator_enforces_futility_row_gate_summaries(self):
        bundle = json.loads(canonical_result_json(self.ready_bundle()))
        bundle["futility_sensitivity"][0]["readiness_diagnostics"][
            "minimum_continuation_effective_sample_size"
        ] = 99.0
        with self.assertRaisesRegex(ValueError, "minimum continuation ESS gate"):
            validate_result_bundle(bundle)

    def test_wire_validator_rejects_prior_gate_summary_drift(self):
        bundle = json.loads(canonical_result_json(self.ready_bundle()))
        bundle["prior_sensitivity"][0]["readiness_diagnostics"][
            "minimum_history_effective_sample_size"
        ] = 399.0
        with self.assertRaisesRegex(ValueError, "differ from family diagnostics"):
            validate_result_bundle(bundle)

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

    def test_wire_validator_requires_numeric_family_weights(self):
        bundle = json.loads(canonical_result_json(self.ready_bundle()))
        del bundle["prior_sensitivity"][0]["families"][0]["prior_weight"]
        with self.assertRaisesRegex(ValueError, "family prior weight must be numeric"):
            validate_result_bundle(bundle)

        bundle = json.loads(canonical_result_json(self.ready_bundle()))
        family = bundle["prior_sensitivity"][0]["families"][0]
        family["posterior_weight"] = str(family["posterior_weight"])
        with self.assertRaisesRegex(
            ValueError, "family posterior weight must be numeric"
        ):
            validate_result_bundle(bundle)

    def test_wire_validator_requires_release_disclosure(self):
        for invalid in (None, "", "   "):
            with self.subTest(invalid=invalid):
                bundle = json.loads(canonical_result_json(self.ready_bundle()))
                if invalid is None:
                    del bundle["release"]["disclosure"]
                else:
                    bundle["release"]["disclosure"] = invalid
                with self.assertRaisesRegex(ValueError, "release disclosure"):
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
        template = result_html_template()
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
        self.assertIn("Min continuation ESS (margin over 100)", html)
        self.assertIn("minimum_continuation_effective_sample_size", html)
        self.assertEqual(extract_embedded_result_bundle(html), bundle)

    def test_published_artifacts_must_match_canonical_public_data(self):
        bundle = json.loads(canonical_result_json(self.ready_bundle()))
        bundle["public_data"]["source"]["sha256"] = "a" * 64
        with TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "bundle.json"
            html_path = root / "explorer.html"
            json_path.write_text(canonical_result_json(bundle), encoding="utf-8")
            html_path.write_text(
                embed_result_bundle(result_html_template(), bundle),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "committed public history"):
                validate_published_artifacts(
                    json_path=json_path,
                    html_path=html_path,
                )

    def test_require_ready_persists_withheld_bundle_before_failing(self):
        prior_rows, futility_rows = sensitivity_results(history_ess=99.0)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "bundle.json"
            html_path = root / "explorer.html"
            html_path.write_text(result_html_template(), encoding="utf-8")
            args = SimpleNamespace(
                nsim=1000,
                seed=20260825,
                workers=1,
                source_revision="abc1234",
                require_ready=True,
                output_json=json_path,
                html=html_path,
            )
            with mock.patch(
                "report.run_regal_forecast_analysis",
                return_value=(prior_rows, futility_rows),
            ):
                stderr = StringIO()
                with redirect_stderr(stderr):
                    exit_code = _build_command(args)
            self.assertEqual(exit_code, 1)
            self.assertIn("diagnostic bundle was written", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            persisted = validate_published_artifacts(
                json_path=json_path,
                html_path=html_path,
            )
            self.assertEqual(persisted["release"]["status"], "withheld")

    def test_custom_futility_grid_fails_before_family_work_starts(self):
        with mock.patch("report._family_worker") as worker:
            with self.assertRaisesRegex(ValueError, "complete configured grid"):
                run_regal_forecast_analysis(nsim=1, thresholds=(None, 1.0))
            worker.assert_not_called()

    def test_nonfinite_proposal_target_fails_before_family_work_starts(self):
        with mock.patch("report._family_worker") as worker:
            with self.assertRaisesRegex(ValueError, "must be finite"):
                run_regal_forecast_analysis(
                    nsim=1,
                    proposal_interim_z_targets=(float("nan"),),
                )
            worker.assert_not_called()

    def test_not_run_bundle_uses_neutral_status_badge(self):
        html = (ROOT / "regal_explorer.html").read_text(encoding="utf-8")
        self.assertIn('notRun=release.status==="not_run"', html)
        self.assertIn('notRun?"":" withheld"', html)


if __name__ == "__main__":
    unittest.main()
