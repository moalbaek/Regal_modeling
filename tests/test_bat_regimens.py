"""Tests for REGAL v2 BAT strata, regimens, and allocation roles."""

import os
import sys
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bat_regimens import (  # noqa: E402
    BATComponent,
    BATDesign,
    BATDesignRole,
    BATPathway,
    BATRegimen,
    BATStratum,
    BEAR_STRONG_BAT_STRESS,
    DEFAULT_COMPONENT_LIBRARY,
    HMA_REGIMEN,
    HMA_VENETOCLAX_REGIMEN,
    LDAC_REGIMEN,
    LDAC_VENETOCLAX_REGIMEN,
    LEGACY_COMPONENT_MIX,
    OBSERVATION_REGIMEN,
    PRIMARY_EQUAL_STRATA,
    VENETOCLAX_DOMINANT_STRESS,
    default_component_library,
)
from survival_models import SurvivalScale  # noqa: E402


def combination_design():
    """Equal strata with two combinations crossing their planned strata."""

    return BATDesign(
        name="combination_test",
        role=BATDesignRole.PRIMARY,
        pathways=(
            BATPathway(
                BATStratum.SUPPORTIVE_CARE_HYDROXYUREA,
                OBSERVATION_REGIMEN,
                0.25,
            ),
            BATPathway(BATStratum.HMA, HMA_VENETOCLAX_REGIMEN, 0.25),
            BATPathway(BATStratum.VENETOCLAX, HMA_VENETOCLAX_REGIMEN, 0.25),
            BATPathway(BATStratum.LDAC, LDAC_VENETOCLAX_REGIMEN, 0.25),
        ),
    )


class CommittedBATDesignTest(unittest.TestCase):
    def test_primary_design_has_four_equal_planned_strata(self):
        self.assertIs(PRIMARY_EQUAL_STRATA.role, BATDesignRole.PRIMARY)
        self.assertEqual(set(PRIMARY_EQUAL_STRATA.stratum_probabilities), set(BATStratum))
        for probability in PRIMARY_EQUAL_STRATA.stratum_probabilities.values():
            self.assertAlmostEqual(probability, 0.25)

    def test_primary_proxy_reproduces_legacy_equal_strata_component_weights(self):
        expected = {
            BATComponent.OBSERVATION: 135.0 / 700.0,
            BATComponent.HYDROXYUREA: 40.0 / 700.0,
            BATComponent.HMA: 0.25,
            BATComponent.VENETOCLAX: 0.25,
            BATComponent.LDAC: 0.25,
        }
        self.assertEqual(
            set(PRIMARY_EQUAL_STRATA.survival_component_probabilities), set(expected)
        )
        for component, probability in expected.items():
            self.assertAlmostEqual(
                PRIMARY_EQUAL_STRATA.survival_component_probabilities[component],
                probability,
            )
            self.assertAlmostEqual(
                PRIMARY_EQUAL_STRATA.component_exposure_probabilities[component],
                probability,
            )
        self.assertAlmostEqual(
            sum(PRIMARY_EQUAL_STRATA.regimen_probabilities.values()), 1.0
        )

    def test_current_component_weights_are_only_a_legacy_comparison(self):
        self.assertIs(LEGACY_COMPONENT_MIX.role, BATDesignRole.LEGACY_COMPARISON)
        expected = {
            BATComponent.OBSERVATION: 0.27,
            BATComponent.HYDROXYUREA: 0.08,
            BATComponent.HMA: 0.22,
            BATComponent.VENETOCLAX: 0.35,
            BATComponent.LDAC: 0.08,
        }
        self.assertEqual(LEGACY_COMPONENT_MIX.survival_component_probabilities, expected)
        self.assertAlmostEqual(
            LEGACY_COMPONENT_MIX.stratum_probabilities[
                BATStratum.SUPPORTIVE_CARE_HYDROXYUREA
            ],
            0.35,
        )

    def test_venetoclax_dominant_and_bear_allocations_are_stress_tests(self):
        expected_venetoclax = {
            VENETOCLAX_DOMINANT_STRESS: 0.60,
            BEAR_STRONG_BAT_STRESS: 0.70,
        }
        for design, probability in expected_venetoclax.items():
            with self.subTest(design=design.name):
                self.assertIs(design.role, BATDesignRole.STRESS_TEST)
                self.assertAlmostEqual(
                    design.survival_component_probabilities[BATComponent.VENETOCLAX],
                    probability,
                )
                self.assertAlmostEqual(sum(design.regimen_probabilities.values()), 1.0)

    def test_default_component_library_uses_documented_overall_survival_inputs(self):
        expected = {
            BATComponent.OBSERVATION: (6.0, 0.03, 1.1),
            BATComponent.HYDROXYUREA: (5.0, 0.02, 1.1),
            BATComponent.HMA: (12.0, 0.10, 1.0),
            BATComponent.VENETOCLAX: (12.0, 0.15, 0.78),
            BATComponent.LDAC: (7.0, 0.08, 1.1),
        }
        self.assertEqual(set(DEFAULT_COMPONENT_LIBRARY), set(expected))
        for key, (median, cure, shape) in expected.items():
            with self.subTest(component=key.value):
                component = DEFAULT_COMPONENT_LIBRARY[key]
                self.assertEqual(component.uncured.median_months, median)
                self.assertEqual(component.cure_fraction, cure)
                self.assertEqual(component.uncured.shape, shape)
                self.assertIs(component.survival_scale, SurvivalScale.OVERALL)

        with self.assertRaises(TypeError):
            DEFAULT_COMPONENT_LIBRARY[BATComponent.HMA] = DEFAULT_COMPONENT_LIBRARY[
                BATComponent.LDAC
            ]
        first = default_component_library()
        second = default_component_library()
        self.assertIsNot(first, second)
        first.pop(BATComponent.HMA)
        self.assertIn(BATComponent.HMA, second)


class CombinationRegimenTest(unittest.TestCase):
    def test_joint_marginals_keep_strata_regimens_and_exposures_distinct(self):
        design = combination_design()
        for probability in design.stratum_probabilities.values():
            self.assertEqual(probability, 0.25)

        self.assertEqual(design.regimen_probabilities[OBSERVATION_REGIMEN], 0.25)
        self.assertEqual(design.regimen_probabilities[HMA_VENETOCLAX_REGIMEN], 0.50)
        self.assertEqual(design.regimen_probabilities[LDAC_VENETOCLAX_REGIMEN], 0.25)
        self.assertEqual(sum(design.regimen_probabilities.values()), 1.0)

        profiles = design.survival_component_probabilities
        self.assertEqual(profiles[BATComponent.OBSERVATION], 0.25)
        self.assertEqual(profiles[BATComponent.VENETOCLAX], 0.75)
        self.assertEqual(sum(profiles.values()), 1.0)

        exposures = design.component_exposure_probabilities
        self.assertEqual(exposures[BATComponent.OBSERVATION], 0.25)
        self.assertEqual(exposures[BATComponent.HMA], 0.50)
        self.assertEqual(exposures[BATComponent.VENETOCLAX], 0.75)
        self.assertEqual(exposures[BATComponent.LDAC], 0.25)
        self.assertEqual(sum(exposures.values()), 1.75)

    def test_sampled_combination_patients_are_not_double_counted(self):
        class FixedRng:
            @staticmethod
            def random(size):
                if size != 4:
                    raise AssertionError("unexpected requested size")
                return np.array([0.10, 0.30, 0.60, 0.90])

        cohort = combination_design().sample(np.int64(4), FixedRng())
        self.assertEqual(cohort.patient_count, 4)
        self.assertEqual(sum(cohort.stratum_counts().values()), 4)
        self.assertEqual(sum(cohort.regimen_counts().values()), 4)
        self.assertEqual(sum(cohort.survival_component_counts().values()), 4)
        self.assertEqual(sum(cohort.component_exposure_counts().values()), 7)
        self.assertEqual(cohort.regimen_counts()[HMA_VENETOCLAX_REGIMEN], 2)
        self.assertEqual(
            cohort.assignments[1].stratum,
            BATStratum.HMA,
        )
        self.assertEqual(
            cohort.assignments[2].stratum,
            BATStratum.VENETOCLAX,
        )
        self.assertIs(
            cohort.assignments[1].regimen.survival_component,
            BATComponent.VENETOCLAX,
        )


class BATValidationTest(unittest.TestCase):
    def test_regimens_reject_duplicate_or_unprofiled_components(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            BATRegimen(
                "duplicate",
                (BATComponent.HMA, BATComponent.HMA),
                BATComponent.HMA,
            )
        with self.assertRaisesRegex(ValueError, "also appear"):
            BATRegimen("unprofiled", (BATComponent.HMA,), BATComponent.VENETOCLAX)

    def test_pathways_reject_regimens_in_the_wrong_planned_stratum(self):
        with self.assertRaisesRegex(ValueError, "compatible"):
            BATPathway(
                BATStratum.SUPPORTIVE_CARE_HYDROXYUREA,
                HMA_REGIMEN,
                1.0,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            BATPathway(BATStratum.HMA, HMA_REGIMEN, True)

    def test_design_requires_unique_cells_all_strata_and_unit_probability(self):
        base = combination_design().pathways
        with self.assertRaisesRegex(ValueError, "all four"):
            BATDesign("missing", BATDesignRole.PRIMARY, base[:-1])
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            BATDesign(
                "bad total",
                BATDesignRole.PRIMARY,
                tuple(
                    BATPathway(path.stratum, path.regimen, path.probability * 0.9)
                    for path in base
                ),
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            BATDesign(
                "duplicate",
                BATDesignRole.PRIMARY,
                base
                + (
                    BATPathway(
                        BATStratum.SUPPORTIVE_CARE_HYDROXYUREA,
                        OBSERVATION_REGIMEN,
                        0.01,
                    ),
                ),
            )
        conflicting_hma = BATRegimen(
            HMA_REGIMEN.key,
            (BATComponent.HMA, BATComponent.VENETOCLAX),
            BATComponent.VENETOCLAX,
            "Conflicting definition",
        )
        with self.assertRaisesRegex(ValueError, "regimen key"):
            BATDesign(
                "conflicting key",
                BATDesignRole.PRIMARY,
                (
                    base[0],
                    BATPathway(BATStratum.HMA, HMA_REGIMEN, 0.125),
                    BATPathway(BATStratum.HMA, conflicting_hma, 0.125),
                    base[2],
                    base[3],
                ),
            )

    def test_sample_size_and_rng_draws_are_validated(self):
        for invalid in (True, False, 0, -1, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    PRIMARY_EQUAL_STRATA.sample(invalid, np.random.default_rng(1))

        class InvalidRng:
            @staticmethod
            def random(size):
                return np.full(size, 1.0)

        with self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
            PRIMARY_EQUAL_STRATA.sample(2, InvalidRng())

    def test_defined_single_component_regimens_remain_compatible(self):
        # A small explicit tripwire for the two non-combination active profiles.
        self.assertIs(HMA_REGIMEN.survival_component, BATComponent.HMA)
        self.assertIs(LDAC_REGIMEN.survival_component, BATComponent.LDAC)


if __name__ == "__main__":
    unittest.main()
