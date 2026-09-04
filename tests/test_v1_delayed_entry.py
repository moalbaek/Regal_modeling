"""Invariants for v1's CR2 -> randomization delayed-entry layer.

REGAL randomizes patients months after they achieve CR2: the last re-induction dose must be
at least 4 weeks earlier (inclusion 7) and consent must fall within 6 months of CR2
(inclusion 8). Patients who relapse or die inside that window never enrol.

That is a genuine left-truncation, but on the CR2 clock rather than the trial clock, which is
what separates it from v1's old outcome clip: the truncation point lands at trial-clock t = 0,
so no guarantee interval appears. These tests pin that distinction.

Two caveats these tests also pin, because both are easy to lose track of:

* The layer conditions on OVERALL survival, not on remaining in CR2. The component library
  carries no relapse hazard, so a patient who relapses inside the window but survives it is
  retained. Relapse-free survival is below overall survival, so the true denominator is smaller
  and the real cohort is healthier: this is a death-only LOWER bound on the enrichment.
* Uniform[1, 6] is an analyst assumption, not a protocol quantity. Inclusion 7 runs on the
  re-induction-dose clock rather than the CR2 clock and sets no floor after CR2; inclusion 8
  gives only an upper bound. The floor is material, so it is pinned as material below.
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import regal_explorer as R  # noqa: E402


COMPONENTS = [(6.0, 0.03, 1.1), (12.0, 0.15, 0.78), (7.0, 0.08, 1.1), (5.0, 0.02, 1.1)]
THETA = 0.53
Q = 0.25
WINDOW = (1.0, 6.0)


class DelayGridTest(unittest.TestCase):
    def test_grid_preserves_the_mean_and_variance_of_the_window(self):
        """Same moment-matched construction as the WP5 reporting-lag PMF."""
        a, b = WINDOW
        grid, weights = R.dgrid(a, b)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=12)
        mean = float(np.sum(weights * grid))
        var = float(np.sum(weights * grid**2) - mean**2)
        self.assertAlmostEqual(mean, 0.5 * (a + b), places=12)
        self.assertAlmostEqual(var, (b - a) ** 2 / 12.0, places=12)

    def test_a_disabled_window_collapses_to_a_single_zero_delay_point(self):
        """dmax=0 must disable the layer outright, whatever dmin is left at.

        The default config carries dmin=1, so turning the window off by setting dmax=0 alone
        used to leave a fixed one-month delay in place rather than no delay at all. The upper
        bound now dominates, and the surviving point must actually be zero -- asserting only
        that the grid has one point would pass for any fixed delay.
        """
        for spec in ((0.0, 0.0), (1.0, 0.0), (6.0, 0.0)):
            with self.subTest(spec=spec):
                grid, weights = R.dgrid(*spec)
                self.assertEqual(len(grid), 1)
                self.assertEqual(float(grid[0]), 0.0)
                self.assertAlmostEqual(float(weights[0]), 1.0, places=12)

    def test_a_zero_width_window_collapses_to_its_own_fixed_delay(self):
        """A degenerate but non-zero window is a fixed delay, not a disabled layer."""
        for spec, point in (((5.0, 5.0), 5.0), ((2.0, 2.0), 2.0)):
            with self.subTest(spec=spec):
                grid, weights = R.dgrid(*spec)
                self.assertEqual(len(grid), 1)
                self.assertEqual(float(grid[0]), point)
                self.assertAlmostEqual(float(weights[0]), 1.0, places=12)

    def test_turning_the_window_off_by_dmax_alone_matches_the_frailty_only_arm(self):
        """The config-level path, not just dgrid: dmax=0 with the default dmin=1 is a no-op."""
        off = R.bat_arm(R.default_cfg(dmax=0.0))
        both = R.bat_arm(R.default_cfg(dmin=0.0, dmax=0.0))
        self.assertAlmostEqual(off["pibat"], both["pibat"], places=12)
        for probe in (1.0, 6.0, 24.0, 60.0):
            self.assertAlmostEqual(
                float(off["Sbat"](probe)), float(both["Sbat"](probe)), places=12
            )


class DelayAssumptionTest(unittest.TestCase):
    """The delay window is an analyst choice; these pin what that choice actually buys."""

    def test_the_lower_bound_is_a_material_assumption(self):
        """dmin is not identified by the protocol, so it must not be silently load-bearing.

        Inclusion 7's four-week rule is measured from the last re-induction dose, not from CR2,
        so nothing establishes a one-month floor on the CR2 clock. Dropping the floor to zero
        while holding the six-month bound moves the BAT median by around half a month, which is
        the same order as the effects this model is fitted to resolve. If a future change makes
        the floor immaterial, that is a real finding about the model and this test should be
        updated deliberately rather than deleted.
        """
        floored = R.median(R.bat_arm(R.default_cfg(dmin=1.0))["Sbat"])
        unfloored = R.median(R.bat_arm(R.default_cfg(dmin=0.0))["Sbat"])
        self.assertGreater(floored, unfloored)
        self.assertGreater(floored - unfloored, 0.25)

    def test_the_window_is_monotone_in_both_bounds(self):
        """A longer window screens more, so enrichment must not fall as either bound rises."""
        med = lambda **o: R.median(R.bat_arm(R.default_cfg(**o))["Sbat"])  # noqa: E731
        for lo, hi in zip((0.0, 0.5, 1.0, 2.0), (0.5, 1.0, 2.0, 3.0)):
            self.assertGreaterEqual(med(dmin=hi), med(dmin=lo))
        for lo, hi in zip((2.0, 4.0, 6.0, 9.0), (4.0, 6.0, 9.0, 12.0)):
            self.assertGreaterEqual(med(dmin=0.0, dmax=hi), med(dmin=0.0, dmax=lo))

    def test_the_layer_conditions_on_death_only_not_on_staying_in_CR2(self):
        """Pins the approximation as an approximation: the denominator is OS, not RFS.

        A relapse hazard would push the entry-survival denominator strictly below the
        overall-survival one, raising the cure lift. Emulating that with an inflated hazard
        shows the direction: the death-only denominator is an upper bound on the true one, so
        the modelled enrichment is a lower bound on the true enrichment.
        """
        med, cure, k = 12.0, 0.15, 0.78
        grid, weights = R.dgrid(*WINDOW)
        os_denom = R.entry_survival(med, cure, k, THETA, Q, grid, weights)
        # A shorter median stands in for "relapse-free survival is below overall survival".
        rfs_denom = R.entry_survival(0.5 * med, cure, k, THETA, Q, grid, weights)
        self.assertLess(rfs_denom, os_denom)
        self.assertGreater(cure / rfs_denom, cure / os_denom)


class DelayedEntryTest(unittest.TestCase):
    def setUp(self):
        self.grid, self.weights = R.dgrid(*WINDOW)

    def _sdel(self, t, med, cure, k, q=Q):
        return R.Sdel(t, med, cure, k, THETA, q, self.grid, self.weights)

    def test_disabling_the_window_reproduces_the_frailty_only_curve_exactly(self):
        ts = np.linspace(0.0, 120.0, 1201)
        off, off_w = R.dgrid(0.0, 0.0)
        for med, cure, k in COMPONENTS:
            with self.subTest(med=med):
                np.testing.assert_allclose(
                    R.Sdel(ts, med, cure, k, THETA, Q, off, off_w),
                    R.Scf(ts, med, cure, k, THETA, Q),
                    rtol=0.0,
                    atol=1e-15,
                )

    def test_survival_starts_at_one_on_the_trial_clock(self):
        """The truncation sits at t = 0, so there is no jump and no guarantee interval."""
        for med, cure, k in COMPONENTS:
            with self.subTest(med=med):
                self.assertAlmostEqual(float(self._sdel(0.0, med, cure, k)), 1.0, places=12)
                self.assertLess(float(self._sdel(0.05, med, cure, k)), 1.0)

    def test_no_flat_segment_and_no_corner(self):
        t = np.linspace(0.001, 90.0, 60000)
        for med, cure, k in COMPONENTS[:2]:
            with self.subTest(med=med):
                d1 = np.diff(self._sdel(t, med, cure, k))
                self.assertTrue(np.all(d1 < 0.0))
                ratio = d1[1:] / d1[:-1]
                self.assertLess(float(np.max(np.abs(ratio - 1.0))), 0.25)

    def test_window_survivors_are_enriched(self):
        """Conditioning on reaching randomization can only help the enrolled cohort."""
        for med, cure, k in COMPONENTS:
            with self.subTest(med=med):
                plain = R.median(lambda t: R.Scf(t, med, cure, k, THETA, Q))
                delayed = R.median(lambda t: self._sdel(t, med, cure, k))
                self.assertGreater(delayed, plain)

    def test_cure_fraction_rises_by_exactly_the_window_survival(self):
        """Uncured patients die during the window, so survivors over-represent cured ones."""
        for med, cure, k in COMPONENTS:
            with self.subTest(med=med):
                denom = R.entry_survival(med, cure, k, THETA, Q, self.grid, self.weights)
                self.assertLess(denom, 1.0)
                self.assertAlmostEqual(
                    float(self._sdel(1e12, med, cure, k)), cure / denom, delta=1e-6
                )

    def test_a_memoryless_component_is_barely_enriched_by_the_window(self):
        """Delayed entry enriches only through heterogeneity.

        With no frailty spread, no cure fraction and shape k = 1, survival is exponential and
        therefore memoryless: surviving the entry window carries no prognostic information, so
        the enrolled curve must be unchanged. Anything else would be enrichment from nowhere.
        """
        ts = np.linspace(0.0, 60.0, 601)
        np.testing.assert_allclose(
            R.Sdel(ts, 9.0, 0.0, 1.0, 0.0, Q, self.grid, self.weights),
            R.Scf(ts, 9.0, 0.0, 1.0, 0.0, Q),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_sampled_times_match_the_analytic_delayed_curve(self):
        """The exact draw and the closed form must describe one model."""
        rng = np.random.default_rng(20260825)
        n = 400000
        for med, cure, k in COMPONENTS[:3]:
            with self.subTest(med=med):
                t = R.sampdel(n, med, cure, k, THETA, Q, self.grid, self.weights, rng)
                self.assertTrue(np.all(t > 0.0), "negative trial-clock survival time")
                for probe in (1.0, 6.0, 12.0, 24.0):
                    self.assertAlmostEqual(
                        float(np.mean(t > probe)),
                        float(self._sdel(probe, med, cure, k)),
                        delta=0.005,
                    )

    def test_uncured_only_draw_matches_the_conditional_curve(self):
        """The plateau panel's responder mixes cure with this non-cured shape."""
        rng = np.random.default_rng(4242)
        n = 400000
        for med, cure, k in COMPONENTS[:2]:
            with self.subTest(med=med):
                t = R.sampdel(
                    n, med, cure, k, THETA, Q, self.grid, self.weights, rng,
                    uncured_only=True,
                )
                self.assertTrue(np.all(np.isfinite(t)))
                denom = R.entry_survival(med, cure, k, THETA, Q, self.grid, self.weights)
                enrolled_cure = cure / denom
                for probe in (1.0, 6.0, 12.0, 24.0):
                    analytic = (
                        float(self._sdel(probe, med, cure, k)) - enrolled_cure
                    ) / (1.0 - enrolled_cure)
                    self.assertAlmostEqual(float(np.mean(t > probe)), analytic, delta=0.006)

    def test_bat_arm_sampler_reproduces_the_analytic_bat_curve(self):
        """End-to-end: the committed default config's engine and curves agree."""
        cfg = R.default_cfg()
        arm = R.bat_arm(cfg)
        grid, weights = R.dgrid(cfg["dmin"], cfg["dmax"])
        rng = np.random.default_rng(99)
        n = 400000
        pick = rng.choice(len(arm["cm"]), size=n, p=arm["w"])
        draws = np.empty(n)
        for j, comp in enumerate(arm["cm"]):
            idx = np.flatnonzero(pick == j)
            draws[idx] = R.sampdel(
                idx.size, comp["med"], comp["cure"], comp["k"],
                cfg["fvar"], cfg["esel"], grid, weights, rng,
            )
        for probe in (1.0, 6.0, 12.0, 24.0, 48.0):
            self.assertAlmostEqual(
                float(np.mean(draws > probe)), float(arm["Sbat"](probe)), delta=0.005
            )
        self.assertAlmostEqual(float(np.mean(draws > 1e8)), arm["pibat"], delta=0.005)


if __name__ == "__main__":
    unittest.main()
