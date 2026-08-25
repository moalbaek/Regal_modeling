"""Invariants for v1's gamma-frailty eligibility selection.

V1 originally modelled trial eligibility as a hard left-truncation of the death-time
distribution, ``min(1, S(t)/(1-q))``. That conditions on the patient's *realized* survival,
which no screening process can do: it produced a zero-hazard guarantee interval, a visible
corner where the clip released, and a cure fraction mechanically inflated to ``c/(1-q)``.

Selection now screens on baseline frailty instead. These tests pin the properties that make
that a legitimate selection model, so a regression back to outcome-conditioning fails loudly.
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import regal_explorer as R  # noqa: E402


COMPONENTS = [(6.0, 0.03, 1.1), (12.0, 0.15, 0.78), (7.0, 0.08, 1.1), (5.0, 0.02, 1.1)]
THETA = 1.43


class FrailtySelectionTest(unittest.TestCase):
    def test_zero_frailty_variance_reproduces_the_unselected_curve_exactly(self):
        """With no prognostic spread there is nothing to select on, at any q."""
        ts = np.linspace(0.0, 120.0, 2401)
        for med, cure, k in COMPONENTS:
            for q in (0.0, 0.25, 0.5):
                with self.subTest(med=med, q=q):
                    np.testing.assert_array_equal(
                        R.Scf(ts, med, cure, k, 0.0, q), R.Sc(ts, med, cure, k)
                    )

    def test_zero_selection_reproduces_the_published_component_median(self):
        """lambda is anchored on the population marginal, so q=0 is the face-value curve.

        Anchoring the conditional (frailty=1) curve instead would double-count the
        heterogeneity already present in the published KM fits.
        """
        for med, cure, k in COMPONENTS:
            with self.subTest(med=med):
                got = R.median(lambda t: R.Scf(t, med, cure, k, THETA, 0.0))
                self.assertAlmostEqual(got, med, places=6)

    def test_selection_never_creates_a_guarantee_interval(self):
        """The old clip pinned S(t)=1 out to the q-quantile; hazard must now be positive at 0."""
        for med, cure, k in COMPONENTS:
            for q in (0.1, 0.25, 0.5):
                with self.subTest(med=med, q=q):
                    S = lambda t: R.Scf(t, med, cure, k, THETA, q)  # noqa: E731
                    self.assertLess(S(0.05), S(0.0))
                    self.assertLess(S(1.0), S(0.05))

    def test_survival_is_smooth_with_no_corner(self):
        """The clip held S flat, then released at a point: a first-derivative discontinuity.

        Both assertions have teeth against the old mechanism. On this grid the clipped curve
        held ~15k consecutive zero first differences before releasing, so it fails the
        strictly-decreasing check outright and its derivative ratio jumps at the release.
        """
        t = np.linspace(0.001, 90.0, 60000)
        d1 = np.diff(R.Scf(t, 48.0, 0.0, 1.15, THETA, 0.25))
        self.assertTrue(np.all(d1 < 0.0), "selected curve has a flat (zero-hazard) segment")
        ratio = d1[1:] / d1[:-1]
        self.assertLess(float(np.max(np.abs(ratio - 1.0))), 0.25)

    def test_selection_does_not_change_the_cure_fraction(self):
        """Screening cannot convert an uncured patient into a cured one.

        The frailty mixture has a polynomial rather than exponential tail, so the plateau is
        approached more slowly than under the plain Weibull; probe far enough out to separate
        that from any actual shift in the cure fraction.
        """
        for med, cure, k in COMPONENTS:
            with self.subTest(med=med):
                for q in (0.0, 0.25, 0.5):
                    self.assertAlmostEqual(
                        float(R.Scf(1e12, med, cure, k, THETA, q)), cure, delta=1e-6
                    )

    def test_bat_cure_fraction_is_flat_in_q(self):
        """pibat used to rise as c/(1-q); it must now be the plain weighted cure."""
        expected = None
        for q in (0.0, 0.2, 0.4):
            arm = R.bat_arm(R.default_cfg(esel=q))
            weighted = sum(
                arm["w"][i] * arm["cm"][i]["cure"] for i in range(len(arm["cm"]))
            )
            self.assertAlmostEqual(arm["pibat"], weighted, places=12)
            if expected is None:
                expected = arm["pibat"]
            self.assertAlmostEqual(arm["pibat"], expected, places=12)

    def test_selection_is_monotone_and_raises_the_median(self):
        """More screening enriches the cohort; it may never make it worse."""
        med, cure, k = 6.0, 0.03, 1.1
        medians = [
            R.median(lambda t, q=q: R.Scf(t, med, cure, k, THETA, q))
            for q in (0.0, 0.15, 0.3, 0.45)
        ]
        for lo, hi in zip(medians, medians[1:]):
            self.assertGreater(hi, lo)

    def test_sampled_event_times_match_the_analytic_selected_curve(self):
        """The Monte-Carlo draw and the closed-form curve must describe one model."""
        rng = np.random.default_rng(20260825)
        n = 400000
        for med, cure, k in COMPONENTS[:2]:
            for q in (0.0, 0.3):
                with self.subTest(med=med, q=q):
                    z = R.sampf(n, THETA, q, rng)
                    t = R.sampNCf(med, cure, k, THETA, rng.random(n), z)
                    cured = rng.random(n) < cure
                    t = np.where(cured, np.inf, t)
                    for probe in (3.0, 6.0, 12.0, 24.0):
                        empirical = float(np.mean(t > probe))
                        analytic = float(R.Scf(probe, med, cure, k, THETA, q))
                        self.assertAlmostEqual(empirical, analytic, delta=0.005)

    def test_enrolled_frailty_mean_matches_the_closed_form(self):
        """E[Z | eligible] = (1-q)^theta is what makes the gamma tilt conjugate."""
        rng = np.random.default_rng(7)
        for q in (0.0, 0.25, 0.5):
            with self.subTest(q=q):
                z = R.sampf(500000, THETA, q, rng)
                self.assertAlmostEqual(
                    float(z.mean()), (1.0 - q) ** THETA, delta=0.01
                )


if __name__ == "__main__":
    unittest.main()
