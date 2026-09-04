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
# Keep mechanism tests at a stronger heterogeneity stress value. The shipped default is pinned
# independently below, so changing a tunable default cannot silently weaken these invariants.
THETA = 1.43


class FrailtySelectionTest(unittest.TestCase):
    def test_evidence_informed_defaults(self):
        cfg = R.default_cfg()
        self.assertEqual(cfg["esel"], 0.20)
        self.assertEqual(cfg["fvar"], 0.35)
        self.assertAlmostEqual((1.0 - cfg["esel"]) ** cfg["fvar"], 0.9248717100187196)

    def test_zero_frailty_variance_reproduces_the_unselected_curve_exactly(self):
        """With no prognostic spread there is nothing to select on, at any q."""
        ts = np.linspace(0.0, 120.0, 2401)
        for med, cure, k in COMPONENTS:
            for q in (0.0, 0.25, 0.5):
                with self.subTest(med=med, q=q):
                    np.testing.assert_array_equal(
                        R.Scf(ts, med, cure, k, 0.0, q), R.Sc(ts, med, cure, k)
                    )

    def test_the_small_theta_limit_is_numerically_continuous(self):
        """theta -> 0 must degrade gracefully, not fall off a numerical cliff.

        lamf's coefficient (A^-theta - 1)/theta and Scf's (1+x)^(-1/theta) both evaluate to a
        rounding-limited constant once theta nears machine epsilon: the first collapsed the scale
        to zero and raised ZeroDivisionError at theta=1e-16, the second returned S=1 at every t.
        The expm1/log1p forms converge to the unselected curve instead.
        """
        ts = np.array([0.5, 3.0, 6.0, 24.0, 96.0])
        for med, cure, k in COMPONENTS:
            base = R.Sc(ts, med, cure, k)
            unselected_scale = R.lam(med, cure, k)
            for theta in (1e-16, 1e-14, 1e-12, 1e-10, 1e-8):
                for q in (0.0, 0.25, 0.5):
                    with self.subTest(med=med, theta=theta, q=q):
                        self.assertAlmostEqual(
                            R.lamf(med, cure, k, theta), unselected_scale, places=6
                        )
                        np.testing.assert_allclose(
                            R.Scf(ts, med, cure, k, theta, q), base, rtol=1e-6, atol=1e-9
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
        """The old clip pinned S(t)=1 out to the q-quantile; S must now fall for every t>0.

        The invariant is the absence of a positive-length flat segment, not the hazard VALUE at
        the origin: that value is infinite for Weibull shape k<1 and zero for k>1, so "positive
        and finite at 0" would be false for the shapes this model actually uses.
        """
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

    def test_frailty_screening_alone_does_not_move_the_cure_fraction(self):
        """With the entry window off, pibat is the plain weighted cure at every q.

        The old clip raised it as c/(1-q) by conditioning on post-randomization survival.
        Screening on baseline frailty cannot do that: it rescales the uncured hazard only.
        """
        expected = None
        for q in (0.0, 0.2, 0.4):
            arm = R.bat_arm(R.default_cfg(esel=q, dmin=0.0, dmax=0.0))
            weighted = sum(
                arm["w"][i] * arm["cm"][i]["cure"] for i in range(len(arm["cm"]))
            )
            self.assertAlmostEqual(arm["pibat"], weighted, places=12)
            if expected is None:
                expected = arm["pibat"]
            self.assertAlmostEqual(arm["pibat"], expected, places=12)

    def test_entry_window_raises_the_cure_fraction_by_the_survival_denominator(self):
        """Delayed entry legitimately enriches for cured patients: c / E_D[S(D)].

        This looks like the old clip's c/(1-q) inflation but is a different quantity: it
        conditions on surviving BEFORE enrolment, which screening actually observes, rather
        than on post-randomization survival, which it cannot.
        """
        cfg = R.default_cfg()
        arm = R.bat_arm(cfg)
        grid, weights = R.dgrid(cfg["dmin"], cfg["dmax"])
        expected = sum(
            arm["w"][i]
            * arm["cm"][i]["cure"]
            / R.entry_survival(
                arm["cm"][i]["med"], arm["cm"][i]["cure"], arm["cm"][i]["k"],
                cfg["fvar"], cfg["esel"], grid, weights,
            )
            for i in range(len(arm["cm"]))
        )
        self.assertAlmostEqual(arm["pibat"], expected, places=12)
        plain = sum(arm["w"][i] * arm["cm"][i]["cure"] for i in range(len(arm["cm"])))
        self.assertGreater(arm["pibat"], plain)

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
