"""Golden regression test for the REGAL Python engine.

Pins the deterministic fits, event accrual, no-GPS-cure verdict, and fixed-seed
Monte-Carlo readouts across all five presets to values captured in golden.json.
A change that shifts P(success), a median, or a verdict fails here — regenerate
with `python3 tests/gen_golden.py` and inspect the diff when the shift is intended.

Runs under either `python -m unittest discover tests` or `pytest`.
"""
import json
import math
import os
import unittest

from _snapshot import compute_snapshot

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden.json")

# Per-field absolute tolerances. Loose enough to absorb cross-platform float
# noise in exp/log and the argmin grid search; tight enough that a real change
# (a shifted fit, a flipped verdict, a P(success) move) still trips the test.
TOL = {
    # probabilities / fractions
    "pibat": 0.03, "presp": 0.03, "pgps": 0.03, "poolCure": 0.03,
    "ps": 0.03, "reach": 0.03,
    # months / event counts / patient counts
    "batMed": 1.5, "gpsMed": 2.5, "poolMed": 1.5, "mG": 2.5,
    "edv": 1.5, "aliveG": 1.5, "aliveB": 1.5,
    # hazard ratios / Weibull shape / residual / ratio
    "medHR": 0.05, "medHR_IA": 0.05, "sG": 0.05, "rmsResid": 0.5, "ratio": 0.5,
}
DEFAULT_TOL = 0.05
EXACT = {"state", "cureReq"}   # categorical: must match exactly


def _is_marker(v):
    return isinstance(v, str) and v in ("inf", "-inf", "nan")


class GoldenSnapshotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(GOLDEN) as fh:
            cls.golden = json.load(fh)
        cls.actual = compute_snapshot()

    def _check(self, path, key, gv, av):
        if key in EXACT or isinstance(gv, str) and not _is_marker(gv):
            self.assertEqual(gv, av, f"{path}: {gv!r} != {av!r}")
            return
        if _is_marker(gv):
            # golden is inf/-inf/nan -> recomputed value must be the same kind
            self.assertTrue(
                _is_marker(av) and av == gv,
                f"{path}: expected non-finite {gv!r}, got {av!r}",
            )
            return
        # golden is a finite number -> recomputed must be finite and within tol
        self.assertFalse(_is_marker(av), f"{path}: expected finite {gv}, got {av!r}")
        tol = TOL.get(key, DEFAULT_TOL)
        self.assertTrue(
            math.isclose(gv, av, abs_tol=tol),
            f"{path}: {av} not within +/-{tol} of golden {gv}",
        )

    def test_presets_match_golden(self):
        self.assertEqual(
            sorted(self.golden), sorted(self.actual),
            "preset set changed vs golden.json",
        )
        for preset, panels in self.golden.items():
            for panel, fields in panels.items():
                for key, gv in fields.items():
                    av = self.actual[preset][panel][key]
                    path = f"{preset}.{panel}.{key}"
                    if isinstance(gv, list):
                        self.assertEqual(len(gv), len(av), f"{path}: length differs")
                        for i, (g, a) in enumerate(zip(gv, av)):
                            self._check(f"{path}[{i}]", key, g, a)
                    else:
                        self._check(path, key, gv, av)


if __name__ == "__main__":
    unittest.main()
