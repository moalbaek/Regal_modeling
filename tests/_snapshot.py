"""Shared golden-snapshot computation for the REGAL engine.

Both the generator (`gen_golden.py`) and the regression test (`test_golden.py`)
import `compute_snapshot()` from here so they measure the engine identically.
The snapshot pins the *deterministic* outputs (the fits, event accrual, and the
no-GPS-cure verdict) plus the fixed-seed Monte-Carlo readouts across all five
BAT-composition presets. Non-finite values (a "not reached" median -> inf, an
undefined interim HR -> nan) are serialised as marker strings so they survive
JSON round-trips.
"""
import math
import os
import sys

# tests/ lives one level below the repo root that holds regal_explorer.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import regal_explorer as R  # noqa: E402

PRESETS = ["base", "low", "dom", "bear", "bull"]
NSIM = 400   # fixed MC budget for reproducible P(success); mc() seeds deterministically

# The five real presets all land in State C, so they never exercise the no-GPS-cure
# verdict's other branches. These synthetic configs — a base scenario with only the
# blinded milestones swapped — deliberately drive the fit to each remaining verdict,
# pinning the categorical A/B/C logic. Each sits well clear of its flip boundary:
#   A_cure_required — milestones stall, so the no-cure Weibull runs to its median cap
#                     AND the heavy tail edge (cure required, cureReq=True).
#   A_light_edge    — bunched milestones want an ever-lighter (increasing-hazard) tail,
#                     so sG pins at the light edge (non-identified, cureReq=False).
#   B_inconsistent  — a mid-burst then stall no single Weibull shape can hit: the best
#                     fit is interior yet the residual clears the tolerance (inconsistent).
VERDICT_FIXTURES = {
    "A_cure_required": {"ev_counts": [70, 72, 73]},
    "A_light_edge": {"ev_dates": [(2024, 12, 10), (2025, 3, 26), (2025, 5, 11)]},
    "B_inconsistent": {"ev_counts": [50, 76, 78]},
}


def num(x):
    """JSON-safe number: finite -> float, else a marker string for inf/-inf/nan."""
    if isinstance(x, bool):
        return x
    f = float(x)
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    if math.isnan(f):
        return "nan"
    return f


def _plateau_row(cfg):
    Mc = R.build_plateau(cfg)
    rc = R.mc(Mc, NSIM)
    return {
        "pibat": num(Mc["pibat"]), "presp": num(Mc["presp"]),
        "pgps": num(Mc["pgps"]), "poolCure": num(Mc["poolCure"]),
        "batMed": num(Mc["batMed"]), "gpsMed": num(Mc["gpsMed"]),
        "poolMed": num(Mc["poolMed"]),
        "edv": [num(Mc["ed"](t)) for t in Mc["MT"]],
        "ps": num(rc["ps"]), "reach": num(rc["reach"]),
        "medHR": num(rc["medHR"]), "medHR_IA": num(rc["medHR_IA"]),
        "aliveG": num(rc["aliveG"]), "aliveB": num(rc["aliveB"]),
    }


def _nogpscure_row(cfg):
    Ml = R.build_no_gps_cure(cfg)
    rl = R.mc(Ml, NSIM)
    return {
        "state": Ml["state"], "cureReq": bool(Ml["cureReq"]),
        "mG": num(Ml["mG"]), "sG": num(Ml["sG"]),
        "rmsResid": num(Ml["rmsResid"]), "ratio": num(Ml["ratio"]),
        "edv": [num(x) for x in Ml["edv"]],
        "ps": num(rl["ps"]), "reach": num(rl["reach"]), "medHR": num(rl["medHR"]),
    }


def _fixture_cfg(spec):
    """Base preset with only the blinded milestones (`ev`) overridden per the fixture."""
    cfg = R.apply_preset(R.default_cfg(), "base")
    ev = [dict(e) for e in cfg["ev"]]
    for i, n in enumerate(spec.get("ev_counts", [])):
        ev[i]["n"] = n
    for i, (y, m, d) in enumerate(spec.get("ev_dates", [])):
        ev[i].update(y=y, m=m, d=d)
    cfg["ev"] = ev
    return cfg


def _verdict_row(cfg):
    """The no-GPS-cure verdict fields only — what a State A/B/C fixture pins."""
    Ml = R.build_no_gps_cure(cfg)
    return {
        "state": Ml["state"], "cureReq": bool(Ml["cureReq"]),
        "mG": num(Ml["mG"]), "sG": num(Ml["sG"]), "rmsResid": num(Ml["rmsResid"]),
    }


def compute_snapshot():
    """Return {"presets": {name: {"plateau", "nogpscure"}}, "verdicts": {label: ...}}."""
    presets = {}
    for name in PRESETS:
        cfg = R.apply_preset(R.default_cfg(), name)
        presets[name] = {
            "plateau": _plateau_row(cfg),
            "nogpscure": _nogpscure_row(cfg),
        }
    verdicts = {label: _verdict_row(_fixture_cfg(spec))
                for label, spec in VERDICT_FIXTURES.items()}
    return {"presets": presets, "verdicts": verdicts}
