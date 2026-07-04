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


def compute_snapshot():
    """Return the full {preset: {"plateau": ..., "nogpscure": ...}} snapshot."""
    snap = {}
    for name in PRESETS:
        cfg = R.apply_preset(R.default_cfg(), name)
        snap[name] = {
            "plateau": _plateau_row(cfg),
            "nogpscure": _nogpscure_row(cfg),
        }
    return snap
