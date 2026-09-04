"""Reproduce the v1 equal-strata interim-efficacy sensitivity.

This is an operating-characteristic replay, not conditioning on REGAL's observed
continuation and not a forecast for the ongoing trial.
"""

import argparse
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import regal_explorer as regal  # noqa: E402
from trial_design import obrien_fleming_two_look  # noqa: E402


def equal_strata_config():
    """Four 25% planned BAT strata; preserve the v1 BSC 27:8 internal split."""

    cfg = regal.default_cfg()
    supportive_total = 25.0
    supportive_split = 27.0 + 8.0
    weights = [
        supportive_total * 27.0 / supportive_split,
        supportive_total * 8.0 / supportive_split,
        25.0,
        25.0,
        25.0,
    ]
    for component, weight in zip(cfg["comp"], weights):
        component["w"] = weight
    return cfg


def replay(nsim=10000, seed=987654321):
    cfg = equal_strata_config()
    model = regal.build_plateau(cfg)
    result = regal.mc(model, nsim=nsim, seed=seed)
    boundary = obrien_fleming_two_look(0.025, cfg["IA"] / cfg["FINAL"])
    return {
        "interpretation": "v1 fixed-scenario operating characteristic",
        "boundary_variant": "classical two-look O'Brien-Fleming; not Lan-DeMets spending",
        "nsim": nsim,
        "seed": seed,
        "bat_weights": [component["w"] for component in cfg["comp"]],
        "interim_information": cfg["IA"] / cfg["FINAL"],
        "interim_efficacy_z": boundary["interim_z"],
        "final_efficacy_z": boundary["final_z"],
        "interim_efficacy_crossing": result["p_IA_efficacy"],
        "interim_reach": result["reach_IA"],
        "final_scenario_rejection_rate_given_reach": result["ps"],
        "final_reach": result["reach"],
        "median_final_hr": result["medHR"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsim", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=987654321)
    args = parser.parse_args()
    print(json.dumps(replay(args.nsim, args.seed), indent=2))


if __name__ == "__main__":
    main()
