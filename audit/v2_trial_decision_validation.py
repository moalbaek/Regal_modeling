"""Reproduce the REGAL v2 trial-decision validation report.

The report validates Lan-DeMets efficacy spending, null type-I error in both the
canonical and patient-level event-driven paths, mutually exclusive branch
conservation, and an explicit sensitivity grid for the unpublished futility
rule.  It does not condition on REGAL's observed continuation and is not a
forecast.
"""

import argparse
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simulation import (  # noqa: E402
    FUTILITY_HR_SENSITIVITY_GRID,
    REGAL_V2_EFFICACY_DESIGN,
    simulate_futility_sensitivity_grid,
    simulate_patient_level_exponential_null,
)
from trial_design import (  # noqa: E402
    lan_demets_obrien_fleming_two_look,
    obrien_fleming_two_look,
)


def validation_report(
    nsim=200000,
    seed=20260823,
    final_z_mean=0.0,
    patient_nsim=5000,
    patient_seed=20260824,
):
    design = REGAL_V2_EFFICACY_DESIGN
    protocol = lan_demets_obrien_fleming_two_look(
        design.alpha, design.interim_information
    )
    legacy = obrien_fleming_two_look(
        design.alpha, design.interim_information
    )
    rows = simulate_futility_sensitivity_grid(
        thresholds=FUTILITY_HR_SENSITIVITY_GRID,
        design=design,
        nsim=nsim,
        seed=seed,
        final_z_mean=final_z_mean,
    )
    patient_null = simulate_patient_level_exponential_null(
        design=design,
        nsim=patient_nsim,
        seed=patient_seed,
    )
    return {
        "interpretation": "v2 canonical trial-decision operating characteristic",
        "conditioning": "unconditional; does not use REGAL's observed continuation",
        "forecast": False,
        "nsim": nsim,
        "seed": seed,
        "final_z_mean": final_z_mean,
        "design": {
            "interim_events": design.interim_events,
            "final_events": design.final_events,
            "interim_information": design.interim_information,
            "one_sided_alpha": design.alpha,
            "protocol_efficacy_boundaries": protocol,
            "legacy_classical_audit_boundaries": legacy,
        },
        "futility_rule_status": (
            "unpublished; rows are assumed one-step-HR thresholds and None disables futility"
        ),
        "futility_sensitivity": [row.as_dict() for row in rows],
        "patient_level_null_validation": patient_null.as_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsim", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--final-z-mean", type=float, default=0.0)
    parser.add_argument("--patient-nsim", type=int, default=5000)
    parser.add_argument("--patient-seed", type=int, default=20260824)
    args = parser.parse_args()
    print(
        json.dumps(
            validation_report(
                args.nsim,
                args.seed,
                args.final_z_mean,
                args.patient_nsim,
                args.patient_seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
