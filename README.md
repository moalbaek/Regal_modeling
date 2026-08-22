# Regal_modeling

[![CI](https://github.com/moalbaek/regal_modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/moalbaek/regal_modeling/actions/workflows/ci.yml)

A public-information scenario explorer for SELLAS Life Sciences' (NASDAQ: **SLS**) blinded Phase 3
**REGAL** trial (NCT04229979) of galinpepimut-S (GPS) vs best available therapy (BAT) in AML second
complete remission.

> [!IMPORTANT]
> The current Python and HTML engines are the **v1 legacy model**. Their `P(success)` output is a
> fixed-scenario Monte-Carlo rejection rate conditional on the model assumptions and on reaching the
> final analysis. It is **not** a posterior probability for the ongoing REGAL trial and does not
> condition on the observed decision to continue after the 60-event interim. The rebuild is tracked
> in [`V2_IMPLEMENTATION_PLAN.md`](V2_IMPLEMENTATION_PLAN.md).

The blinded death-event milestones (60/72/78) constrain only the *pooled* survival trajectory, so the
split between arms is an explicit assumption. The tool calibrates assumed pooled curve families to
the milestones, decomposes them into arms under user-controlled assumptions, and simulates the
pre-specified final test. It compares a GPS cure-mixture scenario with a bounded no-GPS-cure
alternative using the same BAT arm. Their percentages are fixed-scenario rejection rates. The
alternative's State A/B/C result means boundary fit, residual misfit, or adequate interior fit; it is
a model diagnostic rather than a formal hypothesis test or evidence that any biological mechanism
has been established.

## Files

| File | What it is |
|------|------------|
| `regal_explorer.html` | Self-contained legacy scenario explorer — open in any browser, no build or dependencies. It reports scenario rejection rates and fit diagnostics, plus live survival, event-accrual, HR, enrollment, effect-sweep, and BAT-sensitivity charts. |
| `regal_explorer.py` | The same engine in Python. Prints a summary across the five BAT presets and writes the 9-panel `regal_explorer_panel.png`. Requires `numpy` + `matplotlib`. |
| `REGAL_MODEL_DOCUMENTATION.md` | Full methodology, parameter sourcing, and limitations. |
| `BAT_CONTROL_ARM_RESEARCH.md` | The research basis for the default BAT-arm settings: component composition/weights (US/EU/China) and per-component mixture-cure survival parameters (median OS, cure fraction, Weibull shape), and how they map to `DEFAULT_COMP`. |
| `V2_IMPLEMENTATION_PLAN.md` | Scientific and software roadmap for the conditional, protocol-compatible v2 forecast. |
| `trial_design.py` | Dependency-light canonical two-look O'Brien–Fleming boundary calculation shared by the audit harness. |
| `audit/interim_efficacy_replay.py` | Fixed-seed equal-planned-strata replay of the interim boundary-crossing and final scenario-rejection rates. |

```bash
pip install -r requirements.txt  # numpy + matplotlib (the .html needs nothing)
python3 regal_explorer.py        # CLI summary + figure
# or just open regal_explorer.html in a browser
```

## Development

The Python engine is covered by a golden regression test that pins the fits,
event accrual, and fixed-seed Monte-Carlo rejection rate across all five presets,
plus synthetic fixtures that exercise each bounded no-GPS-cure fit-status branch (State A
upper/heavy boundary and light-edge, State B residual misfit; State C is the presets), so a
change that silently moves a number or flips a fit status fails loudly. CI
(`.github/workflows/ci.yml`) runs it on Python 3.9–3.12 and smoke-tests the CLI.

```bash
python3 -m unittest discover -s tests   # run the golden test
python3 audit/interim_efficacy_replay.py --nsim 10000  # reproduce the equal-strata interim result
python3 tests/gen_golden.py             # regenerate golden.json after an INTENDED change, then review the diff
```

**Research/analysis tool operating entirely on public disclosures. Not investment advice, and it
does not access or estimate the confidential trial outcome.**
