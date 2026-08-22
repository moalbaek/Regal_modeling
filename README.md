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
split between arms is an explicit assumption. The tool calibrates an assumed pooled curve family to the milestones,
decomposes it into arms under user-controlled assumptions, and Monte-Carlo simulates the trial's
pre-specified Cox/log-rank test. The **headline** is the **plateau (GPS-cure)** probability of
success. The **second panel is a null test, not a co-equal probability**: it holds the BAT arm
*bit-for-bit identical* and swaps only the GPS **responder** component — a durable-remission cure
versus a fitted heavy-tailed Weibull with **no cure** — to ask whether the milestone plateau
*requires* a GPS-specific durable benefit. It returns a three-state verdict: null **rejected**
(non-identified — GPS cure required), rejected (inconsistent), or **not excluded** (a no-cure GPS
heavy tail also fits, given this BAT). Only the "not excluded" state carries a second P(success), and
it is *conditional on crediting BAT* at the chosen medians/cures — the bear presets and selection
slider are the intended stress controls.

## Files

| File | What it is |
|------|------------|
| `regal_explorer.html` | Self-contained interactive explorer — open in any browser, no build or dependencies. Sliders for BAT composition, enrollment selection (eligibility filter, a left-truncation), non-responder fraction, natural (non-disease) death rate, the no-GPS-cure test's GPS tail shape sG (fitted by default, with a manual override), etc.; the plateau P(success), the no-GPS-cure verdict, plus live charts (survival curves, event-accrual timeline, simulated-HR distribution, GPS-cure-vs-no-GPS-cure divergence, enrollment validation, a P(success)-vs-effect power curve, and a BAT-median-&-cure-vs-selection sweep). |
| `regal_explorer.py` | The same engine in Python. Prints a summary across the five BAT presets and writes the 9-panel `regal_explorer_panel.png`. Requires `numpy` + `matplotlib`. |
| `REGAL_MODEL_DOCUMENTATION.md` | Full methodology, parameter sourcing, and limitations. |
| `BAT_CONTROL_ARM_RESEARCH.md` | The research basis for the default BAT-arm settings: component composition/weights (US/EU/China) and per-component mixture-cure survival parameters (median OS, cure fraction, Weibull shape), and how they map to `DEFAULT_COMP`. |
| `V2_IMPLEMENTATION_PLAN.md` | Scientific and software roadmap for the conditional, protocol-compatible v2 forecast. |

```bash
pip install -r requirements.txt  # numpy + matplotlib (the .html needs nothing)
python3 regal_explorer.py        # CLI summary + figure
# or just open regal_explorer.html in a browser
```

## Development

The Python engine is covered by a golden regression test that pins the fits,
event accrual, and fixed-seed Monte-Carlo P(success) across all five presets,
plus synthetic fixtures that exercise each no-GPS-cure verdict branch (State A
cure-required and light-edge, State B inconsistent; State C is the presets), so a
change that silently moves a number or flips a verdict fails loudly. CI
(`.github/workflows/ci.yml`) runs it on Python 3.9–3.12 and smoke-tests the CLI.

```bash
python3 -m unittest discover -s tests   # run the golden test
python3 tests/gen_golden.py             # regenerate golden.json after an INTENDED change, then review the diff
```

**Research/analysis tool operating entirely on public disclosures. Not investment advice, and it
does not access or estimate the confidential trial outcome.**
