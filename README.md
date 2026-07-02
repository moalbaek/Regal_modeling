# Regal_modeling

A public-information forecasting tool for the outcome of SELLAS Life Sciences' (NASDAQ: **SLS**)
blinded Phase 3 **REGAL** trial (NCT04229979) of galinpepimut-S (GPS) vs best available therapy
(BAT) in AML second complete remission.

The blinded death-event milestones (60/72/78) pin only the *pooled* survival curve, so the split
between arms is an explicit assumption. The tool calibrates the pooled curve to the milestones,
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
| `regal_explorer.py` | The same engine in Python. Prints a summary across the four BAT presets and writes the 9-panel `regal_explorer_panel.png`. Requires `numpy` + `matplotlib`. |
| `REGAL_MODEL_DOCUMENTATION.md` | Full methodology, parameter sourcing, and limitations. |

```bash
python3 regal_explorer.py        # CLI summary + figure
# or just open regal_explorer.html in a browser
```

**Research/analysis tool operating entirely on public disclosures. Not investment advice, and it
does not access or estimate the confidential trial outcome.**
