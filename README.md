# Regal_modeling

A public-information forecasting tool for the outcome of SELLAS Life Sciences' (NASDAQ: **SLS**)
blinded Phase 3 **REGAL** trial (NCT04229979) of galinpepimut-S (GPS) vs best available therapy
(BAT) in AML second complete remission.

The blinded death-event milestones (60/72/78) pin only the *pooled* survival curve, so the split
between arms is an explicit assumption. The tool calibrates the pooled curve to the milestones,
decomposes it into arms under user-controlled assumptions, and Monte-Carlo simulates the trial's
pre-specified Cox/log-rank test. It fits the same milestones under **two survival shapes** — a
*plateau* (cure-mixture) and a *no-plateau* (log-logistic) tail — and reports a P(success) for each;
the gap between them is the irreducible "is the plateau real?" uncertainty.

## Files

| File | What it is |
|------|------------|
| `regal_explorer.html` | Self-contained interactive explorer — open in any browser, no build or dependencies. Sliders for BAT composition, enrollment selection (eligibility filter), non-responder fraction, natural (non-disease) death rate, tail heaviness, etc.; dual P(success) plus live charts (survival curves, event-accrual timeline, simulated-HR distribution, plateau-vs-tail divergence, enrollment validation, a P(success)-vs-effect power curve, and a BAT-median-&-cure-vs-selection sweep). |
| `regal_explorer.py` | The same engine in Python. Prints a summary across the four BAT presets and writes the 8-panel `regal_explorer_panel.png`. Requires `numpy` + `matplotlib`. |
| `REGAL_MODEL_DOCUMENTATION.md` | Full methodology, parameter sourcing, and limitations. |

```bash
python3 regal_explorer.py        # CLI summary + figure
# or just open regal_explorer.html in a browser
```

**Research/analysis tool operating entirely on public disclosures. Not investment advice, and it
does not access or estimate the confidential trial outcome.**
