# Regal_modeling

A public-information forecasting tool for the outcome of SELLAS Life Sciences' (NASDAQ: **SLS**)
blinded Phase 3 **REGAL** trial (NCT04229979) of galinpepimut-S (GPS) vs best available therapy
(BAT) in AML second complete remission.

The blinded death-event milestones (60/72/78) pin only the *pooled* survival curve, so the split
between arms is an explicit assumption. The tool calibrates the pooled curve to the milestones,
decomposes it into arms under user-controlled assumptions, and Monte-Carlo simulates the trial's
pre-specified Cox/log-rank test. Both survival models share **one responder curve** — a Weibull
(shape < 1 = heavier tail) — and differ by a single assumption, whether long-term responders are
*cured*: a **plateau** shape (the shared Weibull plus a cured fraction) and a **no-plateau** shape
(the same Weibull with no cure and a **fitted** tail shape). It fits both to the same milestones and
reports a P(success) for each; the gap between them is the irreducible "is the plateau real?"
uncertainty. When even a free, heavy tail cannot reproduce the decelerating milestones, the
no-plateau fit is flagged **degenerate / excluded** rather than reported as a falsely confident
number.

## Files

| File | What it is |
|------|------------|
| `regal_explorer.html` | Self-contained interactive explorer — open in any browser, no build or dependencies. Sliders for BAT composition, enrollment selection (eligibility filter, a left-truncation), non-responder fraction, natural (non-disease) death rate, the no-plateau Weibull tail shape (fitted by default, with a manual override), etc.; dual P(success) plus live charts (survival curves, event-accrual timeline, simulated-HR distribution, plateau-vs-tail divergence, enrollment validation, a P(success)-vs-effect power curve, and a BAT-median-&-cure-vs-selection sweep). |
| `regal_explorer.py` | The same engine in Python. Prints a summary across the four BAT presets and writes the 9-panel `regal_explorer_panel.png`. Requires `numpy` + `matplotlib`. |
| `REGAL_MODEL_DOCUMENTATION.md` | Full methodology, parameter sourcing, and limitations. |

```bash
python3 regal_explorer.py        # CLI summary + figure
# or just open regal_explorer.html in a browser
```

**Research/analysis tool operating entirely on public disclosures. Not investment advice, and it
does not access or estimate the confidential trial outcome.**
