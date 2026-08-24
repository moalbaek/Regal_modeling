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
| `regal_explorer.py` | The legacy scenario engine in Python, plus audit-only interim-efficacy fields not computed by the browser. Prints five BAT presets and writes the 9-panel `regal_explorer_panel.png`. Requires `numpy` + `matplotlib`. |
| `REGAL_MODEL_DOCUMENTATION.md` | Full methodology, parameter sourcing, and limitations. |
| `BAT_CONTROL_ARM_RESEARCH.md` | The research basis for the default BAT-arm settings: component composition/weights (US/EU/China) and per-component mixture-cure survival parameters (median OS, cure fraction, Weibull shape), and how they map to `DEFAULT_COMP`. |
| `V2_IMPLEMENTATION_PLAN.md` | Scientific and software roadmap for the conditional, protocol-compatible v2 forecast. |
| `trial_design.py` | Keeps the cached classical boundary for the legacy audit and adds isolated v2 Lan-DeMets O'Brien–Fleming spending, protocol-factor stratified log-rank analysis, and explicit futility-rule primitives. |
| `audit/interim_efficacy_replay.py` | Fixed-seed equal-planned-strata replay of the interim boundary-crossing and final scenario-rejection rates. |
| `simulation.py` | Isolated v2 60/80-event decision engine: stratified interim/final analyses, mutually exclusive efficacy/futility/continuation branches, and canonical operating-characteristic validation. |
| `audit/v2_trial_decision_validation.py` | Fixed-seed null type-I-error and futility-threshold sensitivity report for the v2 decision engine. |
| `survival_models.py` | Isolated v2 survival layer: scale-aware cure mixtures, population mortality, and pre-outcome frailty/case-mix selection followed by randomization. It is not wired into the legacy explorer. |
| `bat_regimens.py` | Isolated v2 BAT layer: joint planned-stratum/delivered-regimen pathways, combination exposures with one validated outcome profile per patient, an equal-strata primary proxy, explicit zero-mass policy, and labeled legacy/stress allocations with a reproducible bear survival library. |
| `data/regal_public_history.json` | Versioned enrollment/event evidence with observation and announcement dates, typed counts, source notes, and explicit reporting-lag uncertainty; current through the 11 Aug 2026 event-80 right censor. |
| `event_likelihood.py` | Isolated v2 public-history layer: registry-anchored fixed-N accrual and a joint Poisson-multinomial likelihood over correlated integer event increments and latent patient trajectories. |
| `audit/v2_public_history_validation.py` | Deterministic WP5 audit for schema integrity, reachable 20/104/126 accrual anchors, small-cohort likelihood correctness, and event-80 reporting-lag sensitivity. |
| `posterior.py` | Isolated WP6 fixed-scenario conditioning layer: one consistent latent enrollment/outcome history, exact public-count conditional sampling, continuation-centered mixture importance weights, right-censor enforcement, and forward final projection. It does not yet average across model families or parameters. |
| `audit/v2_interim_conditioning_validation.py` | Fixed-seed WP6 stress audit showing that continuation-centered importance sampling reaches a rare continuation branch while preserving the public-history compatibility estimate. |

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
The v2 survival tests independently verify OS-versus-net background-mortality handling,
non-immortal cured survival, analytic/sampled agreement, and case-mix selection that cannot inspect
future event times or manufacture a guaranteed-survival interval. Non-unit disease frailty is
restricted to net/relative-survival inputs; overall-survival inputs remain at neutral frailty until
they have a validated excess-hazard decomposition. The v2 BAT tests separately pin planned-stratum,
delivered-regimen, component-exposure, and single-outcome-profile marginals, including exact
categorical boundaries, seeded marginal recovery, custom-library coverage, and combination regimens
that count as one patient even when they create more than one component exposure.
The v2 decision tests independently pin the planned 2.339711/2.011777 protocol-spending boundaries,
tied-event stratified score calculations over multiple factor columns, realized-information boundary
recalculation when all deaths tied at a cutoff are retained, all interim/final branches, event-calendar
cutoffs, null type-I error, and paired sensitivity rows for the unpublished futility threshold. The
canonical audit is backed by a separate exponential-null run through the complete patient-level
calendar-trigger and stratified-analysis path. The legacy classical 2.327/2.015 audit boundary
remains unchanged. The v2 public-history tests brute-force the joint integer-count likelihood on
small cohorts, prove that cumulative count marginals are not multiplied as independent, forbid
pre-opening enrollment, pin all 20/104/126 accrual anchors, distinguish threshold and as-of evidence,
and integrate both the unknown event-60 threshold date and the unannounced 80th event over independent,
moment-matched three-point lag grids spanning 0–14 days. The 104-patient
planning projection centers the provisional reference path but is deliberately excluded from
likelihood evidence; the reference path is not an independent Bayesian prior.
The WP6 conditioning tests then keep enrollment evidence, event-count compatibility, the interim
decision, and the final projection on one latent patient history. They pin the exact WP5 likelihood
ratio in a small fixture, require every proposed trajectory to satisfy the event-80 right censor,
verify continuation/final branch conservation, exercise pre-outcome censoring, and reuse identical
importance draws across assumed futility thresholds. They also pin support-preserving range targets,
consistent quota-DP cell budgets, and unbiased base-proposal fallback when a continuation tilt cannot
be fitted for one draw. Reported effective sample size, maximum weight share, and tilt-fallback counts
expose a poorly supported fixed-scenario projection rather than hiding it behind a raw draw count;
tilt iteration/error summaries are `None` when no tilt converged. Direct tilt callers can catch the
exported `TiltProposalError`.

```bash
python3 -m unittest discover -s tests   # run the golden test
python3 audit/interim_efficacy_replay.py --nsim 10000  # reproduce the equal-strata interim result
python3 audit/v2_trial_decision_validation.py --nsim 200000  # validate v2 alpha/branches/futility grid
python3 audit/v2_public_history_validation.py  # validate WP5 data/accrual/joint likelihood
python3 audit/v2_interim_conditioning_validation.py  # validate WP6 latent-history/rare-continuation IS
python3 tests/gen_golden.py             # regenerate golden.json after an INTENDED change, then review the diff
```

**Research/analysis tool operating entirely on public disclosures. Not investment advice, and it
does not access or estimate the confidential trial outcome.**
