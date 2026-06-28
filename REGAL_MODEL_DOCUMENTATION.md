# REGAL Trial Reverse-Engineering Model — Documentation

**Subject:** SELLAS Life Sciences (NASDAQ: SLS) Phase 3 REGAL trial (NCT04229979) of
galinpepimut-S (GPS) vs best available therapy (BAT) as maintenance in AML second complete
remission (CR2), non-transplant.

**Purpose:** Reconstruct what the *blinded, publicly disclosed* trial data imply about the
probability of a positive readout, by (1) calibrating the pooled (both-arm) survival curve to the
disclosed death-event milestones, (2) decomposing it into arm-level survival under explicit
assumptions, and (3) Monte-Carlo simulating the trial's pre-specified test to get a probability of
statistical success. The current tool fits the **same milestones under two competing survival
shapes** — a *plateau* (cure-mixture) and a *no-plateau* (log-logistic) tail — and reports a
P(success) for each. The gap between the two headline numbers is the irreducible "is the plateau
real?" uncertainty that blinded data cannot resolve.

**Deliverables:** `regal_explorer.html` (self-contained interactive explorer) and
`regal_explorer.py` (the same engine in Python, with a CLI summary and a 3-panel figure).

**Last updated:** 2026-06-28 · **Status:** research/analysis tool, not investment advice.

---

## 0. Epistemic frame (read first)

The single most important structural fact: **REGAL is blinded.** Public disclosures give the
*pooled* number of deaths at several dates, never the per-arm split. Consequently:

- The **pooled survival curve is identifiable** from the event milestones + the enrollment curve.
- The **decomposition into GPS and BAT arms is *not* identifiable** from blinded data. It requires
  an assumption about one arm (here: the BAT/control arm). Every "P(success)" number this model
  produces is therefore a function of an explicit, user-controlled **BAT-quality prior**, not a
  claim to know the confidential outcome.

This is a forecast built from public information (press releases, SEC filings, ClinicalTrials.gov,
the published trial-design paper) — the same class of event-driven analysis used in mainstream
biotech equity research. It does not access, infer, or attempt to unblind confidential trial data.

A recurring finding (Section 6): because the blinded data only pin the *pooled* curve, every
structural refinement we add to the arms — component-mixture BAT, immunological non-responders —
gets **absorbed by the pooled fit and barely moves the answer.** The one thing that materially
moves P(success) is whether the pooled *plateau* is real. The explorer makes this concrete by
fitting **two survival shapes** to the same milestones and reporting both P(success) numbers side
by side (Sections 4.7, 7).

---

## 1. Module map

The current tool is a single unified engine, delivered in two equivalent forms:

| File | Role | Key outputs |
|------|------|-------------|
| `regal_explorer.html` | Self-contained interactive explorer (no build, no dependencies): sliders for BAT composition, venetoclax cure, non-responder fraction, enrollment back-loading, tail heaviness β, and the BAT survival-stretch cap; live survival chart and the two headline P(success) numbers. | dual P(success), median HR, fit-check, per-arm curves |
| `regal_explorer.py` | The same engine in Python (`build_cure`, `build_ll`, `mc`), with a CLI summary across the four BAT presets and a 3-panel figure (`regal_explorer_panel.png`). | dual P(success) table, preset/non-responder sweeps, figure |

Both share one enrollment reconstruction, one set of survival primitives, and the same significance
threshold (Section 2.1). Within each, **two survival models are fit to the identical milestones**:

- **`build_cure` / `buildCure` — plateau (cure-mixture).** A weighted mixture of per-component
  Weibull cure-models; the GPS responder plateau and an early-hazard calibration `L` are fit to the
  events (Sections 4.3–4.4).
- **`build_ll` / `buildLL` — no-plateau (log-logistic).** Both arms are log-logistic tails with a
  shared shape β; a single BAT-median scale `k` and a GPS/BAT median ratio `r` are fit to the same
  events (Section 4.7).

> **Lineage.** This explorer consolidates an earlier set of standalone prototypes — `regal_model2.py`
> (first cure-mixture calibration + Monte-Carlo), `regal_verify.py` (full Cox MLE vs one-step, RMST),
> `regal_v3.py` (Bayesian posterior-predictive P(success) under 3 BAT priors), `regal_bat2.py`
> (component-mixture BAT + venetoclax bear map), and `regal_nr.py` (immunological non-responder
> subgroup). Their structure survives as paths inside the unified engine (the cure-mixture fit, the
> BAT composition, the non-responder fraction); the explorer adds the per-component Weibull shape,
> the survival-stretch cap, the back-loading interpolation, and the log-logistic no-plateau model.

---

## 2. Input parameters and their sources

Notation: **[S]** = directly sourced from a public disclosure (see References);
**[A]** = analyst assumption (with the literature anchor that informs it);
**[D]** = derived/calibrated by the model from sourced inputs.

### 2.1 Trial design & statistical analysis plan (SAP)

| Parameter | Value | Type | Source / reasoning |
|-----------|-------|------|--------------------|
| Patients enrolled (N) | 126 (1:1 → 63 GPS / 63 BAT) | [S] | SELLAS disclosures; reiterated in the May 2026 conference coverage [R7]. |
| Primary endpoint | Overall survival (OS) | [S] | Trial-design paper [R1]; interim coverage [R4][R5]. |
| Interim analysis | 60 deaths (efficacy/futility/safety) | [S] | SAP amendment, Nov 2022 [R2]; design paper [R1]. |
| Final analysis trigger | 80 deaths (63.5% of 126) | [S] | SAP amendment [R2]; Q1-2026 8-K [R6]. |
| Primary test | **Stratified Cox PH model**, treatment as only covariate, H0: HR ≥ 1 vs H1: HR < 1 | [S] | Design paper [R1] (explicit). |
| Alpha | **one-sided 0.025** | [S] | Design paper [R1]. |
| Alpha spending | Lan–DeMets **O'Brien–Fleming**, one interim at 60 deaths | [S] | Design paper [R1]; OncLive [R5]. |
| Stratification factors | CR2 vs CR2p; cytogenetic risk; MRD status; CR1 duration (<1 yr vs ≥1 yr) | [S] | Targeted Oncology [R4]. |
| Design effect size | HR 0.636 ⇒ medians 12.6 mo (GPS) vs ~8.0–8.1 mo (BAT) | [S] | SAP/IR [R2]; conference coverage states 12.6 vs 8.1 mo [R7]. |
| **Significance threshold used in code** | observed HR ≤ **0.636**, i.e. z_crit = |ln 0.636|·√80 / 2 = **2.024** (one-sided p ≈ 0.0215) | [D] | Derived from N, 80 events, and one-sided 0.025 with a small OBF interim spend; matches SELLAS's stated 0.636. |
| BAT-arm allowed agents | observation/hydroxyurea, hypomethylating agents (HMA), venetoclax, low-dose ara-C (LDAC); targeted maintenance (e.g. FLT3i) **excluded** | [S] | Targeted Oncology [R4]; OncLive [R5]. |

> **Note on the threshold.** The log-rank test is the score test of the Cox model, so the
> Monte-Carlo significance decision (`score_z > z_crit`) is the operating characteristic of the
> trial's *actual* pre-specified test. `regal_verify.py` confirms a fully-iterated Cox MLE differs
> from the one-step estimate by ≤0.001 in HR here (balanced 1:1, single covariate), so the
> approximation does not affect any conclusion.

### 2.2 Event milestones (pooled deaths, calendar)

| Date | Cumulative deaths | % of 126 | Type | Source |
|------|-------------------|----------|------|--------|
| 2024-12-10 | 60 | 47.6% | [S]* | Interim-analysis trigger; IDMC review completed and announced Jan 2025 [R4][R5]. *Exact 8-K date used in code (2024-12-10) should be reconciled against SELLAS IR; the milestone itself (interim at 60) is firmly sourced. |
| 2025-12-26 | 72 | 57.1% | [S] | SELLAS IR / TipRanks, 29 Dec 2025 [R3]. |
| 2026-05-11 | 78 | 61.9% | [S] | SEC 8-K exhibit 99.1 and Q1-2026 release, 12 May 2026 [R6]. |
| ~2026-06-21 | still 78 | 61.9% | [S] | No 80th-event announcement as of late June 2026 [R8] → near-zero accrual late Q2. |

Inter-milestone accrual ≈ **1 death/month** throughout — the central empirical anomaly that drives
the whole analysis (a cohort mostly 2–4 years past randomization should be dying far faster unless
survival is unexpectedly long).

### 2.3 Enrollment reconstruction

The exact monthly accrual is **not public**; the curve is reconstructed [A] to honor the sourced
anchors below. Its shape is treated as a nuisance parameter via the **back-loading slider**
(Section 2.8), which interpolates between a "flat" and a heavily back-loaded profile (default 0.50,
i.e. the midpoint that the earlier prototypes marginalized over).

| Anchor | Value | Type | Source |
|--------|-------|------|--------|
| Registration | NCT04229979 first posted Jan 2020 | [S] | ClinicalTrials.gov. |
| Early protocol | WT1-positivity required initially, later broadened | [S] | OncLive eligibility note [R5]. |
| Ex-China enrollment target reached | Nov 2023 | [S] | SELLAS PRs (2023). |
| China cohort (via 3D Medicines) | enrolled ~Dec 2023 – Mar 2024 | [S] | SAP/partnership disclosures [R2]. |
| Last patient in | ~March 2024 | [S] | CEO, May 2026 conference [R7]. |
| Original expectation to 80th event | 12–15 months after last patient (~mid-2025) | [S] | CEO, May 2026 conference [R7]. |
| **Code reconstruction (base)** | slow 2020–21 (COVID + WT1-only) → heavy 2022–23 → China bolus to Mar-2024, summing to 126 | [A] | Piecewise monthly rates chosen to match the anchors above. |

### 2.4 Interim disclosures (Jan 2025, at 60 deaths)

| Quantity | Value | Type | Source |
|----------|-------|------|--------|
| Median follow-up at interim | ~13.5 months (range 1 to >36) | [S] | CancerNetwork / Targeted Oncology [R4][R5]. |
| Deaths at interim | <50% of enrolled (i.e. 60/126 = 47.6%) | [S] | [R4]. |
| Pooled median OS | **≥ 13.5 months** (a floor; blinded) | [S] | [R4] — note this is median *follow-up* with <50% dead, so pooled median OS is at least this. |
| IDMC recommendation | continue without modification; futility crossed; no safety concerns | [S] | [R4][R5]. |
| WT1 immune response | ~80% of GPS patients showed a WT1-specific T-cell response | [S] | Interim coverage [R4] → motivates the ~20% default non-responder fraction in the explorer. |
| Historical comparator cited | ~6-month OS in a similar CR2 non-transplant population | [S] | [R4]. |

### 2.5 BAT comparator and the component library (`build_cure`)

The BAT arm is modeled as a weighted mixture of component therapies, each a cure-mixture
parameterized by **(median OS, long-term/"cure" fraction, Weibull shape k)**. These per-component
numbers are **analyst assumptions [A]** anchored to the comparator literature; they are the main
lever and are intended to be edited. The shape **k** generalizes the earlier pure-exponential
non-cured tail (`k = 1` reproduces it): `k < 1` is a heavy tail (more long-term survivors), `k > 1`
accelerates. All components default to `k = 1`.

| Component | Median OS (mo) | Cure fraction | Shape k | Type | Anchor / reasoning |
|-----------|----------------|---------------|---------|------|--------------------|
| Observation / BSC | 6.0 | 0.08 | 1 | [A] | Untreated CR2 relapses fast; long-term survival low. **Non-responders track this row.** |
| Hydroxyurea (palliative) | 6.0 | 0.05 | 1 | [A] | Palliative; poorest durable-remission. |
| HMA (aza/dec) | 10.0 | 0.13 | 1 | [A] | Some durable responses; consistent with HMA-maintenance literature. |
| Venetoclax (± HMA) | 13.0 | 0.22 | 1 | [A] | Best BAT option; a subset achieve durable remission. **This is the key bear/bull knob.** |
| LDAC | 8.0 | 0.09 | 1 | [A] | Modest activity. |

**Presets** (selectable in the explorer; weights auto-normalize):
*Base* (15/5/30/35/15, ven cure 22%) · *Low-venetoclax* (25/15/35/10/15) ·
*Venetoclax-dominant* (5/2/23/60/10) · *Bear corner* (5/2/13/70/10, ven cure 36%) — the bear
corner is the only composition that pushes the plateau-shape P(success) clearly below 50%.

Supporting literature anchors for these assumptions [A]:
- Contemporary non-transplant CR2 maintenance (HMA and/or BCL-2 inhibitor): **~8-month** expected
  median OS, per a REGAL steering-committee member [R3].
- R/R AML on venetoclax+HMA (active disease): median OS ~5.5–6.1 mo; post-Ven/HMA failure ~5.9 mo
  (relapsed 11.2 / refractory 3.1) — a *floor*, since REGAL patients are in remission, not active
  R/R disease. Ven+HMA *responders* (selected) run much longer (~21.6 mo). [comparator search,
  Section 2.7]
- Oral-azacitidine maintenance (QUAZAR AML-001, CR1 context): ~24.7-mo median OS — relevant for
  the *upper* bound on maintenance benefit, not the CR2 control.

**Base composition** (investigator's-choice weights, contemporary; **assumption [A]**, editable):
Observation 0.15 · Hydroxyurea 0.05 · HMA 0.30 · Venetoclax 0.35 · LDAC 0.15 → implied BAT cure
≈ 14%, BAT median ≈ 9.4 mo. Two alternates ("low-venetoclax / early-ex-US", "venetoclax-dominant /
modern US") bracket the range.

### 2.6 Bayesian priors on the BAT plateau (legacy `regal_v3.py`; superseded by the composition lever)

The lineage `regal_v3.py` gave the BAT-arm long-term-survivor fraction (π_c) a Beta prior, with the
GPS plateau following from the data constraint (Section 4.4). The explorer replaces this abstract
prior with the clinically-grounded **BAT composition** (Section 2.5) and the **stretch cap**
(Section 2.8), which together set π_BAT directly; the Beta priors below are retained only as the
historical mapping from a one-number prior to a P(success). Priors are **analyst choices [A]**:

| Prior | Beta(a,b) | Mean π_c | Rationale |
|-------|-----------|----------|-----------|
| Optimistic | Beta(4.55, 30.45) | 0.13 | BAT ≈ historical (6–8 mo, low plateau). |
| Base | Beta(5.10, 24.90) | 0.17 | Steering-committee ~8-mo BAT anchor [R3]. |
| Skeptical | Beta(8.10, 21.90) | 0.27 | Venetoclax-era BAT substantially improved. |

### 2.7 GPS non-responder subgroup (`build_cure`, non-responder path)

| Parameter | Value | Type | Source / reasoning |
|-----------|-------|------|--------------------|
| Non-responder fraction f_nr | swept 0–30% | [A] | Anchored to the ~80% WT1 T-cell response rate [R4] ⇒ ~20% immunological non-responders. |
| Non-responder survival | = Observation component (median 6 mo, cure 8%) | [A] | User's specification: non-responders get no vaccine benefit → behave like best-supportive-care. |
| Responder cure | refit to events given f_nr & BAT | [D] | Increases 62% → 87% as f_nr rises 0 → 30% (Section 6). |

### 2.8 Survival-shape stress controls (the explorer's headline knobs)

These do not change the milestones — they change the *shape* fit to them, which is exactly the
unidentified question. All are user-controlled in the explorer.

| Control | Range / default | Type | Role |
|---------|-----------------|------|------|
| Tail heaviness **β** (log-logistic) | 0.6–2.0, default 1.20 | [A] | Shape of the no-plateau model. Lower β = heavier tail (both arms survive long, smaller implied effect); higher β ≈ a plateau. Controls the orange "no-plateau" number only. |
| Max BAT survival stretch **(×)** | 1.0–3.0×, default 1.5 | [A] | Caps how far the cure-mixture fit may stretch BAT survival past the component medians via the early-hazard multiplier `L` (`L_min = 1/maxStretch`). Tighter cap → less longevity attributed to BAT → more attributed to GPS → **higher** plateau P(success). 3× ≈ unconstrained. |
| Enrollment back-loading | 0–1, default 0.50 | [A] | Interpolates the monthly accrual between a flat profile (0) and a heavily back-loaded-into-2023 profile (1); replaces the earlier fixed 50/50 marginalization (Section 2.3) with a continuous slider. |
| Per-component shape **k** | ≥0.3, default 1 | [A] | Weibull shape of each BAT component's non-cured tail (Section 2.5). |

---

## 3. Calibrated / derived outputs [D]

Representative values at the **base preset** (f_nr = 20%, default β and stretch cap); every number
is a function of the user controls in Sections 2.5–2.8, so treat these as a centre point, not a
fixed result. Monte-Carlo figures carry ±2–3 pp simulation noise at the default sim budget.

| Quantity | Value (base preset) | Source |
|----------|---------------------|--------|
| BAT cure / median | ~14% · ~9 mo (→ ~14 mo after stretch cap) | `build_cure` |
| GPS cure / median | ~51% · not reached within 48 mo | `build_cure` |
| Pooled long-term-survivor fraction | ~0.33–0.37 | `build_cure` |
| Pooled median OS | **~16–21 mo** (above the ≥13.5 floor) | `build_cure` |
| **P(success) — plateau (cure-mixture)** | **~90–95%** | `build_cure` + `mc` |
| **P(success) — no-plateau (log-logistic, β=1.2)** | **~98–99%** | `build_ll` + `mc` |
| 80th event reached in MC | ~80–85% of sims (plateau); ~100% (no-plateau) | `mc` |

Sweeping the BAT composition shows where the two shapes diverge: under the **bear corner**
(venetoclax-dominant *and* ~36% durable) the plateau P(success) falls to ~25–30% while the
no-plateau P(success) stays ~98%. That divergence — not the central estimate — is the analysis's
main output.

> The plateau model's ultimate dead fraction (~63%) nearly coincides with the 80-event trigger
> (63.5%), which is *why* accrual has stalled at 78 — the cohort is essentially at its modeled
> asymptote. The no-plateau model has no asymptote, so it always reaches the 80th event; the real
> trial's stall at 78/80 therefore *mildly* favors the plateau model, though the milestones alone
> cannot adjudicate.

---

## 4. Methodology and reasoning

### 4.1 Survival primitives

The explorer carries **two** primitives, fit independently to the same milestones:

- **Plateau (cure-mixture, Weibull):** `Sc(t) = π + (1−π)·exp(−(L·t / λ)^k)`, a cured/long-term-survivor
  fraction π plus a Weibull-decaying remainder. λ is set so the non-cured median equals the component
  median (`λ = median / A(π)^{1/k}`, `A(π) = −ln[(0.5−π)/(1−π)]`); `k = 1` recovers the original pure
  exponential; `L` is the early-hazard calibration multiplier (Section 4.3). Rationale: cancer-vaccine
  effects classically manifest as a durable-remission (plateau) difference rather than a uniform hazard
  shift, and the steep post-interim event deceleration cannot be fit by any single smooth non-plateau
  curve.
- **No-plateau (log-logistic):** `Sll(t) = 1 / (1 + (t/α)^β)`, a heavy-tailed survival with **no
  asymptote**. This is the explicit "the plateau may not be real" alternative — previously flagged as
  the untested sensitivity, now fit and scored alongside the plateau model (Section 4.7).

Both have matching inverse-CDF samplers used by the Monte-Carlo (`sampNC`, `sampLL`).

### 4.2 Enrollment → expected deaths

For an enrollment cohort enrolled at calendar time `e` with `n` patients, expected cumulative
deaths at calendar time `T` are `Σ_cohorts n · [1 − S(T − e)]`. This convolution is the forward
model linking a survival curve to the disclosed event counts.

### 4.3 Pooled calibration

The pooled curve is `0.5·S_BAT + 0.5·S_GPS`. The explorer fits its free parameters to the three
(date, deaths) milestones by **weighted least squares**, with weights `WT = [1, 2, 4]` that
up-weight the most recent (and most informative) milestone, over a coarse grid followed by three
local-refinement passes. For the plateau model the two free parameters are the GPS responder cure
`π_resp` and the early-hazard multiplier `L` (bounded by `L ∈ [1/maxStretch, 2.2]`); the displayed
"survival calibration" is `1/L`. The enrollment shape is set by the back-loading slider (Section 2.8)
rather than marginalized. (The lineage `regal_v3.py` used a Poisson log-likelihood with a Bayesian
prior on π_BAT; the explorer replaces that with this transparent point-fit + composition lever.)

### 4.4 Arm decomposition (the unidentified step)

Because blinded data fix only the *average* of the arms, the model imposes the constraint
`π_GPS = 2·π_pool − π_BAT`: the data pin `π_pool`; the **BAT prior/composition fixes π_BAT**; the
GPS plateau follows. Different decomposition modes:
- **PH (proportional hazards):** `S_GPS = S_BAT^HR` — but this cannot reproduce a plateau without an
  implausibly extreme HR, evidence *against* simple PH (and ruled out independently by the slow
  accrual; see v2).
- **Cure-difference (preferred):** GPS shares the control's early dynamics but has a higher plateau
  — a biologically motivated, early-and-sustained separation.

### 4.5 Monte-Carlo P(success)

`P(success)` is the fraction of simulated trials whose pre-specified test is significant
(`mc()`). Each simulated trial: draws enrollment per cohort; assigns 1:1 GPS/BAT; draws each
patient's survival from the relevant arm/component (cured patients get an effectively infinite
time); finds the calendar time of the `FINAL`-th (80th) death; censors everyone there; and computes
the **log-rank score statistic = Cox score test = the trial's actual pre-specified test**, declaring
success when `z > z_crit = |ln(HRC)|·√FINAL / 2 = 2.024`. It returns P(significant), the fraction of
sims that reach the 80th event, and the median simulated HR. The same `mc()` runs on both the
plateau and the no-plateau model, producing the two headline numbers.

### 4.6 Component-mixture BAT and non-responders

These replace any abstract π_BAT prior with clinically-grounded structure (Sections 2.5, 2.7).
**They add interpretability, not identifying information** — the blinded data still see only the
pooled curve, so refits absorb the new structure and leave P(success) largely unchanged
(Section 6).

### 4.7 The no-plateau (log-logistic) model and the shape gap

To make "is the plateau real?" answerable rather than merely flagged, the explorer fits a second,
plateau-free model to the **identical** milestones. Both arms are log-logistic (`Sll`) with a shared
tail-heaviness β; the two fitted parameters are a single BAT-median scale `k` (`median_BAT = k·m̄`,
with `m̄` the weight-averaged component median) and the GPS/BAT median ratio `r`. Non-responders in
this mode track a log-logistic at the Observation median. The same Monte-Carlo (Section 4.5) then
scores it.

Because a heavy tail keeps patients alive without ever asymptoting, it generally fits the same three
death counts about as well as the plateau model but implies a **different** (often larger, but
shape-sensitive) survival gap. The signed difference between the two P(success) numbers is the
explorer's headline output: a small gap means the conclusion is shape-robust; a large gap means the
verdict hinges on an assumption the blinded data cannot test. Lowering β (heavier tail) fits the
milestones slightly *worse* but cannot be excluded — the only weak, data-side tilt toward the
lighter tail (and toward the plateau model is the trial's current stall at 78/80, Section 3).

---

## 5. Key functions (reference)

Names are given as **`python` / `javascript`** where they differ between `regal_explorer.py` and
`regal_explorer.html`; the two implementations are line-for-line equivalent.

| Function | Purpose |
|----------|---------|
| `Acoef` / `lam` | Weibull coefficient `A(π) = −ln[(0.5−π)/(1−π)]` and scale `λ = median / A^{1/k}`. |
| `Sc(t, med, cure, k, L)` | Per-component cure-mixture Weibull survival (the plateau primitive, Section 4.1). |
| `Sll(t, α, β)` | Log-logistic survival (the no-plateau primitive). |
| `sampNC` / `sampLL` | Inverse-CDF samplers for non-cured Weibull and log-logistic times (Monte-Carlo draws). |
| `enroll(bl, N)` | Monthly enrollment cohorts summing to `N`, interpolating flat↔back-loaded by `bl` (Section 2.3). |
| `common(cfg)` | Shared setup: normalized weights, clamped per-component params, cohorts, milestones, fit weights. |
| `build_cure` / `buildCure` | Plateau model: fits `π_resp` and the stretch multiplier `L` to the milestones; returns per-arm `Sbat/Sgps/Spool`, cures, medians (Sections 4.3–4.4). |
| `build_ll` / `buildLL` | No-plateau model: fits BAT-median scale `k` and GPS/BAT ratio `r` at fixed β (Section 4.7). |
| `median(S)` | Bisection median of a survival function (`∞`/"NR" if never below 0.5 within 900 mo). |
| `mc(M, nsim)` | Monte-Carlo trial: enrollment → per-arm death draws → censor at the 80th event → **log-rank/Cox score test**; returns P(significant), 80th-event-reached fraction, median HR (Section 4.5). |
| `figure()` / `render` + `chart` | 3-panel figure (py) / live SVG chart and metrics panel (html). |

---

## 6. Principal findings

1. **The framework matches the trial.** The confirmed primary test is a stratified Cox at one-sided
   0.025 with an OBF interim [R1]; the model's significance machinery and HR ≤ 0.636 threshold are
   consistent with it.
2. **Blinded pooled survival is high:** ~33–37% modeled plateau, ~16–21-mo median — far above the
   ~6–8-mo historical/contemporary control. Something is keeping these patients alive.
3. **Under the plateau model, P(success) is governed by the BAT-quality assumption.** With a
   clinically-built BAT composition it stays high (~90%+ at base) and is hard to push below ~50%
   without assuming venetoclax maintenance is both dominant *and* ~30–36% durable — the narrow
   "bear corner," where it falls to ~25–30%.
4. **Structural refinements are absorbed by the pooled fit.** Component-mixture BAT and a 0–40%
   non-responder subgroup each leave P(success) ≈ unchanged, because the data fix the pooled
   plateau and refits merely redistribute it (e.g. raising the non-responder fraction forces the
   responder cure up). This *localizes* the uncertainty rather than resolving it.
5. **The shape assumption — not the arm split — is what moves the answer.** The no-plateau
   (log-logistic) model fits the same milestones and generally returns a *higher* P(success)
   (~98–99%), because a heavy tail with no asymptote implies an even larger arm gap. Only in the
   bear corner do the two shapes sharply diverge (plateau ~25–30% vs no-plateau ~98%). That
   plateau-vs-tail gap is the irreducible, blinded-data-proof uncertainty, and the explorer reports
   it as the headline.

---

## 7. Limitations and the load-bearing assumption

- **The plateau may not be real — now operationalized, not eliminated.** The pooled plateau is
  extrapolated from three event counts under a mixture-cure assumption. The explorer addresses this
  head-on by also fitting a heavy-tailed log-logistic model with no asymptote (Section 4.7) and
  reporting both P(success) numbers. This *quantifies* the sensitivity but does not resolve it: the
  milestones fit both shapes about equally well, so which one is right remains the single
  load-bearing, blinded-data-proof assumption. (In practice the no-plateau fit tends to return an
  even higher P(success), so the plateau is not uniformly the optimistic choice — the divergence is
  composition-dependent and largest in the bear corner.)
- **Decomposition is unidentified.** All arm-level conclusions are prior-/assumption-driven; the
  blinded data cannot adjudicate them.
- **Delayed vs sustained separation.** The cure-difference structure assumes early, sustained
  separation (favorable to the Cox test). A genuinely *delayed* separation would violate PH and the
  committed Cox test could under-detect — the one shape where this risk bites.
- **Per-component BAT survival and composition are assumptions** [A], not patient-level data; they
  are the intended user levers.
- **Promotional bias.** Several anchors (e.g. the ~8-mo BAT figure, the "longer-than-expected
  survival" framing) originate with SELLAS or affiliates and should be discounted accordingly.
- **Not incorporated (conservative):** the interim futility pass; stratification of the Cox model
  (the trial stratifies; the simulation does not — a minor, likely slightly power-increasing,
  difference).

---

## 8. References

Public sources used for sourced [S] inputs. Press/secondary sources are used for facts that
originate in SELLAS disclosures; verify primary 8-K/PR text on SELLAS IR and SEC EDGAR where exact
dates matter.

- **[R1]** REGAL trial-design / methods paper, *PMC* (open access) — primary efficacy analysis
  (stratified Cox, H0: HR ≥ 1, one-sided 0.025, Lan–DeMets O'Brien–Fleming, interim at 60 deaths).
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11760237/
- **[R2]** SELLAS, "Update on Phase 3 REGAL … Interim Analysis Now at 60 Events and Final Analysis
  Now at 80 Events," GlobeNewswire, 14 Nov 2022.
  https://www.globenewswire.com/news-release/2022/11/14/2554907/0/en/SELLAS-Life-Sciences-Announces-Update-on-Phase-3-REGAL-Clinical-Trial-Evaluating-Lead-Asset-Galinpepimut-S-in-Acute-Myeloid-Leukemia.html
- **[R3]** SELLAS, "Update on Pivotal Phase 3 REGAL … 72 events as of December 26, 2025"
  (steering-committee ~8-mo BAT context), 29 Dec 2025 (IR / TipRanks).
  https://www.tipranks.com/news/the-fly/sellas-life-sciences-says-regal-trial-cro-informs-company-72-events-occurred-thefly
- **[R4]** "Galinpepimut-S Completes Phase 3 REGAL Interim Analysis in AML," CancerNetwork (interim:
  median FU ~13.5 mo, <50% dead, pooled median ≥13.5 vs ~6-mo historical, IDMC continue, ~80% WT1
  response). https://www.cancernetwork.com/view/galinpepimut-s-completes-phase-3-regal-interim-analysis-in-aml
- **[R5]** "REGAL Trial Receives Green Light to Continue…," OncLive (OBF spending; eligibility/WT1;
  SAP changes). https://www.onclive.com/view/regal-trial-receives-green-light-to-continue-testing-galinpepimut-s-in-aml
  · and "Phase 3 REGAL Trial … Advances Toward Completion," Targeted Oncology (stratification
  factors; BAT-allowed agents). https://www.targetedonc.com/view/phase-3-regal-trial-of-galinpepimut-s-in-aml-advances-toward-completion
- **[R6]** SELLAS, "Reports First Quarter 2026 Financial Results …" (78 events as of 11 May 2026;
  final analysis at 80th event), 12 May 2026 — SEC 8-K exhibit 99.1.
  https://www.sec.gov/Archives/edgar/data/1390478/000139047826000009/sls-202605128xkexhibit991.htm
- **[R7]** CEO remarks, Stifel 2026 Targeted Oncology Forum, 20 May 2026 (126 patients; 12.6 vs
  8.1-mo design medians; last patient ~Mar 2024; original 12–15-mo expectation to 80th event;
  patients >3 yr on treatment). https://stocktwits.com/news-articles/markets/equity/sls-stock-gps-very-good-chance-beat-earlier-survival-outcomes/cZXDpXKReVe
- **[R8]** Status check, late June 2026 — no 80th-event announcement yet (still 78).
  https://www.merlintrader.com/sellas-life-sciences/

*Comparator literature anchors for [A] component survival (Section 2.5) were drawn from published
AML CR2 / R/R venetoclax-HMA and azacitidine-maintenance outcome studies; the specific
per-component (median, cure) values are analyst estimates, not direct quotations, and should be
treated as editable inputs rather than sourced facts.*

---

*Prepared as a quantitative research tool operating entirely on public information. It explores
assumption-driven scenarios consistent with disclosed aggregate data; it does not estimate the
confidential outcome of the ongoing trial, and it is not investment advice.*
