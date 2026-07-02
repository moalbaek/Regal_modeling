# REGAL Trial Reverse-Engineering Model — Documentation

**Subject:** SELLAS Life Sciences (NASDAQ: SLS) Phase 3 REGAL trial (NCT04229979) of
galinpepimut-S (GPS) vs best available therapy (BAT) as maintenance in AML second complete
remission (CR2), non-transplant.

**Purpose:** Reconstruct what the *blinded, publicly disclosed* trial data imply about the
probability of a positive readout, by (1) calibrating the pooled (both-arm) survival curve to the
disclosed death-event milestones, (2) decomposing it into arm-level survival under explicit
assumptions, and (3) Monte-Carlo simulating the trial's pre-specified test to get a probability of
statistical success. Both survival shapes share **one responder curve** — a Weibull (shape < 1 =
heavier tail) — and differ by a **single assumption**: whether long-term responders are *cured*. The
**plateau** shape adds a cured fraction to the shared Weibull; the **no-plateau** shape uses the same
Weibull with **no cure and a fitted tail shape**. The tool fits both to the **same milestones** and
reports a P(success) for each. The gap between the two headline numbers is the irreducible "is the
plateau real?" uncertainty that blinded data cannot resolve. When the no-plateau fit lands on a
parameter boundary (a light-tailed, no-cure family cannot reproduce the decelerating milestones) it
is reported as **degenerate / excluded** rather than as a falsely confident number.

**Deliverables:** `regal_explorer.html` (self-contained interactive explorer) and
`regal_explorer.py` (the same engine in Python, with a CLI summary and a 9-panel figure).

**Last updated:** 2026-07-02 · **Status:** research/analysis tool, not investment advice.

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
| `regal_explorer.html` | Self-contained interactive explorer (no build, no dependencies): sliders for BAT composition, venetoclax cure, enrollment selection (eligibility filter), non-responder fraction, enrollment median timing, natural (non-disease) death rate, loss-to-follow-up, the no-plateau Weibull tail shape (fitted by default, with a manual override), and per-component shape k, plus an interim futility-HR consistency check and a weighted/unweighted fit toggle; the two headline P(success) numbers and a "Trial dynamics" panel of live charts (survival curves, event-accrual timeline, simulated-HR distribution, plateau-vs-tail divergence band, enrollment validation, a P(success)-vs-effect power curve, and a BAT-median-&-cure-vs-selection sweep). | dual P(success), median HR, implied interim HR, per-arm alive-at-80th, GPS-median Poisson CI, fit-check, degenerate-fit flag, per-arm curves |
| `regal_explorer.py` | The same engine in Python (`build_plateau`, `build_no_plateau`, `mc`), with a CLI summary across the four BAT presets and a 9-panel figure (`regal_explorer_panel.png`). | dual P(success) table, preset/non-responder sweeps, 9-panel figure |

Both share one enrollment reconstruction, one set of survival primitives, and the same significance
threshold (Section 2.1). Both survival models are built on **one shared responder curve** — a
Weibull `Sweib(t) = exp(−(t/scale)^shape)` (shape < 1 = heavier tail) — and fit to the **identical
milestones**; they differ by a single downstream assumption, the cured fraction:

- **`build_plateau` — plateau (cured fraction).** The shared Weibull responder **plus a cured
  fraction** (a weighted mixture of per-component Weibull cure-models). Only the GPS responder cure
  is fit to the events; the BAT arm is fixed by the component medians plus enrollment selection
  (Sections 4.3–4.4).
- **`build_no_plateau` — no-plateau (fitted tail).** The **same** Weibull responder with **no cured
  fraction**. Three parameters are fit to the milestones — a BAT-median scale `k`, a GPS/BAT median
  ratio `r`, and the **Weibull tail shape** itself — and the GPS arm is a *single* responder curve
  (no non-responder mixture). If the fit sits on a parameter boundary it is flagged as
  degenerate/excluded (Section 4.7).

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
| **Significance threshold used in code** | observed HR ≤ **0.636**, i.e. z_crit = \|ln 0.636\|·√80 / 2 = **2.024** (one-sided p ≈ 0.0215) | [D] | Derived from N, 80 events, and one-sided 0.025 with a small OBF interim spend; matches SELLAS's stated 0.636. |
| BAT-arm allowed agents | observation/hydroxyurea, hypomethylating agents (HMA), venetoclax, low-dose ara-C (LDAC); targeted maintenance (e.g. FLT3i) **excluded** | [S] | Targeted Oncology [R4]; OncLive [R5]. |

> **Note on the threshold.** The log-rank test is the score test of the Cox model, so the
> Monte-Carlo significance decision (`score_z > z_crit`) is the operating characteristic of the
> trial's *actual* pre-specified test. A fully-iterated Cox MLE differs from the one-step estimate
> by ≤0.001 in HR here (balanced 1:1, single covariate), so the approximation does not affect any
> conclusion.

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
anchors below. Its shape is controlled by the **enrollment-timing slider** (Section 2.8), which slides
accrual between an earlier (flat) and a later (back-loaded) profile. Because the *median enrollment
date* is the quantity that actually drives time-from-randomization at each milestone, the explorer
now **displays the implied median date** (default ≈ Mar 2023) live, together with the cumulative
patients enrolled at the sourced anchor dates, so drift away from the anchors is visible.

| Anchor | Value | Type | Source |
|--------|-------|------|--------|
| Registration | NCT04229979 first posted Jan 2020 | [S] | ClinicalTrials.gov. |
| Early protocol | WT1-positivity required initially, later broadened | [S] | OncLive eligibility note [R5]. |
| Cumulative enrolled (anchors) | ~20 by Apr 2022 · ~104 by Nov 2023 · 126 by Apr 2024 | [S] | SELLAS PRs (2023) / SAP [R2]. |
| China cohort (via 3D Medicines) | enrolled ~Dec 2023 – Mar 2024 | [S] | SAP/partnership disclosures [R2]. |
| Last patient in | ~March 2024 | [S] | CEO, May 2026 conference [R7]. |
| Original expectation to 80th event | 12–15 months after last patient (~mid-2025) | [S] | CEO, May 2026 conference [R7]. |
| **Code reconstruction (base)** | slow 2020–21 (COVID + WT1-only) → heavy 2022–23 → China bolus to Mar-2024, summing to 126; **implied median ≈ Mar 2023** | [A] | Piecewise monthly rates chosen to match the anchors above. The ~104-by-Nov-2023 anchor pins the median to roughly Q1–Q2 2023. |

### 2.4 Interim disclosures (Jan 2025, at 60 deaths)

| Quantity | Value | Type | Source |
|----------|-------|------|--------|
| Median follow-up at interim | ~13.5 months (range 1 to >36) | [S] | CancerNetwork / Targeted Oncology [R4][R5]. |
| Deaths at interim | <50% of enrolled (i.e. 60/126 = 47.6%) | [S] | [R4]. |
| Pooled median OS | **≥ 13.5 months** (a floor; blinded) | [S] | [R4] — note this is median *follow-up* with <50% dead, so pooled median OS is at least this. |
| IDMC recommendation | continue without modification; futility crossed; no safety concerns | [S] | [R4][R5]. |
| WT1 immune response | ~80% of GPS patients showed a WT1-specific T-cell response | [S] | Interim coverage [R4] → motivates the ~20% default non-responder fraction in the explorer. |
| Historical comparator cited | ~6-month OS in a similar CR2 non-transplant population | [S] | [R4]. |

### 2.5 BAT comparator and the component library (`build_plateau`)

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

### 2.5.1 Enrollment selection (eligibility filter)

The component medians in Section 2.5 describe **all** CR2 transplant-ineligible patients on each
therapy. But a trial's eligibility bar (performance status, organ function, blast counts, …) enrols a
**healthier subset** than the unselected real-world population those medians come from — so the true
comparator arm can outlive its face-value component inputs. The **enrollment-selection slider**
(`esel`, 0–50%, default 25%) makes that gap an explicit lever.

| Parameter | Value | Type | Source / reasoning |
|-----------|-------|------|--------------------|
| Enrollment selection (drop weakest / keep strongest 1−f) | 0–50%, default 25% | [A] | Fraction of the *weakest* patients (by survival) the eligibility criteria are assumed to screen out. 0% = component medians taken at face value; 50% = only the healthiest half of each component is enrolled. |

**Mechanism (left-truncation).** The operation is a **left-truncation**: discard the earliest-dying
`f`, retain the longest-surviving `1−f`. Keeping the healthiest fraction `1−f` of any distribution is
exactly its survival conditioned on outliving its `f`-quantile `t_f` (where `S(t_f)=1−f`):

```
S_sel(t) = min(1, S(t) / (1 − f))
```

for each plateau component (and, in the no-plateau model, each Weibull arm). This lifts every curve
to its "top `100(1−f)`%" shape: the long-term/cure fraction **rises** from `c` to `c/(1−f)` and the
median lengthens, with no re-anchoring of the Weibull scale (so the `c < 0.5` parameterization never
breaks). The `min(1, ·)` clip is **kept on purpose**: the near-flat segment before `t_f` is real
**guarantee time**, with a direct correlate in REGAL's *"estimated life expectancy > 6 months"*
enrolment criterion — it is a feature, not an artifact. The matching Monte-Carlo draw is the
inverse-transform of the same left-truncation: draw `u ~ Unif(0, 1−f)` (which keeps the strongest
`1−f`), so a plateau patient is a long-term survivor with probability `c/(1−f)`, otherwise its
non-cured time is drawn conditioned on exceeding `t_f` (`u → u·(1−f−c)/(1−c)`), and the GPS-responder
non-cured mixture reweights to `w·(1−f−c)`. For the no-plateau Weibull arms the draw is simply
`u = (1−f)·rnd()`. At `f = 0` every expression collapses back to the unselected model exactly.

**Applies to both panels, before the arm split.** Enrollment eligibility screens *every* randomized
patient, so this left-truncation is an **upstream transform of the pooled CR2 pool, applied
identically in both panels and both arms** — it is shared infrastructure, not one of the assumptions
that distinguishes the panels (that one assumption, the cured fraction, lives downstream). Because the
truncation is non-differential across arms it cannot bias the within-trial comparison, so the fitted
**HR is roughly invariant to `f`** at the default settings while the milestone fit and P(success)
shift. (At extreme `f` the re-fit can be pushed onto a parameter boundary in the no-plateau panel, in
which case that panel is flagged degenerate/excluded — see Section 4.7.)

**`q` is the single BAT-side lever.** With the BAT arm otherwise fixed by the component medians, `q`
is what determines how much of the milestone deceleration is attributed to a healthier enrolled cohort
versus to the GPS effect; the plateau fit's *only* free parameter is the GPS responder cure `π_resp`.
The default is **`q = 25%`** (mid-band; see below).

**Effect (base preset).** As `q` rises 0 → 25 → 50% the BAT median OS lifts ~9 → 14 → 22 mo and the BAT
cure fraction climbs ~14 → 19 → 29% (both plotted live in panel *(i)* / the "enrollment selection lifts
the BAT arm" chart, `S_BAT` and `π_BAT/(1−q)`, independent of the Monte-Carlo). To keep the pooled
60/72/78 pinned, the fitted GPS responder cure falls ~0.81 → 0.70 → 0.48, so the **plateau** P(success)
drops steeply ~100 → 94 → 13% — a healthier, harder-to-beat comparator leaves less residual to
attribute to GPS. Note the direction of the fit-check: at `q = 0` the raw medians *over*-produce early
deaths (modeled ~65/74/76 vs 60/72/78) and `π_resp` cannot slow BAT, so a residual misfit at low `q` is
the signal that *some* selection is needed; the fit tightens through the defensible band and, past it,
the first milestone starts to *under*-fire (BAT too healthy). The **no-plateau** number is largely a
different story: with no plateau to lift, the left-truncation only shifts medians and is absorbed by
the Weibull `k`/`r`/shape fit, so the HR stays roughly invariant to `f` — until, at high `f`, the fit
is pushed onto a boundary and the panel is flagged degenerate/excluded. Enrollment selection is
therefore chiefly the *plateau-shape* lever, and the natural companion to the venetoclax-cure and
composition knobs for building a bear case on the comparator arm.

### 2.6 Bayesian priors on the BAT plateau (an alternative to the composition lever)

One way to set the BAT-arm long-term-survivor fraction (π_c) is a Beta prior, with the
GPS plateau following from the data constraint (Section 4.4). The explorer replaces this abstract
prior with the clinically-grounded **BAT composition** (Section 2.5) and the **enrollment-selection
lever** (Section 2.5.1), which together set π_BAT directly; the Beta priors below are retained only as the
historical mapping from a one-number prior to a P(success). Priors are **analyst choices [A]**:

| Prior | Beta(a,b) | Mean π_c | Rationale |
|-------|-----------|----------|-----------|
| Optimistic | Beta(4.55, 30.45) | 0.13 | BAT ≈ historical (6–8 mo, low plateau). |
| Base | Beta(5.10, 24.90) | 0.17 | Steering-committee ~8-mo BAT anchor [R3]. |
| Skeptical | Beta(8.10, 21.90) | 0.27 | Venetoclax-era BAT substantially improved. |

### 2.7 GPS non-responder subgroup (`build_plateau`, non-responder path)

| Parameter | Value | Type | Source / reasoning |
|-----------|-------|------|--------------------|
| Non-responder fraction f_nr | swept 0–40% (default 20%) | [A] | Anchored to the ~80% WT1 T-cell response rate [R4] ⇒ ~20% immunological non-responders. |
| Non-responder survival | = Observation component (median 6 mo, cure 8%) | [A] | User's specification: non-responders get no vaccine benefit → behave like best-supportive-care. |
| Responder cure | refit to events given f_nr & BAT | [D] | Increases ~56% → ~91% as f_nr rises 0 → 40% (base preset, 2% natural death); the GPS *arm* cure stays ~57% because the rising responder cure offsets the larger non-responder share (Section 6). |

### 2.8 Survival-shape stress controls (the explorer's headline knobs)

These do not change the milestones — they change the *shape* fit to them, which is exactly the
unidentified question. All are user-controlled in the explorer.

| Control | Range / default | Type | Role |
|---------|-----------------|------|------|
| No-plateau tail **shape** (Weibull) | 0.35–1.5, **fitted** by default (manual override) | [D]/[A] | Shape of the no-plateau responder Weibull. In **auto** mode it is a *fitted* parameter of the no-plateau model (alongside the BAT median and GPS/BAT ratio) and the slider merely displays the fitted value. Tick **override** to pin it and explore: **shape < 1 = heavier tail** (both arms survive long, smaller implied effect); shape → 1 is exponential. Controls the orange "no-plateau" number only. |
| Enrollment timing (median) | 0–1, default 0.50 (≈ median Mar 2023) | [A] | Slides the monthly accrual between an earlier (flat) and a later (back-loaded) profile; the **implied median enrollment date** and cumulative-at-anchor counts are displayed live (Section 2.3). The sourced anchors hold the median to ~Q1–Q2 2023. |
| Per-component shape **k** | ≥0.3, default 1 | [A] | Weibull shape of each BAT component's non-cured tail (Section 2.5). |

### 2.9 Natural (non-disease) death rate

The REGAL population is an AML second-complete-remission cohort that is **mostly in its sixties**, so
a non-trivial share of deaths is background, age-related mortality rather than disease relapse. The
explorer makes this an explicit, adjustable assumption.

| Control | Range / default | Type | Role |
|---------|-----------------|------|------|
| Natural death rate | 0–10%/yr, default 2% | [A] | All-cause background mortality, applied **equally to both arms** as an independent competing risk. ~2%/yr ≈ the US all-cause rate for ages 60–69; raise to stress-test an older or frailer cohort. |

**Mechanics.** The annual fraction `p` is converted to a constant monthly hazard
`h = −ln(1 − p) / 12` and overlaid as a multiplicative survival factor `S_nat(t) = e^(−h·t)` on every
arm. Because it is common to both arms, the pooled all-cause survival is simply
`S_pool^all(t) = S_pool^disease(t) · S_nat(t)`. This factor enters the milestone fit (Section 3), so the
calibration *attributes the observed 60/72/78 deaths to disease + background mortality*: a higher
natural rate implies disease-specific survival is actually somewhat **better** than the raw milestones
would otherwise suggest. In the Monte-Carlo (Section 4), each subject draws an independent exponential
natural-death time `T_nat = −ln(u)/h` and dies of whichever cause comes first
(`survival = min(disease, T_nat)`); this also caps the "cured" (plateau) subjects, who otherwise never
contribute an event.

**Effect on the readout.** Natural mortality (i) thins the plateau and shortens medians — e.g. it gives the
GPS arm a finite ~78-mo all-cause median where the disease-only plateau never crosses 50% — and (ii) brings
the 80th-event trigger *forward*: the cure-mixture "reached" fraction climbs from ~82% at 0% to ~100% by
2%/yr. Two forces then push P(success) in opposite directions — background deaths are *non-differential*,
which *dilutes* the treatment contrast (downward), but the more-reliable trigger removes stalled sims that
never reached significance (upward). In the explorer's presets the trigger-reliability gain dominates, so the
base-preset plateau P(success) rises gently with the natural rate (≈ **94% → 96% → 99% → 100%** across
0 / 2 / 5 / 10 %/yr, with the GPS arm cure rising 51% → 57% → 66% → 80% over the same range). Dilution would
win only where the trigger already fires in ~100% of sims; at the realistic ~2%/yr default the net move is a
few points.

### 2.10 Interim futility consistency check

A sourced fact that the earlier versions left on the table: at the **60-event interim** the IDMC
reviewed the trial and recommended continuation — i.e. it **cleared the pre-specified futility look**
[R4][R5]. That is information about the arm separation, because a scenario in which GPS shows little
or no benefit by the interim would have been *stopped*, not continued.

The explorer now uses this as a **consistency check on the arm split** rather than leaving the split
entirely free:

| Control | Range / default | Type | Role |
|---------|-----------------|------|------|
| Interim-analysis events | default 60 | [S] | The event count at the IDMC interim (SAP [R2]). |
| Interim futility HR | default 1.00 | [A] | The trial is taken to have been on track for futility-stop only if the *implied* HR at the interim was below this threshold. 1.00 = "no benefit trend"; tighten it (e.g. 0.85) to impose the stronger reading that continuation implied a real interim signal. |

**Mechanics.** In the Monte-Carlo the model already simulates every event time, so it computes the
implied Cox/log-rank HR at the moment the 60th death occurs (median across sims) exactly as it does
for the 80th. If that **implied interim HR exceeds the futility threshold**, the scenario is
inconsistent with the disclosed "continue past futility" and is flagged as implausible in the metrics
panel and fit note. This converts the arm split from a fully free knob into a **bounded** one: BAT
assumptions that imply GPS was barely separating by the interim are ruled out.

**Caveat.** The futility *boundary* itself is an assumption [A], not a published number, so it is an
adjustable input. At the default 1.00 even the pessimistic **bear corner** clears it (implied interim
HR ≈ 0.7), so the constraint mainly excludes extreme anti-GPS scenarios; tightening the threshold
makes it bite harder. It is a soft, user-controlled constraint, deliberately not a hard gate.

### 2.11 Loss to follow-up (administrative censoring)

Distinct from natural death (Section 2.9, which *is* an event), some patients leave the study before
dying — withdrawal, lost to follow-up, administrative censoring. These patients contribute follow-up
but **no death event**, so they slow event accrual.

| Control | Range / default | Type | Role |
|---------|-----------------|------|------|
| Loss to follow-up | 0–10%/yr, default 0 | [A] | Annual dropout rate, applied to both arms as an independent censoring process. 0 = complete follow-up; comparable trials run ~3–10%. |

**Mechanics.** Each subject draws an independent exponential censoring time `T_cens = −ln(u)/h_c`
(monthly hazard `h_c = −ln(1−p)/12`); if it precedes death the subject is censored (no event, but
counted alive "before censoring" in the per-arm 80th-event split). The same thinning enters the
milestone fit: the expected *observed* deaths by a date use
`∫ S_cens(t) dF_death(t) = e^{−h_c τ}(1−S(τ)) + h_c ∫₀^τ (1−S(t)) e^{−h_c t} dt` per cohort
(closed-form reduces to `1−S(τ)` when `h_c = 0`), so the fit stays calibrated to 60/72/78 with the
underlying disease survival adjusted for the censoring. At default 0 the model is unchanged.

**Effect.** Dropout meaningfully lowers P(success) and can stall the trigger: at the base preset the
plateau P(success) falls ~96% → 93% → 88% → 60% across 0 / 3 / 5 / 10 %/yr, and the 80th-event
"reached" fraction starts dropping (~89% at 10%). It is non-differential, so it dilutes the contrast
and removes events; unlike natural death it does not bring the trigger forward.

**Important reading of this control.** Because the censoring is folded into the *fit*, raising the
slider re-infers a **markedly deadlier underlying disease** to still reproduce the fixed 60/72/78
counts (some of those deaths are now "hidden" by dropout) — the GPS median moves ~78 → 38 → 24 mo
across 0 / 5 / 10 %/yr. So the P(success) decline is **not** merely "fewer observed events"; the
slider also reshapes the disease curve. That coupling follows from holding the milestones fixed, but
it is the key thing to internalize about what this control does.

### 2.12 Milestone weighting and fit uncertainty

The pooled fit minimizes a weighted squared error over the three milestones (Section 4.3). Two
controls expose the robustness of that fit:

| Control | Default | Type | Role |
|---------|---------|------|------|
| Milestone weighting | weighted 1 / 2 / 4 (toggle to equal 1 / 1 / 1) | [A] | The default up-weights the most recent (most informative) milestone; the **unweighted** toggle treats 60/72/78 equally, testing whether the weighting choice drives the answer. At base it barely moves the fit (GPS median ~78 → ~79 mo). |
| GPS-median Poisson interval | reported, not set | [D] | The event counts carry Poisson sampling noise, so the explorer refits at each count ±√n and reports the resulting **~68% interval on the derived GPS median** (e.g. ~23–222 mo at base). Its width shows how weakly three counts constrain the tail. |

---

## 3. Calibrated / derived outputs [D]

Representative values at the **base preset** (f_nr = 20%, natural death 2%/yr, fitted no-plateau tail,
enrollment selection q = 25%, 0% loss-to-follow-up, weighted fit); every number is a function of the user controls in Sections
2.5–2.12, so treat these as a centre point, not a
fixed result. Monte-Carlo figures carry ±2–3 pp simulation noise at the default sim budget.

| Quantity | Value (base preset) | Source |
|----------|---------------------|--------|
| Median enrollment date | ≈ Mar 2023 (cumulative ≈ 30 / 102 / 126 by Apr 2022 / Nov 2023 / Apr 2024) | `enroll` |
| BAT cure / median | ~19% · ~14 mo at the q=25% default; left-truncation sweeps it ~14% · ~9 mo (q=0) → ~29% · ~22 mo (q=50%), the cured fraction *rising* with q (Section 2.5.1) | `build_plateau` |
| GPS cure / median | ~58% · ~90 mo all-cause (disease-only plateau is never reached); both fall as selection rises and `π_resp` re-fits down | `build_plateau` |
| GPS median Poisson 68% CI | ~21 – 234 mo (from 60/72/78 ±√n) — wide: three counts barely pin the tail | `fit_ci` |
| Pooled long-term-survivor fraction | ~0.39 (disease plateau; all-cause survival decays below it) | `build_plateau` |
| Pooled median OS | **~20 mo** (above the ≥13.5 floor) | `build_plateau` |
| Implied HR at the 60-event interim | ~0.48 (clears the 1.00 futility threshold); drifts toward 1 as selection rises | `mc` |
| Patients alive at the 80th event | ~33 GPS / ~13 BAT (before censoring) | `mc` |
| **P(success) — plateau (cured fraction)** | **~94% at the q=25% default**; selection sweeps it ~100% (q=0) → ~13% (q=50%) (Section 2.5.1) | `build_plateau` + `mc` |
| **P(success) — no-plateau (fitted Weibull tail)** | **~4%** at base: with the tail free to fit, the deceleration is absorbed by a heavy shape (~0.45) rather than a large effect, so the honest fitted GPS/BAT ratio is only ~1.1× (HR ~0.94). At some presets (e.g. *low-venetoclax*) the fit hits a boundary and is reported **degenerate/excluded** | `build_no_plateau` + `mc` |
| 80th event reached in MC | ~100% of sims (both shapes) at the 2% natural-death default; the plateau drops to ~82% only at 0% natural death | `mc` |

The 2% natural-death default (Section 2.9) raises the GPS arm cure to ~57% and, because it guarantees
the trigger eventually fires, lifts the plateau "reached" fraction from ~82% (at 0%) to ~100%; that
removes the stalled sims and nudges the plateau P(success) up a few points relative to a no-mortality
run.

Sweeping the BAT composition shows how differently the two shapes read the same milestones. At the
**base preset** the plateau P(success) is ~94% but the honest, fitted-tail no-plateau P(success) is
only ~4% — with the tail free to fit, the milestone deceleration is explained by a heavy shape
(~0.45) rather than by a large treatment effect, so the implied GPS/BAT ratio is ~1.1×. At the **bear
corner** the ordering flips (plateau ~6%, no-plateau ~100%), and at the **low-venetoclax** preset the
no-plateau fit is boundary-bound and reported as **excluded**. That shape-dependence — not any single
central estimate — is the analysis's main output.

> The plateau model's ultimate *disease* dead fraction (~63%) nearly coincides with the 80-event
> trigger (63.5%), which is *why* real-world accrual has stalled at 78 — the cohort is essentially at
> its modeled disease asymptote, and the few remaining events are expected to come slowly from
> background (natural) mortality. The no-plateau model has no asymptote, so it reaches the 80th event
> readily; the real trial's stall at 78/80 therefore *mildly* favors the plateau model, though the
> milestones alone cannot adjudicate. (In the Monte-Carlo, the natural-death overlay lets the plateau
> model also reach the trigger in ~100% of sims, on a longer timeline than the disease process alone.)

---

## 4. Methodology and reasoning

### 4.1 Survival primitives

Both survival models are built on **one shared responder curve**, fit independently to the same
milestones. The shared primitive is a **Weibull** `Sweib(t) = exp(−(t/scale)^shape)` whose `scale` is
set so its median equals a target (`scale = median / (ln 2)^{1/shape}`); **shape < 1 gives a heavier
tail** and a monotone non-increasing hazard (no non-monotone hazard "hump"). The two panels differ by
one thing — the cured fraction:

- **Plateau (cured fraction):** `Sc(t) = π + (1−π)·exp(−(t / λ)^k)` — the shared Weibull responder
  **plus** a cured/long-term-survivor fraction π. λ is set so the non-cured median equals the
  component median (`λ = median / A(π)^{1/k}`, `A(π) = −ln[(0.5−π)/(1−π)]`); `k = 1` recovers the pure
  exponential. Rationale: cancer-vaccine effects classically manifest as a durable-remission (plateau)
  difference rather than a uniform hazard shift.
- **No-plateau (fitted Weibull tail):** the **same** Weibull `Sweib` with **no cured fraction** and a
  **fitted `shape`**. This is the explicit "the plateau may not be real" alternative. Freeing the tail
  shape is the key repair: the steep post-interim deceleration can now be absorbed by a heavy tail
  instead of forcing the treatment-effect parameter to a boundary (Section 4.7).

Both share the matching inverse-CDF samplers used by the Monte-Carlo (`sampNC` for the plateau
non-cured Weibull, `sampWeib` for the bare no-plateau Weibull).

### 4.2 Enrollment → expected deaths

For an enrollment cohort enrolled at calendar time `e` with `n` patients, expected cumulative
deaths at calendar time `T` are `Σ_cohorts n · D(T − e)`, where `D(τ)` is the fraction *observed*
dead by `τ`. With complete follow-up `D(τ) = 1 − S(τ)` and `S` is the **all-cause** survival
`S_disease · S_nat` (Section 2.9); under loss-to-follow-up at hazard `h_c` (Section 2.11) the observed
fraction is thinned to `D(τ) = e^{−h_c τ}(1−S(τ)) + h_c ∫₀^τ (1−S(t))e^{−h_c t}dt`. This convolution
is the forward model linking a survival curve to the disclosed event counts; folding background
mortality and censoring into `D` is what lets the fit split the observed deaths between disease,
natural causes, and patients who left before dying.

### 4.3 Pooled calibration

The pooled curve is `0.5·S_BAT + 0.5·S_GPS`. The explorer fits its free parameters to the three
(date, deaths) milestones by **weighted least squares**, with weights `WT = [1, 2, 4]` that
up-weight the most recent (and most informative) milestone (a **toggle** switches to equal weights
`[1, 1, 1]` to check the choice is not load-bearing — at base it shifts the GPS median by ~1 mo), over
a coarse grid followed by three local-refinement passes. Sampling uncertainty in the counts is
propagated by refitting at each milestone ±√n, giving a ~68% Poisson interval on the derived medians
(Section 2.12). For the plateau model there is a **single** free parameter — the GPS responder
cure `π_resp` — fit over a 1-D grid plus local refinement. The BAT arm is fully determined by the
component medians and the enrollment-selection fraction `q` (Section 2.5.1): any longevity the
milestones demand beyond the raw component medians is supplied *explicitly* by `q` — a healthier
enrolled cohort — rather than by any hidden calibration. The enrollment shape is set by the
back-loading slider (Section 2.8) rather than marginalized. (An earlier Bayesian formulation used a
Poisson log-likelihood with a prior on π_BAT; the explorer replaces that with this transparent
point-fit + composition/selection levers.)

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
time); applies an independent exponential natural-death time as a competing risk
(`survival = min(disease, T_nat)`, Section 2.9), which also caps the cured subjects; draws an
independent loss-to-follow-up time and censors the subject (no event) if it precedes death
(Section 2.11); finds the calendar time of the `FINAL`-th (80th) death; censors everyone there; and
computes
the **log-rank score statistic = Cox score test = the trial's actual pre-specified test**, declaring
success when `z > z_crit = |ln(HRC)|·√FINAL / 2 = 2.024`. It returns P(significant), the fraction of
sims that reach the 80th event, and the median simulated HR. The same `mc()` runs on both the
plateau and the no-plateau model, producing the two headline numbers.

The same pass also reports three diagnostics that make the fit auditable: the **implied Cox HR at the
60-event interim** (the futility read-through of Section 2.10), a boolean for whether it clears the
futility threshold, and the mean **per-arm patients alive at the 80th event** (before censoring) —
e.g. ~34 GPS / ~12 BAT at the base preset. The alive-split is the same quantity external modelers use
as a sanity check on the arm decomposition.

### 4.6 Component-mixture BAT and non-responders

These replace any abstract π_BAT prior with clinically-grounded structure (Sections 2.5, 2.7).
**They add interpretability, not identifying information** — the blinded data still see only the
pooled curve, so refits absorb the new structure and leave P(success) largely unchanged
(Section 6).

### 4.7 The no-plateau (fitted-tail) model, the shape gap, and degenerate fits

To make "is the plateau real?" answerable rather than merely flagged, the explorer fits a second,
plateau-free model to the **identical** milestones. Both arms are the shared **Weibull** (`Sweib`)
with **no cured fraction**; **three** parameters are fit — a single BAT-median scale `k`
(`median_BAT = k·m̄`, with `m̄` the weight-averaged component median), the GPS/BAT median ratio `r`,
and the **Weibull tail `shape`** itself. The GPS arm is a **single** responder curve: unlike the
plateau panel it carries no non-responder (`f_nr`) sub-mixture, because under a true no-cure model
"responder vs non-responder" collapses to two medians on the same eventually-fatal family, which a
single fitted-tail curve already captures in aggregate — and dropping it removes the ~18–20 mo seam
(a spurious plateau shoulder) that the old two-component GPS arm produced. The same Monte-Carlo
(Section 4.5) then scores it.

**Why fit the shape (the actual repair).** Previously the tail was held fixed at the slider value,
leaving only `k` and `r` to match three *decelerating* milestones. Deceleration is a plateau signal a
single fixed-shape family cannot reproduce, so the optimizer drove `r` to its cap to fake the late
flattening — a degenerate fit that rendered as a falsely confident "HR ≈ 0.25, P(success) 100%,
median ratio pinned at ~6.9×". Making the **tail shape a free parameter** (range 0.35–1.5, free to go
heavy) lets the deceleration be absorbed by a heavy tail instead of by the effect size. At the base
preset the fit is now well-behaved: shape ≈ 0.45, GPS/BAT ratio ≈ 1.1×, milestones matched to
~59/74/77, giving an honest HR ≈ 0.94 and P(success) ≈ 4%. Two guardrails prevent the old
degeneracy from creeping back: **BAT (`k`) stays a free fitted parameter** (fixing it would delete the
very uncertainty the tool exists to represent), and **the tail is free to go heavy** (a light-tail
floor would re-force the boundary run-away).

**Degenerate-fit detection (the failure mode is the signal).** After fitting, the model tests whether
the solution sits on a parameter boundary — median ratio at its cap, BAT median at a bound, or tail
shape at the edge of its range — and adds a diagnostic that raises the ratio cap and checks whether
the fitted ratio *tracks* it (a sign the milestones do not pin it). If the fit is boundary-bound the
no-plateau panel is **not** shown as an HR/P(success); it is labelled **"degenerate — non-identified
(excluded)"** and the **milestone residual** (modeled vs observed 60/72/78) is surfaced as the
evidence that the data are plateau-shaped and a light-tailed, no-cure family cannot fit them. This is
by design: an unidentified boundary solution is a diagnostic, not a result, and must never render as a
clean number.

The signed difference between the two P(success) numbers (when both are identified) is the explorer's
headline output: a small gap means the conclusion is shape-robust; a large gap means the verdict
hinges on an assumption the blinded data cannot test. The trial's current stall at 78/80 is a weak,
data-side tilt toward the plateau model (a no-plateau curve reaches the 80th event readily; Section 3).

---

## 5. Key functions (reference)

Names below use the Python spelling; the JavaScript in `regal_explorer.html` uses the camelCase
equivalents (`build_plateau` → `buildPlateau`, `build_no_plateau` → `buildNoPlateau`, and the shared
`Sweib`/`sampWeib`/`wscale` primitives). The two implementations are function-for-function equivalent
(the Python `common()` reads its inputs from the `cfg` dict, where the JavaScript reads module-level
state, but the computed results match).

| Function | Purpose |
|----------|---------|
| `Acoef` / `lam` | Weibull coefficient `A(π) = −ln[(0.5−π)/(1−π)]` and scale `λ = median / A^{1/k}`. |
| `Sc(t, med, cure, k)` | Per-component cure-mixture Weibull survival (the plateau primitive = shared Weibull + cure, Section 4.1). |
| `Sweib(t, scale, shape)` / `wscale(med, shape)` | The shared Weibull responder survival (the no-plateau primitive; shape < 1 = heavier tail) and the median→scale map `scale = median/(ln 2)^{1/shape}`. |
| `sampNC` / `sampWeib` | Inverse-CDF samplers for the plateau non-cured Weibull and the bare no-plateau Weibull times (Monte-Carlo draws). |
| `enroll(bl, N)` | Monthly enrollment cohorts summing to `N`, interpolating flat↔back-loaded by `bl` (Section 2.3). |
| `common(cfg)` | Shared setup: normalized weights, clamped per-component params, cohorts, milestones, fit weights. |
| `build_plateau` | Plateau model: fits the single free parameter `π_resp` (GPS responder cure) to the milestones with the BAT arm fixed by components + selection `q`; returns per-arm `Sbat/Sgps/Spool`, cures, medians (Sections 4.3–4.4). |
| `build_no_plateau` | No-plateau model: fits BAT-median scale `k`, GPS/BAT ratio `r`, **and the Weibull tail `shape`** (auto), with a single-curve GPS arm; flags boundary-bound solutions as degenerate/excluded (Section 4.7). |
| `median(S)` | Bisection median of a survival function (`∞`/"NR" if never below 0.5 within 900 mo). |
| `mc(M, nsim)` | Monte-Carlo trial: enrollment → per-arm death draws → censor at the 80th event → **log-rank/Cox score test**; returns P(significant), 80th-event-reached fraction, median HR (Section 4.5). |
| `figure()` / `render` + `chart*` | 9-panel figure (py, 3×3 grid) / live SVG charts and metrics panel (html): `chart` (survival), `chartAccrual`, `chartHist`, `chartDiverge`, `chartEnroll`, `chartPower`, `chartSelect`. |

---

## 6. Principal findings

1. **The framework matches the trial.** The confirmed primary test is a stratified Cox at one-sided
   0.025 with an OBF interim [R1]; the model's significance machinery and HR ≤ 0.636 threshold are
   consistent with it.
2. **Blinded pooled survival is high:** ~33–37% modeled plateau, ~16–21-mo median — far above the
   ~6–8-mo historical/contemporary control. Something is keeping these patients alive.
3. **Under the plateau model, P(success) is governed by the BAT-quality assumption.** With a
   clinically-built BAT composition it stays high (~94% at base) and is hard to push down
   without assuming venetoclax maintenance is both dominant *and* ~30–36% durable — the
   "bear corner," where it collapses (~6%).
4. **Structural refinements are absorbed by the pooled fit.** Component-mixture BAT and a 0–40%
   non-responder subgroup each leave P(success) ≈ unchanged, because the data fix the pooled
   plateau and refits merely redistribute it (e.g. raising the non-responder fraction forces the
   responder cure up). This *localizes* the uncertainty rather than resolving it.
5. **The shape assumption — not the arm split — is what moves the answer.** The no-plateau model
   fits the same milestones with a *fitted* Weibull tail and no cured fraction. When identified, it
   reads the milestones very differently from the plateau model (at base: plateau ~94% vs a fitted-tail
   no-plateau ~4%, because a heavy tail explains the deceleration with almost no arm separation);
   the ordering flips under other compositions (bear corner: plateau ~6% vs no-plateau ~100%). Where a
   free, heavy tail *still* cannot fit the decelerating milestones, the no-plateau fit is boundary-bound
   and reported as **degenerate/excluded** rather than as a number — that exclusion is itself evidence
   the data are plateau-shaped. That plateau-vs-tail gap (or the exclusion) is the irreducible,
   blinded-data-proof uncertainty the explorer reports as the headline.

---

## 7. Limitations and the load-bearing assumption

- **The plateau may not be real — now operationalized, not eliminated.** The pooled plateau is
  extrapolated from three event counts under a mixture-cure assumption. The explorer addresses this
  head-on by also fitting a heavy-tailed, no-cure **Weibull** model with a *fitted* tail shape
  (Section 4.7) and reporting both P(success) numbers. This *quantifies* the sensitivity but does not
  resolve it: whether long-term responders are truly cured remains the single load-bearing,
  blinded-data-proof assumption. The divergence is composition-dependent (the plateau is *not*
  uniformly the optimistic choice), and where even a free heavy tail cannot reproduce the
  decelerating milestones the no-plateau fit is non-identified and reported as excluded — a signal
  that the data themselves lean plateau-shaped, though the milestones alone cannot prove it.
- **Decomposition is unidentified.** All arm-level conclusions are prior-/assumption-driven; the
  blinded data cannot adjudicate them.
- **Delayed vs sustained separation.** The cure-difference structure assumes early, sustained
  separation (favorable to the Cox test). A genuinely *delayed* separation would violate PH and the
  committed Cox test could under-detect — the one shape where this risk bites.
- **Per-component BAT survival and composition are assumptions** [A], not patient-level data; they
  are the intended user levers.
- **Natural mortality is a flat, independent hazard.** Background death (Section 2.9) is modeled as a
  single constant all-cause rate (default 2%/yr), common to both arms and independent of the disease
  process. Real age-related mortality rises across the multi-year follow-up, and non-relapse mortality
  in a post-induction AML CR2 cohort can exceed general-population rates; the 0–10%/yr slider is the
  lever for stress-testing that, but the constant-hazard, disease-independent form is a simplification.
- **Promotional bias.** Several anchors (e.g. the ~8-mo BAT figure, the "longer-than-expected
  survival" framing) originate with SELLAS or affiliates and should be discounted accordingly.
- **Interim futility pass is a soft check, not a hard gate.** The IDMC's continuation past the
  60-event futility look is now used as an adjustable consistency constraint on the arm split
  (Section 2.10), flagging implausible scenarios rather than rejecting them outright; the futility
  boundary itself is an assumed number.
- **Loss to follow-up is modeled as a flat, independent rate.** Administrative censoring (Section 2.11)
  enters both the fit and the simulation, but as a single constant all-cause-independent hazard
  (default 0); real dropout is time- and arm-varying.
- **Enrollment selection is an idealized sharp filter.** The eligibility lever (Section 2.5.1) screens
  on *realized* survival — it assumes the criteria perfectly remove the patients who would in fact die
  soonest. Real criteria select on covariates only *correlated* with survival, so a given `esel` is an
  upper bound on how cleanly eligibility can enrich the cohort; treat it as "how much healthier could
  the enrolled population plausibly be," not a literal drop-rate. It is applied within each component
  (holding the composition weights fixed) and equally to both arms.
- **Not incorporated (conservative):** stratification of the Cox model (the trial stratifies; the
  simulation does not — a minor, likely slightly power-increasing, difference).

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
