# REGAL Trial Reverse-Engineering Model — Documentation

**Subject:** SELLAS Life Sciences (NASDAQ: SLS) Phase 3 REGAL trial (NCT04229979) of
galinpepimut-S (GPS) vs best available therapy (BAT) as maintenance in AML second complete
remission (CR2), non-transplant.

**Purpose:** The legacy v1 model calibrates assumed pooled survival families to the publicly disclosed
death-event milestones, decomposes them into arm-level survival under explicit BAT assumptions, and
Monte-Carlo simulates a fixed-scenario rejection rate. It does **not** estimate a posterior probability
for the actual ongoing trial. The plateau and bounded no-cure panels are alternative parametric
explanations using the same BAT arm. Their boundary and residual classifications are diagnostics, not
formal hypothesis tests, and cannot identify whether the pooled tail is GPS-specific.

**Deliverables:** `regal_explorer.html` (self-contained interactive explorer) and
`regal_explorer.py` (the legacy engine in Python, with additional audit-only interim fields, a CLI
summary, and a 9-panel figure).

> **Legacy-v1 status.** The current `ps` values are fixed-scenario rejection rates conditional on the selected assumptions
> and on reaching the final analysis. V1 does not condition on the observed decision to continue
> after the 60-event interim. See [`V2_IMPLEMENTATION_PLAN.md`](V2_IMPLEMENTATION_PLAN.md) for the
> corrective rebuild.

**Last updated:** 2026-08-22 · **Status:** research/analysis tool, not investment advice.

---

## 0. Epistemic frame (read first)

The single most important structural fact: **REGAL is blinded.** Public disclosures give the
*pooled* number of deaths at several dates, never the per-arm split. Consequently:

- The event milestones constrain several enrollment-weighted integrals of the pooled survival
  curve, but three aggregate counts do **not** identify the curve or its long-term shape.
- The **decomposition into GPS and BAT arms is *not* identifiable** from blinded data. It requires
  assumptions about the BAT arm and the GPS effect family. Every v1 rejection-rate value is therefore a
  fixed-scenario operating characteristic, not a claim to know the confidential outcome.

This is a forecast built from public information (press releases, SEC filings, ClinicalTrials.gov,
the published trial-design paper) — the same class of event-driven analysis used in mainstream
biotech equity research. It does not access, infer, or attempt to unblind confidential trial data.

A recurring v1 finding is that arm-level refinements are absorbed by the assumed pooled-family fit.
This does not identify the pooled tail or either arm. The explorer compares a GPS cure-mixture family
with a bounded no-GPS-cure family using identical BAT assumptions. State A/B/C records boundary,
residual-misfit, or adequate-interior fit; it is a sensitivity diagnostic rather than a hypothesis
test of a biological mechanism.

---

## 1. Module map

The legacy survival, fitting, and final-analysis engine is delivered in two forms:

| File | Role | Key outputs |
|------|------|-------------|
| `regal_explorer.html` | Self-contained legacy explorer with BAT, survival, enrollment, censoring, and shape controls plus live diagnostic charts. | fixed-scenario rejection rates, median HR, implied interim HR, event reach, fit status, per-arm curves |
| `regal_explorer.py` | Python engine (`bat_arm`, `build_plateau`, `build_no_gps_cure`, `mc`) with a CLI, 9-panel figure, and audit-only interim-efficacy fields not present in the browser. | scenario rates, A/B/C fit status, preset/non-responder sweeps, interim audit fields |
| `trial_design.py` | Legacy classical audit boundary plus isolated v2 Lan-DeMets spending and protocol-factor stratified score analysis. | legacy and v2 efficacy boundaries; stratified primary statistic; named one-step diagnostics |
| `audit/interim_efficacy_replay.py` | Fixed-seed equal-strata operating-characteristic replay. | reproducible interim efficacy crossing, final rate, and median HR |
| `simulation.py` | Isolated v2 event-driven decision process; not consumed by the legacy explorer. | 60/80-event cutoffs, efficacy/futility/continuation branches, final decisions, canonical operating characteristics |
| `audit/v2_trial_decision_validation.py` | Fixed-seed v2 design audit. | null type-I error, branch conservation, and paired futility-threshold sensitivity grid |
| `survival_models.py` | Corrected, isolated v2 survival primitives; not consumed by the legacy explorer. | scale-aware OS/net cure mixtures, population mortality, pre-outcome frailty/case mix, post-selection randomization |
| `bat_regimens.py` | Corrected, isolated v2 BAT representation; not consumed by the legacy explorer. | joint planned-stratum/delivered-regimen pathways, combination exposure, one survival profile per patient, and scenario-role labels |
| `data/regal_public_history.json` | Versioned WP5 public evidence, current through 11 Aug 2026. | typed enrollment/event counts, observation and announcement dates, source notes, reporting-lag distributions |
| `event_likelihood.py` | Isolated v2 public-history likelihood; not consumed by the legacy explorer. | registry-anchored fixed-N accrual, correlated Poisson-multinomial event increments, event-80 announcement right censor |
| `audit/v2_public_history_validation.py` | Deterministic WP5 data/likelihood audit. | anchor reachability, integer-count correlation, reporting-lag sensitivity |
| `posterior.py` | Isolated WP6 fixed-scenario conditioning engine; not consumed by the legacy explorer. | consistent latent histories, exact public-count conditional draws, continuation importance weights with reported base fallbacks, conditional final projection, ESS diagnostics |
| `audit/v2_interim_conditioning_validation.py` | Fixed-seed WP6 rare-continuation stress audit. | base-versus-centered proposal coverage and public-history compatibility agreement |

The two legacy implementations share one enrollment reconstruction, one set of survival primitives,
the same significance threshold (Section 2.1), and — critically — **one shared BAT arm** (`bat_arm`): the plateau and bounded-alternative
panels consume byte-identical BAT (same per-component medians, cures, shapes, and left-truncation
selection), so they are literally **one biological lever apart**. Both are fit to the **identical
milestones**; they differ only in the GPS **responder** component. The isolated v2 modules do not
reuse the legacy accrual or fitting path:

- **`build_plateau` — plateau (GPS cure).** The shared BAT plus GPS responders modelled as a Weibull
  **cure-mixture**. Only the GPS responder cure `π_resp` is fit to the events; the BAT arm is fixed by
  the component medians plus enrollment selection (Sections 4.3–4.4).
- **`build_no_gps_cure` — bounded no-GPS-cure alternative.** The **same** BAT; GPS responders swap the cure-mixture
  for a **no-cure Weibull** with two fitted parameters — a GPS responder median `m_G` and a tail shape
  `s_G` (GPS non-responders still track Observation). BAT is *fixed on purpose* here (Section 3). The
  fit yields a three-state diagnostic; a parameter boundary is reported as non-identified (Section 4.7).

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
| **Legacy-v1 significance threshold** | observed one-step HR ≤ **0.636**, i.e. z_crit = \|ln 0.636\|·√80 / 2 = **2.024** (one-sided p ≈ 0.0215) | [D] | Preserved for v1 numerical reproducibility; matches SELLAS's stated design effect size but is not the v2 boundary implementation. |
| **V2 planned efficacy boundaries** | `z60 = 2.339711`; `z80 = 2.011777` | [D] | Calculated from one-sided 0.025 Lan–DeMets O'Brien–Fleming spending at information fractions 60/80 and 1; recalculated from realized event-count information if a tied cutoff overshoots. |
| BAT-arm allowed agents | observation/hydroxyurea, hypomethylating agents (HMA), venetoclax, low-dose ara-C (LDAC); targeted maintenance (e.g. FLT3i) **excluded** | [S] | Targeted Oncology [R4]; OncLive [R5]. |

> **Note on the analyses.** V1 uses an unstratified score and one-step `exp(U/V)` HR with the fixed
> 0.636 threshold. V2 now uses the stratified log-rank score—equivalently the beta-zero score test of
> the treatment-only stratified Cox model—for efficacy at the calculated Lan–DeMets boundaries. It
> accepts all public protocol factors as separate columns and forms risk sets within their combined
> strata. The unstratified score and one-step HR remain available only as named diagnostics.

### 2.2 Event milestones (pooled deaths, calendar)

| Date | Cumulative deaths | % of 126 | Type | Source |
|------|-------------------|----------|------|--------|
| Announced 2024-12-10 | 60 | 47.6% | [S] threshold hit | SELLAS announced that the prespecified threshold had been reached, but did not disclose the exact threshold date [R9]. WP5 integrates a 0/7/14-day PMF with weights 4/21, 13/21, and 4/21; IDMC continuation was announced Jan 2025 [R4][R5]. |
| 2025-12-26 | 72 | 57.1% | [S] exact/as-of | CRO count as of 26 Dec, announced by SELLAS 29 Dec 2025 [R3]. |
| 2026-05-11 | 78 | 61.9% | [S] | SEC 8-K exhibit 99.1 and Q1-2026 release, 12 May 2026 [R6]. |
| 2026-08-11 | 80 not announced; last exact count remains 78 | — | [S]/right censor | SELLAS said REGAL was still approaching event 80 and would announce when it occurs [R8]. WP5 treats this as censoring of the announcement process with the same 0/7/14-day PMF, not as an invented event count on 11 Aug. |

The exact-count increments were 12 deaths over roughly 12.5 months and then six over roughly 4.5
months. Event 80 remained unannounced on 11 Aug 2026. These sparse cumulative disclosures strongly
constrain calendar event accrual but do not identify a unique pooled survival curve or either arm.

### 2.3 Enrollment reconstruction

The exact monthly accrual is **not public**; the curve is reconstructed [A] to honor the sourced
anchors below. Its shape is controlled by the **enrollment-timing slider** (Section 2.8), which slides
accrual between an earlier (flat) and a later (back-loaded) profile. Because the *median enrollment
date* is the quantity that actually drives time-from-randomization at each milestone, the explorer
**displays the implied median date** (default ≈ Mar 2023) live, together with the cumulative
patients enrolled at the sourced anchor dates, so drift away from the anchors is visible.

| Anchor | Value | Type | Source |
|--------|-------|------|--------|
| Registration | Actual study start 2021-02-08 | [S] | ClinicalTrials.gov / NCT04229979. |
| First accrual anchor | First 20 enrolled before protocol version 3 (Apr 2022); month-end is an explicit operational cutoff because the individual date is not public | [S]/[A] | Design paper [R1]. |
| Intermediate accrual anchor | Sponsor projected November 2023 completion outside China with 20-25 of 126 still anticipated from China: interval 101-106, rounded center 104 | [S]/projection | SELLAS, 12 Oct 2023 [R10]. It centers the provisional WP5 reference path but is not likelihood evidence. |
| Enrollment complete | 126 randomized, completed Apr 2024; month-end used because the individual completion date is not public | [S] | Later sponsor reconciliation [R11]. |
| China cohort (via 3D Medicines) | enrolled ~Dec 2023 – Mar 2024 | [S] | SAP/partnership disclosures [R2]. |
| Last patient in | ~March 2024 | [S] | CEO, May 2026 conference [R7]. |
| Original expectation to 80th event | 12–15 months after last patient (~mid-2025) | [S] | CEO, May 2026 conference [R7]. |
| **Legacy code reconstruction (base)** | slow 2020–21 → heavy 2022–23 → China bolus, summing to 126; implied median ≈ Mar 2023 | [A] | Preserved for v1 reproducibility, but starts before study opening and misses the first-20 anchor. |
| **V2 WP5 accrual reference** | fixed N=126; support begins 2021-02-08; piecewise-uniform mass centered exactly on 20/104/126 | [A] | `event_likelihood.py`; no simulated patient can predate opening and every anchor is tested for reachability. It is not an independent Bayesian prior. |

WP5 evaluates enrollment constraints jointly under a fixed-N multinomial model. Event likelihood is
also joint: each patient supplies probabilities for mutually exclusive calendar intervals, and a
dynamic program sums all patient allocations compatible with 60/72/78 plus the event-80 right
censor. These are the WP5 likelihood ingredients for `P(public history | fixed scenario)`, not by
themselves a quantity conditioned on the IDMC decision and not a posterior forecast. WP6 now samples
one enrollment history and uses those same entry dates for the enrollment constraints and every
patient calendar event CDF, avoiding an unjustified product of mismatched marginal and conditional
quantities. It conditions patient event intervals on an allowed 60/72/78/right-censor integer
trajectory, requires a finite stratified 60-event statistic in the continuation region, and projects
the unresolved conditional event-time tails to the final 80-event analysis. The resulting quantities
remain fixed-scenario conditional projections until WP7 averages across models and parameters.

The two unknown threshold-to-announcement lags are modeled as independent draws. Each uses a
three-point PMF at 0/7/14 days with weights 4/21, 13/21, and 4/21. This exactly preserves the mean
and variance of the original discrete-uniform 0–14 day sensitivity distribution while reducing the
joint lag mixture from 225 to nine branches; it remains a computational sensitivity assumption, not
a sponsor disclosure.

The convenience calendar-CDF adapter assumes complete death ascertainment apart from administrative
censoring; it does not add an independent loss-to-follow-up or withdrawal process. That assumption
is explicit because any future attrition model must enter through an adjusted calendar event CDF
rather than silently alter the observed event history.

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
lever and are intended to be edited. The full sourcing for the default weights and per-component
survival parameters — regional composition (US/EU/China) and the CR2-discounted mixture-cure
literature — is collected in [`BAT_CONTROL_ARM_RESEARCH.md`](BAT_CONTROL_ARM_RESEARCH.md). The shape **k** generalizes the non-cured tail beyond a pure
exponential (`k = 1` reproduces the exponential): `k < 1` is a heavy tail (more long-term survivors),
`k > 1` accelerates. The palliative/observation components default to `k = 1.1` (mildly increasing
hazard as untreated relapse accrues) and venetoclax to `k = 0.78` — the published decreasing-hazard
Weibull OS shape reconstructed from VIALE-A (VEN+AZA), the plateau-forming tail.

| Component | Median OS (mo) | Cure fraction | Shape k | Type | Anchor / reasoning |
|-----------|----------------|---------------|---------|------|--------------------|
| Observation / BSC | 6.0 | 0.03 | 1.1 | [A] | Untreated CR2 relapses fast (mOS ~5–7 mo); durable-survivor tail ~2–5%. **Non-responders track this row.** |
| Hydroxyurea (palliative) | 5.0 | 0.02 | 1.1 | [A] | Palliative count-control in more symptomatic disease; poorest durable-remission. |
| HMA (aza/dec) | 12.0 | 0.10 | 1.0 | [A] | QUAZAR-style maintenance signal (CR1 plateau ~15–25%) discounted to CR2; exponential hazard per HTA fits. |
| Venetoclax (± HMA) | 12.0 | 0.15 | 0.78 | [A] | Best BAT option (VIALE-A mOS 14.7 / 24-mo OS 37.5%) discounted to CR2; k=0.78 published VEN+AZA tail. **Key bear/bull knob.** |
| LDAC | 7.0 | 0.08 | 1.1 | [A] | Modest activity (mOS ~5–10 mo); small responder tail. |

**Legacy-v1 presets** (selectable in the explorer; weights auto-normalize):
*Base* (27/8/22/35/8, ven cure 15%) · *Low-venetoclax* (33/12/30/15/10, ven cure 15%) ·
*Venetoclax-dominant* (8/4/18/60/10, ven cure 15%) · *Bear corner* (5/3/12/70/10, ven cure 25%) · *Bull corner*
(40/10/25/15/10, ven cure 10%) — the bear corner is the only legacy composition that pushes the
plateau scenario rejection rate clearly below 50%. The **bull corner** is the opposite extreme:
it credits BAT as little as is clinically defensible — venetoclax demoted to a floor weight and to a
poorly-durable 10% cure, its remaining weight relocated onto the low-cure active (HMA/LDAC) and
palliative (Observation/Hydroxyurea) components, with observation/hydroxyurea now the plurality of the
arm (~50%). It drives the plateau scenario rate toward its ceiling and pushes the bounded no-GPS-cure
alternative toward a boundary or residual-misfit status. Those statuses indicate sensitivity to the
chosen parameter box and BAT assumptions; they do not establish cure.

**Protocol-compatibility correction.** The trial publication describes approximately equal BAT
strata. Under a literal four-stratum interpretation, the primary allocation is approximately 25%
each to supportive care/hydroxyurea, HMA, venetoclax, and LDAC. A legacy equal-strata run makes BAT
slightly weaker because of the LDAC weight and is therefore more bullish: approximately 99.9%
scenario power, median HR 0.30, and a 94% probability of crossing the interim efficacy boundary.
The last quantity is reproducible with
`python3 audit/interim_efficacy_replay.py --nsim 10000`. This is one interpretation of the published
stratification statement, not proof of the realized regimen mix: the publication's "and/or" wording
allows combinations, and balanced planned strata need not equal delivered treatments. Accordingly,
venetoclax-dominant and bear remain clearly labeled allocation stress tests rather than primary
protocol reconstructions. The isolated v2 `bat_regimens.py` layer now stores planned stratum
separately from received regimen. A combination such as HMA + venetoclax records both component
exposures but selects one outcome profile, so it remains one patient in regimen and survival
marginals. Its primary configuration has four 25% planned strata; the current 27/8/22/35/8 component
weights are retained only as a legacy comparison. The primary's single-profile delivered-regimen
mapping is provisional until patient-level or realized-regimen evidence is available. In particular,
the venetoclax survival profile is reconstructed from VEN+azacitidine and is best supported for the
explicit HMA+VEN combination. Reusing that profile when co-therapy is unspecified—especially for
venetoclax monotherapy—can overstate BAT survival and should be revised when realized regimens are
known. V2 therefore exposes combination-regimen primitives without committing an evidence-free
combination allocation constant.

The v2 `BEAR_STRONG_BAT_STRESS` allocation pairs explicitly with the separate immutable
`BEAR_STRONG_BAT_COMPONENT_LIBRARY`, which changes only the venetoclax cure fraction from 15% to
25%. Keeping allocation and survival parameters separate makes the legacy bear assumption fully
reproducible without hiding it in component weights. Primary BAT designs require positive mass in
all four planned strata. Comparison and stress designs may retain a zero-probability pathway for an
absent stratum; such a pathway is never sampled and does not require an unused survival profile.

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
Observation 0.27 · Hydroxyurea 0.08 · HMA 0.22 · Venetoclax 0.35 · LDAC 0.08 → implied BAT cure
≈ 9%, BAT median ≈ 8.1 mo before enrollment selection (≈ 12% / ≈ 12 mo at the default 25% keep-strongest
filter). Weights follow the US/EU/China BAT-composition review (observation ~35%, venetoclax ~35%,
HMA ~22%, LDAC ~8%); two alternates ("low-venetoclax / access-constrained", "venetoclax-dominant /
US-heavy") bracket the range.

### 2.5.1 Enrollment selection (eligibility filter)

> **Legacy v1 mechanism.** The top-survivor truncation documented below remains only so the v1
> scenario outputs stay reproducible. V2 does not select patients using future survival. Its
> `survival_models.py` layer selects on baseline frailty before randomization, retains positive
> early-event probability, and does not mechanically increase the cure fraction. Non-unit disease
> frailty requires a net/relative-survival input; current OS inputs remain neutral until they are
> refitted or given a validated excess-hazard decomposition.

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

for each BAT component, for the plateau panel's GPS responders (cure-mixture), and for the no-GPS-cure
panel's GPS responder Weibull alike. This lifts every curve to its "top `100(1−f)`%" shape: the
long-term/cure fraction **rises** from `c` to `c/(1−f)` and the median lengthens, with no re-anchoring
of the Weibull scale (so the `c < 0.5` parameterization never breaks). The `min(1, ·)` clip is **kept
on purpose**: the near-flat segment before `t_f` is real **guarantee time**, with a direct correlate in
REGAL's *"estimated life expectancy > 6 months"* enrolment criterion — it is a feature, not an artifact.
The matching Monte-Carlo draw is the inverse-transform of the same left-truncation: draw
`u ~ Unif(0, 1−f)` (which keeps the strongest `1−f`), so a cured patient survives with probability
`c/(1−f)`, otherwise its non-cured time is drawn conditioned on exceeding `t_f`; the no-cure GPS
responder simply draws `u = (1−f)·rnd()`. At `f = 0` every expression collapses back to the unselected
model exactly.

**Applies to both panels, before the arm split.** Because BAT is **shared code** (`bat_arm`), the
selection is literally an **upstream transform of the pooled CR2 pool, applied identically in both
panels** — there is no second BAT copy to keep in sync. It is shared infrastructure, not one of the
assumptions that distinguishes the panels (the GPS responder family is the downstream distinction).
The truncation is non-differential across arms, so applied to a *fixed* arm split it
cannot bias the within-trial comparison. **Note, though, that the fitted HR is *not* strictly invariant
to `f`** here: because the milestones are held fixed and the arm split is *re-fit* at each `f` (the
plateau's `π_resp`, the bounded alternative's `m_G`/`s_G`), selection re-attributes survival to BAT and the fitted HR
drifts — e.g. the plateau `medHR` moves ~0.29 → 0.42 → 0.61 as `f` goes 0 → 0.25 → 0.40. This drift is
inherited from the (unchanged) plateau fit and is the correct consequence of pinning the blinded
milestones while the split re-calibrates; what selection cannot do is bias the comparison *at a fixed
split*. What clearly *does* move with `f` is the milestone fit, the scenario rejection rate, and the BAT cured
fraction (which rises as `π_BAT → π_BAT/(1−f)`). (At extreme `f` the re-fit can be pushed onto a
parameter boundary in the no-GPS-cure panel, which then reports State A (boundary/non-identified) — see
Section 4.7.)

**`q` is the single BAT-side lever.** With the BAT arm otherwise fixed by the component medians, `q`
is what determines how much of the milestone deceleration is attributed to a healthier enrolled cohort
versus to the GPS effect; the plateau fit's *only* free parameter is the GPS responder cure `π_resp`.
The default is **`q = 25%`** (mid-band; see below).

**Effect (base preset).** As `q` rises 0 → 25 → 50% the BAT median OS lifts ~8 → 12 → 19 mo and the BAT
cure fraction climbs ~9 → 12 → 18% (both plotted live in panel *(i)* / the "enrollment selection lifts
the BAT arm" chart, `S_BAT` and `π_BAT/(1−q)`, independent of the Monte-Carlo). To keep the pooled
60/72/78 held fixed, the fitted GPS responder cure falls ~0.87 → 0.78 → 0.59, so the plateau scenario rate
drops steeply ~100 → 94 → 13% — a healthier, harder-to-beat comparator leaves less residual to
attribute to GPS. Note the direction of the fit-check: at `q = 0` the raw medians *over*-produce early
deaths (modeled ~65/74/76 vs 60/72/78) and `π_resp` cannot slow BAT, so a residual misfit at low `q` is
the signal that *some* selection is needed; the fit tightens through the defensible band and, past it,
the first milestone starts to *under*-fire (BAT too healthy). Because BAT is shared, the **no-GPS-cure
alternative** rides the same BAT: as `q` rises its fitted GPS median `m_G` and tail `s_G` re-fit, and the
fit status can change (a healthier BAT can make an adequate interior State-C fit easier; an extreme
`q` can instead push a parameter to the State-A boundary or produce State-B residual misfit).
Enrollment selection is therefore chiefly the *plateau-shape* lever, and the natural companion to the
venetoclax-cure and composition knobs for building a bear case on the comparator arm.

### 2.6 Bayesian priors on the BAT plateau (an alternative to the composition lever)

One way to set the BAT-arm long-term-survivor fraction (π_c) is a Beta prior, with the
GPS plateau following from the data constraint (Section 4.4). The explorer replaces this abstract
prior with the clinically-grounded **BAT composition** (Section 2.5) and the **enrollment-selection
lever** (Section 2.5.1), which together set π_BAT directly; the Beta priors below are an alternative
one-number mapping from a prior to a scenario rejection rate. Priors are **analyst choices [A]**:

| Prior | Beta(a,b) | Mean π_c | Rationale |
|-------|-----------|----------|-----------|
| Optimistic | Beta(4.55, 30.45) | 0.13 | BAT ≈ historical (6–8 mo, low plateau). |
| Base | Beta(5.10, 24.90) | 0.17 | Steering-committee ~8-mo BAT anchor [R3]. |
| Skeptical | Beta(8.10, 21.90) | 0.27 | Venetoclax-era BAT substantially improved. |

### 2.7 GPS non-responder subgroup (`build_plateau`, non-responder path)

| Parameter | Value | Type | Source / reasoning |
|-----------|-------|------|--------------------|
| Non-responder fraction f_nr | swept 0–40% (default 20%) | [A] | Anchored to the ~80% WT1 T-cell response rate [R4] ⇒ ~20% immunological non-responders. |
| Non-responder survival | = Observation component (median 6 mo, cure 3%) | [A] | User's specification: non-responders get no vaccine benefit → behave like best-supportive-care. |
| Responder cure | refit to events given f_nr & BAT | [D] | Refits upward as f_nr rises 0 → 40% (base preset, 2% natural death); the GPS *arm* cure stays ~55–60% because the rising responder cure offsets the larger non-responder share (Section 6). |

### 2.8 Survival-shape stress controls

These do not change the milestones — they change the *shape* fit to them, which is exactly the
unidentified question. All are user-controlled in the explorer.

| Control | Range / default | Type | Role |
|---------|-----------------|------|------|
| No-GPS-cure GPS tail shape **s<sub>G</sub>** (Weibull) | 0.15–1.5, **fitted** by default (manual override) | [D]/[A] | Shape of the bounded alternative's GPS responder Weibull. In **auto** mode it is fitted alongside the GPS responder median m<sub>G</sub>; the slider displays that fit. Tick **override** to pin it and explore: **s<sub>G</sub> < 1 = heavier tail**; s<sub>G</sub> → 1 is exponential. A State-C result means only that this family fits adequately inside the selected parameter box. Controls the alternative's fit status only. |
| Enrollment timing (median) | 0–1, default 0.50 (≈ median Mar 2023) | [A] | Slides the monthly accrual between an earlier (flat) and a later (back-loaded) profile; the **implied median enrollment date** and cumulative-at-anchor counts are displayed live (Section 2.3). The sourced anchors hold the median to ~Q1–Q2 2023. |
| Per-component shape **k** | ≥0.3, default 1 | [A] | Weibull shape of each BAT component's non-cured tail (Section 2.5). |

### 2.9 Natural (non-disease) death rate

The REGAL population is an AML second-complete-remission cohort that is **mostly in its sixties**, so
a non-trivial share of deaths is background, age-related mortality rather than disease relapse. The
explorer makes this an explicit, adjustable assumption.

| Control | Range / default | Type | Role |
|---------|-----------------|------|------|
| Natural death rate | 0–10%/yr, default 2% | [A] | Legacy v1 overlays this competing risk on every component. V2 will apply it according to whether an input is OS or net survival. |

**Legacy-v1 mechanics.** The annual fraction `p` is converted to a constant monthly hazard
`h = −ln(1 − p) / 12` and overlaid as a multiplicative survival factor `S_nat(t) = e^(−h·t)` on every
arm. Because it is common to both arms, the pooled all-cause survival is simply
`S_pool^all(t) = S_pool^disease(t) · S_nat(t)`. This factor enters the milestone fit (Section 3), so the
calibration *attributes the observed 60/72/78 deaths to disease + background mortality*: a higher
natural rate implies disease-specific survival is actually somewhat **better** than the raw milestones
would otherwise suggest. In the Monte-Carlo (Section 4), each subject draws an independent exponential
natural-death time `T_nat = −ln(u)/h` and dies of whichever cause comes first
(`survival = min(disease, T_nat)`); this also caps the "cured" (plateau) subjects, who otherwise never
contribute an event.

**Survival-scale correction.** The background-mortality layer is appropriate for the cured fraction:
a flat plateau extrapolated indefinitely would imply immortality. However, the current literature
inputs for the non-cured components are already **overall survival**, so multiplying those terms by
`S_nat` double-counts background mortality. With the existing OS inputs the corrected mixture is
`S(t) = c·S_bg(t) + (1−c)·S_uncured,OS(t)`. If components are instead fitted to net/relative survival,
the correct form is `S(t) = S_bg(t)·[c + (1−c)·S_uncured,net(t)]`. V2 will store that scale explicitly
per component rather than removing background mortality altogether.

**Legacy sensitivity.** In the reproduced base run, changing the legacy overlay from 0% to 2% moves
the 80-event reach fraction from about **70% to 100%** and the fitted median HR from about **0.359 to
0.316**, while the fixed-scenario rejection rate barely moves, about **98.7% to 99.7%**. The overlay
therefore changes the event-stall narrative and fitted effect much more than the scenario
power. These values characterize the flawed v1 comparison; the corrected v2 mixture must be refitted.
The earlier documentation's ~82% reach value at 0% was stale.

### 2.10 Interim futility consistency check

A sourced fact that the earlier versions left on the table: at the **60-event interim** the IDMC
reviewed the trial and recommended continuation — i.e. it **cleared the pre-specified futility look**
[R4][R5]. That is information about the arm separation, because a scenario in which GPS shows little
or no benefit by the interim would have been *stopped*, not continued.

V1's primary output only compares its median simulated interim HR with an assumed futility threshold;
it does not stop trials at the interim or condition on the observed continuation region. The committed
audit path now also records the 60-event score and compares it with a classical two-look
O'Brien–Fleming efficacy boundary, solely to reproduce the equal-strata operating characteristic.
That discrete-look `c/√t` boundary is not the protocol's Lan-DeMets alpha-spending construction;
the numerical difference is small here, and it remains intact only for v1 reproducibility.

The isolated v2 path now calculates the planned Lan-DeMets spending boundaries
(`z60 = 2.339711`, `z80 = 2.011777`) and applies the protocol-factor stratified score at both
event-driven cutoffs. Every death tied at a cutoff is retained; observed events divided by the
planned 80 are used as the information proxy, and both sequential boundaries are recalculated to
preserve one-sided alpha. It simulates mutually exclusive efficacy-stop, assumed-futility-stop, and
continuation branches; continued trials can reject, not reject, or fail to reach the 80th event. This
implements the trial mechanics but does not yet use the fact that REGAL actually continued as
likelihood information—that conditioning remains WP6.

| Control | Range / default | Type | Role |
|---------|-----------------|------|------|
| Interim-analysis events | default 60 | [S] | The event count at the IDMC interim (SAP [R2]). |
| Interim futility HR | v1 default 1.00; v2 has no default and audits disabled/0.80/0.90/1.00/1.10/1.20 | [A] | A sensitivity row assumes futility stop when the diagnostic interim HR is at or above the stated threshold. Smaller thresholds impose a stronger continuation requirement. No row is asserted to be the unpublished protocol rule. |

**Mechanics.** V1 computes an unstratified score and implied one-step HR at the 60th death, then only
flags the median simulated HR against its UI threshold. In v2, `evaluate_event_driven_trial` excludes
patients not yet randomized at each calendar cutoff, retains every death tied at the cutoff, forms
risk sets separately within the combined protocol strata, recalculates alpha spending from realized
event-count information, and applies efficacy first, assumed futility second, and continuation
otherwise.
The unstratified score and one-step HR are named diagnostics; efficacy uses the stratified score.
`audit/v2_trial_decision_validation.py` uses paired canonical-normal draws to validate one-sided null
type-I error and show how the assumed futility threshold reallocates branches without Monte-Carlo
noise between sensitivity rows. It separately sends identical exponential GPS/BAT outcomes through
the complete patient-level enrollment, event-trigger, and protocol-factor stratified path as a
non-circular null check.

**Caveat.** The futility rule's form and boundary are assumptions [A], not published numbers. The v2
efficacy design therefore commits no futility default. Every HR threshold is labeled as sensitivity,
and the no-futility row preserves the nominal 0.025 efficacy design. Later conditioning must average
or stress-test these rules rather than treating any one threshold as known.

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

**Effect.** Dropout meaningfully lowers the scenario rejection rate and can stall the trigger: at the base preset the
plateau rate falls ~100% → 99% → 97% → 85% across 0 / 3 / 5 / 10 %/yr, and the 80th-event
"reached" fraction starts dropping (~83% at 10%). It is non-differential, so it dilutes the contrast
and removes events; unlike natural death it does not bring the trigger forward.

**Important reading of this control.** Because the censoring is folded into the *fit*, raising the
slider re-infers a **markedly deadlier underlying disease** to still reproduce the fixed 60/72/78
counts (some of those deaths are now "hidden" by dropout) — the GPS median moves ~78 → 38 → 24 mo
across 0 / 5 / 10 %/yr. So the rejection-rate decline is **not** merely "fewer observed events"; the
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

Representative values at the **base preset** (f_nr = 20%, natural death 2%/yr, fitted GPS tail s<sub>G</sub>,
enrollment selection q = 25%, 0% loss-to-follow-up, weighted fit); every number is a function of the user controls in Sections
2.5–2.12, so treat these as a centre point, not a
fixed result. Monte-Carlo figures carry ±2–3 pp simulation noise at the default sim budget.

| Quantity | Value (base preset) | Source |
|----------|---------------------|--------|
| Median enrollment date | ≈ Mar 2023 (cumulative ≈ 30 / 102 / 126 by Apr 2022 / Nov 2023 / Apr 2024) | `enroll` |
| BAT cure / median | ~12% · ~12 mo at the q=25% default; left-truncation sweeps it ~9% · ~8 mo (q=0) → ~18% · ~19 mo (q=50%), the cured fraction *rising* with q (Section 2.5.1) | `build_plateau` |
| GPS cure / median | ~63% · ~140 mo all-cause (disease-only plateau is never reached); both fall as selection rises and `π_resp` re-fits down | `build_plateau` |
| GPS median Poisson 68% CI | ~28 – 276 mo (from 60/72/78 ±√n) — wide: three counts barely pin the tail | `fit_ci` |
| Pooled long-term-survivor fraction | ~0.38 (disease plateau; all-cause survival decays below it) | `build_plateau` |
| Pooled median OS | **~19 mo** (above the ≥13.5 floor) | `build_plateau` |
| Implied HR at the 60-event interim | ~0.38 (clears the 1.00 futility threshold); drifts toward 1 as selection rises | `mc` |
| Patients alive at the 80th event | ~36 GPS / ~10 BAT (before censoring) | `mc` |
| **Plateau fixed-scenario rejection rate** | **~100% at the q=25% default**; selection sweeps it ~100% (q=0) → ~65% (q=50%) (Section 2.5.1) | `build_plateau` + `mc` |
| **Bounded no-cure diagnostic** | At base, the bounded no-cure GPS responder fit has median m<sub>G</sub> ≈ 48 mo and shape s<sub>G</sub> ≈ 1.15 with milestone residual RMS ≈ 1.7. Boundary and residual labels are legacy diagnostics, not a test that can prove or reject cure. | `build_no_gps_cure` + `mc` |
| 80th event reached in MC | ~100% of sims (both panels) at the 2% natural-death default; because BAT is shared, the bounded alternative inherits the plateau scenario's event-stall sensitivity and can also stall if BAT cure is pushed hard | `mc` |

In the legacy implementation, the 2% natural-death default lifts the plateau reach fraction from
about 70% at 0% to ~100%. Because the non-cured inputs are already OS, this is a diagnostic of the v1
overlay rather than a defensible estimate of the corrected model's reach probability.

Sweeping the legacy BAT composition illustrates sensitivity. At base, the plateau scenario rate is
~100% and the bounded alternative has an adequate State-C fit (median ~48 mo, shape ~1.15). At the
bear stress case, the plateau rate falls below 50% (~43%) while the alternative remains State C.
Other settings can produce a boundary or residual-misfit status; those remain conditional model-fit
diagnostics.

> In the plateau scenario, the modeled disease dead fraction (~63%) nearly coincides with the
> 80-event trigger (63.5%), so v1 can reproduce a late event stall. That coincidence does not establish
> why observed accrual slowed: the disease inputs are OS and the blanket mortality overlay double-counts
> background deaths outside the cured fraction. The bounded alternative shares BAT and therefore much
> of the same sensitivity. In v1 the overlay raises eventual trigger reach to ~100%; v2 must first fit
> net/relative-survival components or apply background mortality only to the plateau fraction.

---

## 4. Methodology and reasoning

### 4.1 Survival primitives

Both panels share a **Weibull** primitive `Sweib(t) = exp(−(t/scale)^shape)` whose `scale` is set so
its median equals a target (`scale = median / (ln 2)^{1/shape}`); **shape < 1 gives a heavier tail**
and a monotone non-increasing hazard (no non-monotone hazard "hump"). The BAT arm is identical in both
panels (`bat_arm`); the panels differ only in the GPS **responder** family:

- **Plateau — GPS cure:** GPS responders (and every BAT component) use the cure-mixture Weibull
  `Sc(t) = π + (1−π)·exp(−(t / λ)^k)` — a Weibull **plus** a cured/long-term-survivor fraction π. λ is
  set so the non-cured median equals the component median (`λ = median / A(π)^{1/k}`,
  `A(π) = −ln[(0.5−π)/(1−π)]`); `k = 1` recovers the pure exponential. Rationale: cancer-vaccine effects
  classically manifest as a durable-remission (plateau) difference.
- **No-GPS-cure — GPS responder Weibull:** GPS responders use the **bare** Weibull `Sweib` with **no
  cured fraction**, fitted median `m_G` and **fitted shape `s_G`**. This is the explicit "the plateau
  may not be *GPS-specific*" alternative: with `s_G` free to go heavy, a no-cure GPS heavy tail can
  *try* to reproduce the milestone deceleration on top of BAT's own plateau (Section 4.7).

Both share the matching inverse-CDF samplers used by the Monte-Carlo (`sampNC` for the cure-mixture
non-cured Weibull, `sampWeib` for the bare no-GPS-cure GPS responder Weibull).

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
back-loading slider (Section 2.8) rather than marginalized.

### 4.4 Arm decomposition (the unidentified step)

Within the selected mixture family, v1 fits an average of the arms and imposes
`π_GPS = 2·π_pool − π_BAT`. The sparse public counts do not identify `π_pool`; the fitted family and
BAT assumptions jointly determine the resulting GPS plateau. Different decomposition modes:
- **PH (proportional hazards):** `S_GPS = S_BAT^HR` — but this cannot reproduce a plateau without an
  implausibly extreme HR, evidence *against* simple PH (and ruled out independently by the slow
  accrual).
- **Cure-difference (preferred):** GPS shares the control's early dynamics but has a higher plateau
  — a biologically motivated, early-and-sustained separation.

### 4.5 Monte-Carlo fixed-scenario rejection rate

The legacy `ps` output is the fraction of simulated trials whose final test is significant
(`mc()`). Each simulated trial: draws enrollment per cohort; assigns 1:1 GPS/BAT; draws each
patient's survival from the relevant arm/component (cured patients get an effectively infinite
time); applies an independent exponential natural-death time as a competing risk
(`survival = min(disease, T_nat)`, Section 2.9), which also caps the cured subjects; draws an
independent loss-to-follow-up time and censors the subject (no event) if it precedes death
(Section 2.11); finds the calendar time of the `FINAL`-th (80th) death; censors everyone there; and
computes
the **log-rank score statistic = Cox score test = the trial's actual pre-specified test**, declaring
success when `z > z_crit = |ln(HRC)|·√FINAL / 2 = 2.024`. It returns P(significant), the fraction of
sims that reach the 80th event, and the median simulated HR. The same `mc()` runs on both panels; for
the bounded alternative it draws GPS responders from the no-cure Weibull (all other draws identical to the
plateau branch), and its rejection rate is reported only when the fit is State C (Section 4.7).

The same pass also reports three diagnostics that make the fit auditable: the **implied Cox HR at the
60-event interim** (the futility read-through of Section 2.10), a boolean for whether it clears the
futility threshold, and the mean **per-arm patients alive at the 80th event** (before censoring) —
e.g. ~33 GPS / ~13 BAT at the base preset. The alive-split is the same quantity external modelers use
as a sanity check on the arm decomposition.

### 4.6 Component-mixture BAT and non-responders

These replace any abstract π_BAT prior with clinically-grounded structure (Sections 2.5, 2.7).
**They add interpretability, not identifying information** — the blinded data still see only the
pooled curve, so refits absorb this structure and leave the scenario rejection rate largely unchanged
(Section 6).

### 4.7 Bounded no-GPS-cure alternative and fit status

The second panel holds BAT **bit-for-bit identical** to the plateau scenario and changes only the GPS
responder family to a no-cure Weibull. Its fitted parameters are responder median
`m_G ∈ [median_BAT, 120]` months and shape `s_G ∈ [0.15, 1.5]`; non-responders continue to track
Observation. Because BAT is fixed and the parameter box and residual tolerances are analyst choices,
this is a bounded sensitivity analysis rather than a formal statistical null.

The A/B/C status is deliberately descriptive:

- **State A — boundary / non-identified.** At least one fitted parameter reaches the box boundary.
  The legacy `cure_req` field distinguishes upper/heavy from light-edge boundary subtypes for
  regression compatibility, but neither subtype proves or rejects cure. No scenario rejection rate
  is reported for a boundary fit.
- **State B — residual misfit.** The best interior fit exceeds the legacy RMS or maximum-residual
  tolerance. This diagnoses incompatibility of this bounded family with these BAT assumptions; it is
  not a biological hypothesis-test rejection. No scenario rejection rate is reported.
- **State C — adequate interior fit.** The bounded family matches the milestones within the legacy
  tolerances. The panel reports its parameters, HR, and fixed-scenario rejection rate. This remains
  conditional on the BAT arm, parameter bounds, enrollment model, and all other v1 assumptions.

These statuses say how one chosen parametric family behaves inside one chosen box. They cannot resolve
whether the blinded pooled tail is GPS-specific. V2 replaces this fit-status logic with explicit model
families, likelihoods, priors, and posterior sensitivity.

---

## 5. Key functions (reference)

Names below use the Python spelling; the JavaScript in `regal_explorer.html` uses the camelCase
equivalents (`bat_arm` → `batArm`, `build_plateau` → `buildPlateau`, `build_no_gps_cure` →
`buildNoGPSCure`, and the shared `Sweib`/`sampWeib`/`wscale` primitives). Their legacy survival,
fitting, and final-Monte-Carlo outputs match (the Python `common()` reads its inputs from `cfg`, while
JavaScript reads module-level state). Python `mc()` additionally returns `reach_IA`,
`p_IA_efficacy`, `p_IA_efficacy_given_reach`, and `z_IA_efficacy` for the committed replay. Those
audit-only fields are intentionally absent from the browser; automated full parity is v2 WP8 work.

| Function | Purpose |
|----------|---------|
| `Acoef` / `lam` | Weibull coefficient `A(π) = −ln[(0.5−π)/(1−π)]` and scale `λ = median / A^{1/k}`. |
| `Sc(t, med, cure, k)` | Per-component cure-mixture Weibull survival (BAT components + plateau-panel GPS responders, Section 4.1). |
| `Sweib(t, scale, shape)` / `wscale(med, shape)` | The bare Weibull survival (the no-GPS-cure GPS responder; shape < 1 = heavier tail) and the median→scale map `scale = median/(ln 2)^{1/shape}`. |
| `sampNC` / `sampWeib` | Inverse-CDF samplers for the cure-mixture non-cured Weibull and the bare no-GPS-cure Weibull times (Monte-Carlo draws). |
| `enroll(bl, N)` | Monthly enrollment cohorts summing to `N`, interpolating flat↔back-loaded by `bl` (Section 2.3). |
| `common(cfg)` | Shared setup: normalized weights, clamped per-component params, cohorts, milestones, fit weights. |
| `bat_arm(cfg)` | The **shared BAT arm** consumed byte-identically by both panels: per-component cure-mixture with left-truncation selection; returns `Sbat/Snc/Ssel`, `pibat`, `obs`. Guarantees "BAT identical" by construction. |
| `build_plateau` | Plateau (GPS-cure) scenario: shares `bat_arm`, fits `π_resp` to the milestones; returns per-arm curves, cures, and medians. |
| `build_no_gps_cure` | Bounded no-GPS-cure alternative: shares `bat_arm`, fits GPS responder median `m_G` and tail shape `s_G` (auto) with GPS non-responders tracking Observation; emits the three-state fit status (A/B/C), boundary flags, and milestone residual (Section 4.7). |
| `median(S)` | Bisection median of a survival function (`∞`/"NR" if never below 0.5 within 900 mo). |
| `mc(M, nsim)` | Monte-Carlo trial: enrollment → per-arm death draws → censor at the 80th event → **log-rank/Cox score test**; returns P(significant), 80th-event-reached fraction, median HR (Section 4.5). |
| `lan_demets_obrien_fleming_two_look` | V2 one-sided alpha-spending solve for the sequential 60/80 efficacy boundaries; separate from the legacy classical audit function. |
| `stratified_logrank` | V2 beta-zero score test with independent risk sets inside combined protocol-factor strata and hypergeometric tie variance. |
| `evaluate_event_driven_trial` | V2 patient-level 60/80-event decision path: interim efficacy/futility/continuation followed, when applicable, by the final stratified test. |
| `simulate_futility_sensitivity_grid` | Paired canonical-normal operating-characteristic rows for no futility and explicit assumed one-step-HR thresholds. |
| `simulate_patient_level_exponential_null` | Independent null validation of the full enrollment-calendar, event-trigger, and stratified-analysis path. |
| `condition_on_public_history` | WP6 fixed-scenario importance sampler: one latent enrollment/assignment/censoring history, exact public integer-count conditioning, observed interim continuation, and forward final projection. |
| `condition_futility_sensitivity_grid` | Paired WP6 conditional projections that reuse identical latent histories and importance weights across explicit futility-rule assumptions. |
| `figure()` / `render` + `chart*` | 9-panel figure (py, 3×3 grid) / live SVG charts and metrics panel (html): `chart` (survival), `chartAccrual`, `chartHist`, `chartDiverge`, `chartEnroll`, `chartPower`, `chartSelect`. |

---

## 6. Principal findings

1. **The v2 decision mechanics match the public trial design.** The confirmed primary test is a
   stratified Cox at one-sided 0.025 with a Lan–DeMets OBF interim [R1]. V2 now calculates the
   planned 2.339711/2.011777 boundaries, recalculates them after a tied-event overshoot, and applies
   the equivalent stratified score test. V1's unstratified HR ≤ 0.636 rule remains a
   reproducibility-only approximation.
2. **WP6 can now produce a conditional fixed-scenario projection.** Exact integer-count conditioning
   keeps 60/72/78, the event-80 right censor, the finite continuation-region interim statistic, and
   the forward final outcome on one latent patient history. Range targets preserve every compatible
   count interval, and a failed draw-specific continuation tilt falls back to the exact base proposal
   without aborting the run. Compatibility, ESS, maximum-weight, and tilt-fallback diagnostics expose
   scenarios for which the public history or continuation branch is poorly supported. The quota-DP
   safety limit has one definition throughout: logical patient-by-state cells. Tilt iteration/error
   diagnostics are unavailable (`None`) when no component converged, and direct callers can catch the
   exported `TiltProposalError`. This is not yet the WP7 posterior model average.
3. **Blinded pooled survival is high:** ~33–38% modeled plateau, ~16–21-mo median — far above the
   ~6–8-mo historical/contemporary control. Something is keeping these patients alive.
4. **Under the plateau model, the scenario rejection rate is governed by the BAT-quality assumption.** With a
   clinically-built BAT composition it stays high (~100% at base) and is hard to push down
   without assuming venetoclax maintenance is both dominant *and* durable at the top of its
   frontline range — the "bear corner" (70% venetoclax at a 25% cure), where it still only
   falls to ~43%.
5. **Structural refinements are absorbed by the pooled-family fit.** Component-mixture BAT and a
   0–40% non-responder subgroup each leave the legacy scenario rate approximately unchanged because
   fitted parameters redistribute the assumed pooled trajectory. This localizes assumptions rather
   than identifying the arms.
6. **The bounded alternative is a sensitivity diagnostic.** At the base preset it has an adequate
   interior fit (median ~48 mo, shape ~1.15). Other settings can drive it to a parameter boundary or
   residual misfit. None of those statuses determines whether the pooled tail is GPS-specific.

---

## 7. Limitations and the load-bearing assumption

- **Whether the pooled tail is GPS-specific is unresolved.** Three event counts do not identify its
  shape. The bounded no-GPS-cure alternative quantifies one parametric sensitivity, but its A/B/C
  status is conditional on BAT assumptions, parameter bounds, and arbitrary residual tolerances.
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
- **The interim futility rule remains unknown.** V1 treats continuation as a soft HR consistency
  check. V2 conditions on actual continuation for no futility and paired explicit one-step-HR
  thresholds, but does not pretend that any one unpublished rule is the protocol truth.
- **Loss to follow-up is modeled as a flat, independent rate.** Administrative censoring (Section 2.11)
  enters both the fit and the simulation, but as a single constant all-cause-independent hazard
  (default 0); real dropout is time- and arm-varying.
- **Enrollment selection is an idealized sharp filter.** The eligibility lever (Section 2.5.1) screens
  on *realized* survival — it assumes the criteria perfectly remove the patients who would in fact die
  soonest. Real criteria select on covariates only *correlated* with survival, so a given `esel` is an
  upper bound on how cleanly eligibility can enrich the cohort; treat it as "how much healthier could
  the enrolled population plausibly be," not a literal drop-rate. It is applied within each component
  (holding the composition weights fixed) and equally to both arms.
- **V1 remains unstratified.** The isolated v2 decision and WP6 conditioning engines perform the
  protocol-factor stratified score test and accept a sampled patient-level factor distribution, but
  REGAL's realized distribution is not public and still requires an explicit WP7 prior/sensitivity.

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
  (steering-committee ~8-mo BAT context), 29 Dec 2025.
  https://ir.sellaslifesciences.com/news/News-Details/2025/SELLAS-Life-Sciences-Provides-Update-on-Pivotal-Phase-3-REGAL-Trial-of-Galinpepimut-S-GPS-in-Acute-Myeloid-Leukemia-AML/default.aspx
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
- **[R8]** SELLAS, Q2 2026 results and corporate update, 11 Aug 2026 — REGAL described as
  approaching the prespecified 80th event; company will announce when it occurs.
  https://ir.sellaslifesciences.com/news/News-Details/2026/SELLAS-Life-Sciences-Reports-Second-Quarter-2026-Financial-Results-and-Provides-Corporate-Update/default.aspx
- **[R9]** SELLAS, "Triggers Interim Analysis …", 10 Dec 2024 — prespecified 60-event threshold
  reached.
  https://ir.sellaslifesciences.com/news/News-Details/2024/SELLAS-Life-Sciences-Triggers-Interim-Analysis-in-Phase-3-REGAL-Trial-of-GPS-in-Acute-Myeloid-Leukemia/
- **[R10]** SELLAS, REGAL enrollment update, 12 Oct 2023 — projected November completion outside
  China with 20-25 patients anticipated from China.
  https://ir.sellaslifesciences.com/news/News-Details/2023/SELLAS-Life-Sciences-Provides-Update-on-Phase-3-REGAL-Clinical-Trial-for-Galinpepimut-S-in-Acute-Myeloid-Leukemia/default.aspx
- **[R11]** SELLAS, IDMC periodic review, 7 Aug 2025 — 126 randomized; enrollment completed in
  April 2024.
  https://ir.sellaslifesciences.com/news/News-Details/2025/SELLAS-Life-Sciences-Announces-Independent-Data-Monitoring-Committee-Periodic-Review-and-Positive-Recommendation-to-Continue-Pivotal-Phase-3-REGAL-Trial-of-GPS-in-AML-Without-Modification/default.aspx

*Comparator literature anchors for [A] component survival (Section 2.5) were drawn from published
AML CR2 / R/R venetoclax-HMA and azacitidine-maintenance outcome studies; the specific
per-component (median, cure) values are analyst estimates, not direct quotations, and should be
treated as editable inputs rather than sourced facts.*

---

*Prepared as a quantitative research tool operating entirely on public information. It explores
assumption-driven scenarios consistent with disclosed aggregate data; it does not estimate the
confidential outcome of the ongoing trial, and it is not investment advice.*
