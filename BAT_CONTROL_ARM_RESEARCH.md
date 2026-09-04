# The REGAL BAT Control Arm: Composition and Survival Parameters

*Research basis for the model's default `DEFAULT_COMP` / `PRESETS` settings. This reference covers both
the component **composition/weights** of a REGAL-style "best available therapy" (BAT) control arm across
the US, EU, and China, and the per-component **mixture-cure survival parameters** (median OS, cure
fraction, Weibull shape) for AML in second complete remission (CR2), transplant-ineligible. All figures
are analyst assumptions extrapolated from public disclosures and adjacent-setting literature — REGAL
itself is blinded, so no realized arm composition or arm-level OS has ever been published.*

## TL;DR

- A realistic REGAL-style BAT arm is dominated by **observation/watchful waiting** and **hypomethylating
  agents (HMAs)**, with **venetoclax-based regimens** the fastest-growing and most survival-relevant
  active component. The single most important driver of the arm's median OS is the **venetoclax weight**,
  because it is the only BAT option that plausibly pushes mOS toward the low-to-mid teens.
- The most defensible base-case parameters are a **blended median OS of roughly 9–13 months** and a
  **blended cure (long-term-survivor) fraction of only ~8–15%**, with an uncured-fraction Weibull shape
  **k ≈ 0.8–1.1** (near-exponential to mildly decreasing hazard).
- Only two components carry any directly relevant long-term-survivor (plateau) signal —
  **venetoclax+azacitidine** (24-month OS 37.5% in newly-diagnosed unfit; VIALE-A) and **HMA maintenance**
  (QUAZAR long-term-survivor ≈23–35%) — but both come from CR1/frontline settings and must be discounted
  for the more fragile CR2 state; observation and LDAC contribute essentially no cure fraction.
- The single published *numeric* Weibull OS shape for a BAT-relevant regimen is **k ≈ 0.78 for
  venetoclax+azacitidine** (decreasing hazard, Patel et al. 2021); most other AML HTA models chose
  exponential (k = 1) or log-normal fits, so **k ≈ 0.8–1.1 per uncured component** is the best-supported
  modeling choice.
- The single most important **structural** fact: formal maintenance in CR2 is **not guideline-codified
  anywhere** — oral azacitidine's approval (QUAZAR AML-001) is strictly a first-remission (CR1) label —
  so a large fraction of real-world CR2 transplant-ineligible patients receive observation or off-label
  continuation, not a labeled maintenance drug.

## 1. Setting and Trial Design

REGAL (NCT04229979; SELLAS SLSG18-301) enrolls AML patients in CR2/CRp2 after second-line salvage who
are **not** transplant candidates, randomized 1:1 to galinpepimut-S (GPS) vs investigator's-choice BAT.
Per the trial design and SELLAS's public description, the BAT menu permits exactly **four** options, as
monotherapy or in combination:

1. **Observation** with possible palliative hydroxyurea (Droxia);
2. an **HMA** — azacitidine (Vidaza) or decitabine (Dacogen);
3. **venetoclax** (Venclexta);
4. **low-dose cytarabine** (LDAC, Ara-C).

Patients whose CR2 can be maintained with **FLT3 or IDH inhibitors are explicitly excluded** from
eligibility. Targeted maintenance agents therefore do **not** populate the REGAL BAT arm, even though
they are used in wider real-world relapsed AML. The setting is data-poor by design: there is essentially
no dedicated randomized survival dataset for "maintenance in CR2, transplant-ineligible," so every
parameter below is extrapolated from an adjacent setting (CR1 maintenance, newly-diagnosed unfit, or
relapsed/refractory), with the mismatch flagged for each.

### The CR2-maintenance evidence vacuum

The standard of care for AML patients in remission not proceeding to transplant was historically
**observation**. Oral azacitidine (Onureg/CC-486) — the only NCCN Category 1 maintenance option — is
approved and studied **only in first remission** after intensive induction (QUAZAR AML-001; FDA Sept 1,
2020; EC June 2021). There is **no labeled maintenance therapy specifically for CR2**, so continuation
therapy in CR2 is one of: (a) continuation of the low-intensity regimen used to achieve CR2 (e.g.,
venetoclax + HMA), (b) off-label use of a CR1 maintenance drug, or (c) observation.

### Anchoring benchmark

SELLAS and secondary sources cite a historical mOS of **~6 months** for AML patients in CR2 who do not
undergo transplant. Per the SELLAS press release (January 23, 2025): *"Fewer than 50% of Enrolled
Patients Confirmed Deceased After the Median Follow-Up of 13.5 Months, Indicating a Median Survival of
Over 13.5 Months in the Trial vs. Historical Median Survival of 6 Months for Conventional Therapy, as
Reported in Similar Phase 2 Study."* The interim analysis was triggered at 60 death events; the final
analysis is set at 80. This pooled doubling (across **both** arms, still blinded, and GPS-favorable) is
the strongest empirical clue that the real-world BAT arm is outperforming the ~6-month anchor — most
likely because of venetoclax-based salvage and generally longer-lived enrolled patients. It **cannot** be
decomposed into GPS vs BAT contributions and should not be read as the control arm's realized OS.

## 2. Component Composition (weights)

No component weights have ever been disclosed — REGAL remains blinded (72 of 80 death events as of
December 26, 2025). SELLAS disclosed that US and European sites accounted for ~75% of the 127 randomized
patients, with the **US the highest-enrolling country**, and that target enrollment was reached
"ex-mainland China" (mainland China was excluded from primary enrollment). All weights below are modeled
from guidelines, approvals, and real-world evidence — not from REGAL.

**Real-world "no active treatment" is common.** In US real-world AML the no-active-treatment share runs
~30% (weighted average); in the EU ~24–35%; SEER-Medicare (2008–2015) found 31% of AML patients aged ≥65
received no active antileukemic or supportive care. In relapsed/refractory AML specifically, US Optum
claims (2016–2022) found ~1/3 received no active treatment, and R/R salvage registries put best
supportive care at ~20–26% (PETHEMA ~20%; DATAML ~26%).

### Regional ranking of BAT components (most → least common)

| Rank | United States | European Union | China |
|------|---------------|----------------|-------|
| 1 | Observation / watchful waiting | Observation / watchful waiting | Observation (amplified by cost/reimbursement) |
| 2 | Venetoclax + HMA (most common *active* regimen; 2L venetoclax-based ~17% in Optum) | Venetoclax + azacitidine (reimbursed frontline-unfit; NICE TA765/TA787) | Venetoclax + HMA (NMPA 2020; NRDL-listed 2022) |
| 3 | HMA monotherapy (aza/dec; oral-aza off-label in CR2) | HMA monotherapy | Decitabine / azacitidine monotherapy (cheap, widely used) |
| 4 | LDAC ± venetoclax | Oral azacitidine (CR1 label; off-label CR2) | Low-/reduced-intensity chemo (incl. HHT regimens) |
| 5 | Oral azacitidine (only if CR1; off-label in CR2) | LDAC ± venetoclax; IL-2 + histamine (rare) | Targeted agents where mutation-applicable (exclude from REGAL) |

**Regulatory/reimbursement differences that shape the arm.** Oral azacitidine's CR1-only label (FDA/EC)
is the single biggest reason CR2 maintenance is not codified — in CR2 its use is off-label everywhere,
pushing patients toward observation or continuation of the salvage regimen. Venetoclax's on-label
indication is frontline-unfit disease in all regions (FDA Oct 2020; EMA May 2021; China NMPA Dec 2020,
NRDL 2022), so its use as CR2 continuation is off-label but common. China's 2022 NRDL listing of
venetoclax (average 60.1% price cut that round) materially improved affordability.

### Modeled base-case weights

Given the eligibility exclusion of targeted maintenance and the CR1-only status of oral azacitidine, a
realistic (soft, **not** REGAL-disclosed) blend is:

- **Observation / watchful waiting: 30–45%** — the highest single component.
- **Venetoclax-based (VEN + HMA or VEN + LDAC): 25–40%** — fastest-growing active component; higher at US sites.
- **HMA monotherapy: 15–25%.**
- **Low-dose cytarabine (± venetoclax): 5–10%.**

Because US-weighted enrollment tilts the pooled arm toward more venetoclax and less pure observation than
a China- or EU-heavy arm would, a **US-heavy blend** is the more defensible base case for REGAL
specifically.

## 3. Component Survival Parameters

A mixture-cure model writes **S(t) = π + (1 − π)·S_u(t)**, where π is the cured (long-term-survivor)
fraction and S_u(t) is uncured survival, taken as a Weibull with shape k. The table gives the base-case
(median OS, π, k) for each component; the two rows below the rule are **excluded** from REGAL BAT and
shown for context only.

| Component (CR2, transplant-ineligible) | mOS (months) | Cure fraction | Weibull k (uncured) | Key source(s) / setting |
|----------------------------------------|--------------|---------------|---------------------|-------------------------|
| Observation / BSC (± hydroxyurea) | ~5–7 (CR2) | ~2–5% | ~1.0–1.3 (steep early, mildly increasing hazard) | QUAZAR placebo mOS 14.8 mo (CR1 ceiling); untreated elderly AML mOS ~2 mo; relapsed AML 4–6 mo |
| HMA monotherapy (aza/dec) | ~10–15 (CR1 maint.) / ~7–10 (active) | ~10–18% (CR1) → ~8–12% (CR2) | ~1.0 (exponential per HTA) | QUAZAR oral-AZA 24.7 vs 14.8 mo; AZA-AML-001 10.4 mo; DACO-016 7.7 mo |
| Venetoclax-based (VEN+HMA / VEN+LDAC) | ~12–15 (frontline) / ~6–10 (r/r) | ~15–25% (frontline) → ~12–18% (CR2) | **~0.78 (decreasing hazard, published)** | VIALE-A VEN+AZA mOS 14.7 mo, 24-mo OS 37.5%; r/r VEN combos ~6.1 mo |
| Low-dose cytarabine (LDAC) | ~5–10 (frontline unfit) | ~5–12% | ~1.0–1.2 | LDAC mOS ~5 mo (Burnett); 9.6 mo single-center; 3-yr OS 12% |
| *FLT3i — gilteritinib (EXCLUDED)* | *9.3 (r/r, ADMIRAL)* | *~10% (2-yr OS 20.6%)* | *~1.0* | *ADMIRAL mOS 9.3 mo (context only)* |
| *IDHi — ivosidenib+AZA / enasidenib (EXCLUDED)* | *29.3 (ivo frontline) / 6.5 (ena r/r)* | *~15–25% (ivo)* | *~0.8–1.0* | *AGILE 29.3 mo; IDHENTIFY 6.5 mo, negative (context only)* |

### 3.1 Observation / watchful waiting / BSC

The weakest-survival arm and closest analog to "no active maintenance." Bracketed by an upper bound
(QUAZAR CR1 placebo: mOS 14.8 mo, 3-yr OS 27.9% — optimistic ceiling) and a lower bound (untreated
elderly AML mOS ~2 mo; relapsed AML 4–6 mo, 3-yr survival <10%). The CR2-no-transplant anchor is ~6 mo.
A small durable tail exists (favorable-cytogenetics CR2, NPM1-mutated late relapsers), but without
maintenance most relapse. **Chosen: mOS ~5–7, π ~2–5%, k ~1.0–1.3.**

### 3.2 Hypomethylating agent (HMA) monotherapy

Two settings collide. As **maintenance** (closest analog), QUAZAR AML-001 (Wei et al., NEJM 2020) tested
oral azacitidine in CR1 transplant-ineligible patients ≥55: *"Median overall survival … was 24.7 months
… in the azacitidine group vs 14.8 months … in the placebo group (P<.001)"*; long-term survivors (≥3 yr)
~23–35% depending on definition, strongly enriched for intermediate-risk cytogenetics, NPM1 mutation, and
on-study MRD conversion. As **active treatment** (frontline unfit): AZA-AML-001 mOS 10.4 mo; DACO-016 7.7
mo; real-world ~7–8 mo. AML HTA models generally fit HMA-alone OS with an **exponential** (k = 1). The
QUAZAR plateau is a CR1 phenomenon; CR2 is more fragile, so discount. **Chosen: mOS ~10–15, π ~8–12%
(discounted from CR1's ~15–25%), k ~1.0.**

### 3.3 Venetoclax-based regimens

The component that determines whether the BAT arm beats its historical benchmark. **Frontline unfit**
(VIALE-A, Pratz et al. AJH 2024, 43.2-mo follow-up): *"median OS was 14.7 months … with
venetoclax-azacitidine, and 9.6 months … with placebo-azacitidine (HR 0.58…); the estimated 24-month OS
rate was 37.5% and 16.9%."* MRD<10⁻³ patients reached mOS 34.2 mo; ~30% long-term survivors. **R/R**
(closer to CR2 biology): VEN+HMA mOS ~6.1 mo; post-HMA/VEN-failure survival dismal (1–3 mo). The one
published numeric Weibull OS shape is **k = 0.78** (Patel et al., Blood Advances 2021, reconstructed from
VIALE-A KM) — decreasing hazard, consistent with a forming plateau; competing individual-patient-data
models chose log-normal/exponential, so genuine uncertainty remains. **Chosen: mOS ~12–15 (frontline
ceiling), discounting toward ~9–12 for CR2; π ~12–18%; k ~0.78.**

### 3.4 Low-dose cytarabine (LDAC)

The weakest active option, likely a small minority of the arm. Historical mOS ~5 mo (Burnett; CR ~13–18%);
a single-center series 9.6 mo with 3-yr OS 12%. Steep, roughly exponential decline with a small responder
tail. **Chosen: mOS ~5–10, π ~5–12%, k ~1.0–1.2.**

### 3.5 FLT3 / IDH inhibitors — EXCLUDED (context only)

REGAL excludes patients whose CR2 is maintained on molecularly targeted agents, so these do **not**
contribute to BAT weighting. Reported for completeness: gilteritinib (ADMIRAL, r/r) mOS 9.3 mo, 2-yr OS
20.6%; ivosidenib+AZA (AGILE, frontline IDH1) mOS 29.3 mo (a genuinely favorable, plateau-forming
subgroup); enasidenib (IDHENTIFY, r/r IDH2) **negative** (mOS 6.5 vs 6.2 mo ITT). Using AGILE's 29-month
ivosidenib mOS in the control arm would badly overstate it — **do not**.

## 4. Cure-Model Framing and Curve Shape

- **The "3-year plateau = cure" convention.** Yanada et al. (Cancer 2007) and the large ECOG analysis
  support treating AML patients still in remission at ~3 years as "potentially cured"; a long KM plateau
  with sufficient follow-up is the empirical signature of π. In transplant-based CR2 series the cured
  fraction can be large (CBF-AML CR2 5-yr OS 58.2%), but those are transplanted, favorable-biology
  patients — not the REGAL BAT population.
- **Published AML mixture-cure work** (Fu et al., Statistics in Medicine 2022) found a ~30% plateau in
  younger intensively-treated CR patients — again an optimistic ceiling relative to older CR2
  non-transplant patients.
- **Weibull shape interpretation.** k < 1 = decreasing hazard (early deaths, then plateau; matches VEN+AZA
  k = 0.78); k = 1 = constant hazard (exponential; matches most HMA/LDAC HTA fits); k > 1 = increasing
  hazard. For the uncured fraction of relapsed/re-remitted AML, an early-steep-then-flattening pattern
  (**k ≈ 0.8–1.1**) is the most defensible default.

## 5. Synthesis: The Blended BAT Arm

Using illustrative weights (observation 30–45%, venetoclax 25–40%, HMA 15–25%, LDAC 5–10%) and the
component estimates above:

- **Blended median OS.** Medians don't average linearly, but a weighted-mean-of-medians gives a sighting
  shot. With mid-point medians (observation 6, HMA 12, venetoclax 12, LDAC 7) and mid-point weights
  (observation 37.5%, venetoclax 32.5%, HMA 20%, LDAC 7.5%): ≈ **9.1 months**. A venetoclax-heavy mix
  pushes this to ~10–11; an observation-heavy mix pulls it to ~8. Blending the curves (rather than
  medians) lets the heavy early mortality of observation/LDAC lower the composite median somewhat, so a
  base-case **blended mOS of ~9–13 months** is consistent both with this arithmetic and with the observed
  pooled REGAL interim (>13.5 months, both arms, GPS-favorable).
- **Blended cure fraction.** 0.375×3.5% + 0.325×15% + 0.20×10% + 0.075×8% ≈ **~8.8%**, i.e., a plateau of
  roughly **8–15%** depending on venetoclax/HMA weight — well below the CR1 QUAZAR plateau (~24–35%) and
  the frontline VEN+AZA plateau (~30%), reflecting the CR2 discount.
- **Composite curve shape.** Steep early drop (observation + LDAC) → slower decline → low plateau
  (venetoclax + HMA). The mixture is itself non-Weibull; if a single uncured-fraction shape is required,
  **k ≈ 0.9–1.1** captures it, with the venetoclax subgroup specifically warranting **k ≈ 0.78**.

## 6. How This Maps to the Model Defaults

The model implements the BAT arm as an explicit component mixture (`DEFAULT_COMP` in
`regal_explorer.py` / `.html`), each a cure-mixture with its own (median, cure, Weibull k), then weights
them. The recommended base case above is encoded as the **base preset**. The model applies two
pre-randomization selection layers on top of these components — frailty-based eligibility screening
(evidence-informed defaults `q` = 20%, `θ` = 0.35) and the CR2 → randomization entry window
(default 1–6 months) — so the cure
fractions below are the **pre-selection** values. Frailty screening leaves them unchanged; the entry
window raises them to `π / E_D[S(D)]` at runtime.

The selection defaults are deliberately modest. A 599-patient first-relapse AML validation found
one-year OS of 64%, 38%, and 17% across favorable, intermediate, and poor prognostic groups
(https://pubmed.ncbi.nlm.nih.gov/16803568/). In 1,042 AML patients transplanted in CR2, 28% were
MRD-positive; two-year relapse was 40% versus 24% for MRD-negative patients, with cytogenetics and
time to transplant also prognostic (https://doi.org/10.1038/s41408-021-00479-3). These cohorts show
real heterogeneity but do not directly estimate a gamma-frailty variance for REGAL's
transplant-ineligible population; `θ = 0.35` is therefore a central working assumption and
0.20–0.60 is the sensitivity range.

Published AML trial ineligibility rates are higher than the modeled `q`: 26.7–35.9% by reported
race across 13 FDA registration trials (https://doi.org/10.1016/j.clml.2023.03.012), and 41% in a
post-transplant AML/MDS maintenance study (https://doi.org/10.1182/bloodadvances.2020002544).
Those totals include mutation mismatch, consent, logistics, treatment choice, and timing failures.
Because this model's `q` represents only exclusions correlated with baseline frailty—and the entry
window separately handles deaths before randomization—the working default is 20%, with 10–30% used
for sensitivity analysis. The default pair yields `E[Z | eligible] = 0.8^0.35 = 0.925`, a 7.5%
reduction in mean uncured disease hazard among accepted patients.

| Component | Weight | Median OS | Cure π | Weibull k |
|-----------|--------|-----------|--------|-----------|
| Observation | 27% | 6.0 mo | 3% | 1.1 |
| Hydroxyurea | 8% | 5.0 mo | 2% | 1.1 |
| HMA | 22% | 12.0 mo | 10% | 1.0 |
| Venetoclax | 35% | 12.0 mo | 15% | 0.78 |
| LDAC | 8% | 7.0 mo | 8% | 1.1 |

Observation + hydroxyurea (35% combined) implement the review's ~35% observation stratum; venetoclax at
35% and a 15% cure is the CR2-discounted midpoint of the frontline 15–25% range; the venetoclax k = 0.78
is the one published shape. This yields a pre-selection blended cure ≈ 9% and median ≈ 8 mo (≈ 12% /
≈ 10.7 mo after the default selection layers), squarely inside the review's ~8–15% / ~9–13-month
base case.

The four stress presets map to scenario corners: **low-venetoclax** (access-constrained /
observation-heavy), **venetoclax-dominant** (US-heavy delivered-regimen stress), **bear** (70%
venetoclax at a 25% cure — an intentionally strong-BAT allocation stress), and **bull**
(observation-heavy weak-BAT corner). They are sensitivities, not reconstructions of the protocol's
planned stratum balance.

## 7. Recommendations

1. **Model the BAT arm as an explicit 4-component mixture** (plus a hydroxyurea split of observation),
   each with its own (mOS, π, k), then weight. Base-case weights: observation ~35%, venetoclax ~35%,
   HMA ~22%, LDAC ~8%.
2. **The venetoclax weight is the dominant swing factor.** Vary it 25%→45% and its mOS 9→15 months — this
   single-handedly determines whether the blended arm clears ~12 months. Monitor it as the key
   sensitivity because it directly threatens the trial's assumed hazard ratio.
3. **Keep planned stratum and delivered regimen separate.** The publication describes approximately
   balanced randomization strata, while its "and/or" wording permits combination regimens. Use an
   equal-planned-strata sensitivity and a geography-informed delivered-regimen sensitivity; do not
   treat either as known patient-level composition. With ~75% of REGAL from the US/EU, a US-heavy
   venetoclax blend remains useful as a stress test rather than the sole primary weighting.
4. **Vary the CR2 discount** applied to CR1/frontline cure fractions (test 0.5×–0.8× of frontline π), and
   test k = 1.0 (exponential) as an alternative to 0.78 for venetoclax.
5. **Flag oral azacitidine carefully** — do not weight Onureg as a labeled maintenance component in CR2;
   fold any HMA use into a generic off-label "HMA monotherapy" bucket.
6. **Do not** include FLT3/IDH-inhibitor survival in the BAT weighting — those patients are excluded from
   REGAL by protocol.
7. **Benchmarks that would change the model.** If REGAL's eventual unblinded BAT-arm mOS is ≥12 months (or
   venetoclax >40% of the arm), raise the venetoclax weight and cure assumptions; if BAT mOS lands at
   ~6–8 months (or observation >50%), revert toward the observation-heavy, low-cure base case.

## 8. Caveats and Data Limitations

- **No disclosed REGAL BAT composition or arm-level OS exists.** All weights and parameters are modeled
  from guidelines/approvals/RWE and adjacent-setting literature, not from REGAL.
- **Every component parameter is extrapolated.** The closest analogs (QUAZAR for HMA maintenance, VIALE-A
  for venetoclax) are CR1/frontline and systematically optimistic. CR2 is more fragile than CR1 (median
  time-to-relapse after CR2 ~16.5 months in one late-relapse series), so all CR1/frontline cure fractions
  are discounted ~0.6–0.8×.
- **Cure fractions are soft.** QUAZAR "long-term survivor" figures (23–35%) depend on definition and are
  confounded by censoring. Treat π as a modeling assumption, not a measured constant.
- **Weibull shape is thinly evidenced.** Only VEN+AZA has a published numeric shape (k ≈ 0.78, fit to
  digitized KM); competing IPD models preferred log-normal/exponential. HMA/LDAC k values are inferred
  from the general HTA preference for exponential fits.
- **No clean CR2-specific real-world distribution is published;** quantitative shares come from adjacent
  settings (R/R registries, general AML undertreatment, R/R claims).
- **The REGAL pooled interim (>13.5 months, both arms blinded) cannot be decomposed** into GPS vs BAT and
  is GPS-favorable; it is consistent with a BAT arm above the ~6-month anchor but does not prove it.
- **IDHENTIFY was negative** — do not assume IDH2 inhibition confers survival benefit in unselected r/r
  disease.

---

*Research/analysis reference operating entirely on public disclosures. Not investment advice, and it
does not access or estimate the confidential trial outcome.*
