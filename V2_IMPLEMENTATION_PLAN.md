# REGAL model v2 implementation plan

This document tracks the scientific and software changes required before the model can report a
forecast for the actual REGAL trial. The existing Python and HTML implementations remain the
reproducible **v1 legacy scenario explorer** while this work is in progress.

## Output contract

V2 must keep three quantities separate:

1. **Scenario operating characteristics** — the probability of interim efficacy, futility,
   continuation, final rejection, and overall trial success if a fixed parameter set generated a
   new trial.
2. **Conditional scenario projection** — the probability of final rejection under a fixed
   parameter set after conditioning on the public event history and the actual interim decision to
   continue.
3. **Posterior forecast** — the conditional projection integrated over survival-model, enrollment,
   BAT-composition, and parameter uncertainty.

Only the third quantity may be described as the model's headline forecast for REGAL. Until it is
implemented, v1's `P(success)` is a fixed-scenario simulated rejection rate, not the posterior
probability that the ongoing trial succeeds.

## Work packages

### 1. Preserve v1 and correct its language

- Tag and retain the current numerical behavior as `v1-legacy`.
- Replace ambiguous output labels with explicit branch probabilities:
  `P(interim efficacy)`, `P(futility)`, `P(continue)`,
  `P(final rejection | continue)`, and `P(overall trial success)`.
- Rename the "no-GPS-cure null" to a **bounded no-cure alternative**.
- Remove automatic claims that a boundary fit statistically rejects no cure or proves cure is
  required. Boundary and residual checks are model diagnostics, not hypothesis tests.
- Correct stale enrollment, background-mortality, and UI documentation.
- Keep the golden tests as regression tests for legacy behavior, not evidence of scientific
  validity.

### 2. Correct the survival primitives

Current non-cured literature inputs are overall survival (OS), while the cured fraction should
still experience population mortality. For OS inputs use

\[
S(t)=cS_{bg}(t)+(1-c)S_{uncured,OS}(t).
\]

If a component is refitted to net or relative survival, use

\[
S(t)=S_{bg}(t)\{c+(1-c)S_{uncured,net}(t)\}.
\]

PR 2 implements these requirements in the isolated `survival_models.py` layer:

- [x] Add an explicit `survival_scale = overall | net` to every component.
- [x] Default the current literature inputs to `overall`.
- [x] Replace the unconditional multiplier with the scale-aware equations above: for OS inputs, overlay
  background mortality only on the cured fraction; for net/relative-survival inputs, overlay it on
  the whole mixture.
- [x] Ensure the cured fraction follows population mortality rather than forming an immortal plateau.
- [x] Replace top-survivor truncation with a pre-outcome case-mix/frailty model applied symmetrically to
  both randomized arms. No inclusion rule may depend on future realized survival.

The frailty model draws baseline prognosis, applies a probabilistic eligibility rule, and only then
randomizes the selected cohort. Its neutral default has zero frailty variance and no prognostic
enrichment, exactly reproducing the component curve; both quantities must later receive priors or
external calibration rather than inheriting v1's arbitrary 25% top-survivor setting. The legacy
Python/HTML engines remain unchanged, and the v2
primitives are not a trial forecast until subsequent BAT, likelihood, and simulation work wires them
into the canonical engine.

Non-unit disease frailty is permitted only for `net` inputs. An `overall` curve already contains
population mortality, so scaling its full hazard would inconsistently scale background mortality
for uncured patients but not cured patients. Existing OS inputs therefore remain at neutral frailty
until they are refitted to net/relative survival or supplied with a validated excess-hazard
decomposition. Cure probability is provisionally frailty-independent and must be revisited when the
case-mix prior is calibrated.

### 3. Represent BAT randomization strata and regimens

- [x] Represent the four planned strata separately: supportive care/hydroxyurea, HMA, venetoclax, and
  LDAC.
- [x] Use approximately 25% per stratum as the primary protocol-compatible configuration.
- [x] Store both the randomization stratum and the patient-level regimen/components, so "and/or"
  combinations do not double-count patients.
- [x] Treat bear and venetoclax-dominant allocations as stress tests rather than primary protocol
  reconstructions unless realized-regimen evidence supports them.
- [x] Treat the current component weights as a legacy comparison.
- [x] Resolve each patient assignment to exactly one validated survival component, with named
  failures when a custom component library is incomplete or invalid.
- [x] Permit explicit zero-mass strata only in non-primary designs; keep all four primary strata
  positive.

`bat_regimens.py` implements a joint distribution over planned stratum and delivered regimen. Each
patient pathway carries both values, while each regimen carries all known component exposures and
exactly one survival-profile key. An HMA + venetoclax patient therefore contributes once to the
regimen and outcome distributions but to both exposure marginals. The primary
`PRIMARY_EQUAL_STRATA` design uses 25% per planned stratum and preserves the legacy 27:8 internal
observation/hydroxyurea split within the supportive-care stratum. Its single-profile regimen mapping
is an explicit proxy until realized combination evidence is available, not a claim about delivered
REGAL treatments.

The committed `LEGACY_COMPONENT_MIX` is classified as `legacy_comparison`; the 60% venetoclax-dominant
and 70% bear allocations are classified as `stress_test`. The bear preset's separate 25% venetoclax
cure assumption is reproduced by the separately named, immutable
`BEAR_STRONG_BAT_COMPONENT_LIBRARY` rather than being hidden in allocation. The component library
connects the documented Observation, Hydroxyurea, HMA, venetoclax, and LDAC inputs to the scale-aware
work-package-2 survival API, all explicitly on the overall-survival scale. `component_for()` resolves
one patient assignment to its outcome profile, while `BATDesign.validate_library()` checks all
positive-mass pathways before simulation.

The venetoclax profile is VEN+azacitidine-derived. Applying it where co-therapy is unknown,
particularly to monotherapy, is a provisional BAT-favorable mapping that may overstate survival.
The tested public combination-regimen machinery is therefore retained without an illustrative
production allocation constant until evidence supports one. Primary designs require positive mass
in every planned stratum; non-primary comparison and stress designs may use an explicit zero-mass
pathway to represent an absent stratum.

The legacy equal-strata run is more bullish than the current default (about 99.9% scenario power,
median HR about 0.30, and about 94% interim efficacy crossing). Reproduce it with
`python3 audit/interim_efficacy_replay.py --nsim 10000`. These are characterization checks, not v2
forecast targets. Equal planned strata are only one interpretation because "and/or" combinations
can make realized regimens differ from stratification balance.

### 4. Implement the trial decision process

- [x] Calculate the one-sided 0.025 Lan-DeMets/O'Brien-Fleming efficacy boundaries from the spending
  function and the 60/80 information fraction. Expected validation values are approximately
  `z60 = 2.340` and `z80 = 2.012`. The legacy replay's classical discrete-look values
  (`2.327` / `2.015`) are characterization snapshots, not v2 validation targets.
- [x] Simulate all interim branches: efficacy stop, futility stop, and continuation.
- [x] Keep the unknown futility rule configurable and report a sensitivity grid.
- [x] Replace the unstratified one-step approximation with the protocol-compatible stratified log-rank
  or Cox analysis; keep the approximation only as a diagnostic.
- [x] Validate the null type-I error and decision branches by simulation.

`trial_design.py` now keeps the legacy classical boundary intact while adding the protocol spending
function and sequential boundary solve. At 60/80 information the v2 values are
`z60 = 2.339711` and `z80 = 2.011777`; the correlated probability of crossing either boundary under
the null is one-sided 0.025. The v2 primary statistic is the stratified log-rank score, equivalently
the score test at beta zero for a treatment-only stratified Cox model. It accepts either combined
stratum labels or multiple protocol-factor columns and uses tied-event hypergeometric variance. The
unstratified score and `exp(U/V)` one-step HR are returned only as named diagnostics.

`simulation.py` applies those analyses at the 60th and 80th observed deaths, excluding patients not
yet randomized at each event-calendar cutoff. It retains every death tied at the cutoff and uses the
realized event count divided by 80 as the public design's information proxy, recalculating both
sequential boundaries so overshoot does not silently make the design conservative or inflate alpha.
Interim efficacy, assumed futility, and continuation are mutually exclusive; a continued trial can
then reject, not reject, or fail to reach the final look. No futility rule is embedded in the
committed efficacy design because its form and boundary are unpublished. `HazardRatioFutilityRule`
makes an assumed one-step-HR cutoff explicit, and
`audit/v2_trial_decision_validation.py` reports paired sensitivity rows for no futility and thresholds
0.80 through 1.20. Its canonical correlated-normal null simulation pins branch conservation and
approximately 0.025 overall type-I error. A separate exponential-null audit runs the complete
patient-level calendar-trigger and stratified-analysis path. Neither is a REGAL survival forecast or
conditions on the observed continuation.

### 5. Build a public-history likelihood

- Re-anchor accrual to the registry's 2021-02-08 study start. The v1 September 2020 window creates
  pre-opening patients and cannot reach its own approximately 20-patient April 2022 anchor at any
  enrollment-slider setting. Require all published enrollment anchors to be reachable and test them.
- Store enrollment and event disclosures in `data/regal_public_history.json`, including observation
  date, announcement date, observation type, reporting-lag uncertainty, source, and notes.
- Distinguish exact/as-of counts, threshold-hitting observations, announcement dates, and right
  censoring of the 80th event.
- Replace weighted least squares and independent Poisson error bars with a joint likelihood over
  integer enrollment cohorts and correlated event increments.
- Represent uncertainty over event trajectories. Three aggregate event counts do not uniquely
  identify the pooled survival curve.

### 6. Condition on the observed interim continuation

For each parameter/model draw:

1. Generate latent enrollment, randomization strata, treatment assignments, survival, and censoring.
2. Weight or retain datasets compatible with the disclosed 60/72/78-event history.
3. Calculate the interim statistic at 60 events.
4. Require the statistic to lie between the futility and efficacy boundaries.
5. Apply the latest right-censoring constraint on the 80th event.
6. Simulate unresolved patients forward and apply the final stratified analysis.

Use importance sampling or sequential Monte Carlo so rare continuation-compatible draws can be
handled efficiently. Report `P(public history | scenario)` or an equivalent compatibility measure;
this prevents a scenario with a high unconditional early-efficacy probability from being presented
as a high conditional forecast without penalty.

### 7. Average across GPS effect structures

Include at least: no effect, proportional hazards, delayed proportional hazards, cure-fraction
difference, delayed cure, and waning/piecewise effect families. Treat the current responder/cure
construction as one exploratory family. Report model-family weights and prior sensitivity rather
than declaring cure from an arbitrary parameter boundary.

### 8. Frontend and validation

Use Python as the canonical engine and generate a versioned JSON result bundle for the self-contained
HTML interface. The target module split is:

```text
regal_data.py
trial_design.py
survival_models.py
bat_regimens.py
event_likelihood.py
posterior.py
simulation.py
report.py
```

Required tests include survival-scale handling, non-immortal cure survival, no future-outcome
selection, BAT weights and combinations, interim branch conservation, O'Brien-Fleming type-I error,
stratified-analysis parity, small-cohort likelihood checks, continuation-boundary conditioning,
80th-event right censoring, and Python/HTML consistency.

## Release gates

- **v1.1:** truthful labels, documentation corrections, and UI consistency fixes.
- **v2-scenario:** corrected survival, BAT strata, and trial mechanics.
- **v2-forecast:** public-history likelihood, interim-continuation conditioning, and posterior model
  averaging.

No single output should be described as REGAL's probability of success before the `v2-forecast`
gate is complete.
