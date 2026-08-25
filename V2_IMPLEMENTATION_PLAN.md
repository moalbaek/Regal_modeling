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

- [x] Re-anchor accrual to the registry's 2021-02-08 study start. The v1 September 2020 window creates
  pre-opening patients and cannot reach its own approximately 20-patient April 2022 anchor at any
  enrollment-slider setting. Require all published enrollment anchors to be reachable and test them.
- [x] Store enrollment and event disclosures in `data/regal_public_history.json`, including observation
  date, announcement date, observation type, reporting-lag uncertainty, source, and notes.
- [x] Distinguish exact/as-of counts, threshold-hitting observations, announcement dates, and right
  censoring of the 80th event.
- [x] Replace weighted least squares and independent Poisson error bars with a joint likelihood over
  integer enrollment cohorts and correlated event increments.
- [x] Represent uncertainty over event trajectories. Three aggregate event counts do not uniquely
  identify the pooled survival curve.

`event_likelihood.py` implements the isolated WP5 likelihood layer. The default fixed-N accrual reference
starts on 2021-02-08 and is piecewise uniform between the explicit 20/104/126 anchor centers; every
draw therefore contains exactly 126 patients and none can predate study opening. The intermediate
104 anchor is correctly classified as the rounded center of the sponsor's forward-looking 101-106
projection (126 less 20-25 anticipated China patients). It centers the provisional reference path but is excluded
from the observation likelihood. The first-20 and completed-126 constraints remain integer-valued
likelihood evidence, and tests require every anchor to be both reachable and centered. This reference
parameterization is not an independent Bayesian prior; WP7 posterior work must not reuse the same
likelihood evidence when specifying the accrual-parameter prior.

The event likelihood accepts one cumulative event probability per patient at each disclosure cutoff.
It converts those CDFs to mutually exclusive calendar intervals and uses dynamic programming to sum
the Poisson-multinomial mass of every latent patient/event allocation compatible with the cumulative
constraints. Thus 60/72/78 are evaluated jointly rather than as independent residuals, while
heterogeneous enrollment dates and survival profiles remain available to the caller. The public data
distinguish the 60-event threshold hit from the exact 72- and 78-event as-of counts. Because the
60-event announcement did not disclose the exact threshold date, that occurrence date is integrated
over the same explicit lag sensitivity model used for event 80. Both use independent three-point
PMFs at 0/7/14 days with weights 4/21, 13/21, and 4/21, which preserve the mean and variance of the
original discrete-uniform 0-14 day model while reducing the joint mixture from 225 to nine branches.
SELLAS's 2026-08-11 statement that the study was still approaching event 80 is encoded as
announcement-process right censoring, integrated over that explicit reporting-lag sensitivity
distribution. The lag prior is an assumption, not a company disclosure. WP5 deliberately exposes the marginal
enrollment-anchor likelihood and the event likelihood conditional on patient-level calendar CDFs as
separate components. WP6 now integrates them with the same sampled enrollment history rather than
silently multiplying mismatched marginal and conditional quantities.

The likelihood retains independent safety guards for lag combinations and DP states. The default
four-million-state cap clears the natural unconstrained three-cutoff REGAL boundary of
`127^3 = 2,048,383` while still failing loudly before an unbounded allocation; callers can lower
either guard for tighter runtime or memory budgets.

`audit/v2_public_history_validation.py` pins the schema, accrual gates, a brute-force-verifiable
small-cohort correlation example, and the right-censor lag path. This package evaluates
the component likelihoods needed for `P(public history | fixed scenario)`; `posterior.py` performs
the consistent latent-history integration and continuation conditioning. The WP5 component alone
still does not condition on interim continuation, average over parameter/model uncertainty, or
produce the headline REGAL forecast.

### 6. Condition on the observed interim continuation

- [x] Generate one internally consistent latent enrollment, randomization-stratum, treatment,
  survival, and censoring history per fixed-scenario draw.
- [x] Condition patient event intervals on every allowed cumulative integer trajectory compatible
  with the disclosed 60/72/78-event history and the event-80 announcement right censor.
- [x] Calculate the stratified statistic at the realized 60-event look and retain only finite
  statistics in the continuation branch between efficacy and any explicit assumed futility rule.
- [x] Draw unresolved event times from their original conditional tails and apply the final
  event-driven stratified analysis to that same patient history.
- [x] Use exact count conditioning plus continuation-centered mixture importance sampling so a rare
  continuation branch is sampled directly rather than reached only by rejection.
- [x] Report fixed-scenario public-history compatibility, continuation and final conditional
  probabilities, effective sample sizes, maximum weight share, and paired futility sensitivity.

`posterior.py` implements the isolated WP6 conditioning engine. For each draw it samples one fixed-N
enrollment history from the supplied accrual model, calls a target-scenario generator for protocol
strata, randomized arm, censoring, and patient death-time CDF/quantile functions, and selects one of
the same nine disclosure-lag branches used by WP5. Enrollment evidence is checked against those
actual entry dates. The corresponding entry dates also determine every patient calendar event CDF,
so the engine never multiplies a marginal accrual likelihood by an event realization from a
different enrollment history.

The event proposal enumerates every allowed monotone cumulative integer-count vector. For natural
REGAL branches these are exactly 60/72/78 followed by either 78 or 79 at the event-80 censor cutoff.
A scaled backward dynamic program draws patient interval assignments exactly conditional on the
selected vector. The base component is the target event model itself; optional exponential-tilt
components also target treated-event shares corresponding to continuation-region interim scores.
The proposal is a known mixture, and the exact target/proposal category-density ratio removes both
the count conditioning and the continuation tilt. Within-interval event times are then drawn from
the target conditional quantiles. Deaths after the last public cutoff remain unresolved rather than
being discarded, and are projected forward to the actual 80th-event analysis. Censoring is sampled
before outcomes and caps each patient's observable-event CDF and risk time. Range-valued count
targets are projected into a support-preserving monotone interior so no allowed event interval is
accidentally removed. If a continuation tilt cannot be fitted for one latent draw, that component is
omitted and the exact base proposal remains in force; attempts, fallbacks, and affected draws are
reported. `max_quota_states` consistently limits logical patient-by-state DP cells in both the quota
probability and conditional-sampling paths. A non-selected component with zero mass on the sampled
count vector contributes zero mixture density. If that component was selected, the outcome is kept
in the Monte Carlo denominator as a zero-weight draw, preserving the full nominal mixture law rather
than conditioning on proposal success. `ConditioningResult.proposal_infeasible_draws` reports how
often that occurs. A mathematically positive quota probability that underflows remains a distinct,
loud numerical error rather than being misclassified as structural zero mass. The public
`TiltProposalError` supports direct low-level callers.

`ConditioningResult` keeps `P(public history | fixed scenario)`,
`P(continue | public history, scenario)`, and
`P(final rejection | public history, continue, scenario)` separate. It also reports history and
continuation effective sample sizes and the maximum normalized history weight. The public futility
rule remains unknown. It additionally reports tilt attempts, fallback components, affected draws,
the fallback rate, and selected proposal/count pairs that were structurally infeasible; per-draw and
run-level maximum errors, along with run-level mean iterations, are `None` if no tilt converged rather
than misleadingly reporting zero.
`condition_futility_sensitivity_grid()` reuses identical importance
draws for no futility and explicit one-step-HR thresholds rather than selecting one as the protocol
truth.

`audit/v2_interim_conditioning_validation.py` deliberately uses an uncalibrated strong-effect
scenario where the exact-count base proposal produced no continuation draws in the pinned
300-draw run. Adding the continuation-centered component produced 71 raw continuation draws
(weighted ESS 3.89) while giving a consistent public-history compatibility estimate. This validates
rare-branch access; none of those illustrative probabilities is a REGAL forecast. WP6 itself still
operates under one fixed scenario (or one supplied prior-predictive family) at a time. WP7 now
supplies effect-family and parameter draws and averages those conditional projections.

### 7. Average across GPS effect structures

- [x] Include no effect, proportional hazards, delayed proportional hazards, cure-fraction
  difference, delayed cure, and waning/piecewise effect families.
- [x] Retain the current responder/cure construction as a separately labeled exploratory family.
- [x] Draw effect parameters from explicit within-family priors rather than refitting an arbitrary
  parameter boundary and calling the boundary evidence for cure.
- [x] Use a prior-predictive accrual model that does not recycle the 20/104/126 anchor centers as
  prior information; apply public enrollment evidence once in the WP5/WP6 likelihood.
- [x] Calculate posterior family weights from each family's joint compatibility with the public
  history and observed continuation, then average the family-specific final projections.
- [x] Report skeptical, balanced, and cure-favoring model-weight sensitivity while reusing the same
  family likelihood estimates.
- [x] Preserve WP6's paired futility-rule sensitivity through the complete family average so no
  unpublished rule is silently selected as protocol truth.

`posterior.py` now implements the WP7 prior-predictive and model-averaging layer. The six required
families are joined by a seventh `responder_cure_exploratory` family. No effect, PH, delayed PH, and
waning start from the exact scale-aware marginal cure-mixture survival curve from WP2. Their hazard
ratios transform that component-level all-cause cumulative hazard, making PH genuinely proportional
on the marginal curve. This statistical construction also transforms the population-mortality
contribution and therefore is not a biological disease- or excess-hazard effect. Delayed PH begins
that transformation after a sampled landmark; cure difference moves a sampled fraction of otherwise
uncured GPS patients onto population mortality; delayed cure preserves their original hazard until
the landmark and removes only disease hazard thereafter; and the waning family uses distinct early
and late piecewise hazard ratios. The exploratory responder family samples an immune-response
fraction, gives responders their own cure probability and the BAT non-cured component mixture, and
keeps non-responders on the Observation profile. Latent cure/response states are drawn before event
times and are integrated rather than fitted to the same event history.

Every family is run through the unchanged WP6 latent-history engine. Its parameter prior is sampled
inside the scenario generator, so the resulting `ConditioningResult` integrates parameter and
patient-level BAT-composition uncertainty within that family but still has
`is_posterior_forecast = False`. Across families, Bayes' rule uses

\[
w_j(H,C) \propto w_j\,P(H\mid M_j)\,P(C\mid H,M_j),
\]

where `H` is the public enrollment/event history and `C` is the actual interim continuation. The
headline conditional final-rejection probability is the posterior-weighted mean of
`P(final rejection | H, C, M_j)`. `posterior_model_average()` refuses incomplete or duplicate
family sets. A complete `PosteriorForecastResult` has `is_posterior_forecast = True` only when every
family also has history and continuation ESS of at least 100 and maximum history weight share no
greater than 5%; otherwise `forecast_readiness_issues` identifies the failed numerical gates.

The default WP7 accrual prior draws a log-linear calendar slope over the registry opening-to-close
window. It uses the known opening and completed-enrollment boundary as support but does not center
itself on the intermediate public counts; those counts remain likelihood evidence. Default effect
parameter ranges and the three model-family weight profiles are transparent analyst priors, not
company disclosures. `condition_effect_families_futility_sensitivity_grid()` reuses identical
importance draws within each family across no-futility and explicit HR-threshold assumptions before
the family average. `audit/v2_effect_model_averaging_validation.py` exercises all seven families,
the complete WP6 conditioning path, paired futility rows, Bayesian weight normalization, and prior
sensitivity on a compact synthetic trial. Its printed values are validation fixtures, not a REGAL
forecast.

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

- **v1.1 — complete:** truthful labels, documentation corrections, and UI consistency fixes.
- **v2-scenario — complete:** corrected survival, BAT strata, and trial mechanics.
- **v2-forecast backend — complete:** public-history likelihood, interim-continuation conditioning,
  and posterior model averaging. WP8 still has to publish a versioned result bundle and expose it in
  the interface.

Only a complete, numerically ready `PosteriorForecastResult` may be described as the v2 posterior
forecast. Legacy scenario rates, one-family WP6/WP7 projections, synthetic audit values, incomplete
family sets, and complete averages that fail the ESS/weight-concentration gates must not be described
as REGAL's probability of success.
