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

Implementation requirements:

- Add an explicit `survival_scale = overall | net` to every component.
- Default the current literature inputs to `overall`.
- Remove the unconditional background-mortality multiplier from the legacy survival mixture.
- Ensure cured survival follows population mortality rather than forming an immortal plateau.
- Replace top-survivor truncation with a pre-outcome case-mix/frailty model applied symmetrically to
  both randomized arms. No inclusion rule may depend on future realized survival.

### 3. Represent BAT randomization strata and regimens

- Represent the four planned strata separately: supportive care/hydroxyurea, HMA, venetoclax, and
  LDAC.
- Use approximately 25% per stratum as the primary protocol-compatible configuration.
- Store both the randomization stratum and the patient-level regimen/components, so "and/or"
  combinations do not double-count patients.
- Remove the bear preset as a protocol-plausible primary case. Retain venetoclax-dominant allocation,
  if at all, only as an explicitly off-protocol stress test.
- Treat the current component weights as a legacy comparison.

The legacy equal-strata run is more bullish than the current default (about 99.9% scenario power,
median HR about 0.30, and about 94% interim efficacy crossing). Those numbers are characterization
checks, not v2 forecast targets. Their main implication is that the observed interim continuation
has greater evidential weight under equal allocation.

### 4. Implement the trial decision process

- Calculate the one-sided 0.025 Lan-DeMets/O'Brien-Fleming efficacy boundaries from the spending
  function and the 60/80 information fraction. Expected validation values are approximately
  `z60 = 2.33` and `z80 = 2.02`.
- Simulate all interim branches: efficacy stop, futility stop, and continuation.
- Keep the unknown futility rule configurable and report a sensitivity grid.
- Replace the unstratified one-step approximation with the protocol-compatible stratified log-rank
  or Cox analysis; keep the approximation only as a diagnostic.
- Validate the null type-I error and decision branches by simulation.

### 5. Build a public-history likelihood

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
