# WT1 biology-informed responder prior

This note documents the external biological evidence used by the optional REGAL
responder/cure prior in `biology_priors.py` and `biology_informed_posterior.py`.
It is deliberately kept separate from the blinded REGAL public-history
likelihood.

## Two separate questions

The model separates:

1. **Immune-response probability** — the probability that a GPS-treated patient
   mounts a measurable WT1-specific cellular immune response.
2. **Responder durable-benefit probability** — conditional on such a response,
   the probability that the patient enters the responder/cure family's
   durable-remission / very-low-disease-hazard state.

This prevents the reported REGAL 80% immune-response rate from being treated as
though 80% of GPS patients necessarily obtain a survival benefit. SELLAS did not
publicly disclose the immune sample size; the default 8/10 translation used below
is therefore a working modeling assumption, not a reported patient count.

## Immune-response prior

The direct GPS evidence is pooled with a Beta(1,1) reference prior:

| Cohort | Responders | Evaluable | Denominator status |
| --- | ---: | ---: | --- |
| GPS phase 2 AML | 9 | 14 | reported |
| REGAL interim randomly selected GPS patients | 8 | 10 | working assumption from reported 80% |
| **Combined default** | **17** | **24** | includes assumed REGAL n=10 |

Under that default assumption, the resulting response probability is
**Beta(18,8)** with mean **69.2%**. The phase-2-only Beta(10,6) and REGAL-only
Beta(9,3) distributions remain available for exchangeability sensitivity.

Because the public disclosure gives 80% but not the denominator, the code also
exposes this denominator sensitivity:

| Assumed REGAL n | Implied responders | Pooled posterior | Pooled mean |
| ---: | ---: | --- | ---: |
| 5 | 4 | Beta(14,7) | 66.7% |
| 10 | 8 | Beta(18,8) | 69.2% |
| 15 | 12 | Beta(22,9) | 71.0% |
| 20 | 16 | Beta(26,10) | 72.2% |

GPS phase 2 source:
https://pmc.ncbi.nlm.nih.gov/articles/PMC5812332/

REGAL source reporting 80% in a randomly selected sample, without sample size:
https://www.sec.gov/Archives/edgar/data/1390478/000110465925005648/tm254291d1_ex99-1.htm

## Why responder survival is not meta-analyzed directly

The available WT1 responder studies are not exchangeable. They use different
vaccines, response definitions, populations, endpoints, transplant settings,
and landmark rules. Several of the most dramatic responder-versus-nonresponder
hazard ratios come from very small non-randomized subsets and are vulnerable to
prognostic and guarantee-time bias.

Accordingly, the model does **not** plug those published HRs into REGAL. Instead,
it uses an explicit three-component mixture describing the probability of a
truly durable responder state.

## Evidence anchors

### 1. OCV-501 randomized placebo-controlled AML trial — skeptical anchor

In elderly AML CR1, OCV-501 did not improve outcomes overall:

- DFS HR **0.933** (95% CI 0.590–1.477)
- OS HR **0.956** (95% CI 0.592–1.544)
- long-term 5-year DFS **36.0% vs 33.7%**

Post-hoc immune responders did better, but the randomized treatment comparison
was negative. This is the strongest reason to retain substantial probability
that measurable WT1 immunoreactivity is prognostic or insufficient rather than
causally producing a large survival effect.

Sources:
https://pubmed.ncbi.nlm.nih.gov/34677647/
https://pmc.ncbi.nlm.nih.gov/articles/PMC10123586/

### 2. GPS phase 2 AML — direct but small responder association

Among 14 immunologically evaluable patients, 9 had a CD4 and/or CD8 response.
Responder and nonresponder survival curves separated:

- DFS median: **not reached vs 15.6 months**, p=0.11
- OS median: **not reached vs 35.8 months**, p=0.08

This is directly relevant to GPS but is a small non-randomized correlative
analysis, so it supports the moderate-benefit component rather than being used
as a literal efficacy estimate.

Source:
https://pmc.ncbi.nlm.nih.gov/articles/PMC5812332/

### 3. WT1 mRNA dendritic-cell vaccine — independent platform

In 30 high-risk AML patients, 13 had an antileukemic response. Reported outcomes
included:

- 5-year OS **53.8% in responders vs 25.0% in nonresponders**, p=0.01
- in CR1, 5-year RFS **50% vs 7.7%**, p<0.0001

This independently supports a durable WT1-responder phenotype, but responder
classification remains non-randomized and the platform differs from GPS.

Source:
https://pubmed.ncbi.nlm.nih.gov/28830889/

### 4. 2026 pediatric post-allo-HSCT WT1 peptide study — strong mechanism, low transferability

The week-6 immune landmark analysis reported:

- 3-year OS **90.9% in immune responders vs 40.0% in nonresponders**

The study also showed expansion of WT1-specific CTLs and memory phenotypes in
blood and bone marrow. It is compelling mechanistic evidence for a durable-tail
mechanism, but the study is very small, pediatric, and post-transplant. The model
therefore gives the strong-benefit component only a small weight.

Source:
https://pubmed.ncbi.nlm.nih.gov/42308229/

## Elicited durable-benefit mixture

The existing responder/cure family previously used a flat
`Uniform(0.20, 0.85)` prior for `responder_cure_probability`, with mean 52.5%.
That is broad but surprisingly optimistic in expectation.

The evidence-informed alternatives use three beta components:

| Component | Beta | Mean | Interpretation |
| --- | --- | ---: | --- |
| Skeptical | Beta(1.5, 8.5) | 15% | response may be prognostic or insufficient |
| Moderate | Beta(4, 6) | 40% | causal benefit exists but is far smaller than raw responder associations |
| Strong | Beta(7, 3) | 70% | genuine large durable-tail biology |

Mixture sensitivities:

| Prior | Skeptical | Moderate | Strong | Mean durable probability |
| --- | ---: | ---: | ---: | ---: |
| Skeptical | 60% | 35% | 5% | **26.5%** |
| Balanced | 45% | 45% | 10% | **31.75%** |
| Mechanism-favoring | 30% | 50% | 20% | **38.5%** |

All three are below the legacy flat prior's 52.5% mean. This is intentional. The
new biological evidence narrows the immune-response rate, but the clinical
literature argues for **more skepticism about how often an immune response becomes
a truly durable survival state**.

## Interpretation in REGAL

With the pooled response mean of 69.2%, the implied prior mean fraction of all
GPS patients entering the durable state is approximately:

- skeptical survival prior: **18.3%**
- balanced survival prior: **22.0%**
- mechanism-favoring survival prior: **26.7%**

These are prior-predictive quantities before conditioning on REGAL's blinded
60/72/78-event history and interim continuation. The public trial history remains
responsible for deciding whether such responder-tail models gain or lose
posterior weight.

The side-by-side audit runner is:

```bash
python audit/biology_informed_posterior_comparison.py \
  --nsim 10000 \
  --output data/biology_informed_posterior_comparison.json
```

It integrates the default non-responder families once, recomputes only the
responder family for each biology prior on the same family-specific random seed,
and reports posterior rejection probability across the complete futility-HR
sensitivity grid. By default the CLI refuses to print or write results if any
forecast-readiness gate fails. `--allow-diagnostic-output` may be used to inspect
such a run, but the table and JSON are then explicitly marked `diagnostic_only`
and retain the readiness failures and ESS/weight diagnostics.

## What this prior does not claim

- It does not claim the WT1 studies prove causality.
- It does not use the Japanese responder HR or other extreme small-study HRs as
  direct REGAL efficacy inputs.
- It does not assume immune nonresponders receive zero benefit as a biological
  truth; that remains a simplifying feature of the exploratory responder/cure
  family.
- It does not alter the validated default WP7 priors. Biology-informed analyses
  remain explicit opt-in sensitivity runs.
