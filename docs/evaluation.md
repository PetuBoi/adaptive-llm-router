# Experimental protocol

## Hypothesis and baselines

Hypothesis: prompt-only per-model Ridge estimates can select a cheaper candidate with useful
task performance compared with a validation-fixed strong model. This is a measured tradeoff;
the implementation does not assume it beats every baseline on every task or latency metric.
The default utility weights (quality=1, cost=0.15, latency=0.1), rule threshold (400 characters or
code fences), quality target (0.7) and latency target (3,000 ms) are fixed protocol settings.
They were not optimized using held-out test outcomes. Users may configure other weights.

Compare cheapest, strongest, reproducible random, a length/code heuristic, learned ML routing
and an evaluation-only quality oracle. The oracle maximizes observed quality; equal-quality
ties choose lower observed reference cost and then model ID. It is not deployable.

## Authentic data provenance and normalization

[xRouteBench](https://huggingface.co/datasets/ulab-ai/xRouteBench), generic configuration and
candidate metadata, is attributed to Feng et al., *LLMRouter: Unified Infrastructure for
Developing, Evaluating, and Deploying LLM Routers* (2026). The adapter is implemented independently.
The inspected card does not declare an explicit dataset license. Source terms govern downloads;
this repository's MIT license applies only to original code and fixtures. Raw downloads and
per-observation derived files are excluded from Git.

The adapter inspected the actual schema: `query`, `task_name`, `task_id`, `metric`, `model_name`,
`response`, `performance`, `input_tokens`, `output_tokens` and `response_time`. Candidate prices
are `input_price_per_1m` / `output_price_per_1m`. Latency seconds are converted to milliseconds.
Published unit-interval task scores are preserved, with metric names. No model response/code is
executed. Missing outcomes stay missing; they are never assigned invented quality scores.

Original train/test files remain unchanged in `data/external/`. Normalization collapses exactly
identical observations. Conflicting duplicate prompt/model outcomes cause every outcome of that
prompt to be quarantined from that split; no averaging or favorable outcome selection occurs.
For the verified revision: one exact training duplicate was removed and 71 remaining rows from
two conflicting prompts were quarantined. Another 198 training rows crossed test prompt identity
or source task family and were quarantined. No test records were removed in this revision.

Normalized text hashes provide stable prompt IDs. Available task IDs define families within their
task; absent IDs fall back to normalized text identity. Training/test prompt identities and families
must be disjoint. GroupShuffleSplit creates validation only within training; TF-IDF/scaling fit
only on fit prompts during alpha selection. Near duplicates lacking usable family metadata may
remain; there is no claim of comprehensive semantic deduplication.

Full run: 4,474 cleaned training prompts (3,578 fit / 896 validation), 3,729 original test prompts,
18 candidates, 13 tasks. Quick mode chooses whole families in seeded hash order **per original
split**, with all model observations retained. Large families may slightly exceed the requested
cap. Model metadata records sampling, prompt IDs, source revision, hashes, features and alphas.
Evaluation uses the exact saved test cohort for that trained artifact.

## Estimates and recorded outcomes

Decisions use prompt-only predicted quality, token estimates and latency estimates. Input/output
token estimates are rounded before reference cost arithmetic. Actual retrospective costs use
recorded selected-model tokens and published reference prices:

`USD = (input_tokens * input_price_per_million + output_tokens * output_price_per_million) / 1e6`

These reference prices are historical and do not imply current billing. Recorded latency is
historical, provider-specific and affected by execution conditions. Log-target regression gives
a central point estimate that may underestimate arithmetic mean latency in skewed distributions;
it has no calibrated tail guarantee. Estimated hard constraints cannot guarantee final outcomes.

Every strategy uses the same complete prompt/model matrix. Missing cells remove the prompt
uniformly for comparisons and are reported. If constraints cause routing failures, failure rates
use the entire comparable cohort, while observed outcome tables use the intersection of successful
prompts across all strategies. Failed requests do not receive fabricated zero quality/cost/latency.
The default unconstrained benchmark has no failures. Its 100% constraint attainment is therefore
vacuous; behavioral tests separately verify hard estimated constraints.

## Quality interpretation and uncertainty

The primary aggregate is a **macro mean of task means of published [0,1] task scores**. It treats
each task equally and assumes a unit score is a useful routing utility proxy across tasks.
It is not universal accuracy and does not make exact match, code evaluation and F1 semantically
equivalent. Always inspect `task_performance.csv`, which preserves the task/metric definitions.
Target attainment compares each recorded unit score with the fixed 0.7 threshold under that
same proxy assumption. Cost/latency means use prompt-weighted selected outcomes.

Confidence intervals use 200 seeded bootstrap resamples within tasks (prompt independence
assumption), then macro aggregation. They are descriptive percentile intervals. Related test
prompts may violate independence; no statistical significance claim is made. Quality predictor
MAE is also reported so aggregate routing performance does not hide weak calibration.

`reports/real/` stores aggregate recorded results and charts. `reports/synthetic/` stores explicitly
SIMULATED results from fictional economy/balanced/premium models. Per-prompt decisions and
predictor errors are generated locally, ignored by Git and available to the dashboard after evaluation.
Synthetic generator families share templates; synthetic performance is a reproducibility demo.

## Reproduction

Run the commands in README.md. No paid provider or GPU is needed. `uv.lock` fixes software
versions. Seeds, predictor configuration, training counts and dataset revision are included in
the computed summary. Tests validate selected outcome joins, common cohorts, missing observations,
oracle restrictions, cost arithmetic, split isolation, serialization and constraints.
