# GenePT/GO + matched-control flow v11: experiment draft

Draft, not a training execution contract. Numeric GenePT conversion and
source-bound feature transforms are now completed and replayed, as recorded in
[the feature report](GENEPT_NUMERIC_V11_RESULTS_20260910.md). The new model and
execution runner still require implementation, tests and registration before
GPU fitting. No automatic submission,
Feng admission or PromoterAI use is authorized by this document.

## Question and fixed comparisons

Does GenePT carry useful target information beyond GO and availability, after
separating input-branch scale from information content? v10 public held-target
MSE was4.83%/9.05% worse than controls; the exploratory VCC result was-0.06402.
Neither establishes a useful model. v10 branch RMS is a scale hypothesis,
not causal feature importance. Do not select new settings from the VCC score.

Proposed2x3factorial, source-specific for K562 and Jurkat, seeds20260926 and
20260927. All six configurations per source/seed have identical parameter
shapes and initial parameters:

| Target input | Original projection scale | Fit-control-calibrated scale |
| --- | --- | --- |
| GO + GenePT availability; zero GenePT vector | A | D |
| GO + correct GenePT vector and availability | B | E |
| GO + shuffled GenePT vector; unchanged availability | C | F |

Proposed24fits x600steps=14,400AdamW updates, batch128, hidden128,2blocks,
learning rate1e-4, weight decay1e-4, gradient clip1, Heun16, fixed-final
checkpoint only. This is fresh public pretraining, not resumed post-training.
Resource intent:one verified free RTX2080Ti on an approved BioHPC compute node,
6GiB GPU allocator cap,16GiB RSS,8GiB output, one-hour total execution cap.
Final execution registration must pin exact device, code, dependencies, gains
and inputs. No login/head-node training; no new cloud compute expenditure.

## Feature and response boundaries

Keep the authenticated v10 public cache,7,097 measured normalization genes and
482 modeled genes. All98 old held-out target identities stay excluded from
every response fit, as do the original56panel and772protected identities.
K562 and Jurkat remain separate source checkpoints, not pooled-response fits.

GO uses each source's unchanged fit-target-only vocabulary. GenePT uses the
verified numeric artifact; exact duplicate representations are quarantined as
documented in[its report](GENEPT_NUMERIC_V11_RESULTS_20260910.md). Do not guess
aliases, rename targets or zero-fill response genes. Missing/quarantined GenePT
does not mean no perturbation:GO and availability still condition a real target.

Fit GenePT coordinate means and one scalar RMS using only available source-fit
target vectors after quarantine; persist fitting identities, statistics and
hashes. No PCA or data-dependent transform is fitted on held/official targets.
Reuse the source's frozen transform for opposite-source and official names.
GO and GenePT have separate named projections. Instantiate both projections
in every arm, including masked arms. Preserve feature availability flags in all
three information arms. Derange GenePT only, independently within fit/held
roles and availability strata; report singleton strata and unchanged missing
rows. Never shuffle GO or alter expression/time sampling through the shuffle RNG.

Calibrate branch gains once per source/seed from available source-fitting
target features and fitting-control pools only, without treated responses.
Use the same frozen gains across correct/shuffled/masked arms. The exact
calibration formula and any bound must be registered before execution. Log
both unscaled/scaled branch activation RMS and gradient RMS; scale alone is
not signal. Hash named initial parameters and row/time sampling per arm.

An explicit perturbation indicator/null fast path must return original
non-targeting controls exactly, including count identity. A zero embedding
does not enforce this in a network with state/context/time branches and biases.
Keep the nonnegative decoder and native unmodeled-count anchoring. This null
path is an architectural invariant, not a learned guarantee for real targets.

## Validation and movement to later stages

Report source-local held targets and opposite-source held targets separately,
using only the training source's vocabulary/transforms. These are public
development diagnostics:old holdouts have already been inspected, so do not
call them an untouched test. Compare unchanged controls, masked GenePT and
shuffled GenePT. Report target-balanced centroid MSE, distribution MMD,
zero-fraction error, variance and target sensitivity. Preserve per-target and
per-control-pool results; repeated controls are not independent cells.

Public centroid MSE and MMD are not interchangeable with official competition
metrics. In particular, VCC FID is DE direction fidelity, not a distribution
distance. Score-aligned public diagnosis must examine effect directions, DE
significance/yield, and pseudobulk discrimination with the pinned evaluator;
improved MMD alone would not establish leaderboard improvement. See the
[primary-source metric audit](research/NEXT_ROUTE_PRIMARY_SOURCE_AUDIT.md).

Primary evidence for GenePT is E beating both D and F consistently across
seeds/sources, not merely beating unbalanced A. Record every optimizer step,
branch diagnostics and final evaluations in W&B; offline recording must not
be described as a synced cloud dashboard. Save replayable model+AdamW state.

482modeled genes are only2.6%of18,533submission genes. Holding the other18,051
native counts unchanged is safe but limits modeled response coverage; improving
target embeddings alone may not solve full-transcriptome distribution errors.
Investigate full-output modeling as a separate axis after this diagnostic,
without mixing its effect into the GenePT comparison.

Feng follows a successful, interpretable public-only conditioning test plus
raw gene-axis/count admission and exact donor/control-role validation. A true
post-training run restores model AND AdamW parent state; random initialization
is not post-training. Its smaller control pools require a new protocol.
Promoter composition descriptors are not PromoterAI embeddings; actual model
features need their own acquisition/license and output-axis ablation.

User-authorized exploratory VCC attempts remain available after technical
validation; do not let failure of a scientific promotion threshold masquerade
as an API prohibition. Preserve official daily/in-flight checks and submit a
meaningfully distinct, provenance-bound candidate, not a renamed duplicate.
