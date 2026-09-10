# GO target × control-context kernel v5: frozen public protocol

2026-09-10. Freeze this protocol, code and input hashes before reading the existing
expression cache for this experiment. This is previously exposed public-development
data, not an untouched test. Preserve all completed v3/v4 artifacts and seals.

## Question and fixed cohort

Does a GO-by-control-context product feature improve a sparse control-anchored
predictor beyond additive GO and a shared nonlinear context basis?

Reuse exactly the authenticated v3 roles/vocabulary and v2 cache: 56 common
K562/Jurkat targets, 398 target/source/batch groups, 512 genes, four target folds,
42 fit/14 held targets, two source directions, and 772 protected exclusions.
Held-target treated cells in BOTH sources and every held-source response remain
excluded from fitting. Fit-only priors deduplicate original source/row identities.
Recipient control context is allowed for prediction, never for fitting transforms.
No new raw expression, Feng/H1/RPE1/HepG2/challenge-treated data or downloads.

Seven fixed arms in order: control, context, additive, true, shuffled_20260921,
shuffled_20260922, shuffled_20260923. GO values AND coverage masks move together
using exactly v3's role-contained shuffle assignments. Missing annotations remain
in the target roster. Additive consumes the unshuffled true-GO packet.

## Exact product-feature model

Let c be the v3 training-only weighted-standardized matched-control context
(mean and SD summaries, scale floor0.1), divided by sqrt(context width).
Let z be the v3 stable training-only standardized GO features plus availability
and unknown-target indicators, divided by sqrt(the whole target block width).
No PCA, target-ID one-hot representation, pretrained GenePT acquisition or learned
embedding is introduced.

For fitting/inference rows define C=c c' and T=z z'. Kernel products below are
elementwise; they exactly represent tensor-product features, without allocating
the full expanded feature matrix:

- Control: exact zero response.
- Context: K=1+C+C*C.
- Additive GO: K=1+C+C*C+T.
- True GO and each shuffled arm: K=1+C+C*C+T+T*C.

The common nonlinear context basis prevents attributing any generic nonlinear
context benefit solely to GO. It does NOT match effective degrees of freedom
or kernel energy between arms. The additive comparison isolates the T*C increment
within this new experiment. V4 is historical context, not an identical-basis
comparator, because v5 adds C*C.

Fit concatenated occupancy/amplitude responses with the same normalized
source/target/batch weights and fixed lambda10 as v4. The constant kernel is
penalized, preserving a zero-effect prior. For W=diag(weights), solve
(sqrt(W) K sqrt(W)+10I) A=sqrt(W) Y. Save B=sqrt(W) A; inference is Knew,fit B.
This is equivalent to explicit-feature weighted ridge. All components have
fixed unit kernel weight; no hyperparameter, gain or checkpoint selection.

Only one source is fitted per direction. Thus interaction learning observes
within-source/batch variation and extrapolates to the held source; this cannot
establish learned cross-cell-line interaction transfer from multiple fitting
cell lines. Context may encode technical differences, not just biological state.

## Frozen sparse decoder and descriptive diagnostics

Reuse v4's response smoothing, deduplicated positive prior, +/-4 occupancy and
+/-log4 amplitude clipping, integer positive-support COUNT rounding, unchanged
support preservation, deterministic donor ranking and exact zero/NTC bypass.
These remain continuous log1p-CP10k values, NOT integer UMI counts.

Record training-only effective degrees of freedom and weighted kernel trace,
occupancy/amplitude response RMS and fitted RMS, and source/target/batch-weighted
fraction of gene/groups whose predicted positive count is unchanged by rounding.
These are in-sample descriptive diagnostics, not held-out validation, automatic
lambda selection or alternate candidate predictions. Effective df describes the
linear smoother per response, not the nonlinear sparse decoder's capacity.

Retain all v4 outer pooled/batch mean and occupancy errors, MMD, zero fractions,
clipping, support activation/deactivation and requested/realized occupancy metrics.
No outcome-based tolerances, gain tuning, rounding changes or emission alternatives.

## Fixed progression screen

In BOTH held sources, true GO must:

- Improve target-pooled centroid MSE by at least1% versus unchanged controls and
  EACH of the three shuffles.
- Strictly beat both nonlinear context-only and additive-GO primary MSE.
- Have batch-macro MMD no worse than EVERY other arm.
- Have target-pooled occupancy MSE no worse than control and context-only.
- Have aggregate absolute zero-rate error no worse than unchanged controls.
- Have batch-macro centroid MSE no worse than control and context-only. This
  additional preregistered guard addresses v4's pooled/batch disagreement.
- Pass finite/nonnegative, exact zero/NTC and saved-artifact replay checks,
  with zero unresolved activations and each head's clipping fraction<=1%.

All gates remain reported independently. Comparisons use the exact declared
inequalities; no scientific tolerance is added after scoring. This is an engineering
screen, not a significance test, and small differences are not proof of efficacy.

## Execution, artifacts and monitoring

Run 56 fold/arm evaluations: 48 two-head kernel fits plus8 unchanged-control
references on an approved BioHPC compute host, two numerical threads, GPUs
disabled. Do not run this workload on any login/head node. No H100 allocation,
GCP transfer, paid embedding API or new raw-data download is needed.

Save dual coefficients, normalized fitting features and weights, transforms,
feature packets/receipts, training-only responses/priors, donor/output arrays,
original row mappings and per-fold/aggregate metrics. Independently reconstruct
training transforms and check the saved dual-system equation; replay sparse
predictions and every metric without a ridge refit. Require new output directories
and authenticate every implementation/input/protocol hash before and after.

Record63 offline W&B events, nine per arm (eight folds +one aggregate).
Optional training diagnostics share the fold event; its source label denotes the
HELD-source evaluation, while its training diagnostics belong to the opposite
fitting source. No raw expressions, credentials or arbitrary fields are tracked.
No cloud synchronization is claimed.

A separate one-shot v5 readiness report authenticates the completed suite and
recomputes the screen. Keep the existing v4 watcher pinned to v4, unchanged; it
does not discover v5 or promote/train/submit. No mutable latest-pass pointer.

A scientific/safety pass can permit a separately registered promoter ablation.
Feng post-training additionally needs a compatible authenticated trainable flow
parent plus an eligible donor/batch-split Feng consumer. These ridge weights
are not such a parent. VCC submission additionally needs full official raw-count
output, exact axes/panel, provenance, QC, dry run and fresh authorization/quota
checks. None is created by a diagnostic pass. No automatic promotion or upload.

