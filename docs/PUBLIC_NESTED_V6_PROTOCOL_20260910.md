# Public nested regularization v6: frozen protocol

2026-09-10. Register metadata, new inner GO vocabularies and execution code before
decoding the existing expression cache. Preserve completed v3/v4/v5 artifacts.
This is exposed-public development, not an untouched final test.

## Review fixes, not retrospective score repairs

V6 rejects overflowed context means/scales before normalization; an infinite
scale must not silently erase a finite context block. It binds each lambda and
validation kind into saved state, checks fitting row counts, and verifies exact
array dtypes, full fold/group/target rosters and typed provenance. Boolean0/1
coercions and malformed role summaries are rejected.

Nested replay uses JSON-normalized trees consistently, including returned inner
reports, so tuple/list serialization does not falsely break valid reconstruction.
Enormous/nonfinite numeric values fail with controlled validation errors.

Biased squared RBF MMD is mathematically nonnegative but can be slightly negative
under floating-point summation. Preserve every raw group MMD. Canonicalize only
raw values in[-1e-12,0) to0; reject more negative or nonfinite values. Record the
correction and recompute aggregates from canonical group values. This is NOT
a tolerance on model comparisons. Positive MMD values are unchanged, and v5's
failed distribution gates are not altered.

These are robustness fixes, not evidence that the completed v5 results were
corrupted or that weaker regularization necessarily improves prediction.

## Same outer cohort and new inner metadata roles

Reuse the authenticated v5 parent:
`26e03fa1563f783e4c2fedfc74b38ae100f1e8943665da90474a97c0ef0a757f`.
The parent binds v3 roles and the v2 cache: 56 common K562/Jurkat targets,
398 source/target/batch groups,512 genes, four42-fit/14-held target folds,
eight source directions and772 protected exclusions.

Held-target treated cells in BOTH sources and every outer held-source response
remain excluded from all model fitting and lambda selection. No new raw
expression, challenge-treated, H1/RPE1/HepG2/Feng expression, download or cloud
transfer is admitted. The cached log1p-CP10k representation remains unchanged.

Inner target roles use seed20260925 and SHA256 over compact UTF-8 JSON arrays
`[seed,"nested-target-v6",outer_target_fold_id,target]`.
From each outer fold's42 fitting targets, rank by digest then exact symbol:
first14 tune, remaining28 fit; serialize both roles alphabetically.
The target partition is shared across both source directions and all arms.

For each of97 unique source/batches, split the original32 anchor-reference
cells once by original source/row identity. Rank SHA256 of
`[seed,"nested-anchor-v6",source,batch,original_row]`; first16 are inner fitting
references and remaining16 are inner tuning anchors. Serialize row lists
numerically. Use one global assignment consistently across every target,
outer fold, lambda and comparison arm. Never split controls separately per target.

Within each outer fitting source:

- Inner fitting responses/positive priors use only28 inner-fitting targets'
  treated cells and their16 fitting-reference cells.
- The14 inner-tuning targets' treated cells never enter fitting. Their16 tuning
  anchors are disjoint from every fitting-reference identity, even when batches
  are shared.
- Independent32-cell context pools supply input summaries and the frozen
  decoder's allowed positive-expression donors. They never fit response/prior
  moments. Eight sham cells remain unused audit-only data.
- Views physically omit outer-only expression rows while retaining group metadata.
  Control/treated rows preserve parent-cache indices and original-row identities;
  context donors preserve original identities. Unused context-summary rows are
  zero placeholders, never consumed.

Inner fitting and tuning share the SAME biological source. Use an explicit
InnerTargetFold with honest same-source labels and an outer-fold identifier;
do not fake another held-source label to bypass old validators.

Freeze the inner row/target plan before parsing public GO. Reparse the same pinned
GAF/OBO and build four new vocabularies from the28 inner-fitting targets only,
transforming only the42 outer-fitting targets. Recompute present/in-vocabulary/
unknown masks, so a tuning gene with no term in this vocabulary is marked missing.
Inner shuffle packets are constructed afresh within28/14 roles. Outer42-fit
vocabularies and v3 outer shuffle roles remain unchanged.

## Models, selection and output

Seven fixed arms: control, context, additive, true, and the three v3 packet-shuffle
seeds20260921/20260922/20260923. Keep the v5 kernel geometry:

- C = normalized-context inner product; T = normalized-GO/mask inner product.
- Context: 1+C+C*C.
- Additive: 1+C+C*C+T.
- True/shuffles: 1+C+C*C+T+T*C.
- Products are elementwise; intercept and all implicit features are penalized.

All transforms and weights are fitted only on admitted fitting targets/groups.
Context scale floor0.1; source/target/batch weights sum to1. No source-ID or
target-ID one-hot model is introduced.

Each learned arm and outer direction tests EXACTLY lambda{0.1,1.0,10.0} on the
same inner roles and its own inner feature packets. Fit both v4 hurdle response
heads together and evaluate the actual frozen sparse decoder on tuning anchors.

Select the lambda with minimum inner target-pooled centroid MSE; exact score
ties choose the STRONGER regularization. No distribution metric, outer score,
sham outcome, future leaderboard result, feature-coverage subgroup or gain
chooses lambda. The grid, budget and tie rule are identical for all learned arms.
Control performs no inner tuning/fit; its zero coefficients use a bookkeeping
lambda10 only, without claiming selection.

After selection, refit that arm once on all42 outer-fitting targets and original
32-cell training-reference pools. Evaluate only the original outer held-source/
held-target groups. Record every inner candidate score and selected lambda.
This is genuine inner hyperparameter selection, explicitly reported as such.
It is not a claim of no fitting or a pure interaction-only ablation when arms
select different lambdas.

Reuse frozen v4 occupancy/amplitude smoothing, clipping, sparse support changes,
donor ranking, original-row deduplicated positive priors and exact zero/NTC
bypass. Outputs remain512-gene continuous log-normalized values, NOT raw UMI
counts, a full-gene emitter or a resumable flow pretraining checkpoint.

## Metrics and fixed progression gates

Outer metrics retain the v5 definitions. Target-pooled centroid/occupancy MSE
averages matched-batch vectors within target BEFORE squaring, then macro-averages
targets per source. Also retain batch-macro MSE, canonical/raw MMD, zeros,
requested/realized occupancy, clipping, unresolved activations and support changes.

True GO must satisfy, in BOTH outer held sources:

- At least1% primary MSE improvement versus control and each shuffle.
- Strict primary MSE improvement versus context AND additive.
- Canonical batch-macro MMD no worse than every comparator.
- Occupancy MSE and batch-macro centroid MSE no worse than control/context.
- Aggregate absolute zero-rate error no worse than control.
- Finite/nonnegative outputs, exact zero/NTC identities, full artifact replay,
  zero unresolved activations, and each head-clipping fraction<=1%.

No post-score tolerances or threshold changes. The1e-12 MMD canonicalization
is applied before aggregation equally to every arm and never to a comparison
difference. Report all gates independently, even if another fails.

Training-only df, kernel trace, head response/fitted RMS and positive-count
rounding diagnostics remain descriptive. Small df does not prove that lower
lambda helps; observed responses contain noise.

## Bounded execution, tracking and evidence

Run on an approved BioHPC compute host with two numerical threads, GPUs disabled;
never compute on AIDA/Anvil/BioHPC login nodes. No H100 allocation, paid embedding
call, new cloud resource or download is required.

Budget: six learned arms ×eight directions ×(three inner fits +one outer refit)
=192 two-head kernel fits, plus eight outer control references. Total200
fold/candidate evaluations. Save inner and outer coefficients, transforms,
responses/priors, predictions, donors, original and parent-cache row mappings
and all candidate scores. Safe NPZ archives stay within existing512MiB per-file
compressed/expanded limits; no pickle/object arrays.

Independently reconstruct every inner/outer fitting state, verify serialized
lambda and dual equations without refitting, replay donor outputs and all
metrics, then reproduce the selection decision before replaying the outer model.
Authenticity and successful execution are not scientific success.

W&B remains offline and scalar-only. For each learned arm:24 inner candidate
events +eight outer events +one aggregate =33. Control:eight outer +one aggregate
=9. Total207 events. Inner and outer namespaces are separate; the scope label
denotes the OUTER held source, while inner training/tuning belong to its opposite
fitting source. No raw expressions, identities, credentials or arbitrary fields
are logged; no cloud synchronization is claimed.

Use fresh artifact directories, bind metadata/protocol/code hashes before
cache decoding and revalidate them on completion. New metadata readers reject
symlink/FIFO leaves and use bounded regular-file reads. Frozen legacy parent
validation remains unchanged; this does not claim universal filesystem-race
hardening of older dependencies.

## Limits and downstream readiness

A single inner28/14 split tunes unseen-target performance within one source,
not cross-source generalization. Shared batches and control pools still induce
dependence. No independent-replicate significance claim is justified.

Sixteen-cell tuning anchors have different noise and occupancy quantization
from32-cell outer anchors. Report that mismatch; never duplicate cells to
simulate32. Explicit NTC bypass is not learned null calibration. The common
nonlinear context basis is not exactly capacity-matched to GO arms.

A separate authenticated one-shot readiness report recomputes the completed
gate. A pass can permit a separately registered promoter ablation. Feng
post-training still requires an authenticated compatible trainable public-flow
parent and registered eligible Feng consumer; these ridge coefficients do not
supply them. Submission still requires a validated full-gene raw-count emitter,
official prep/dry-run evidence and current credential/quota checks. No automatic
promotion, training, submission or new background watcher occurs here.
