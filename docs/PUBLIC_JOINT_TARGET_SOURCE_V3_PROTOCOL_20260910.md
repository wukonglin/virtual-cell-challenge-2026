# Public joint target/source validation v3: frozen protocol

2026-09-10. Register this protocol, metadata roles, fold-local GO tables and
implementation hashes before decoding the existing expression cache or fitting.
This is a prospective engineering comparison on previously exposed public
development data, not an untouched test or an estimate of leaderboard gain.

## Scope and fixed roles

Reuse exactly the authenticated public control-anchor v2 cache: 56 common
K562/Jurkat targets, 398 target/source/batch groups, 512 fixed output genes,
CP10k over the 7,563 shared measured genes followed by log1p. All 772 protected
target exclusions remain in force. No new raw expression, downloads, cloud
transfers, H1/RPE1/HepG2/challenge-treated cells or Feng counts are admitted.

Rank target symbols by SHA256 of UTF-8 `20260920|joint-target-source-v3|TARGET`,
breaking ties by symbol; consecutive blocks of 14 define four held-target
folds. Sort each role alphabetically. In each fold, fit the other 42 targets in
one source only, then evaluate the 14 held targets in the opposite source.
Repeat both source directions: eight outer evaluations per arm. No treated
cells of a held target, from either source, enter that fold's fit. Source-held
controls may be used only for recipient context, prediction anchors and
evaluation. Shared reference-cell noise cannot cross sources. This is target-ID
and source holdout, not GO-family-disjoint evaluation. All 56 targets are retained.

Write the expression-free fold plan before constructing new features. Build
four GO vocabularies from the 42 fitting targets only, using the pinned local
2026-08-05 human GAF/OBO acquisition. Transform all 56 external annotations in
each fitting vocabulary. Keep exact-symbol, active direct primary-term rules;
no IEA, ND, NOT, obsolete terms, aliases or ancestor expansion. Held-only terms
cannot enter the vocabulary. Explicit missingness is an input, not a reason to
exclude a target. Report coverage and nearest-fitting-target Jaccard similarity
without outcome-based selection. These are GO features, not acquired GenePT,
Lingshu or PromoterAI embeddings.

## Fixed models and negative controls

Six arms: unchanged control, context-only ridge, true-GO-plus-context ridge,
and three GO-plus-context shuffles with seeds 20260921, 20260922, 20260923.
Shuffles separately permute the fitting 42 and held 14 target feature packets;
values and all consumed coverage/unknown flags move together. No packet crosses
roles. Use source-independent SHA-seeded permutations, identical in both source
directions of a target fold. Report identity fixed points and unchanged packets
(including duplicate annotations). Three shuffles are descriptive controls,
not a permutation significance test.

Fitting response is treated mean minus independent anchor/reference mean in the
training source. Context is mean and population standard deviation of the
separate 32-cell matched NTC pool. Fit context scaling on training groups only
(standard-deviation floor 0.1), GO scaling on fitting target annotations only,
using the stable exact-constant-column transform. No PCA. Equal target and
batch weighting, feature-block width scaling, ridge lambda 10, penalized
intercept and gain 1 are fixed. The unchanged control arm has zero effect.
The saved transform must accept genuinely unseen target annotations; a learned
target-ID lookup is not sufficient. No hyperparameter search or post-training.

Predictions are continuous normalized expression:
`raw = matched_anchor + predicted_effect`, `prediction = maximum(raw, 0)`.
NTC and zero-effect paths preserve the anchor exactly. This is not a count
emitter. No rounding, thresholding, positive background removal, per-target
gain calibration or threshold selection after scores are observed.

## Metrics and fixed engineering screen

Primary: within each held source/target, equally average predicted and observed
batch centroid vectors FIRST, then compute mean squared gene error. Average
all 56 out-of-fold targets equally per source, and both sources equally for the
global summary. Never pool vectors across sources or weight by cell counts.
This target-pooled metric differs from the v2 batch-macro MSE; do not relabel
old v2 results as improved.

Secondary: equal-batch then equal-target then equal-source centroid MSE and
fixed per-batch biased RBF MMD squared, both raw and projected. Report shift
direction, negative fraction, exact zeros in predictions/controls/treated,
projection fraction/magnitude and centroid distortion. Pooling can conceal
opposite batch errors, so retain every per-group and per-target result.

The true-GO screen passes only when, in BOTH held sources:

- Primary MSE improves at least 1% relative to unchanged controls and to EACH
  of the three predeclared shuffled arms.
- Primary MSE is strictly lower than context-only.
- Secondary MMD is no worse than controls, context-only and EACH shuffle.
- Every predicted value is finite/nonnegative; saved raw/projected replay,
  exact NTC identity and exact zero-effect identity checks pass.

Sparsity and raw-versus-projected metrics remain explicit safety diagnostics,
not a claim that nonnegativity solves count calibration. Even a passing screen
does not automatically authorize promotion, Feng training, promoter ablations,
GPU scale-up or a challenge submission. A failing screen is not repaired by
selecting a weaker shuffle, dropping difficult targets or retuning this gate.

## Execution, records and limits

Run on an approved BioHPC compute host using two CPU threads and no GPU. Never
run on Anvil/AIDA login nodes; an environment variable is not an allocation.
Six arms times eight evaluations give 48 fold/arm evaluations, including eight
zero-effect control references (40 ridge fits). Use fresh directories only.
Preserve v7/v1/v2/reliability seals and all frozen implementation files.

Hash-pin registration, protocol, feature tables, source/cache identities and
the complete runtime module roster before fitting. Save safe NPZ weights and
predictions with original cache row/group mappings, fold-local transforms,
feature/permutation receipts and JSON summaries. Verify artifacts after writing.
Record each arm with offline W&B at validation stage: eight fold events plus
one aggregate event, 54 events total. These are analytical fit/evaluation
events, not optimizer epochs. No credentials, cell IDs, matrices or arbitrary
paths enter W&B; local provenance artifacts may contain public row mappings.
No online synchronization is claimed without a separate authenticated sync.
