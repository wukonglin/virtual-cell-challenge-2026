# Expanded public-cohort nested validation v8

2026-09-10. Freeze this protocol and the new implementation before fitting.
This is public-only model fitting for diagnostic validation, not flow pretraining,
Feng post-training, promoter ablation, a count emitter or a VCC submission.
Preserve completed v2–v7 code and artifacts unchanged.

## Question and fixed inputs

Does the existing GO-target/control-context model show useful transfer when
trained and evaluated on all registered replication for the same 56 targets?
The v6 conditioning and distribution gates failed; its inner selections all
favored lambda 10. The v7 data expansion removed only the four-batch cap and
was registered before decoding newly admitted expression.

Use only the completed v7 cache SHA-256:
`292e52e7e2818e754f8c9176ff17d9b1542c07d7f1de2a43ed40a0912132f783`.
Bind its row contract, source manifest, receipt and full frozen parent lineage.
There are 1,767 groups, 22,213 treated cells, 101 source/batches and 56 targets
in K562/Jurkat. Keep all 772 protected target exclusions. No raw H5AD X is
decoded in this stage; only the authenticated prepared cache is loaded.
Do not read RPE1, H1, HepG2, Feng or challenge-2026 treated expression.

The cache uses 512 fixed output genes in float32 log1p(CP10000), with normalization
over 7,563 shared genes. Outputs are not full-gene integer UMI counts.
No new embedding model, feature source, output axis or expression preprocessing.

More groups are not independent biological replicates. Many groups reuse the
same controls; the source/batch count increased only from 97 to 101. This is
previously exposed public development data, not an untouched test set.

## Registration and leakage boundaries

Create fresh metadata-only roles and execution registrations before fitting.
Preserve the exact four outer target partitions: 42 fitting / 14 held targets,
evaluated in both source directions (eight outer evaluations per arm).
Reconstruct every group ID and direction membership from the expanded roster;
old flat IDs are not an expanded-prefix mapping.

Within each outer fitting source, preserve v6's deterministic 28 fitting / 14
tuning target split. Opposite-source treated cells and all outer-held responses
must not affect fitting, transforms, priors or regularization selection.
Every original 32-cell anchor pool is split once by the frozen v6 original
source/batch/row hash into 16 fitting-reference and 16 tuning-anchor cells.
Reuse this globally across all targets, arms and folds; extend it to all 101
pools. Context cells (32) and sham cells (8) remain disjoint from anchors and
each other. Context summaries are inference conditioning, not response/prior
training donors. Sham values are not used for fitting.

Reuse the authenticated outer GO vocabularies fitted on each outer 42-target
set, and inner GO vocabularies fitted only on each inner 28-target set. This
reuse is valid because targets and target partitions are identical; group
replication does not change a GO vocabulary. Explicitly declare reuse, bind
both existing feature receipts, and revalidate membership and parent code.
Do not substitute a vocabulary fitted on inner tuning or outer held targets.

## Fixed experiment

Run the same seven arms as v6:

- Unchanged sampled-control reference, with no fitted head or selection.
- Control-context conditioning only.
- Additive GO target plus context conditioning.
- True GO target-by-context interaction.
- Three target-feature shuffles with frozen seeds 20260921, 20260922, 20260923.

Retain v6's float64 kernel calculations, train-only transforms, two-part
occupancy/positive-amplitude responses, target-balanced fitting weights and
nonnegative control-anchored sparse output decoder. Availability masks remain
attached in all true/shuffled comparisons. No architecture or GPU-backend change.

For each learned arm and outer direction, fit all three fixed regularizations
0.1, 1.0 and 10.0 on the inner fitting roles. Select minimum inner target-pooled
MSE; exact ties select the strongest regularization. Refit once on the outer
42-target fitting source using the selected value. The control has no selection.
This totals 192 two-head fits plus eight control references, not 200 neural
training epochs. Do not select across arms or use outer scores to retune this run.

Expanded flat group IDs also enter the frozen decoder's deterministic donor
ranking. Thus old-group predictions need not match v6 even for identical heads;
absolute v6/v8 metric differences are not a clean data-only causal comparison.
All within-v8 comparisons use identical expanded groups and ranking rules.

## Metrics and unchanged progression gates

Retain target-pooled expression MSE, target-pooled occupancy MSE, batch-centroid
MSE, biased RBF MMD, sparsity and clipping/identity diagnostics. Keep frozen
equal batch/target/source macro aggregation and finite-value requirements.
Only negative MMD roundoff in [-1e-12, 0) is canonicalized, with raw values kept.

The true interaction must, in BOTH sources:

- Improve primary MSE by at least 1% versus controls and each shuffled arm.
- Strictly improve primary MSE versus context-only and additive conditioning.
- Have MMD no worse than every comparator, batch-centroid and occupancy MSE no
  worse than control/context, and zero-rate error no worse than controls.
- Preserve nonnegativity, exact NTC/zero-effect identities, zero unresolved
  activation, and occupancy/amplitude clipping fractions no greater than 1%.

Versioned execution validates native v8 metadata/provenance, then explicitly
projects only schema/policy labels into the frozen v6 report checker and screen.
Never change metric values or thresholds in that adapter. A screen is not an
artifact authentication substitute: every saved candidate must also replay.
No favorable exit status is evidence of scientific quality.

## Artifacts, audits and failure handling

Write fresh model and prediction NPZ files separately per fold/candidate, with
exact expected key sets, ordered gene axes, hashes and compressed/expanded sizes.
Limit each archive to 512 MiB compressed and expanded including headers. Never
accumulate all candidate arrays into a single arm-level NPZ. Authenticate safe
regular nonsymlink leaves and bounded ZIP/NPY headers before loading, no pickle.
Record the exact shard roster and candidate-to-report mapping; reject missing,
extra, swapped, traversing or inconsistent shards and unsupported dtypes.

Replay saved heads and every prediction/metric from authenticated cache/feature
inputs without fitting or solving a new kernel system. Verify training-only
transforms, priors and original-row maps; independently reproduce inner selection
and source/aggregate reports. Compact summary JSON remains bounded at 16 MiB.
Keep failure evidence; never overwrite a partial experiment as completed.
Publish completion only after all seven arms, tracking receipts and audits pass.

## Compute and tracking

Run on the approved BioHPC compute host, never a head/login node. Initial live
inspection found 40 physical cores, approximately 478 GiB available RAM and four
11-GiB RTX 2080 Ti GPUs; GPU 0 was active and all cards had resident allocations.
Availability is a snapshot, not exclusive ownership or a reservation.

This small float64 kernel experiment uses eight numerical CPU threads, no GPU,
a 16-GiB process peak-RSS budget, 32-GiB total output budget and four-hour elapsed
budget. Validate thread settings and active BLAS pools before training; sample
RSS/output/time between fits and after audits. These are checkpointed stop
conditions, not a kernel-enforced instantaneous memory or wall-time guarantee.
The user permits more resources, but a larger budget or GPU backend requires a
new recorded configuration rather than an unreported mid-run change.

Log actual fitting/validation events with pinned W&B SDK 0.19.11, offline mode,
group `public-expanded-regularization-v8`, separate run per arm. Retain private
allowlisted scalar journals, input/code/role hashes and failures: 33 events per
learned arm, nine for control, 207 total. Track the stage honestly as validation,
not flow pretrain/posttrain. No raw matrix, model artifact, code, environment or
console upload. Do not claim cloud synchronization; none is requested here.

No new download, GCP request, embedding API, H100 allocation or external write.

## After this experiment

Report conditioning, distribution and safety separately, including failures.
Only successful public validation can motivate a separately registered promoter
ablation. Feng post-training still lacks a compatible trainable flow parent,
and submission still lacks a validated full-gene raw-count emitter. No automatic
training/submission watcher or promotion is introduced by this experiment.
