# Public replication-expanded cache v7: frozen data-preparation protocol

2026-09-10. This stage registers and materializes additional public replication
for the existing 56-target K562/Jurkat panel. It is not model fitting, a new
scientific validation result, promoter/Feng post-training or a VCC submission.
Freeze the protocol and implementation before decoding newly admitted rows.
Keep completed v2–v6 artifacts and implementations unchanged.

## Why this data step

V6 selected lambda 10 in all 48 learned arm/directions; weaker candidates were
worse on its inner target splits. Outer quality gates still failed. A separate
metadata-only audit found that the four-batch cap, not the 64-target ceiling,
discarded replication of the same 56 targets. More replication may help estimate
noisy effects, but no performance improvement or independent-replicate claim
follows from these counts.

The proposed expansion was chosen before reading additional expression values.
Do not rank new batches or cells by observed effects, model predictions, counts
of detected genes or any expression-based quality statistic.

## Frozen lineage and changed sampling rule

The parent v2 row contract is
a94565b29d14d98588e8dbf58ad25481e1ebc5e20b209db643667d8a5344bbf1.
Its source manifest is
b7e7d0a300fe4baeaf8d8963894ef2aa6b6681c1e8cb346f7e31c2515e7a78e7.
The parent cache is
13ed1ea60e0dff4d516098e9ef5c4d0520c7847ce2bdfc733082f645a5ee7186.
The completed v6 execution contract is
cf8968e580a362b9fe552a5e0eb7757385c7f3172a24b845a25f61508541af4c.
Registration must authenticate these parents and their referenced provenance.

Remove ONLY v2's maximum of four batches per target/source:

- Keep all 56 exact common targets and all 772 protected target exclusions.
- Keep minimum eight treated cells per exact source/target/gem_group.
- Keep minimum two eligible batches per target/source.
- Keep at least 72 explicitly non-targeting controls per source/gem_group.
- Keep the 32 treated-cell cap per group.
- Keep seed 20260909, v2 target/batch/cell-identity ranking and row order.
- Keep source/batch-global disjoint pools: 32 context, 32 anchor/reference,
  eight sham controls, selected independently of targets and expressions.
- Admit every eligible batch for these targets, with no top-four truncation.
- Exclude RPE1, H1, HepG2, challenge-2026 and Feng expression from this stage.

The expected full metadata roster is:

| Quantity | K562 | Jurkat | Total |
|---|---:|---:|---:|
| Matched groups | 953 | 814 | 1,767 |
| Source/batches | 48 | 53 | 101 |
| Retained treated rows | 10,334 | 11,879 | 22,213 |
| Unique context rows | 1,536 | 1,696 | 3,232 |
| Unique anchor rows | 1,536 | 1,696 | 3,232 |
| Unique sham rows | 384 | 424 | 808 |
| Unique physical source/rows | 13,790 | 15,695 | 29,485 |

The expansion admits 18,418 additional physical rows. The per-group treated cap
removes 199 Jurkat rows; K562 has no eligible group exceeding 32 treated cells.
Do not increase that cap after seeing values. Existing 398 parent groups must
retain exact pool/treated row lists. Expanded flat indices are not an old-prefix
extension: publish explicit parent-group to expanded-group mappings keyed by
source, target and batch.

## Registration before expression

Use source obs/var metadata and X shape/encoding attributes only. Reconstruct
the full eligibility intersection, row ranking, axes, mappings and resource
estimates; compare source metadata hashes with the original authenticated
sources. Store sources, row contract, axes and audit in a fresh directory.
Registration must not decode X/data values, build GO features, fit models,
contact remote services or modify parent artifacts.

Use exact typed policy/provenance comparisons. New metadata and source readers
must reject symlink/FIFO leaves and parse bounded same-open authenticated bytes.
Do not claim universal race hardening of untouched legacy dependencies.

## Authorized, bounded extraction

Before any X value read, open both pinned raw sources on regular file handles,
check their full byte hashes/sizes, decode and revalidate complete metadata,
reconstruct every admitted row role and check unchanged file identity/stat
signatures. A successful first-source hash is not permission to read its
expression before the second source and all roles are authenticated.

Prevalidate the entire sorted unique source-row request and gene columns.
Validate dense/CSR dimensions, dtypes and selected CSR row spans before payload
slicing. Selected rows only: do not materialize a whole X matrix and filter later.
Opaque whole-file hashing is distinct from decoding expression values.

Extract at most 256 selected rows per numeric chunk. Normalize nonnegative
integer raw UMI counts using float64 row sums over the SAME ordered 7,563
uniquely measured shared genes. Require finite positive sums. Apply the frozen
v2 log1p(CP10000) arithmetic, convert to float32, then select the SAME 512 output
genes. No new variable-gene selection or source-specific output axis.

Keep a deduplicated source/original-row normalized table, then fill the flat
group arrays. Calculate context mean/population-standard-deviation in float64
from the independent 32-cell context pool and store float32, exactly as v2.
Sham controls are stored for future audits, not used for context or fitting.

Bound resources: two numerical threads on an approved BioHPC compute node,
GPUs disabled; never a login/head node. Peak-RSS ceiling eight GiB, cache and
receipt disk budget two GiB, and 512 MiB each for compressed archive and expanded
NPY members including headers. Check planned allocation before large buffers
and actual archive/member sizes before publishing. No new downloads, GCP
transfers, embedding APIs or H100 jobs.

The estimated array payload is about 301.224 MiB. This is not peak RSS;
normalization, compression and audits need temporary memory. Chunking bounds
dense normalization temporaries but does not eliminate cache storage.

## Completion evidence

Before publishing the receipt:

1. Validate exact schema, sorted axes, dtypes, finite/nonnegative values, row/group
   order, repeated original-row consistency and independent context reconstruction.
2. Authenticate the old v2 cache and verify ALL 398 retained parent groups have
   bit-identical normalized values for every pool and identical context summaries.
   Parent conservation is a regression check, not independent verification of
   expression values for the newly admitted rows.
3. Recheck source signatures and frozen registration/code/protocol bindings.
4. Write a fresh no-pickle NPZ, verify its written SHA and sizes, then publish the
   completion receipt. Preserve failed attempts; never overwrite them.
5. Load the completed cache through the versioned verifier, recheck bounded
   ZIP/NPY headers and roles, and independently repeat parent-subset conservation.

The receipt must distinguish data preparation from model training and record
source/row/chunk counts, population entries, sizes, peak RSS and exclusions.
W&B pretrain/posttrain events are not emitted for a stage that performs neither.
Any later model fitting must retain the requested W&B tracking.

## Next model stage — not authorized by this cache receipt alone

More batches are not independent biological replicates; controls are reused
within source/batch. Keep source/target/batch weighting and dependence explicit.
This expanded public cohort is already development-exposed, not an untouched test.

The frozen v6 runner cannot consume this cache: it pins 398 groups and stores
all inner/outer artifacts per arm in single archives. A new training execution
must bind the expanded rows, preserve target/source holdouts, reconstruct inner
reference splits for all 101 pools, and use sized per-fold/candidate artifacts.
Do not silently increase archive limits, loosen gates or edit v6 in place.

No promoter ablation, Feng training or submission follows automatically.
A compatible trainable flow parent and a full-gene count emitter remain absent.
