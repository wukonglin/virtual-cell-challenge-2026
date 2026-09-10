# Public auxiliary v9: metadata-only candidate registration

2026-09-10. This protocol registers a reproducible candidate-row plan, not
expression materialization, model architecture, pretraining, or post-training.
It follows the [completed v8.1 validation](PUBLIC_EXPANDED_V81_RESULTS_20260910.md),
which passed artifact/safety replay but failed conditioning/distribution gates.
The [prior capacity audit](PUBLIC_AUXILIARY_TARGET_OPTIONS_20260910.md) identified
the additional targets before v8.1 completed; the new plan does not alter that
experiment or relax its filters.

## Frozen lineage and public sources

Bind the completed v7 row/source lineage through the frozen v8.1 execution and
role registrations. Keep all 56 panel targets and 772 protected target exclusions
unchanged. The parent protocols, scripts, caches, roles, GO artifacts and old
results remain immutable. Authentication may hash parent cache bytes opaquely;
it must not decode them here.

Read only annotations from the already-local Replogle K562 and Nadig Jurkat
H5AD sources, using their exact acquisition paths/byte sizes and full metadata
fingerprints. X shape/encoding metadata may be inspected, but no expression
values from X or X/* may be read. Source file sizes and open-handle signatures
must stay unchanged during annotation reads. This stage does not freshly hash
the complete raw sources and must say so explicitly; future expression reads
require full-file authentication of both sources and metadata reconstruction.

No RPE1, H1, HepG2, challenge-2026 or Feng expression, new download, network
request, credential access, paid embedding call, W&B initialization or submission
belongs to this stage.

## Candidate selection and source routing

Apply the unchanged minimums: eight treated cells per target/batch, 72 exact
non-targeting controls in that batch, and two qualified batches per target/source.
Retain at most 32 treated cells per group using the unchanged v2 seeded cell-ID
ranking (20260909). There is no four-batch cap and no target selection by GO
coverage, expression response or v8.1 model error.

Remove all 772 protected targets and all 56 validation-panel targets globally.
Check that independently qualified source intersection still equals the original
56 before removing the panel. Expected candidate capacities are fixed:

| Source | Extra targets | Target/batch groups | Treated rows | Existing control pools |
|---|---:|---:|---:|---:|
| K562 | 374 | 3,689 | 38,928 | 48 |
| Jurkat | 113 | 1,386 | 18,279 | 53 |
| Total | 487 | 5,075 | 57,207 | 101 |

These cohorts are source-eligibility-specific, not biologically exclusive genes.
Original row identities include source; equal row numbers across different files
are not the same cell. Group and pool identifiers hash canonical semantic keys
(schema/namespace/source/target/batch), not concatenated cache indices. Store
groups and each source/batch control pool once, with explicit references.

Route K562 candidates only to K562-fitting/Jurkat-held fold clones; route Jurkat
candidates only to Jurkat-fitting/K562-held clones. Preserve the four original
outer 42/14 target partitions and nested 28/14 tuning partitions. No shared
cross-source treated warm start is allowed under this evaluation claim.

## Control roles

Reconstruct the globally fixed v8.1 source/batch reference splits exactly:
16 fitting-reference rows, 16 tuning-anchor rows, 32 independent context rows,
and eight sham rows. The candidate plan contains only fitting-reference16 and
context32. Verify tuning/sham identities against the parent but do not emit them
as candidate expression rows. Context rows are conditioning donors, never
response/prior targets. No target-dependent pools or original-row role overlap.

Across both sources this permits a candidate capacity of 62,055 unique physical
rows: 57,207 treated, 1,616 fitting-reference and 3,232 context rows. A future
deduplicated float32 array for the frozen 512 output genes alone would occupy
127,088,640 bytes (121.20 MiB), excluding metadata, normalization workspace,
masks and caches. This is an estimate, not an allocated or decoded array.

## Serialization, validation and resources

Use new versioned code, a fresh directory, private exclusive-create JSON files,
complete dependency hashes and protocol hash. Seal exact source acquisitions,
metadata fingerprints, target exclusions, gene axes, candidate rows, control
pools, source/fold routing, capacity and explicit false training/materialization
flags. Reauthenticate parents/code/protocol before publishing completion.
A loader must verify the plan hash and completion binding and reconstruct the
entire record from authenticated metadata, rejecting edits even if rehashed.

Use only the approved BioHPC compute host, two numerical CPU threads, no GPUs,
at most 4 GiB peak process RSS, 30 minutes at checkpoints and 16 MiB per JSON
artifact. These RSS/time checks are checkpointed, not instantaneous OS limits.
Do not run annotation processing or tests on a login/head node.

Tests must cover no X-value reads, metadata mutation, source/panel/protected
exclusions, fitting/tuning/sham/control overlap, original source identities,
stable keys, source routing, artifact/protocol/dependency mutation, exclusive
outputs, completion binding and resource checks. Production registration and a
separate metadata replay happen only after tests and independent code review.

## Explicit non-progression

All records must state `expression_materialization_authorized=false`,
`training_authorized=false`, and `architecture_registered=false`. This candidate
plan cannot be passed directly to the old 56-target cache or training consumer.
A separate implementation must specify a deduplicated cache schema and bounded
normalization, authenticate raw files, and prove row conservation before use.

Subsequent source-specific GO/GenePT feature artifacts, train-only transforms,
flow objective/architecture, negative controls, regularization/early-stopping
selection, checkpoint routing and resource budget require a new training
registration. All learned state must obey source/target/control roles, not just
the prediction head. W&B recording applies when actual fitting resumes.
Promoter/Feng/VCC submission remain gated; no background promotion is permitted.
