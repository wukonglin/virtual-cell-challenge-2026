# Public auxiliary v9: candidate registration and metadata replay completed

2026-09-10. The already-local K562/Jurkat auxiliary candidate plan is now sealed
and has passed a separate full metadata reconstruction. No expression was
materialized and no new model fitting or training job was launched.
See the [frozen metadata-only protocol](PUBLIC_AUXILIARY_V9_PROTOCOL_20260910.md)
and [preceding model diagnosis](PUBLIC_EXPANDED_V81_RESULTS_20260910.md).

## Verified candidate capacity

| Candidate source | Extra targets | Target/batch groups | Treated cells | Unique cells including allowed controls |
|---|---:|---:|---:|---:|
| K562 | 374 | 3,689 | 38,928 | 41,232 |
| Jurkat | 113 | 1,386 | 18,279 | 20,823 |
| Total | 487 | 5,075 | 57,207 | 62,055 |

All 56 validation-panel genes and 772 protected targets are excluded. The
candidate plan retains only 1,616 fitting-reference controls and 3,232 context
controls across 101 source/batch pools. Tuning-anchor and sham rows are checked
against the frozen parent but excluded from candidate rows. Context remains
conditioning-only, not a response or prior fitting target.

Routing is source-specific: K562 auxiliaries belong only to K562-fit/Jurkat-held
fold clones, and Jurkat auxiliaries only to Jurkat-fit/K562-held clones. There is
no shared cross-source treated warm start. Stable semantic group/pool hashes
avoid a dependence on concatenated cache indices. The plan does not select
targets by model errors, expression responses or annotation coverage.

## Implementation and verification

New files: `scripts/prepare_public_auxiliary_v9.py` and
`tests/test_prepare_public_auxiliary_v9.py`. Frozen v2-v8.1 implementations,
artifacts and scientific protocols were not edited.

Independent review found no blocking issue. **341 focused tests passed**:
67 new candidate-registration tests plus 274 frozen-lineage/monitor regression
tests. Coverage includes dense/CSR annotation-only access, resealed tampering,
source/target/control leakage, stable identities under roster reordering,
changed code/protocol/completion, fresh outputs, symlinks and resource limits.

The real invocation guarded HDF5 Dataset `__getitem__`, `read_direct`, and
`__array__` against X/X/* access and blocked all NumPy `load` calls. Registration
and separate loader reconstruction made 24 annotation-value reads, **zero
expression-access attempts and zero NPZ-decoding attempts**. Both raw-source
metadata fingerprints and source byte sizes matched their frozen records.
This does not constitute a fresh hash of every raw-source byte; both full-file
hashes must be checked again before a future materializer reads expression.

A separate reviewer also passed an independent standard-library artifact/role
audit without importing the registrar or reading raw expression/cache values.
It checked all36 implementation hashes, protocol/completion/parent seals,
semantic IDs, axes/exclusions, all101 exact control pools and source/fold routes.
All57,207 treated source-row identities are unique and disjoint from panel-treated
and control rows. No tuning/sham rows are emitted. The plan hash was unchanged
before and after this audit.

Registration plus replay took 22.98 seconds, peaking at 862,674,944 bytes
(822.71 MiB) process RSS, with two numerical CPU threads on the approved BioHPC
compute host. There was no GPU use, head-node computation, new download, GCP
transfer, W&B initialization, external message, submission or cloud charge from
this operation. Existing v8.1 W&B records remain offline and unchanged.

## Next implementation boundary

Both `expression_materialization_authorized` and `training_authorized` remain
false. This is data planning, not completed flow pretraining. The next bounded
implementation is a separately contracted, deduplicated auxiliary cache with
full-source authentication and exact row/axis conservation, followed by new
source-specific feature/training registration. A float32 512-gene matrix for
these 62,055 unique candidates alone is estimated at 121.20 MiB; this excludes
metadata, normalization workspace and model state.

Do not pass this plan to the old repeated-pool 56-target cache consumer. Before
new fitting, fix GO/GenePT provenance and fit-only vocabularies/transforms,
warm-start architecture/objective, negative controls, selection rules and
resource budgets. Record actual fitting in W&B. The v8.1 failure does not
justify skipping transfer validation: tiny same-source tuning gains did not
generalize. Promoter/Feng training and VCC submission remain gated; no process
is monitoring this candidate for automatic promotion.

## Seals and locations

Root: `artifacts/public_flow/auxiliary_v9_20260910/`.

- Candidate file: `candidate_plan.json`
- Candidate SHA256: `bde224820c144517dbd49f8740c9d3da750f4c688b2081607ebbca25032df060`
- Completion SHA256: `a16877d20c6bf57e3081eab128b42d5de6348d053bd32a73f6f94d3e57da85c5`
- Protocol SHA256: `3bfaa25d9c3d00739000c9076befd3789fb6f3c43f047192b9ea57650d2a9672`
- Implementation SHA256: `f0fb475cc24efc79bcd397a2e80ca1c15aac9a2249b1a36eecd2854fb9c0692d`
- Test file SHA256: `0cc4588552de5656ab72838452ba5249c82e14033481c9d2610a4ed1e9f01509`

API: `register(repo, output_dir, protocol_path)` creates only fresh private JSON
files; `load_registration(path, sha, repo=...)` reauthenticates lineage and
reconstructs the complete plan from source annotations. Neither API fits a
model or authorizes future expression reads by itself.
