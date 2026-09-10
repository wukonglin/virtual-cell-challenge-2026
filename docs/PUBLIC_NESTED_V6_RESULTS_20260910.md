# Public nested regularization v6: review and execution record

2026-09-10. **Completed: execution and safety passed; conditioning and
distribution failed.** All seven arms and artifact replays completed at
2026-09-10T15:00:35.205721+00:00. No model was promoted, no promoter/Feng training
occurred, and no VCC submission or new challenge score was produced.

See the [frozen protocol](PUBLIC_NESTED_V6_PROTOCOL_20260910.md).
This is public-development validation, not flow pretraining, Feng post-training,
an untouched test, or a VCC submission.

## Review fixes

The new v6 implementation rejects nonfinite context normalization statistics,
verifies exact fold/group/target roles and typed provenance, and binds each
selected regularization value into saved fitting state. It normalizes JSON
containers for replay while retaining exact array dtype/value checks for saved
fitting and prediction fields. Huge numeric inputs fail with controlled errors.

Biased squared MMD can be slightly negative through floating-point summation.
V6 preserves raw group values and canonicalizes only values in [-1e-12, 0) to
zero before aggregation. This is not a tolerance on model comparisons. Positive
MMD deterioration in v5 is not erased or retroactively explained by this fix.

Frozen v3/v4/v5 implementations and results remain unchanged. New tests reproduce
the serialization issue and check tampered/rehashed artifacts, row-role leakage,
overflow, exact selection/ties, tracking failures and fail-closed readiness.

## Registered experiment

Reuse the authenticated 56-target K562/Jurkat cohort: 398 groups, 512 output
genes, four 42-fit/14-held target folds, eight source directions, 772 exclusions.
All challenge-treated cells and H1/RPE1/HepG2/Feng expression are excluded.

Within each outer fitting source, split its 42 permitted targets into 28 fitting
and 14 tuning targets. For each of 97 source/batches, globally partition the
32 original reference controls into disjoint 16-fit/16-tuning rows. The independent
32-cell context pool remains an input and permitted frozen inference donor,
never a response/prior fitting pool. Eight sham cells stay audit-only.

Metadata and row identities were frozen before rebuilding four inner GO
vocabularies. No targets were discarded based on annotation coverage.

| Inner fold | GO terms | Fitting GO present | Tuning GO present |
|---|---:|---:|---:|
| 0 | 268 | 25/28 | 14/14 |
| 1 | 250 | 27/28 | 14/14 |
| 2 | 310 | 26/28 | 14/14 |
| 3 | 271 | 25/28 | 14/14 |

Jurkat has 51 source/batches and K562 has 46. An independent metadata audit
recomputed target/reference rankings and verified original-row disjointness.
No expression arrays were decoded for registration or this coverage audit.

Each of six learned arms tests lambda {0.1, 1, 10} on the same inner roles.
Minimum inner target-pooled centroid MSE selects lambda; exact ties choose the
stronger value. Refit on all 42 outer fitting targets with original 32-cell
reference pools, then evaluate the unchanged outer holdout. Control has no
selection. No outer result selects lambda or changes the registered grid.

The v5 kernels and frozen v4 sparse decoder remain unchanged. All 192 two-head
fits plus eight unchanged-control references completed: 200 evaluations.
W&B recorded 33 scalar-only offline events per learned arm and nine for control,
207 total with zero errors. These are fit/validation events, not optimizer
epochs or cloud sync.

## Inner selection and outer validation

All 48 learned arm/direction selections choose lambda 10, strictly beating both
weaker candidates within each inner split; there are no selection ties. Control
has no selection. Descriptive mean inner primary losses for true GO across four
folds are below. Selection occurred per fold, not
from these averages; the folds are not independent replicates.

| Inner fitting/tuning source | Lambda 0.1 | Lambda 1 | Lambda 10 |
|---|---:|---:|---:|
| K562 (outer held Jurkat) | 0.008619879436 | 0.008204060825 | 0.008165011702 |
| Jurkat (outer held K562) | 0.011803469090 | 0.011193308904 | 0.011063551253 |

Strong training shrinkage in v5 did not imply that weaker regularization would
improve held-target prediction. At this scale, the registered inner test supports
retaining lambda 10; it does not prove that 10 is optimal outside this grid or
on a different dataset. Across all 48 comparisons, lambda 0.1 increased inner
MSE by 3.74–12.03% relative to lambda 10; lambda 1 increased it by 0.038–3.37%.
These are descriptive fold comparisons, not independent uncertainty estimates.

| True-GO outer diagnostic | Jurkat | K562 |
|---|---:|---:|
| Primary target-pooled MSE | 0.010444368442 | 0.007268511144 |
| Unchanged-control primary MSE | 0.010447375813 | 0.007270140965 |
| Relative primary improvement | 0.0287859% | 0.0224180% |
| Model batch-macro MMD squared | 0.023915631661 | 0.019097735708 |
| Control batch-macro MMD squared | 0.023915418383 | 0.019093274167 |

The gains remain far below the registered 1% margin; MMD is worse than controls.
An independent JSON audit found all 714 shared source/aggregate metric values
(34 metrics × three scopes × seven arms) exactly equal to v5. Every evaluated
inner/outer raw MMD was positive and every roundoff correction was zero. The
robustness fixes therefore did not change these outer results. These are
public-development metrics, not official challenge scores.

## Verification and resource boundaries

2,454 related synthetic tests passed across 45 modules: 2,367 in the combined
regression plus 87 readiness-checker tests. All seven real-data saved-artifact
audits passed, including all 144 inner candidate replays, selection decisions,
56 outer evaluations, original/parent row mappings and saved dual equations.
Audits performed zero ridge refits. Each arm covers 398 outer groups and 112
source/target pairs. Parent and execution source seals revalidated at completion.

| Arm | Offline W&B run ID | Events |
|---|---|---:|
| control | `89d744b8a15b` | 9 |
| context | `a4f75397b4c1` | 33 |
| additive | `441dbcab28a7` | 33 |
| true | `474015208927` | 33 |
| shuffled_20260921 | `6f383448f01d` | 33 |
| shuffled_20260922 | `6953a4df4e74` | 33 |
| shuffled_20260923 | `4df729484c8c` | 33 |

Execution uses two numerical threads on cbsuvlaminck3.biohpc.cornell.edu, with
GPUs disabled. No login/head-node computation, new cloud download, paid embedding
call or H100 allocation is required. Existing v4 monitoring remains separate and
pinned to v4; it cannot promote this new result.

## Interpretation limits and downstream gates

Inner tuning tests unseen targets within one fitting source, not unseen-source
transfer. Reused batches/control pools induce dependence. Sixteen-cell inner
anchors have different noise and occupancy quantization from 32-cell outer
anchors. This public development set has already been exposed; no independent
test or significance claim follows. Different lambdas across arms would also
prevent interpreting their contrast as a pure interaction-only ablation.

Promoter ablation requires all registered scientific/distribution/safety gates.
Feng post-training additionally needs a compatible authenticated trainable flow
parent and its own public-only training contract. Submission needs a validated
full-gene raw-count emitter and official preparation/dry-run evidence. These
512-gene continuous log-normalized ridge outputs do not supply either artifact.

The authenticated one-shot check at 2026-09-10T15:01:13.766078+00:00 verified
completed suite evidence and recomputed every scientific gate. Evidence is
valid; conditioning and distribution fail, while safety passes. All three
readiness branches (promoter ablation, Feng post-training, submission) are false.
MMD and batch-centroid MSE are worse than controls in both sources. K562 also
fails primary improvement over additive GO and occupancy noninferiority to
context-only. The fixed 1% primary margins fail in both sources.

`artifact_bytes_verified=false` in this one-shot report means it skips a second
large-array rehash after scientific failure, not that execution replay failed.
It authenticated completed replay receipts and metadata without decoding arrays,
reading credentials or using the network. No new daemon was started. The older
v4 watcher still watches only v4; its last inspected heartbeat was 14:45UTC,
108 checks, with the original 2026-09-11 05:50UTC deadline.

## Next data step: replication expansion, not a larger target-cap setting

A separate metadata-only audit matched both source metadata hashes from the v2
contract without reading expression values. The 56 targets are the FULL
intersection passing the current rules, not a top-56 truncation: target ceiling
64 is not binding. Raw target overlap is 2,055; after the same 772 exclusions,
1,424 remain. Requiring at least eight treated cells in each of two matched
`gem_group` batches, with at least 72 NTCs per batch, reduces the intersection
to 56. Lowering the NTC requirement to 64 or 32 still leaves 56.

The current cache instead limits batches per target/source to four and treated
cells per group to 32. More replication of the SAME targets is available:

| Metadata quantity | K562 | Jurkat |
|---|---:|---:|
| All eligible matched groups for these 56 targets | 953 | 814 |
| Groups retained in current cache | 207 | 191 |
| Treated cells across all eligible groups | 10,334 | 12,078 |
| Treated cells retained in current cache | 2,019 | 2,064 |
| Targets with more than four eligible batches | 41/56 | 33/56 |

The next bounded design should expand replication before relaxing per-group
cell thresholds. These are feasibility counts, not evidence that more batches
improve prediction or independent biological replicate counts. Preserve the
772 exclusions and disjoint control roles; register row identities, batch weights,
cell caps, resource limits and validation roles before response extraction.

Do not run the frozen consumers with larger arguments: v2 fixes two-to-four
batches, v3 pins its cache, and v6 verifies 398 groups. A new versioned cache/role
consumer and sized artifact layout are needed. Any later target expansion also
needs a newly registered split and training-only GO vocabulary. No expanded
expression cache, new model run or relaxed filtering was created by this audit.

## Reproducibility

Root: `artifacts/public_flow/nested_regularization_v6_20260910/`.

- `registration/role_manifest.json`: `4e5d87a1790f2e5062cb735fef66d1dcf9d8fb83e31366f872d6d2720ebd2d85`
- `execution/contract.json`: `cf8968e580a362b9fe552a5e0eb7757385c7f3172a24b845a25f61508541af4c`
- `validation/suite.json`: `5ca298bf5088d66b7150e4c6b07ed9ef04e1e2d825b43d38619bf6758e9a6cd2`
- `validation/complete.json`: `ddc6bc8d13da36fc19751afd4973f5b949ff83df0538947974f8ff1165dc74ef`
- `readiness.json`: `0476ef7695ed1ea86afab9a5dd0e412992d838075897197939778fe746080930`
