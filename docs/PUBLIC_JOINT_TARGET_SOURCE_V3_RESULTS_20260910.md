# Completed joint target/source validation v3

2026-09-10. Execution completed successfully; the predeclared quality screen
**failed in both sources**. This was 40 small fixed ridge fits and eight
unchanged-control references, not flow pretraining or Feng post-training.
No checkpoint was promoted and no challenge submission was made.

The [frozen protocol](PUBLIC_JOINT_TARGET_SOURCE_V3_PROTOCOL_20260910.md) was
registered before feature construction/expression decoding. Previously exposed
public data remain development data: these results are not an untouched-test
score, a significance test, or an estimate of leaderboard improvement.

## What changed and what was actually run

- Four deterministic target folds: 42 fitting and 14 held targets each, with
  both K562-to-Jurkat and Jurkat-to-K562 directions. Every target/source pair
  appears out of fold exactly once; every one of the 398 groups is retained.
- Fitting excludes the held targets' treated cells in both sources and all
  treated responses from the held source. The independent control anchor is
  the training response reference; a separate NTC pool supplies context.
- Four GO vocabularies fitted on the fitting targets only. The decoder applies
  saved training-only transforms to unseen annotations, not a target-ID lookup.
- Six fixed arms: control, context-only, true GO and three role-contained GO
  packet shuffles. Masks move with their feature values. No hyperparameter
  search, threshold selection, target removal or score-driven feature changes.
- Continuous log-normalized predictions remain explicitly control-anchored and
  nonnegative. These are not integer counts or a production submission emitter.

Execution was CPU-only on `cbsuvlaminck3.biohpc.cornell.edu`, with two numerical
threads and GPUs disabled. No login/head-node compute, H100 allocation, new
download, GCP transfer, paid embedding service or new raw-expression read.

## Primary result

Primary MSE averages matched-batch error vectors within each target before
squaring, then averages targets equally within each held source. Lower is
better. This is different from v2's batch-macro MSE; changing the metric is not
itself a model gain.

| Fixed arm | Jurkat target-pooled MSE | K562 target-pooled MSE |
|---|---:|---:|
| Unchanged control | 0.010447375813 | 0.007270140965 |
| Context-only | 0.010428953772 | 0.007250647195 |
| True GO + context | 0.010428475370 | 0.007250194688 |
| Shuffled GO 20260921 | 0.010428869571 | 0.007250530928 |
| Shuffled GO 20260922 | 0.010428923561 | 0.007250752366 |
| Shuffled GO 20260923 | 0.010428730898 | 0.007250763770 |

True GO improves unchanged-control MSE by **0.1809% in Jurkat and 0.2744% in
K562**, below the frozen 1% requirement. Its incremental improvement over
context-only is just **0.00459% / 0.00624%**. True GO is numerically better than
each of the three shuffles, but the relative gains are only **0.00245–0.00430%
in Jurkat and 0.00464–0.00785% in K562**, also below the 1% requirement. These
tiny differences are descriptive, not evidence of statistically established
GO generalization. Most of the small gain is already present without GO.

The true-GO arm passes the strict-context-improvement and secondary-MMD
noninferiority conditions, but fails every required 1%-margin comparison.
No comparator or threshold was changed after observing these results.

## Distribution and projection diagnosis

| Diagnostic | Jurkat | K562 |
|---|---:|---:|
| True-GO batch-macro centroid MSE | 0.026372645900 | 0.020562428642 |
| Control batch-macro centroid MSE | 0.026396218156 | 0.020587577772 |
| True-GO batch-macro MMD squared | 0.023897157285 | 0.019072695418 |
| Control batch-macro MMD squared | 0.023915418383 | 0.019093274167 |
| True-GO exact-zero fraction | 26.31% | 27.46% |
| Control exact-zero fraction | 54.49% | 54.05% |
| Observed treated exact-zero fraction | 56.12% | 55.02% |
| Pre-projection negative fraction | 26.31% | 27.46% |
| Mean absolute projection correction | 0.00029847 | 0.00035843 |
| Projection centroid-distortion RMS | 0.00058095 | 0.00067360 |
| Target-pooled shift cosine | 0.06530 | 0.06130 |

All final values are finite and nonnegative; forced NTC and zero-effect paths
preserve controls exactly. Those identity checks are constructions, not learned
null robustness. Tiny positive offsets turn many control zeros into nonzeros,
while negative offsets require clipping. Thus the nearly halved zero fraction
is still a substantial output-distribution problem despite the small MSE gain.

Raw target-pooled MSE is 0.010427679721 in Jurkat and 0.007251921565 in K562.
Projection slightly worsens this pooled metric in Jurkat and improves it in
K562. Nonnegativity therefore is neither a universal metric improvement nor a
solution to sparsity/count calibration. No post-score thresholding was applied.

This additive mean-residual model does not explicitly model target-by-context
interactions or single-cell response heterogeneity. Its weak result does not
establish that GO is useless in a richer model, nor does it establish flow-model
mode collapse. There is still no justification here for expensive flow scale-up.

## Fold-local feature coverage

| Target fold | GO terms | Fitting targets present | Held targets present |
|---|---:|---:|---:|
| 0 | 370 | 39/42 | 14/14 |
| 1 | 377 | 41/42 | 11/14 |
| 2 | 419 | 40/42 | 13/14 |
| 3 | 379 | 39/42 | 14/14 |

All targets stay in the comparisons. C9orf16, RGPD6 and SMN2 lack accepted
source features; PSTK has accepted annotations but none in fold 1's fitting
vocabulary. It is retained with missing-GO indicators. Every fold saves coverage
and nearest-fitting-target GO-overlap metadata. Shuffle receipts record donor
fixed points and unchanged feature packets. Splits are not GO-family-disjoint.
No verified GenePT or promoter embedding is used in these results.

## W&B and verification

Six offline validation runs each contain eight fit/evaluation events and one
aggregate event: **54 recorded events, zero tracking errors**. These are
analytical evaluation events, not optimizer epochs. `cloud_synced=false` in
every receipt; no online W&B synchronization is claimed.

| Arm | Local W&B run ID |
|---|---|
| Control | `32d247bba81a` |
| Context | `b7f1d1c8e6db` |
| True GO | `8f9ccd88d398` |
| Shuffle 20260921 | `d80c3e45f095` |
| Shuffle 20260922 | `9efbbdc6faa2` |
| Shuffle 20260923 | `e299cab5aa1b` |

Each run passed saved-weight/transform-to-effect replay, exact control-plus-
effect projection replay, original cache row/group mapping checks, and
fold/source/global primary and secondary metric recomputation. Control pooled
MSE also reproduces the independent preceding reliability audit. Authenticated
NPZ loading refuses unsafe/object arrays and unexpected archive members.

**1,088 related synthetic regression tests passed** across 29 modules, including
40 full-schema artifact-corruption tests. Independent opaque-byte verification
confirmed v7/v1/v2/reliability inputs, implementations, protocols and outputs
remain unchanged. The earlier failed v2 screen was not revised. New code and
reports are local workspace changes; no Git push was performed.

## Provenance and artifacts

Root: `artifacts/public_flow/joint_target_source_v3_20260910/`.

- `fold_plan.json`: `4f5030decb3981d5103840ca877277c98b515661490c47b64efb8c188554aa84`
- `role_manifest.json`: `53954afe6a04c4bf9f6ff35fef9b8d25c5c91c6da434e70cc629c5e8dbe9bf34`
- `execution/contract.json`: `5513a0e751e500d298c4b6e29f5b6205a29160f5444da08a2e4e5a6e2d1cfafb`
- `validation/suite.json`: `24757e61e87860fd081ddddc2f1023c68eee3d716495e21798916d880d06b204`
- `validation/complete.json`: `87bf82240475a2b3ee47a46b46de765bcb9b95d2bbd25569db9256581585161e`

The role manifest pins four `go/target_fold_*/` tables, source acquisition and
all old cache identities. The execution contract pins the complete runtime
module roster. `validation/runs/<arm>/` contains safe weights/predictions,
step records and summaries; `validation/tracking/<arm>/` contains scalar-only
journals, W&B offline data and completion receipts. Do not overwrite or edit
the frozen protocol, model/preparation/tracking/runner code, or these artifacts
to turn a failed screen into a pass.

## Next bounded design, not yet run

The next useful experiment should address the observed sparsity defect and
weak GO-specific effect before adding promoter features or Feng post-training:

1. Register a new public-only comparison for a two-part output decoder: model
   zero/nonzero probability separately from positive-expression magnitude,
   retaining an exact control/null path. This is a hypothesis to test, not a
   demonstrated improvement or permission to threshold these frozen outputs.
2. Test explicit target-by-control-context interactions against an equally
   capable context-only arm and all fixed shuffled-feature controls. Restrict
   model/transform selection to inner public training folds; retain outer
   source/target exclusions and report distributional as well as mean metrics.
3. Promote no parent checkpoint until conditioning benefit and distribution
   safeguards both pass. Promoter ablations, Feng post-training and H100 flow
   scale-up remain deferred; no new cloud download is needed for the next
   design work.
