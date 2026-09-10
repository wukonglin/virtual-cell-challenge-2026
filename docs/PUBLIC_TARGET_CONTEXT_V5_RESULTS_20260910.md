# Public GO × control-context v5: completed diagnostic

2026-09-10. **Execution and safety passed; conditioning and distribution failed.**
Completed at 2026-09-10T06:11:02.894051+00:00. No checkpoint was promoted, no Feng/promoter
training occurred, and no VCC submission or new competition score was produced.
This result does not establish that GO interactions improve the route.

Seven fixed arms completed on the BioHPC compute host: 48 two-head kernel fits
plus8 unchanged-control references, across four target folds and two source
directions. The authenticated cohort remains56 K562/Jurkat targets,398 groups,
512 genes and772 exclusions. No new raw-expression reads, downloads, GCP
transfers, GPU allocations or paid embedding calls occurred.
See the [frozen protocol](PUBLIC_TARGET_CONTEXT_V5_PROTOCOL_20260910.md).

## What changed

V4's sparse control-anchored decoder, response smoothing, source/target/batch
weights and lambda10 stayed fixed. V5 implements exact GO-by-context product
features via kernel ridge, without materializing the expanded feature matrix.

With normalized context c and normalized GO/mask block z, C=c c' and T=z z':

- Nonlinear context comparator: 1+C+C*C.
- Additive GO comparator: 1+C+C*C+T.
- True GO and three role-contained packet shuffles: 1+C+C*C+T+T*C.
- Control: exact zero effect.

Products are elementwise kernel products. The additive comparison isolates
the incremental interaction. All learned arms share the nonlinear context basis,
but their effective capacity is not matched. V4 lacks the new C*C block, so
cross-version differences cannot be attributed only to T*C.

GO vocabulary, transforms, positive priors and ridge fits remain training-only;
held-target treated cells in both sources and all held-source responses are
excluded from fitting. Missing GO stays in the evaluation roster. No GenePT or
protein/DNA embedding model was acquired here. Because each fit sees only one
source, the interaction learns within-source variation and extrapolates across
sources; it is not evidence of learning cross-cell-line interactions from
multiple training cell lines.

## Fixed public-development results

Primary MSE is target-pooled: equally average matched-batch centroid error vectors
within each target, then square and macro-average targets per source. Lower is
better. These are previously exposed public-development results, not untouched
test results or official leaderboard metrics.

| Arm | Jurkat primary MSE | K562 primary MSE |
|---|---:|---:|
| control | 0.010447375813 | 0.007270140965 |
| context | 0.010444658260 | 0.007269055968 |
| additive | 0.010444372290 | 0.007268487466 |
| true | 0.010444368442 | 0.007268511144 |
| shuffled_20260921 | 0.010444539137 | 0.007269095568 |
| shuffled_20260922 | 0.010444811756 | 0.007269211654 |
| shuffled_20260923 | 0.010444611000 | 0.007268983110 |

True-GO improvements versus controls are **0.0287859% / 0.0224180%**
(Jurkat/K562), far below the fixed1% threshold. It narrowly beats each shuffle
but none by the required1% margin. Versus additive GO, Jurkat improves only
**0.00003684%**, while K562 is **0.00032575% worse**. Thus the interaction does
not consistently improve on the additive comparator.

The fitted feature relation is not supported as a useful improvement at the
registered scale. These minute differences are descriptive, not statistically
established gains, and the validation gates were not loosened after observing them.

| Diagnostic | Jurkat | K562 |
|---|---:|---:|
| True exact zeros | 54.48821% | 54.05064% |
| Control exact zeros | 54.48756% | 54.04938% |
| Treated exact zeros | 56.11979% | 55.02321% |
| Unchanged positive-count fraction | 99.97936% | 99.95989% |
| True target-pooled occupancy MSE | 0.009566127749 | 0.008437632610 |
| Control target-pooled occupancy MSE | 0.009566657434 | 0.008438943137 |
| True batch-macro MMD squared | 0.023915631661 | 0.019097735708 |
| Control batch-macro MMD squared | 0.023915418383 | 0.019093274167 |
| True batch-macro centroid MSE | 0.026398604988 | 0.020594121504 |
| Control batch-macro centroid MSE | 0.026396218156 | 0.020587577772 |

Both MMD and batch-macro centroid MSE are worse than unchanged controls.
K562 also fails occupancy noninferiority to context-only and MMD
noninferiority to additive GO. All safety gates pass: finite/nonnegative
predictions, exact NTC/zero identities, zero unresolved activations, and
no occupancy/amplitude HEAD clipping. Probability clipping into[0,1] is a
separate reported diagnostic, not head clipping.

No positive activation occurred in the true arm; rare deactivations and small
magnitude changes produced the differences. Roughly99.98% /99.96% of gene/group
positive counts remain unchanged. Exact zeros remain around54.49% /54.05%.

## What the new training diagnostics establish

Ranges below are over the four true-GO fits for each HELD-source direction.
The training observations actually belong to the opposite source; the labels
must not be read as training on the held source.

| Training-only diagnostic | Held Jurkat / fit K562 | Held K562 / fit Jurkat |
|---|---:|---:|
| effective_df | 0.27045–0.27880 | 0.31631–0.32363 |
| kernel_trace | 2.80368–2.88730 | 3.26848–3.34256 |
| occupancy_response_rms | 1.11578–1.13445 | 1.11373–1.14445 |
| occupancy_fitted_rms | 0.02067–0.02202 | 0.02164–0.02444 |
| amplitude_response_rms | 0.17998–0.18559 | 0.19862–0.20176 |
| amplitude_fitted_rms | 0.00187–0.00241 | 0.00260–0.00305 |

Effective df is the per-response linear kernel smoother's df, not the sparse
decoder's df or the number of biological effects. Its range0.270–0.324,
together with occupancy fitted RMS about0.021–0.024 versus response RMS
about1.12–1.14, demonstrates strong regularization. Only about2% of occupancy
response RMS is retained before the sparse decoder. Almost all training positive-count
rounding is also unchanged.

This is a useful diagnosis, **not proof that lower regularization generalizes
better**: training responses contain substantial noise, and the held source
requires extrapolation. No lambda, kernel weight, gain, rounding rule or output
candidate was selected from the outer results.

## Verification and W&B

All seven saved-artifact audits passed. They reconstruct training-only feature
transforms and positive priors, verify the dual ridge equation without a solve,
replay saved heads and deterministic sparse donor/output arrays, and reproduce
original-row mappings, training diagnostics and pooled/batch metrics.
Each audit covers398 out-of-fold groups and112 source/target pairs, with
zero ridge refits. Completed artifact replay is execution evidence, not a
scientific success claim.

**1,787 related synthetic tests passed across38 modules**:
1,721 in the combined regression plus66 kernel tests. New coverage includes
expanded-feature/primal equivalence, source/target isolation, all-seven-arm
no-refit replay, altered/rehashed artifacts, exact gates, tracking failures and
fail-closed readiness evidence. Prior v4 contract/source seals were revalidated.
No frozen v3/v4 code or results were rewritten.

**63 offline W&B events, zero errors**, nine per arm. These record fits and
validation diagnostics, not optimizer epochs. No cloud synchronization is claimed.

| Arm | Offline run ID |
|---|---|
| control | `c4bad297eb3c` |
| context | `9d559be0498e` |
| additive | `1307c9c06033` |
| true | `e7be94c83f33` |
| shuffled_20260921 | `c7c72abb5c13` |
| shuffled_20260922 | `fe1deccaa6be` |
| shuffled_20260923 | `7fcb69628d53` |

## Readiness and monitoring

The new one-shot report at
`artifacts/public_flow/target_context_v5_20260910/readiness.json`
authenticates the actual completed suite, all summaries/tracking receipts and
critical artifact-audit fields, then independently recomputes its decision.
A separate agent also verified the actual schema and evidence read-only.

- V5 evidence valid: yes.
- Scientific gate: failed; safety passed.
- Promoter ablation ready: no.
- Feng post-training ready: no; also lacks an authenticated compatible trainable
  public-flow parent and registered Feng donor/batch training contract.
- VCC submission ready: no; also lacks a validated full-gene raw-count emitter,
  official preparation/dry-run evidence and current submission checks.

These are ridge diagnostic weights, not a resumable flow model. The512-gene
log-normalized predictions are not a competition package. No credentials or
network were used by the new readiness checker. On a failed scientific gate,
it does not rehash the large array artifacts; those were authenticated/replayed
by the completed execution. Its `artifact_bytes_verified=false` records that
distinction, not a failed execution audit.

The existing v4 watcher (PID1789427, fixed suite faf88f82…) remained unchanged;
its06:05UTC heartbeat showed4 checks and1 state transition. It is a24-hour
local read-only watcher until2026-09-11 05:50UTC, not an automatic trainer or
submission service. It watches v4 only and does not discover this new v5 result.
The v5 readiness file is a one-shot snapshot, not a new background monitor.

## Next bounded design — proposed, not registered or run

Test regularization through INNER public-data validation without sharing noisy
reference cells between fitting and tuning. Do not simply lower lambda because
an outer result looks weak.

1. Within each outer fit source and its42 fitting targets, pre-register an inner
   target split (for example28 fitting /14 tuning). All inner-tuning treated
   cells stay excluded from fitting.
2. Assign each unique source/batch's32 reference controls globally by original
   row identity to16 inner-fit references and16 inner-tuning anchors. Reuse the
   assignment across targets; never split separately per target. Exclude the32
   context cells from response/prior fitting and keep8 sham cells audit-only.
   Context cells supply input summaries; retaining their frozen inference-donor
   use must also be explicit in the new registration so this remains a lambda-only
   assessment rather than an unacknowledged decoder change.
3. Refit inner GO vocabulary, all transforms and positive priors on the28 inner
   fitting targets and permitted pools. Inner shuffles must be role-contained.
4. Pre-register a small lambda grid, equal budgets and selection rule for every
   learned arm, with stronger regularization on ties. A possible grid is
   {0.1,1,10}; it is NOT currently frozen or executed. No outer-response selection.
5. Refit the selected setting on all42 outer-fitting targets, then evaluate
   the unchanged outer held-source/held-target roles.

Sixteen-cell tuning has different noise and occupancy quantization from the
32-cell outer decoder; report that mismatch, never duplicate cells to fake32.
Shared batches still induce dependence, so no independent-replicate confidence
claims follow. This tunes unseen targets within one fitting source, not
cross-source transfer. The NTC hard bypass is not learned null calibration.

A future promoter ablation or Feng run needs its own authenticated readiness
and experiment plan. No scale-up, submission or new watcher is scheduled by
this report.

## Reproducibility

Root: `artifacts/public_flow/target_context_v5_20260910/`.

- `execution/contract.json`: `26e03fa1563f783e4c2fedfc74b38ae100f1e8943665da90474a97c0ef0a757f`
- `readiness.json`: `8bf8ca00f62bf77a1dffdaf16927c37cccf813204cdd3ac36263b8e1ee3f2091`
- `validation/complete.json`: `434f2a6dfb06034c694a0056250ffce8d808b6a499c49bfc66c2ec83bfad5fd0`
- `validation/suite.json`: `78ac99a6da2c9e4a15c2dafb7ad77141d4440a35bde8769cb0b06ad9a65ff96a`

Additional frozen hashes:

- Protocol: `20ff1c075af1d6c959f9b13e68a5ee95e21f956095c089026f11383a970f8b81`
- Kernel: `8b5443f7da4c58ee4ca6f8d7548c0f904babe754845b0fbe0ad057e949691de5`
- Core: `eb3dde1123a0b9651a72ffa219bfc24d4a9e50dcd6f69e950de8762d33e82ee3`
- Runner: `fe815a6987a591854754b27914baa36a167ac2c9bef60472e94c80890730b854`
- Readiness checker: `eae0657f479572b9bb03eff6a4a6ff86957ce9347d2784d116d11002e9e5ba2c`

All changes are local. Unrelated existing worktree changes are preserved; no
Git commit/push, external project mutation or competition upload was performed.
