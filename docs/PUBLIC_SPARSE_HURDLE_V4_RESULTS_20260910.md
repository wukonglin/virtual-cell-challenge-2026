# Sparse public decoder v4 and progression monitoring

2026-09-10. **Execution and safety passed; conditioning and distribution gates
failed.** The new decoder preserves realistic baseline sparsity but does not
establish enough response-prediction benefit to advance to promoter/Feng
training or a VCC submission.

All six arms completed: 40 fixed two-head ridge fits and eight unchanged-control
references, using the same 56 targets, 398 groups, 512 genes, four target folds
and two source directions as v3. This is public-development validation, not
flow pretraining, post-training or an official competition score.
See the [frozen protocol](PUBLIC_SPARSE_HURDLE_V4_PROTOCOL_20260910.md).

## Implemented and tested

The output model separates occupancy (zero/nonzero expression) from positive
expression magnitude, while retaining the frozen v3 GO/context design and
lambda10. Target-by-context interactions were deliberately left for a separate
experiment so this comparison does not conflate output and conditioning changes.

Both treated and reference occupancy responses use the same fixed smoothing
of empirical proportions, `(32*q+0.5)/33`. This avoids the proposed
sample-size-specific smoothing artifact for all-zero/all-positive genes.
It does not eliminate every finite-sample nonlinear bias.

Magnitude priors use only unique fitting-source/fitting-target anchor and
treated cells, with repeated source/row identities deduplicated. Prediction
donors come only from permitted anchors, matched context controls or the
training-only prior; never held-treated cells. Existing positive support stays
unchanged when the requested count is unchanged. Explicit zero-effect and NTC
paths preserve exact values/support after lossless float64 conversion.

Saved artifacts passed independent replay of training priors, ridge normal
equations, both heads, deterministic donor choices, output values, original-row
mappings and pooled/batch metrics, without refitting a model. This verifies
the implementation, not scientific usefulness or learned null calibration.

## Results

Primary MSE is target-pooled: average matched-batch centroid error vectors
within each target before squaring, then macro-average targets per source.
Lower is better. All 56 targets, including missing-GO targets, remain present.

| Fixed arm | Jurkat primary MSE | K562 primary MSE |
|---|---:|---:|
| Control | 0.010447375813 | 0.007270140965 |
| Context-only | 0.010444607810 | 0.007269053575 |
| True GO | 0.010444352583 | 0.007268523346 |
| Shuffle 20260921 | 0.010444493033 | 0.007269078601 |
| Shuffle 20260922 | 0.010444768796 | 0.007269160863 |
| Shuffle 20260923 | 0.010444544072 | 0.007269020962 |

True GO improves primary MSE over controls by only **0.02894% / 0.02225%**
(Jurkat/K562), below the predeclared 1% margin. It improves over context-only
by **0.00244% / 0.00729%**, and narrowly beats all three shuffles, again well
below the required margins. These tiny differences are descriptive, not
statistically established conditioning gains.

| Distribution diagnostic | Jurkat | K562 |
|---|---:|---:|
| True-GO exact zeros | 54.4882% | 54.0504% |
| Control exact zeros | 54.4876% | 54.0494% |
| Treated exact zeros | 56.1198% | 55.0232% |
| Previous v3 true-GO exact zeros | 26.3144% | 27.4640% |
| Unchanged positive-count fraction | 99.9811% | 99.9669% |
| True-GO target-pooled occupancy MSE | 0.009566185290 | 0.008437752417 |
| Control target-pooled occupancy MSE | 0.009566657434 | 0.008438943137 |
| True-GO batch-macro MMD squared | 0.023915597163 | 0.019097619001 |
| Control batch-macro MMD squared | 0.023915418383 | 0.019093274167 |
| True-GO batch-macro centroid MSE | 0.026398535692 | 0.020593977478 |
| Control batch-macro centroid MSE | 0.026396218156 | 0.020587577772 |

The large loss of exact zeros from v3 is avoided. However, almost every
gene/group keeps its original nonzero count: fixed 32-cell occupancy rounding
erases most of the small predicted probability changes. No new positive
activations occurred in this true-GO evaluation; rare deactivations and small
magnitude changes account for its output differences. The fallback donor
branches are synthetic-tested but should not be described as empirically
validated activation behavior from this run.

Both MMD values are numerically worse than unchanged controls, failing the
frozen distribution condition. K562 occupancy MSE also fails the context-only
comparison. Batch-macro centroid MSE is slightly worse than control even though
the target-pooled primary MSE slightly improves, illustrating why both metrics
are retained. None of these tiny differences is claimed to be statistically
significant. V4's primary-MSE gains are smaller than v3's; v4 is not a universal
replacement or a demonstrated leaderboard improvement.

All final values were finite/nonnegative, no activation was unresolved, no
occupancy-head or amplitude-head clipping occurred, and exact null/zero checks
passed. Probability clipping into [0,1] is separately recorded and is not the
same diagnostic as clipping occupancy heads at +/-4.

## Progression decision

| Stage | Current readiness | Additional evidence required |
|---|---|---|
| Promoter ablation | No | Conditioning, distribution and safety gates all pass |
| Feng post-training | No | Those gates plus a compatible authenticated trainable flow parent and Feng donor/batch eligibility |
| VCC submission | No | Those gates plus a full official count emitter, provenance/QC, dry run and fresh credential/quota checks |

No checkpoint was promoted. This two-head ridge diagnostic is not the flow
pretraining checkpoint required for the intended Feng route. No promoter/Feng
expression training, new raw-data read, new download, GPU allocation, GCP
transfer, paid embedding call or competition upload occurred.

VCC currently requires raw integer-valued counts with the official 18,533 genes,
validation contexts A/B/C and the complete target/cell panel (360,000 rows for
the current 300-target panel at 400 cells per context/target). Our 512-gene
log-normalized public diagnostics are not valid submissions. Exponentiating
and rounding them is not a validated count-space emitter.
[Official VCC guide](https://vcc-cli-wiki.virtualcellchallenge.org/),
[CLI reference](https://vcc-cli-wiki.virtualcellchallenge.org/cli-reference).

The local installed VCC CLI reports 0.2.0 while the existing requirements pin
0.1.0; reconcile this reproducibly before packaging a future candidate. The
default profile did not resolve a credential during the read-only audit.
No fresh authenticated leaderboard score or quota was fetched, no token was
printed or saved by this work, and no new VCC score is claimed. The user's
conditional request to submit will be actionable after the actual candidate
and required evidence exist; it does not waive data/format/scientific checks.

## W&B and regression verification

**54 offline W&B validation events, zero tracking errors**, nine events per arm.
No online synchronization is claimed. These are fit/evaluation records, not
optimizer epochs. Run IDs:

| Arm | Local run ID |
|---|---|
| Control | `b120b2054ff7` |
| Context | `9a44a41a88c9` |
| True GO | `e230e4c7aa5c` |
| Shuffle 20260921 | `9d0eb5abd251` |
| Shuffle 20260922 | `763dbc4e96a8` |
| Shuffle 20260923 | `65d85dcea8a4` |

**1,410 related synthetic tests passed across 33 modules**. The new coverage
includes unequal-sample smoothing, deduplication, leakage/role checks, sparse
support, missing donors, finite arithmetic, rehashed artifact mutations,
exact-1% screen boundaries, tracking failures and readiness-watch safeguards.
Previous v7/v1/v2/reliability/v3 seals are retained; old failures were not
rewritten. New files and status changes are local; no Git push was performed.

The first expression-free registration attempt encountered a missing parent
directory and stopped before writing a contract or fitting. Creating that
experiment directory resolved it; the registered real run completed normally.
This was not a sandbox/GPU failure.

## Active local readiness watcher

A detached, read-only watcher was started and its live OS process and first
heartbeat verified on `cbsuvlaminck3.biohpc.cornell.edu`:

- PID: `1789427` (verify identity before signaling it).
- Started: `2026-09-10 05:50:01 UTC` / `01:50:01 EDT`.
- Stops by: `2026-09-11 05:50:01 UTC` / `01:50:01 EDT`.
- Interval: 300 seconds; duration bound: 24 hours.
- Output: `artifacts/public_flow/sparse_hurdle_v4_20260910/readiness_watch_24h/`.
- Initial state: evidence valid; promoter/Feng/submission readiness all false.

`run.json` records PID, host, deadline and fixed hashes. `heartbeat.json` updates
at each check; `state.json` and `states.jsonl` record state changes. A final
`receipt.json` is written on the time bound or graceful SIGTERM. The watcher
never trains, uploads, reads credentials or sends chat/online notifications.
It is a local status process, not an autonomous research or submission service.

The watcher authenticates this fixed candidate. A new experiment or eligibility
bundle requires a new pinned watcher; it does not discover or trust a mutable
"pass" flag. Waiting cannot turn this immutable failed experiment into a pass.
Use the recorded heartbeat and actual process state to distinguish a healthy
unchanged result from a stalled monitor. Do not kill broad SSH/session matches.

## Provenance and next bounded experiment

Artifact root: `artifacts/public_flow/sparse_hurdle_v4_20260910/`.

- Protocol SHA: `8d2b7b4922d3e1c80ddba0ed954e45394380cf74e8e3272e51e385684490626d`
- Execution contract SHA: `0fb50b88c0ba2e7d8d2db1051c9f7b1cb42aefc4c4842c77c416f1761be39544`
- Suite SHA: `faf88f82b694590527f2be63dc12c56a5b0f2cbb0d3195e9e9bc39936e62b0be`
- Completion SHA: `ea3db7c5a09ac8458e673dc6fd198de64ef2c839b8e9e8c1901ba50e2d575205`
- Model SHA: `3a70248be6423e22d6be80f77b4d98d61f207dac8c50e2f92dfe00f3c1218904`
- Watcher SHA: `02922513329a61f5c8672d6f0de439ff6debae8ad219dfc8c287d9a2cbd7c73f`

The execution contract binds the prior v3 registration and complete runtime
module roster. Per-arm weights, predictions, priors, support/donor mappings,
summaries and W&B receipts are retained under `validation/`.

Next is a separately registered target-by-context interaction experiment with
an equally capable context-only comparator, plus training-only assessment of
shrinkage and occupancy quantization. Any hyperparameter choice must use inner
public fitting data and disjoint reference roles, never the outer held-source
responses. Preserve this completed v4 result; do not retune its gains, thresholds
or comparator roster. Promoter/Feng scale-up and a scored VCC submission remain
deferred until their evidence gates actually pass.
