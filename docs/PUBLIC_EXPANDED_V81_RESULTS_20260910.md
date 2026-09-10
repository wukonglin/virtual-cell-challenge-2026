# Expanded public validation v8.1: completed, not promoted

2026-09-10. Execution completed at **16:52:58 UTC**. The bounded monitor completed
an independent saved-artifact and tracking replay at **17:17:50 UTC**.
Integrity and safety passed; conditioning and distribution gates failed.
There is no new VCC score, promoted checkpoint, flow pretraining, Feng
post-training, or promoter ablation from this run.

## Completed experiment

The unchanged registered kernel/two-part sparse decoder was evaluated on 1,767
public K562/Jurkat groups, 56 target genes, and 512 output genes. Four target
folds and two source directions preserve the 772 protected exclusions.
The run completed 192 two-head fits and eight control references across seven
arms. Inner tuning used only the outer fitting source; outer evaluation held
out both targets and source. See the [scientific protocol](PUBLIC_EXPANDED_V8_PROTOCOL_20260910.md)
and [logging-only recovery](PUBLIC_EXPANDED_V81_PROTOCOL_20260910.md).

Primary loss is target-pooled expression MSE (lower is better). These are
public-development proxy metrics, not official challenge metrics or scores.

| Arm | Held Jurkat MSE | Held K562 MSE |
|---|---:|---:|
| Matched control | 0.008215721754 | 0.004871439097 |
| Context only | 0.008211674833 | 0.004870703342 |
| Additive GO/context | 0.008273361372 | 0.004911959247 |
| True GO/context interaction | 0.008273204238 | 0.004911820218 |
| Shuffled GO, seed 20260921 | 0.008274292324 | 0.004870664824 |
| Shuffled GO, seed 20260922 | 0.008211741447 | 0.004870744709 |
| Shuffled GO, seed 20260923 | 0.008211622046 | 0.004870688058 |

True GO is **0.699664% worse on Jurkat and 0.828936% worse on K562** than
controls. It is slightly better than additive GO, but worse than context-only
and most shuffled comparisons. A successful process exit is not evidence of
useful conditioning.

True-GO batch MMD squared is 0.024262552482 / 0.019056615873 (Jurkat/K562),
versus controls 0.024232195177 / 0.019022530477. Batch-centroid and occupancy
MSE are also worse than control/context. Aggregate zero-rate error improves
over controls, but this alone does not pass the distribution gate.
Predictions are nonnegative, have no unresolved activations, and pass the
registered head-clipping and exact-identity checks. Final occupancy-probability
clipping is a distinct diagnostic, about 2.63% / 2.51%, not zero.

## Diagnosis and limits

Of 48 learned-arm regularization selections, 43 choose lambda10, five choose
lambda1 and none choose lambda0.1. True and additive choose lambda1 in the same
two directions. Their tiny inner gains did not transfer to the opposite source:

| True-GO direction | Inner lambda1 improvement over lambda10 | Outer worsening versus control |
|---|---:|---:|
| Target fold 2, K562 to Jurkat | 0.01625% | 2.88691% |
| Target fold 3, Jurkat to K562 | 0.07487% | 3.27591% |

These two folds account for essentially the aggregate worsening. This suggests
fragile same-source model selection under cross-source transfer, not proof that
an uncomputed lambda10 outer fit would rescue the result. No unselected outer
fit was added and the selection rule was not retroactively changed.

Remaining true fits are strongly shrunk: fitted amplitude RMS is about
0.93-1.38% of training response RMS, and occupancy RMS 1.79-2.07%. In lambda1
fits these increase to 6.50-7.03% / 10.42-10.45%. True predicted expression-shift
RMS is 0.00510 / 0.00485 versus observed 0.16070 / 0.14208, with directional
cosine 0.0064 / -0.0122. Effects remain small and poorly aligned. Variance ratios
1.023 / 1.050 do not support calling this strong distributional mode collapse.

More batches did not establish useful transfer for this design. This is not a
clean data-volume causal comparison against v6: the expanded cohort enters the
frozen decoder's group-index donor ranking, and selected regularization changes.
The sources/control pools and development targets have been reused; folds are
not independent replications or an untouched final test.

## Verification, tracking, and resources

The managed watcher reached `verified_complete` with `scientific_gate_passed=false`.
It replayed all 192 saved fits, eight control references, and 207 tracking events
without refitting, new raw-expression reads, or W&B initialization. The earlier
failed v8 attempt and stopped detached monitor remain preserved. Training and
the managed v8.1 monitor are now finished, not background-running.

Seven offline W&B runs completed with zero tracking errors: nine control events
and 33 events per learned arm. These are fit/validation events, not optimizer
epochs. No cloud synchronization occurred.

Execution used eight CPU threads on the approved BioHPC compute node for
3,517.8 seconds (58.6 minutes), peaking at 1,522,733,056 bytes (1.42 GiB) process
RSS, with 5,495,872,379 bytes (5.12 GiB) recorded output. There was no GPU use,
head-node computation, new download, GCP transfer, or external submission.
Prior implementation verification comprised 2,875 passing related tests.

## Next-stage boundary

Do not promote this candidate, submit it, or use it as a demonstrated useful
Feng/flow parent. The next bounded preparation step is a separately versioned,
metadata-only candidate-row plan for already-local auxiliary public targets:
374 K562 targets and 113 Jurkat targets, outside all 56 panel targets and 772
protected exclusions. See the [capacity and leakage audit](PUBLIC_AUXILIARY_TARGET_OPTIONS_20260910.md).

Auxiliary fitting must be source-specific. Never pool both treated sources into
shared pretrained weights and then claim an unseen-source test. Use only the
globally fixed fitting-reference controls for auxiliary responses/priors;
tuning/sham controls are not auxiliary training rows. A metadata candidate plan
does not authorize expression reads or training: separate cache/training
contracts, feature provenance, resource bounds and leakage tests remain
necessary. Full-gene raw-count generation is still unavailable.

## Evidence

Root: `artifacts/public_flow/expanded_v81_20260910/`.

- Execution contract: `24d4581d9637cf082a028938fe425e515bf3b9a86a8cd4c6c08659419615cab8`
- Suite: `6beafb1a92e6c232d07be1c368827321417b0d08f780ddfabc8610784dde985f`
- Completion: `a1fd3dc49be4076f9c31bbca60a7b61721e2ec9891bdd63079513137cf7b5ec2`
- Frozen watcher: `d40ebe94fca006f642bd8931d24fc87d18e8a82a5d076763d316ac86163f7269`
- Final monitor files: `monitor_managed_60s/{state,verification,receipt,heartbeat}.json`
- Scalar receipts: `execution/validation/tracking/<arm>/tracking.json`

Suite/completion hashes were rechecked against the monitor state at 18:12 UTC.
Frozen models, protocols and completed artifacts are unchanged by this report.
