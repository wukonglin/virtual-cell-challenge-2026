# Expanded public validation v8.1: completed

Final update, 2026-09-10: training completed at 16:52:58 UTC and the managed
monitor independently verified saved fits/tracking at 17:17:50 UTC. Integrity
and safety passed; conditioning/distribution failed. Neither process remains
running. No promoter/Feng training or VCC submission was launched. See the
[completed results and next-stage boundary](PUBLIC_EXPANDED_V81_RESULTS_20260910.md).

The text below preserves the earlier running snapshot; its process/progress
statements are historical, not current status.

Snapshot: 2026-09-10 16:06:55 UTC. **Training is running, not completed.**
No final scientific gate or VCC score is available. The live monitor files below
supersede this static snapshot as subsequent arms finish.

## Current progress

On `cbsuvlaminck3.biohpc.cornell.edu`, process **2078078** is running the fixed
eight-thread CPU experiment. No head/login-node computation, GPU allocation,
new download, GCP transfer or external submission was started.

- Control reference: all eight folds completed and saved artifacts replayed;
  nine offline W&B events, zero tracking errors.
- Context-only model: all 24 inner candidates and eight outer fits completed
  and saved artifacts replayed; 33 offline W&B events, zero tracking errors.
- Additive GO/context: fitting is in progress.
- True GO interaction and three shuffled comparisons: not yet complete.

The full plan remains 192 two-head fits plus eight control references. At the
completed context-arm checkpoint, elapsed time was 684.24 seconds, process peak
RSS was 1,405,263,872 bytes (1.31 GiB), and project output was 1,019,512,242 bytes.
The registered budgets remain eight numerical CPU threads, 16 GiB peak process
RSS, 32 GiB project output and four hours checked between fits/audits. No GPU is
needed for this unchanged small float64 kernel implementation.

W&B is **offline**, under group `public-expanded-regularization-v8`. Local
scalars and journals are recorded; cloud synchronization has not occurred.
Completed run IDs: control `7ae8394b1023`, context `db4cce3e8598`.

## Logging failure fixed without changing science

The initial v8 attempt wrote the first control-reference shards, then stopped
because W&B created an external `debug-core.log` link despite `symlink=False`.
It produced no learned fit, no complete arm and zero tracking events.
The original directory and frozen code are preserved, not overwritten.

V8.1 permits only that narrowly defined SDK debug link, bound to the offline
tracking run ID, and counts the link itself without following or reading its
target. All scientific artifacts still reject symlinks. A real SDK initialization
smoke test passed before the retry. See the [logging-only recovery protocol](PUBLIC_EXPANDED_V81_PROTOCOL_20260910.md)
and [unchanged scientific protocol](PUBLIC_EXPANDED_V8_PROTOCOL_20260910.md).

There are **2,875 passing related tests across 53 modules**: 2,767 in the full
regression, 50 recovery tests, and 58 monitor tests. The recovery tests include
AST checks that fitting, validation, gates and tracking logic are unchanged
apart from the versioned entrypoint.

## Active bounded monitor

Monitor process **2093590** started at 2026-09-10T16:09:40.833651+00:00 and checks
the sealed experiment every 60 seconds. Its wait deadline is
2026-09-10T20:09:40.833651+00:00. The original v4 watcher is unchanged and does
not monitor this new candidate.

Monitor directory: `artifacts/public_flow/expanded_v81_20260910/monitor_managed_60s/`.

The first shell-detached monitor, PID 2090300, stopped after one heartbeat
without a terminal receipt. Its cause is not established; its files remain in
`monitor_60s/` and are not a live status source. The unchanged watcher was
restarted in a managed long-running execution session. Training was unaffected.
The replacement's second heartbeat was verified at 16:10:44 UTC (63.65 seconds
after initialization), with both training and monitor PIDs still alive.

- `run.json`: PID, code/contract hashes and limits.
- `state.json`: pending, recorded failure, unverified completion or verified completion.
- `heartbeat.json`: latest check, phase and whether the monitor is running.
- `verification.json`: created only after successful isolated saved-fit and W&B replay.
- `receipt.json`: terminal reason and final status.

After a valid completion marker appears, the monitor authenticates the suite
and starts only the fixed verifier, not training. This child has eight numerical
threads, a 16-GiB address-space cap, an 8-MiB log cap and a four-hour wall timeout.
Only its own verifier child can be terminated; the training process is never
signaled. Maximum wait plus verification is eight hours plus shutdown grace.

Successful artifact verification is separate from `scientific_gate_passed`.
No promoter/Feng training or submission launches automatically. The monitor
writes local status only; it sends no chat or online notifications. Do not edit
the sealed watcher or training code while either process is active.

## Future data option, not part of this run

A separate metadata-only audit found **487 additional eligible public targets**
with capacity for **57,207 additional treated cells** and no new download.
They are not admitted by the present contract. The clean proposed use is
source-specific auxiliary pretraining, excluding all 56 validation-panel genes
and 772 protected targets, without contaminating opposite-source holdouts or
inner tuning anchors. See the [audit and leakage constraints](PUBLIC_AUXILIARY_TARGET_OPTIONS_20260910.md).

Promoter progression still requires successful public validation. Feng needs a
compatible trainable flow parent, and VCC submission needs a validated full-gene
raw-count emitter. None is established by a successful process exit alone.

## Reproducibility

Root: `artifacts/public_flow/expanded_v81_20260910/`.

- Role manifest: `8ec29334bfb385e002f681f6b8129a47ed19f7df3d8ed7df31cb9185503dc7d0`
- Execution contract: `24d4581d9637cf082a028938fe425e515bf3b9a86a8cd4c6c08659419615cab8`
- Frozen model core: `80a423c50f7281c01c43df597488a3ae0ba6132dab710a92fb04af1acdd6121e`
- V8.1 runner: `eb13c228bfbcef83d33d79b3375e0ca12185744a9914494b234b21b2c6cb28aa`
- Recovery protocol: `475beedf5f0315dacd8d068a75c3cfe0787d6d15c5085db988a01fd409faf52d`
- Watcher: `d40ebe94fca006f642bd8931d24fc87d18e8a82a5d076763d316ac86163f7269`

Prior failed v8 contract:
`e4a21b250f780b9acb02a3ccab62d5152975b89930d4e6093966c6d3ecb54bb9`.
Its failed tracking receipt:
`8be1cebef0d8dcdd32156a63a446fa3bbae566173caf723c5cf776f3fe3a8052`.
