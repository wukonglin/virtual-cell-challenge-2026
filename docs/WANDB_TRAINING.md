# W&B tracking for pre-training, post-training, and validation

Use `scripts/run_training_with_wandb.py` for every **new** training stage.
It wraps an explicit command and leaves frozen model code, Slurm histories,
checkpoint selection and scientific gates unchanged. It does not resubmit old
failed experiments or automatically promote any model.

## Configuration and authentication

The default project is `virtual-cell-challenge-2026`; the default mode is
**offline**. No W&B API key was configured in the inspected environment.
Offline recording is real W&B recording but is **not a cloud dashboard**.
Online mode requires an explicit entity/team and `WANDB_API_KEY` in the process
environment. Configure that key securely on the compute system, not in chat,
Git, a command argument, or an `sbatch` script. No login or key is required for
offline runs. An existing `WANDB_MODE=online` alone cannot enable uploads:
pass `--mode online` or set `VCC_WANDB_MODE=online` for the new launcher.

The tracking environment pins `wandb==0.19.11` in `requirements/tracking.txt`,
matching the already installed and tested local SDK. Anvil creates a separate
`.venv-tracking` inside the allocation. It never installs the logger into the
frozen model environment. All buffers live in project scratch, not Anvil home.

## Recorded information

- Training loss/CFM/MMD, gradient norm and elapsed time when emitted.
- Explicit numeric hyperparameters and feature mode. Add the fresh training
  provenance file to capture allowlisted defaults such as learning rate.
- Validation scores and selected steps; aggregate public context/target
  centroid MSE, MMD and clipping when a fresh summary supplies them.
- Target-CV fold progress, aggregate true/shuffled/constant errors, and the
  conditioning-screen decision.
- Stage (`pretrain`, `posttrain`, or `validation`), group, unique run ID, optional
  declared parent checkpoint SHA-256, and entry-point source hash.
- Child execution exit code and logging errors, separately from model quality.

Historical trainers print step 1 and every 25 steps, plus validation events.
This wrapper captures **every emitted record**, not every optimizer step; it
does not invent missing measurements. Future trainers can emit the same scalar
JSON schema each step or call the helper in a fresh dedicated tracking process.
Learning rate must be emitted for a changing schedule; a configured base rate
is not a measured schedule. Validation and training at the same optimization
step are retained using a separate monotonic event index and custom step axis.

No raw reports, cell/gene/target tables, predictions, expression matrices,
checkpoints, environment dictionaries, command arguments, or model code are
uploaded. SDK console/code/Git/requirements/host metadata/system-stat capture is
disabled; this wrapper currently does **not** log GPU utilization. It initializes
from a fresh empty directory so an unrelated `config-defaults.yaml` cannot
enter run config. Unrecognized/nonfinite telemetry is discarded. W&B control
environment variables are isolated; child training receives W&B disabled and
does not inherit its API key. Use a fresh wrapper process, not an existing
interactive process with a previous W&B setup.

## Stage usage

From the project root on BioHPC **compute** or an allocated node:

```bash
.venv-state/bin/python scripts/run_training_with_wandb.py \
  --stage pretrain --mode offline --group public-route-v5 \
  --output-dir artifacts/tracking/pretrain-v5 \
  --summary-file artifacts/new-run/training_summary.json \
  --summary-file artifacts/new-run/training_provenance.json \
  -- .venv-lingshu-scdfm/bin/python scripts/YOUR_APPROVED_TRAINER.py YOUR_TRAINING_ARGUMENTS
```

This is a template, **not an executable model recipe**. Substitute the approved
future trainer, fresh model-output paths and its actual arguments. For
post-training use `--stage posttrain`, the same `--group`, a new tracking and
model-output directory, and `--parent-checkpoint-sha256` with the selected
pretraining checkpoint digest. This records lineage; it does not load or verify
that checkpoint, which remains the trainer's responsibility.

Every requested summary path must be absent before the child starts. Missing,
invalid or oversized (>32 MB) final summaries mark tracking incomplete. Only
numeric aggregates and typed allowed provenance fields are extracted, never
the whole report. A failed child remains failed even when W&B finalization
fails. An initialization failure stops before training rather than silently
running untracked. Later W&B logging failures retain the local scalar journal
and are reported explicitly; they do not rerun the model.

## New Anvil H100 launcher

`slurm/anvil_h100_tracked_training.sbatch` requests one H100 in `ai`, eight CPUs,
32 GB RAM and a one-hour cap under `bio250247-ai`. A real CUDA probe must pass.
Create `logs/` before submission; Slurm opens those paths before script startup.
The launcher accepts `pretrain|posttrain`, followed by the command and arguments
as separate shell arguments (no `eval`). Set the following as needed:

- `VCC_PROJECT_DIR`: actual scratch project root.
- `VCC_TRAIN_PYTHON`: existing model environment Python used for the CUDA probe.
- `VCC_WANDB_MODE`, `WANDB_PROJECT`, `WANDB_ENTITY`: tracking destination/mode.
- `VCC_WANDB_GROUP`, `VCC_WANDB_NAME`: stage grouping and optional display name.
- `VCC_TRAINING_SUMMARY`, `VCC_TRAINING_PROVENANCE`: fresh final report paths.
- `VCC_PARENT_CHECKPOINT_SHA256`: declared pretraining lineage for post-training.

No new job is submitted by this integration. The old expanded-v3 launchers and
STATE launchers remain historical and are **not retroactively W&B-enabled**.
For distributed training, run one wrapper/controller around `torchrun`, and
make the trainer emit rank-zero scalar records. Interleaved rank output cannot
be reliably attributed by a generic stdout wrapper. If wrappers are invoked on
individual global ranks, only rank zero creates a run; other ranks still train
with W&B disabled. `LOCAL_RANK=0` alone is not global rank zero.

## Offline files and deliberate sync

Each exclusive tracking directory contains `metrics.jsonl`, `tracking.json`,
and `sdk/wandb/offline-run-*` (online runs use the corresponding SDK run path).
`tracking.json` records execution state, stage and logging status; it is not a
model-quality gate. Preserve the directory after preemption. A hard kill may
prevent final receipts; an incomplete run must not be called successful.

Once the destination is configured and its account authenticated, sync only
the intended SDK offline-run directory:

```bash
wandb sync --entity YOUR_ENTITY --project virtual-cell-challenge-2026 \
  artifacts/tracking/YOUR_RUN/sdk/wandb/offline-run-EXACT_RUN_DIRECTORY
```

Inspect the run before sync. Do not use broad `--sync-all` or upload unrelated
historical directories. Cloud sync is a separate action and does not change
model validation status. If SDK recording failed but the scalar journal
survived, syncing the SDK directory alone may omit those journal-only events;
repair/replay must be explicit and labeled, not silently treated as live logs.

References: [W&B settings](https://docs.wandb.ai/models/ref/python/experiments/settings),
[environment configuration](https://docs.wandb.ai/models/track/environment-variables),
[custom axes](https://docs.wandb.ai/models/track/log/customize-logging-axes),
[distributed logging](https://docs.wandb.ai/models/track/log/distributed-training),
[offline sync](https://docs.wandb.ai/models/ref/cli/wandb-sync).

## Verified integration: 2026-09-09

123 tests and 35 launcher subtests passed, including 38 new tracking tests,
four new launcher tests, unchanged historical launcher hash checks, and the
public-only adapter regression suite. The real pinned SDK was tested offline;
synthetic secret markers in external defaults, environment, command arguments
and unrecognized telemetry were absent from the saved W&B files. Signal tests
verified that SIGTERM reaches only the wrapper's owned child process group and
preserves interrupted exit status 143.

Two retained **synthetic integration tests, not scientific training** are at:

- `artifacts/tracking/synthetic_pretrain_smoke_20260909/`: three events,
  execution exit 0, no tracking errors, offline run `e08e6dea346e`.
- `artifacts/tracking/synthetic_posttrain_smoke_20260909/`: four events,
  execution exit 0, no tracking errors, offline run `e39f2592edeb`. Its deliberately
  failed synthetic public gate is correctly recorded as false, separately
  from successful command execution.

Both receipts report `cloud_synced: false`. No scientific run was replayed,
no H100 job was submitted, and no W&B cloud upload occurred during setup.
