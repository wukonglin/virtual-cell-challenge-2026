# Anvil STATE H100 Runbook

This runbook records the portable Anvil path for the VCC 2026 STATE
reconstruction. It does not authorize or create a challenge submission.

## Location and scheduler

- Local source/data root:
  `/fs/cbsuvlaminck3/workdir/yz3482/virtual-cell-challenge-2026`
- Anvil working root:
  `/anvil/scratch/x-yzhang142/virtual-cell-challenge-2026`
- Slurm account: `bio250247-ai`
- Slurm partition: `ai`
- Runtime: one H100 per task

Anvil's AI nodes provide four H100 GPUs per node. The production launcher uses
four independent one-GPU array tasks rather than DDP. The pinned STATE trainer
sets `use_distributed_sampler=False`; treating four devices as one distributed
fit would therefore need a separately validated data-sharding change.

## Environment

Keep the environment and caches in scratch because Anvil home storage is small.
The model runtime and the evaluator are intentionally separate.

```bash
module load conda/2026.03
export CONDA_PKGS_DIRS=/anvil/scratch/x-yzhang142/virtual-cell-challenge-2026/.cache/conda_pkgs
export PIP_CACHE_DIR=/anvil/scratch/x-yzhang142/virtual-cell-challenge-2026/.cache/pip

conda create -y \
  -p /anvil/scratch/x-yzhang142/virtual-cell-challenge-2026/.venv-state \
  python=3.11

PROJECT_DIR=/anvil/scratch/x-yzhang142/virtual-cell-challenge-2026
"$PROJECT_DIR/.venv-state/bin/python" -m pip install \
  -r "$PROJECT_DIR/requirements/model.txt"
"$PROJECT_DIR/.venv-state/bin/python" -m pip install \
  --constraint "$PROJECT_DIR/requirements/model.txt" \
  -r "$PROJECT_DIR/requirements/state-training.txt"
"$PROJECT_DIR/.venv-state/bin/python" -m pip install --no-deps -e \
  "$PROJECT_DIR/external/state_runtime_9bbfe78a"
```

`arc-state` declares `cell-eval`, but current `cell-eval` requires
`anndata>=0.12.10` while the registered model environment pins
`anndata==0.11.4`. The training environment therefore omits `cell-eval`;
run scoring from the dedicated scoring environment.

## Authenticated support data

The archive is staged at
`dataset/state_support/competition_support_set.zip`. Its SHA-256 is:

```text
f1d5fd56a2eee240e5585fac52bb75b0b830a7be6a5f6de1bf773bff4f507f7c
```

Extract with `-j` so the archive's top-level directory is stripped and the
repository-relative paths are created directly, then build the metadata-only
HDF5 wrappers:

```bash
cd /anvil/scratch/x-yzhang142/virtual-cell-challenge-2026
mkdir -p dataset/state_support/extracted
unzip -q -j dataset/state_support/competition_support_set.zip \
  -d dataset/state_support/extracted
.venv-state/bin/python scripts/prepare_state_support_links.py \
  --source-dir dataset/state_support/extracted \
  --output-dir dataset/state_support/state_compatible
```

The wrapper files contain external links. They do not rewrite or duplicate the
authenticated matrices.

The active Anvil `ESM2_pert_features.pt` is a regular 410,886,729-byte file
with SHA-256
`a210e1cc7901513999b2bca3836ba9e2f203cd008be4e9a9d6412a2267de9748`.
The former symlink is retained beside it as
`ESM2_pert_features.pt.symlink_pre_hardened_20260903`; launchers reject the
symlink and authenticate the materialized file.

## Launch sequence

Submit from the Anvil repository root so Slurm can open the tracked `logs/`
directory. The smoke, production, and inference launchers are all the hardened
v1 route:

```bash
cd /anvil/scratch/x-yzhang142/virtual-cell-challenge-2026
smoke_job="$(sbatch --parsable --kill-on-invalid-dep=yes \
  slurm/anvil_h100_train_state_smoke_v1.sbatch)"
train_job="$(sbatch --parsable \
  --kill-on-invalid-dep=yes \
  --dependency="afterok:$smoke_job" \
  slurm/anvil_h100_train_state_sm_20k_array_v1.sbatch)"
infer_job="$(sbatch --parsable \
  --kill-on-invalid-dep=yes \
  --dependency="aftercorr:$train_job" \
  --export="ALL,VCC_TRAINING_ARRAY_JOB_ID=$train_job" \
  slurm/anvil_h100_generate_state_hepg2_reconstruction_array_v1.sbatch)"
printf 'smoke=%s training=%s inference=%s\n' \
  "$smoke_job" "$train_job" "$infer_job"
```

The smoke uses seed 42 for 20 steps. The dependent array registers seeds
`42`, `20260901`, `20260902`, and `20260903`, with at most four
concurrent one-H100 tasks. `aftercorr` is required for inference: inference
task N may run only after training task N succeeds, and the explicit exported
training job ID must match that dependency exactly. Every training task:

- validates all six HDF5 wrappers and their external links;
- requires exactly one visible H100;
- records Slurm, GPU, runtime, source, launcher hash, and seed metadata;
- refuses to overwrite or resume an existing run directory;
- disables Weights & Biases;
- sets `training.devices=1` explicitly; and
- runs `scripts/select_state_checkpoint.py` before it can exit successfully.

### Queue snapshot (2026-09-03)

- `20378649`: hardened v1 smoke, pending for `Priority`; the scheduler currently
  reports no start estimate (an earlier transient projection was
  2026-09-05 04:45 EDT).
- `20378650_[0-3]`: four one-H100 training tasks, held by
  `afterok:20378649`.
- `20378651_[0-3]`: four one-H100 inference tasks, held by task-correlated
  `aftercorr:20378650`; each task also has
  `VCC_TRAINING_ARRAY_JOB_ID=20378650`.

These jobs are already queued; do not submit the example sequence again.
All three launchers have requeue disabled because their run directories are
immutable. Invalid dependencies are killed rather than left pending forever.
At the requested wall-time limits, the complete chain is bounded by 56.25
H100-hours; `bio250247-ai` had 395.1 H100-hours remaining immediately before
submission.

The submitted launcher SHA-256 identities are:

```text
93e610f8a4d2d927d4badfd7a77d08cf7f032556bbc3109df86b65758fc4ec11  smoke v1
7bcea6d18b55b1b55624be297638828c333837f595e4397156747f16c19af5d6  training v1
e8066ae8aa071819f157aee48b4ba20d18203493b3a3140632f599010d6da918  inference v1
```

Earlier jobs `20377299`, `20377394`, `20378227`, and `20378270` were canceled
at zero elapsed time after provenance flaws were found. They are not part of
this lineage and produced no accepted candidate.

Monitor without running work on the login node:

```bash
squeue -u x-yzhang142
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,AllocTRES
```

Do not trust STATE's automatic `best.ckpt` at a validation boundary. Each
v1 array task archives all ten validation boundaries as regular
`stepNNNNN.ckpt` files, authenticates the embedded archive-callback state, and
then creates `checkpoints/selected.ckpt` and
`checkpoints/selected_checkpoint.json` for the minimum recorded validation
loss.

The local 40-step RTX integration proof produced `step00020.ckpt` at
`val_loss=20.491031646728516` and `step00040.ckpt` at
`val_loss=15.510403633117676`. The selector authenticated both archives and
selected step 40. In the same run, STATE's automatic `best.ckpt` metadata was
still tied to the earlier loss, which is why inference accepts only the v1
archive-and-selector receipt.

## Pull and materialize locally

Anvil produces truth-blind reconstruction candidates only. Never transfer
`dataset/public_v51/*sealed_truth*` (or any treated-truth equivalent) to
Anvil. Pull each completed candidate back and score it on the local project,
where sealed truth already resides.

For one seed (repeat for `42`, `20260901`, `20260902`, and `20260903`):

```bash
LOCAL_PROJECT=/fs/cbsuvlaminck3/workdir/yz3482/virtual-cell-challenge-2026
REMOTE_PROJECT=/anvil/scratch/x-yzhang142/virtual-cell-challenge-2026
# Uses the `Host anvil` entry and dedicated key in ~/.ssh/config.
ANVIL_LOGIN=anvil
SEED=42
TAG="state_sm_anvil_reconstructed_hepg2_seed${SEED}_v1"
REMOTE_RUN="$REMOTE_PROJECT/artifacts/anvil_state_reconstruction/candidates/hepg2/$TAG"
EXPORT_ROOT="$LOCAL_PROJECT/artifacts/anvil_candidate_exports/20378651"
SOURCE_RUN="$EXPORT_ROOT/$TAG"
ANCHOR_FILE="$EXPORT_ROOT/$TAG.anvil_candidate_completion.remote.sha256"
REMOTE_SELECTION="$REMOTE_PROJECT/artifacts/state_runs/state_sm_anvil_20k_seed${SEED}_v1/checkpoints/selected_checkpoint.json"

mkdir -p "$SOURCE_RUN/transport"

# Obtain the trust anchor independently, before downloading the bundle.
ssh "$ANVIL_LOGIN" sha256sum "$REMOTE_RUN/anvil_candidate_completion.json" \
  > "$ANCHOR_FILE"
ANCHOR="$(awk '{print $1}' "$ANCHOR_FILE")"

# Pull only the truth-blind candidate artifacts; omit model scratch shards.
# `-a` is required because the completion receipt binds nanosecond mtimes.
rsync -a --prune-empty-dirs \
  --include='/anvil_candidate_completion.json' \
  --include='/anvil_inference_runtime.json' \
  --include='/spec.json' \
  --include='/generation.json' \
  --include='/generation_verified.json' \
  --include='/prediction.h5ad' \
  --include='/transport/' \
  --include='/transport/split_manifest.json' \
  --include='/transport/hepg2_state_residual_top100_v1.json' \
  --exclude='*' \
  "$ANVIL_LOGIN:$REMOTE_RUN/" "$SOURCE_RUN/"

# The final completion receipt also binds the selected-checkpoint receipt,
# which lives in the matching training run rather than the candidate folder.
rsync -a "$ANVIL_LOGIN:$REMOTE_SELECTION" \
  "$SOURCE_RUN/transport/selected_checkpoint.json"

"$LOCAL_PROJECT/.venv-state/bin/python" \
  "$LOCAL_PROJECT/scripts/materialize_public_candidate_transport.py" materialize \
  --project-dir "$LOCAL_PROJECT" \
  --source-run-dir "$SOURCE_RUN" \
  --source-completion-expected-sha256 "$ANCHOR" \
  --destination-public-root \
    "$LOCAL_PROJECT/artifacts/anvil_state_reconstruction" \
  --expected-context HepG2 \
  --expected-output-tag "$TAG"

```

Do not delete, move, modify, or re-sync `SOURCE_RUN` after materialization.
The destination uses hard links for byte preservation, and the scoring gate
re-authenticates both the source staging tree and its independently obtained
completion SHA-256 trust anchor. The completion receipt authenticates the
inference runtime, checkpoint selection, spec, generation report, prediction,
and generation verification as one atomic six-artifact set. Keep the anchor
file as part of the audit trail. Materialize all four registered seeds before
opening any public treated truth or running any score.

## Build the registered ensemble, then score five arms

Once all four sources are materialized, create the one pre-registered uniform
ensemble while the treated truth is still unopened. The plan has file SHA-256
`4278bd9032373fe1cf705c78b602e0087fa2d494e2990bb50b3f919e78303296`
and semantic SHA-256
`f0d4d7ce123e9fea00193c3db8fc649dc1f9ab7773a47671747693421ceb95ae`.

```bash
cd /fs/cbsuvlaminck3/workdir/yz3482/virtual-cell-challenge-2026
PUBLIC_ROOT="$PWD/artifacts/anvil_state_reconstruction"
.venv-state/bin/python scripts/ensemble_state_seed_candidates.py build \
  --project-dir "$PWD" \
  --public-root "$PUBLIC_ROOT" \
  --plan configs/state/anvil_seed_ensemble_v1.json
```

Only after that succeeds, score exactly the four individual seeds and the
derived ensemble:

```bash
for SEED in 42 20260901 20260902 20260903; do
  TAG="state_sm_anvil_reconstructed_hepg2_seed${SEED}_v1"
  VCC_PROJECT_DIR="$PWD" \
  VCC_PUBLIC_V6_ROOT="$PUBLIC_ROOT" \
  VCC_SCORE_PYTHON="$PWD/.venv-score/bin/python" \
  VCC_CONTRACT_PYTHON="$PWD/.venv-state/bin/python" \
  VCC_CONTEXT=HepG2 \
  VCC_OUTPUT_TAG="$TAG" \
  bash slurm/cpu_score_public_candidate_v6.sbatch
done

VCC_PROJECT_DIR="$PWD" \
  bash slurm/cpu_score_public_state_seed_ensemble_v1.sbatch
```

The dedicated ensemble scorer recomputes the exact four-source mean before it
can access truth, then authenticates the canonical sealed-truth hash. Individual
scores are seed-variance diagnostics only: never cherry-pick the best seed.
Compare the ensemble only with fixed seed 42. Promote it only if all/direct MSE,
NMAE, and reach do not regress; all/direct PDS or Jaccard strictly improves;
and every evaluable held-target metric does not regress. Otherwise retain seed
42.

## Regression status

Before submission, the isolated STATE/model test split passed 441 tests with
2 skips, and the evaluator-only split passed 27 tests. The focused
training-through-transport suite passed 84 tests and 75 subtests; the dedicated
ensemble/transport suite passed 28 tests. All five Slurm launchers passed Bash
syntax checks and the worktree diff passed whitespace validation.

This workflow ends with local score evidence under
`artifacts/anvil_state_reconstruction/scoring/hepg2/$TAG` and
`artifacts/anvil_state_reconstruction/ensemble_scoring/hepg2/`. It does not
package or submit a leaderboard entry. A leaderboard submission requires
separate, explicit approval after the five registered arms have been compared.
