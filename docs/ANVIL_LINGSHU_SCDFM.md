# Anvil Lingshu + scDFM expanded-public-v3 runbook

This is a platform migration of the registered
[expanded-public-v3 experiment](LINGSHU_SCDFM_EXPANDED_V3.md), not a new model,
an AIDA checkpoint resume, or a challenge submission. The AIDA v3 outcome is
unverified because access was lost; do not claim that its queued jobs failed
or were cancelled. Preserve the AIDA jobs and receipts for reconciliation when
access returns. Earlier v1/v2 public-quality failures remain failures.

## Placement and scheduling

- Canonical BioHPC code/data root:
  `/fs/cbsuvlaminck3/workdir/yz3482/virtual-cell-challenge-2026`.
- Anvil working root:
  `/anvil/scratch/x-yzhang142/virtual-cell-challenge-2026`.
- Anvil launcher: partition `ai`, account `bio250247-ai`, QoS `ai`, one H100,
  eight CPUs, explicit 32 GB host memory, one-hour limit including setup,
  integrity tests, fitting, and comparison. The GPU probe rejects non-H100
  devices. Explicit memory avoids requesting Anvil's much larger per-GPU
  default on partially occupied nodes.
- Live association checks found only AI/GPU allocations; AI accounts cannot
  submit to `shared`, and CPU-only `ai` requests fail `QOSMinGRES`. Therefore
  all phases share one authorized H100 allocation. The setup/test/comparison
  helper files retain the `.sbatch` suffix for organization but contain no
  scheduler directives and must not be submitted separately.
- Environment and download caches stay under the scratch project, never the
  small Anvil home filesystem. Installation, unpacking, integrity hashing,
  tests, training, and comparisons run in the Slurm allocation. Staging uses
  Anvil's documented [rsync/SFTP file-transfer workflow](https://docs.rcac.purdue.edu/userguides/anvil/file_management/)
  for ordinary SSH file I/O. No unpacking, installation, hashing, tests, or
  model computation runs on the login node; other login-node activity is
  authentication, submission, and small scheduler/status reads.

The user requested priority. The verified `ai` QoS has priority zero and the
scheduler's fair-share priority weight is zero; changing between this user's
AI accounts does not supply a verified priority boost. These launchers do not
set privileged `--priority`, a negative nice value, or an invented high-priority
QoS. Report the scheduler's actual priority and pending reason; priority is not
a guarantee of immediate H100 allocation. Consult the official
[Anvil user guide](https://docs.rcac.purdue.edu/userguides/anvil/) for current
site policy rather than assuming a queue or allocation name confers priority.

## Frozen experiment and minimal staging

The existing cache has 28,579 public K562 training cells across the same 146
training targets. All parent validation arrays, 512 selected genes, official
indices, and 18,077-gene normalization axis remain fixed. Challenge-treated
cells are not read. No source expression files need to be copied for this
training run: verification is source-free and authenticates the existing cache.

Stage only these necessary artifacts, preserving their recorded identities:

| File | SHA-256 |
| --- | --- |
| `artifacts/lingshu_scdfm/public_cache_v1.npz` | `3be669b7fc0966ec49870ecc5927b606b5f2b2fc352f3d8e46d1b5409e4095b7` |
| `artifacts/lingshu_scdfm/public_cache_expanded_train_v3.npz` | `39d04694b579776e8c5fa9d73097287de2927629e1a01e19dbfcb2fc26341e77` |
| `artifacts/lingshu_scdfm/public_cache_expanded_train_v3.json` | `615b6867b72d92f7d406b4ae53cfcc3adb86b619cccc88d61bdbde255a2d5e5e` |
| `artifacts/lingshu_scdfm/v1/lingshu_embeddings.npz` | `66d9f299212472102d292cb4e94c69a39f5ed700cd407b4f0fa5ac51d438f4cb` |
| `artifacts/lingshu_scdfm/v1/lingshu_embeddings.json` | `8918387497f21c67de5cb85423a2e3a4000f6a46c14f897b418b013a4c9bf46a` |

Also stage the following code, not an unreviewed whole-workspace tarball:

- `requirements/lingshu.txt`, `requirements/lingshu-scdfm-route.txt`.
- `scripts/{lingshu_scdfm_model,train_lingshu_scdfm,lingshu_scdfm_effect_model,train_lingshu_scdfm_effect,prepare_lingshu_scdfm_data,prepare_lingshu_scdfm_expanded_train,check_lingshu_scdfm_public_quality,compare_lingshu_scdfm_effect_v2}.py`.
- `tests/{test_lingshu_scdfm_effect,test_check_lingshu_scdfm_public_quality,test_compare_lingshu_scdfm_effect_v2,test_prepare_lingshu_scdfm_expanded_train,test_anvil_lingshu_launchers}.py`.
- The four `slurm/anvil_*lingshu_scdfm_expanded_v3.sbatch` launchers and this runbook.
- Optionally `external/scdfm_lingshu_2cf6bca1/`, including genuine Git metadata:
  commit `2cf6bca1f044e74c4e1dc586892c0495880cf125`, tree
  `a04130b07020505a609158cbe31e9e65083c0d79`. If absent, setup clones it inside
  the allocation and checks out that exact revision. If present, setup verifies
  it without resetting anything. Untracked or ignored files are rejected.
- Create `logs/` before submission because Slurm opens log paths before the
  script starts. No private VCC state, tokens, passwords, unrelated datasets,
  previous checkpoints, or challenge-treated truth belong in the transfer.

The frozen Lingshu-7B embeddings and their authenticated extraction receipt are
sufficient for adapter training. Copying/downloading the 16.6 GB model again
is unnecessary. This does not substitute synthetic or new embeddings.

## Single-allocation phases

After verified staging, submit only
`slurm/anvil_h100_lingshu_scdfm_expanded_v3.sbatch`. Its sequential phases are:

1. `anvil_setup_lingshu_scdfm_expanded_v3.sbatch`: load `conda/2026.03`, create
   scratch-local Python 3.11 `.venv-lingshu-scdfm`, install the pinned route,
   verify dependency consistency, and save the installed package manifest.
2. `anvil_test_lingshu_scdfm_expanded_v3.sbatch`, only after setup succeeds:
   check literal artifact/code hashes, byte-identical validation, and integration
   tests including the real official PAD. Any skipped test blocks training.
3. The main H100 launcher's fitting phase, only after tests succeed:
   repeat immutable input checks after queueing; require exactly one visible
   H100; execute a real CUDA tensor operation; record node/device/runtime; run
   true, shuffled, and constant feature arms sequentially at 4,000 steps,
   validation every 400 steps, batch 32, seed 20260909. All model defaults,
   optimizer settings, selection rules, and public-quality gates remain the
   existing v2/v3 implementation, checked with literal hashes.
4. `anvil_compare_lingshu_scdfm_expanded_v3.sbatch`, invoked by an EXIT trap
   after fitting begins: compare all three arms even when scientific quality
   failed, preserving a nonzero prior status. This is the single-allocation
   equivalent of `afterany`, not a separately queued job. A scheduler hard kill
   may prevent the trap; missing/incomplete receipts must not count as success.

Submit from the Anvil project root, or set `VCC_PROJECT_DIR` to the actual root
and use `sbatch --chdir` consistently for the pre-opened log paths. The scripts
use the resolved `VCC_PROJECT_DIR` / `SLURM_SUBMIT_DIR`, never an AIDA path.
No job is submitted merely by reading this runbook. Actual job IDs and current
allocation availability belong in a separately timestamped launch record.

## Artifacts and quality interpretation

Anvil fits are isolated in
`artifacts/lingshu_scdfm/anvil_effect_v3_expanded/{true,shuffled,constant}/fit/`.
The parent run directory is reserved exclusively; a repeat invocation fails
instead of overwriting or resuming an existing experiment. `runtime.json`
records the actual assigned H100 and a successful CUDA computation. Setup/test
receipts are under `artifacts/lingshu_scdfm/anvil_environment_expanded_v3/`.

Each feature arm retains its training summary, provenance, checkpoints, and
public-quality preflight. The final GPU exit status reflects the true-feature
quality gate, not just successful optimizer steps. Both public holdout kinds
must strictly beat unchanged controls in centroid MSE and MMD2; high-domain
clipping must be at most 1%. These thresholds are not weakened for migration.

`comparison.json` with `status=pass` means the experiment arms are comparable,
not that their predictions passed the quality gate. Useful Lingshu-specific
signal additionally requires the true arm to beat both feature controls; these
are reused public development groups, not an independent final test. There is
no automatic prediction generation, package creation, challenge upload, or API
token use. Even a passing result needs the compatible v2-schema generator and
genuine-inference/official-format audits before a challenge submission.

## Actual launch: 2026-09-09

Anvil access was verified through the existing project-documented SSH alias
as `x-yzhang142`; no new password or verification code was required. The web
dashboard uses ACCESS username `yzhang142`, while the cluster username has the
documented `x-` prefix. No supplied password or VCC token is stored in this
repository or the new job scripts.

At the 13:18–13:23 EDT resource check, Anvil reported 21 AI nodes with 84 H100s:
five nodes were drained/draining (20 GPUs unavailable), and 55 of the other
64 GPUs were allocated. Nine unallocated GPUs were on partially occupied,
scheduler-planned nodes; this did not mean they were immediately schedulable.
`bio250247-ai` had approximately 391.1 GPU-hours remaining. The two AI accounts
both allowed only the `ai` QoS; its priority was zero and the scheduler's
fair-share priority weight was zero. No privileged priority change was made.

The source-free public caches and genuine frozen embeddings were transferred
successfully using the documented rsync file-transfer workflow. No challenge
treated truth, credentials, model checkpoints, or full source expression files
were included. Installs, checkout authentication, hashing, tests, and model work
remain inside the queued compute allocation.

Submitted at **2026-09-09 13:26:19 EDT**:

- Job **20532815**, `vcc-anvil-effect-v3-expanded`.
- Account `bio250247-ai`, partition/QoS `ai`, explicit `--nice=0`.
- One GPU, eight CPUs, 32 GB requested host memory, one-hour cap; the runtime
  must verify that the allocated device is an H100.
- At 13:26:38 EDT: **PENDING**, reason **Priority**, priority **75**, no assigned
  node or start estimate. This is a real queued Slurm job, not a completed fit.
- Work directory: `/anvil/scratch/x-yzhang142/virtual-cell-challenge-2026`.
- Logs: `logs/anvil_effect_v3_expanded_20532815.{out,err}`.

Do not submit this experiment again while that job's outcome is unresolved.
The prior unrelated Anvil STATE smoke `20378649` timed out on `h017`; its
dependent training/inference jobs were cancelled. They were inspected but not
resumed or modified by this migration. AIDA jobs 890585/890586 remain unverified
and untouched, so future reconciliation is necessary when AIDA returns.

Local validation before publication: 57 route tests passed, with three genuine
PAD integration tests skipped because the isolated local test environment lacks
their dependencies/source; 62 subtests passed. All ten Anvil launcher tests
passed separately (35 subtests). The production Anvil prerequisite explicitly
requires the real pinned PAD test and rejects every skip. No remote training
success or quality improvement is inferred from these local tests.

Submitted main launcher SHA-256:
`e432af8efdf63e609566e2580d2e1e45567ed60c8cfb26ff5d4f512fdc1e5670`.
Setup, test, and comparison helper SHA-256 values respectively:

```text
c8cad66d886bdc4bb1275576998e637db8403697ac0d852f72aee2d371dc8f10
42e35f191e7ef4b9c4d87ece7886111fe70cdc01afbbf45451c8f89bdb8490c5
fbfdeecc50e30a10660dc40b8d8357c0d48d39df9dc8fce5a8a03b1a0adf123b
```

The scoped GitHub publication branch is
`codex/anvil-lingshu-public-v3-20260909`, based on upstream main `21011ca`.
The original BioHPC worktree and its unrelated edits are preserved; datasets,
weights, environment files, credentials, and private upload state are excluded.

## Verified outcome: 2026-09-09 15:09 EDT

The earlier pending record is historical. Job **20532815** ran on
`h001.anvil.rcac.purdue.edu` from 14:03:03 for 9 minutes 49 seconds and ended
`FAILED`, exit `1:0`. This was a scientific-gate failure, not an infrastructure
failure: the CUDA tensor probe passed on an NVIDIA H100 80GB HBM3, all 38 tests
passed with zero skips, and true/shuffled/constant training each completed
4,000 steps. Final gate step `.11` returned 1; comparison step `.12` completed.

True-Lingshu public-development centroid MSE was 0.06128875 versus unchanged
controls 0.05635522 in RPE1 (+8.8%), and 0.07529116 versus 0.06031950 on held-out
K562 targets (+24.8%). Both MMD comparisons also failed. True features slightly
beat shuffled features but lost to constant features in all four metrics.
Constant features themselves failed K562 MMD. No challenge submission was made.

Runtime, comparison, per-arm provenance/diagnostics and selected checkpoints
were recovered to BioHPC under
`artifacts/lingshu_scdfm/anvil_effect_v3_expanded/`; all three checkpoint hashes
match their training summaries. Anvil queue inspection found no active user
jobs at 15:15 EDT. Do not resubmit this completed recipe unchanged.

The next bounded experiment is the separately documented
[public-only conditioning-v4 screen](LINGSHU_CONDITIONING_V4.md). It does not
modify this experiment, its quality gates, or any historical outcome.
