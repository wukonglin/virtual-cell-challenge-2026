# AIDA Lingshu + scDFM v1 launch record

## Outcome update: 2026-09-09 04:34 EDT

The H100 workload completed successfully. The earlier pending states below are
historical, not current resource failures:

- Embeddings **889951** and training **889957** completed on H100 node `c0003`.
- Full 360,000-cell generation **889976** completed in 30 minutes 4 seconds.
- Official validation/packaging **889977** completed in 8 minutes 36 seconds:
  360,000 × 18,533 counts, all targets/contexts verified, 2,125,165,452 nonzeros.
- Conditional submission **889983** stopped in its audit, before the upload
  command. No challenge submission was made and no entry ID was created.

The frozen public-quality safeguards rejected the model:

| Holdout | Model centroid MSE | Control centroid MSE | Model MMD | Control MMD |
| --- | ---: | ---: | ---: | ---: |
| RPE1 context | 0.3374933 | 0.0563552 | 0.2282802 | 0.0130981 |
| K562 held-out targets | 0.0610879 | 0.0603195 | 0.0167524 | 0.0015879 |

The selected checkpoint was step 1,800. Low teacher-forced velocity loss did
not establish successful control-start prediction. The gate is not being
weakened and the predictions are not being manually revised. Checkpoints,
embeddings, logs, and diagnostic receipts have been returned to BioHPC under
`artifacts/lingshu_scdfm/v1/`; the large prediction/package remain on AIDA.
The new early public-quality preflight also rejects this run on all four
model/control comparisons, before generation would be attempted.

See [the controlled v2 plan](LINGSHU_SCDFM_EFFECT_V2.md) for the next public-only
training experiment. No v2 result is implied by this outcome record.

## Historical launch record

Recorded 2026-09-09, Eastern time. This is a launch record, not a claim that
training or challenge scoring has finished. Query Slurm and inspect receipts
for subsequent changes.

Runtime root: `/home/fs01/yl4259/yl/virtual-cell-challenge-2026`.
Main BioHPC root: `/fs/cbsuvlaminck3/workdir/yz3482/virtual-cell-challenge-2026`.
Method, reproducibility details, caveats, and future experiments:
[LINGSHU_SCDFM_V1.md](LINGSHU_SCDFM_V1.md).

## Completed preparation

- CPU environment setup: job **889946**, completed.
- Public K562/RPE1 sources and released A/B/C controls staged: job **889947**, completed.
- Pinned Lingshu-7B snapshot download and verification: job **889949**, completed.
- Public-only cache: 9,074 training cells, 2,313 whole-target validation cells,
  3,915 RPE1 context-validation cells; 512 modeled genes.
- Production encoder/adapter/trainer/generator and cache SHA-256 values agree
  between BioHPC and AIDA; checked in allocated CPU job **889978**.
- Final CPU integration/audit test job **889982**: **26 passed**, zero skipped,
  exit code 0. This includes genuine official PAD training, resume, and
  controls-only generation tests, but not the still-pending full Lingshu run.

## Submitted dependency chain

| Job | Work | Resources | State at recording |
| --- | --- | --- | --- |
| 889951 | Frozen Lingshu features for 467 symbols | 1 H100, 1-hour limit | Pending: resources |
| 889957 | Public-data projection + PAD fit, 2,000 steps | 1 H100, 2-hour limit | Pending: dependency |
| 889976 | Generate all 360,000 challenge prediction cells | 1 H100, 4-hour limit | Pending: dependency |
| 889977 | Official VCC dry run and package | CPU, 2-hour limit | Pending: dependency |
| 889983 | Preupload audit, conditional upload, score-status wait | CPU, 2-hour limit | Pending: dependency |

The scheduler estimated **2026-09-09 10:47:16 EDT** for job 889951 at the last
check. This is not a reservation or guarantee. The downstream jobs use
`afterok` dependencies and cancel on invalid dependencies; they do not run
after upstream failure. Training also depends on successful integration tests;
submission additionally depends on the final 26-test job.

No challenge entry ID or score exists for this route at the time of recording.
The upload job is queued, not already submitted to the challenge.

## Automatic upload safeguards

The user authorized public-data adapter fitting while keeping challenge-treated
cells unseen, and requested a first challenge submission. The upload stage
runs only if byte identities, genuine-model/public-only provenance, and official
format receipts pass, and both public validation kinds improve over sampled
controls in centroid MSE and MMD. It also requires high-domain clipping at most
1% and every group library ratio in [0.5, 2.0]. These are predeclared empirical
safeguards, not proof of improved challenge scores. They never edit predictions.

The supplied VCC token was validated earlier and passed via hidden input to the
Slurm job environment; no token was embedded in source, scripts, or plaintext
CLI login storage. Keep the token valid until the queued upload finishes.
Private resumable-upload state is under the ignored, mode-0700 directory
`artifacts/lingshu_scdfm/v1/private_vcc_state`.

The job invokes `vcc submit` once, retaining official daily-limit and
one-in-flight checks. It does not blindly retry or create a second entry if an
upload/launch result is ambiguous. An entry ID is saved before polling for up
to one hour. Status waiting timing out does not imply scoring failed.

## Where to inspect results

Under AIDA `artifacts/lingshu_scdfm/v1/`:

- `fit/training_summary.json` and `fit/best_rollout_diagnostics.json`:
  selected checkpoint and public validation diagnostics.
- `prediction.h5ad.receipt.json`: count-generation provenance and numerical checks.
- `prep_validation.json`, `prep_package.json`, `prediction.vcc.sha256`:
  official format validation and packaged artifact.
- `audit.json`: whether conditional upload was permitted, with failure reason.
- `submission.json`: actual challenge entry ID, only after a successful upload/launch.
- `submission_status.json`: returned scoring status, not necessarily published.

All workload logs are under `logs/aida_lingshu_*` or
`logs/aida_lingshu_scdfm_*`. Use the head node only for lightweight status and
submission commands; installations, downloads, transfers, tests, model work,
hashing, packaging, and upload run inside CPU/GPU allocations.

If a gate fails, inspect its actual reason and public-data evidence before
changing the experiment. Do not weaken the gate or modify predictions merely
to obtain an upload. If an upload was interrupted, recover its existing entry
from private CLI state and resume that entry instead of creating a duplicate.
