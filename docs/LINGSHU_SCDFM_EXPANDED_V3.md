# Expanded-public-data Lingshu/scDFM experiment

Registered on 2026-09-09 before v3 fitting. V2 ran successfully on a real AIDA
H100 but failed public quality checks; no v2 challenge generation or submission
was made. This is a single controlled data change, not a new model architecture.

## Hypothesis and fixed controls

The initial public cache retained at most 64 K562 cells per training target
(9,074 cells across 146 targets). The already-downloaded public K562 source
contains substantially more cells. Increasing the number of training cells and
matched public control cells may reduce noise in group-mean effect supervision.
This hypothesis is not established by the previous failures; this experiment
may also fail and is not a promised score improvement.

- Expand training only to at most 512 cells per existing K562 training target,
  preserving every original treated training row and sampling additional rows
  reproducibly with seed 20260909. Expand the public control pool to at most
  512 controls per source gem group, retaining the original batch-matching rule.
- Keep exactly the same 146 training target identities. Do not admit held-out
  K562 targets. Do not load RPE1 or challenge expression during expansion.
- Preserve every parent `val_*` array, the fixed 512-gene panel, official gene
  indices, and the 18,077-gene normalization axis byte-for-byte. Do not reselect
  features using the enlarged training sample or validation data.
- Authenticate the public K562 file and public gene axis against the original
  provenance. Preserve the base cache and record its hash, source row lineage,
  frozen-array hashes, and expanded-cache hash in new receipts.
- Retain the unmodified v2 effect wrapper, official pinned scDFM PAD, genuine
  frozen Lingshu embeddings, optimizer, 4,000 steps, selection rule, and
  true/shuffled/constant feature comparison. The only changed fit input is the
  public training cache. A training-only split-half diagnostic may describe
  effect-label noise but must not choose a model or tune quality thresholds.

## Execution and interpretation

Work directory: `/home/fs01/yl4259/yl/virtual-cell-challenge-2026` on AIDA.
Canonical code and a preserved copy of results remain on BioHPC.

New cache: `artifacts/lingshu_scdfm/public_cache_expanded_train_v3.npz` with an
adjacent JSON receipt. New fits:
`artifacts/lingshu_scdfm/effect_v3_expanded/{true,shuffled,constant}/fit/`.

Preparation may execute on the BioHPC compute host. AIDA transfer, integration
tests, and cache verification execute in Slurm CPU allocations. Training uses
one explicitly requested H100 in `full`, eight CPUs, 32 GB host RAM, and a
one-hour cap. All three arms run sequentially; the existing small model does
not need multiple H100s. No model workloads, downloads, or preprocessing run
on the AIDA login/head node.

On 2026-09-09 at 11:46 EDT, scheduler metadata listed 20 H100s across
`c0001`–`c0005`, with all 20 allocated. A pending new job is queueing for a real
resource; it is not evidence of a sandbox failure. The local sandbox launcher
has a separate permission failure; approved scoped execution and pinned-key
SSH have successfully reached AIDA and its completed H100 runs.

The original strict-improvement gates remain: both public holdout kinds must
improve centroid MSE and MMD2 relative to unchanged controls, and high-domain
clipping must be at most 1%. Comparison receipt success is not quality success.
There is no automatic challenge generation or upload in this job chain. A
passing primary arm would still require a compatible v2-schema generator,
genuine-inference provenance audit, and official format validation before upload.

These reused public development groups are not an independent test set. The
comparison must report both feature controls, and a useful overall prediction
does not by itself demonstrate benefit from Lingshu. Challenge-treated cells
remain unseen throughout this experiment.

## Prepared data and verification

BioHPC preparation completed on 2026-09-09. Training expanded from **9,074 to
28,579 cells**, preserving all original treated rows and exactly 146 targets.
All ten frozen arrays (seven validation arrays and three gene axes) are
byte-identical. Validation remains 2,313 K562 cells and 3,915 RPE1 cells.
No training rows needed a context-level control fallback.

The training-only split-half mean-effect discrepancy decreased from 0.063881
to 0.030413 (about 52.4%). This is descriptive evidence of more stable sampled
effect labels, not model performance, independent validation, or a challenge
score. It does not change the registered training recipe or gates.

- Parent cache SHA-256: `3be669b7fc0966ec49870ecc5927b606b5f2b2fc352f3d8e46d1b5409e4095b7`.
- Expanded cache SHA-256: `39d04694b579776e8c5fa9d73097287de2927629e1a01e19dbfcb2fc26341e77`;
  26,677,740 bytes.
- Preparation code SHA-256: `46ccc08d63944da1a000ebcad291ce6e32d77f3ad697c5cc0a8266eb1d40d348`.
- Original normalization/source helper SHA-256:
  `fe062caa4f148023e493df52cffd3d6c4b0dc1a42131e62a4f1e81364d35fbb0`.

The independent review found no leakage or unintended model change; 24 scoped
tests plus 50 subtests passed on BioHPC. The AIDA CPU prerequisite additionally
tests the genuine installed scDFM PAD, verifies the staged cache, and checks
that all v2 model/trainer code hashes and the quality-gate hash are unchanged.
The H100 launcher pins the literal parent/expanded cache hashes and repeats
the source-free cache verification when its allocation starts.

## AIDA launch record

Submitted 2026-09-09 at 11:50:42 EDT. Latest status check: 11:51 EDT.

| Job | Task | Actual status at check |
| --- | --- | --- |
| 890584 | CPU integration and staged-cache verification | Completed on `c0014`, exit 0, 9 seconds; 28 tests passed, none skipped; all integrity checks passed |
| 890585 | Three sequential feature-arm fits, one H100 | Pending: `Resources`; CPU dependency satisfied; no start estimate available |
| 890586 | Public quality/feature comparison | Pending: dependency on termination of 890585 |

The GPU job requests `gres/gpu:h100=1`, eight CPUs, 32 GB RAM, account
`allaccess`, partition `full`, QoS `normal`, one-hour limit, and the exact
requested AIDA work directory. No GPU is allocated to it yet. A queued job
is not a completed model or a score improvement. The job script checks the
allocated H100/CUDA environment, cache identity, and frozen validation contract
before fitting. Scientific preflight failure produces a nonzero final job exit
while retaining successful fit receipts and the separate comparison job.

Logs: `logs/aida_effect_v3_test_890584.{out,err}`,
`logs/aida_effect_v3_expanded_890585.{out,err}`, and
`logs/aida_effect_v3_comparison_890586.{out,err}`.
Read `artifacts/lingshu_scdfm/effect_v3_expanded/comparison.json` and each arm's
`public_quality_preflight.json` after completion; comparison `status=pass`
alone does not mean any quality gate passed. No challenge generation or upload
is queued by this chain.

## Continuation checkpoint: 2026-09-09 12:58 EDT

The previous authenticated SSH control connection expired. Renewing the exact
authorized AIDA account with the previously supplied password was rejected;
the pinned ED25519 host key still matches. No further password attempts were
made. A fallback check for other SSH sessions/agent identities was blocked by
the safety review and was not bypassed. Renewed user authentication is needed
before reading the current AIDA job state.

Therefore the last verified states remain those recorded above at 11:51 EDT;
do not assume jobs 890585/890586 are still pending or have completed. No jobs
were cancelled or duplicated during this continuation, and no challenge
submission was made. Once access is restored, inspect `squeue`, `sacct`, and
the existing receipts before deciding whether any new job is needed.

Local follow-up is restricted to a training-only conditioning diagnostic on
the already-expanded public K562 cache. It does not change the queued model,
read validation expression, or replace the required public-quality audit.

The local diagnostic subsequently completed with authenticated expanded-cache
and genuine-embedding hashes. `scripts/diagnose_lingshu_training_signal.py`
reads only `train_x`, `train_control`, `train_targets`, and `train_contexts`;
its result is `artifacts/lingshu_scdfm/public_training_signal_v3.json`.
The six focused tests and the combined 26-test/50-subtest suite passed.

Using fixed five-fold outer / three-fold inner target CV, fold-training-only
normalization, and six fixed ridge penalties, mean effect MSE was:

| Predictor | Exploratory target-CV MSE |
| --- | ---: |
| Zero effect | 0.01332761 |
| Fold-training global mean | 0.01301215 |
| Genuine Lingshu, linear ridge | 0.01300832 |
| Genuine Lingshu, RBF kernel ridge | 0.01299943 |
| Shuffled Lingshu, linear ridge | 0.01301854 |
| Shuffled Lingshu, RBF kernel ridge | 0.01301384 |

The genuine-feature improvements over global mean were only approximately
0.03–0.10%; exploratory paired-target bootstrap intervals included zero, as
did true-versus-shuffled differences. Weak regularization nearly interpolated
training targets but worsened held-out-target error. Useful Lingshu-specific
signal is not established by this probe. This is not scDFM, a cell-level MMD
evaluation, an independent final test, or a challenge-score result. The fixed
gene panel was selected using original public training cells, and matched
controls may recur across target groups, limiting independence.

Decision: do not replace the requested model with this baseline, loosen gates,
or launch additional flexible fits before examining jobs 890585/890586. If the
expanded-data model also fails, stronger biological conditioning/public-data
coverage should demonstrate target-level generalization before more GPU use.
The local diagnostic does not alter either queued job or its training inputs.
