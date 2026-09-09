# Lingshu features and a public-data scDFM adapter

Outcome update, 2026-09-09: v1 training, full prediction, and official packaging
completed, but the public-quality audit blocked upload. No v1 challenge entry
was created. See [actual outcomes](LINGSHU_SCDFM_V1_STATUS.md) and the
[completed v2 comparison](LINGSHU_SCDFM_EFFECT_V2.md), which also failed its
quality gate, and the [expanded-public-data v3 launch](LINGSHU_SCDFM_EXPANDED_V3.md).
Recipes below describe
the implemented v1 workflow, not a claim that it achieved a passing score.

This route trains a compact perturbation-response model on public K562 data,
using frozen Lingshu language features to condition the official scDFM PAD
backbone. It then predicts the challenge A/B/C contexts from their released
non-targeting controls. The public training cache has been prepared and
verified. GPU training, candidate generation, official validation, and a
published submission are separate outcomes; their completion must be established
from their receipts and scheduler records. This document does not assert that
those outcomes have occurred.

## Working locations and compute boundary

| Location | Role |
| --- | --- |
| `/fs/cbsuvlaminck3/workdir/yz3482/virtual-cell-challenge-2026` | Main BioHPC code, source datasets, prepared cache, and returned results |
| `/home/fs01/yl4259/yl/virtual-cell-challenge-2026` | Requested AIDA working directory: staged code/data, environment, weights, checkpoints, and predictions |
| AIDA `.venv-lingshu-scdfm/` | Isolated Python 3.11 model environment |
| AIDA `models/Lingshu-7B/` | Pinned official Lingshu snapshot |
| AIDA `external/scdfm_lingshu_2cf6bca1/` | Pinned, unmodified official scDFM source |

The BioHPC path may resolve locally to `/NFS4/workdir/yz3482/...`; receipts
record the resolved path. An AIDA copy is a separate staged copy, not a shared
mount or an automatic mirror. Preserve immutable artifacts and record transfer
hashes when bringing results back.

Use the AIDA head node only for submission and lightweight status commands.
Downloads, installation, extraction, testing, preprocessing, embedding
extraction, training, generation, and packaging belong inside Slurm CPU/GPU
allocations. The provided AIDA launchers use account `allaccess`, QoS `normal`,
and one H100 via partition `full` and `--gpus=h100:1`. CPU setup/download
launchers use partition `regular`. No BeeGFS path is required by this route.

## Model and zero-shot definition

The frozen encoder is
[`lingshu-medical-mllm/Lingshu-7B`](https://huggingface.co/lingshu-medical-mllm/Lingshu-7B),
revision `b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9`. Each human gene symbol is
placed in the same short CRISPRi description. The extractor averages the final
hidden states over non-padding text tokens in FP32 and L2-normalizes the result,
producing a 3,584-dimensional vector. It performs text-only feature extraction,
not biological response prediction or language-model fine-tuning. The snapshot
files, prompt, tokenizer settings, and resulting feature matrix are recorded in
the extraction receipt. See the [Lingshu project](https://alibaba-damo-academy.github.io/lingshu/).

`scripts/lingshu_scdfm_model.py` imports the unmodified PAD implementation from
the [official scDFM repository](https://github.com/AI4Science-WestlakeU/scDFM),
commit `2cf6bca1f044e74c4e1dc586892c0495880cf125`, tree
`a04130b07020505a609158cbe31e9e65083c0d79`. A learned projection maps frozen
Lingshu features into the PAD conditioning space. The default adapter has
512 modeled genes, hidden dimension 128, four attention heads, two layers,
dropout 0.1, differential-perceiver fusion, and no fitted gene-correlation graph.
The PAD and projection are trained from scratch; the released Norman checkpoint
is not used. This is a project-specific adaptation of the official backbone,
not a reproduction claim for the paper's complete experimental setup.

Here **zero-shot** means no measured challenge perturbation responses enter
fitting or selection, and no response fitting occurs on A/B/C. Public-source
supervision is still required to learn the Lingshu-to-response mapping. This is
not training-free inference. Eleven of the 300 challenge target identities
occur in the K562 training split; six others occur in public K562 target
validation. Thus not every challenge target identity is unseen during fitting
or model selection. Encoding released target names is a covariate input, not a
perturbation-response label.

## Prepared public cache

The source files are the released STATE support matrices
`dataset/state_support/extracted/k562_gwps.h5` and `rpe1.h5`. Despite the GWPS
filename, this K562 support subset contains 183 treated target identities,
not a complete genome-wide raw-count training atlas.

| Split | Context | Target identities | Selected treated cells |
| --- | --- | ---: | ---: |
| Training | K562 | 146 | 9,074 |
| Whole-target validation | K562 | 37 | 2,313 |
| Entire-context validation | RPE1 | 68 | 3,915 |

Seed `20260909` selects a deterministic whole-target holdout of
`ceil(0.2 × 183) = 37` K562 targets. These target identities are excluded from
all training rows. RPE1 is entirely excluded from training. Each target uses
at most 64 cells. Controls are sampled within `gem_group`, with at most 64
candidate controls per batch. All current selected rows have matching-batch
controls; no context-level fallback was needed. These are sampled, unpaired
controls, not experimentally matched individual cell pairs. Shared control
conditioning between K562 train and target-validation groups is allowed.

The 512 features are chosen by variance over sampled K562 **training treated
rows only**, then restored to official gene order. Validation expression does
not determine the feature ranking. The candidate gene and normalization axes
use public/challenge gene metadata, not challenge outcomes.

Both source files contain log1p-transformed continuous abundances, not raw
integer counts. Their identical source axis contains 18,080 genes. We exclude
`HSPA14-1`, `TBCE-1`, and `TMSB15B-1`, which are absent from the official axis,
leaving a shared 18,077-gene normalization axis:

```text
z_ig = log1p(10000 × expm1(X_ig) / sum_{h in shared_18077} expm1(X_ih))
```

Normalization precedes selection of the 512 model features. Challenge controls
use the same 18,077-gene denominator on raw counts. The metadata field
`normalization.space=log1p_library_normalized_full_axis` is further qualified
by `full_axis_scope=shared_public_challenge_gene_axis`; it does not mean all
18,533 challenge genes or recovered full-cell raw UMI counts. The cache carries
`normalization_gene_names` so inference can enforce this exact correspondence.

Prepared artifacts under `artifacts/lingshu_scdfm/`:

| File | Contents | SHA-256 |
| --- | --- | --- |
| `public_cache_v1.npz` | 12,013,592 bytes; paired train/validation arrays, gene axes, row identities, metadata | `3be669b7fc0966ec49870ecc5927b606b5f2b2fc352f3d8e46d1b5409e4095b7` |
| `public_cache_v1.json` | Source/split/normalization receipt | `72b2d13f8aeac938f491efa5684340bcf8fa681ab84be19d60b317bba7ca4594` |
| `embedding_genes_v1.csv` | Header `gene_name`; 467 unique source/challenge symbols plus the control label | `1ac3883b8aa3b6a5bc549bf218492f7c654dbe09f770b451baf4d1bb0f1228d9` |

Source SHA-256 identities are
`b16f446d6aa8ca6f2bfecb65bb4a716f7cfa7dabfb5e876a7fb349ba5d0aa38b`
for K562 and
`51ebd598c3f6f171e4c0668fa3539e943aec6b77514aa1061773e4297696dff9`
for RPE1. `metadata_json` is a Unicode scalar; all arrays load with
`allow_pickle=False`. The normalization, holdout, schema, and validation-blind
feature-selection tests pass in `tests/test_prepare_lingshu_scdfm_data.py`.

## Fitting and count generation

Training samples one target group at a time, starts from sampled control
expression plus Gaussian noise, and fits a conditional flow-matching velocity
loss plus a weighted MMD endpoint loss. Defaults are 2,000 steps, batch size 16,
AdamW learning rate `1e-4`, MMD weight 0.5, noise standard deviation 0.1, and BF16
on H100. Validation runs every 200 steps, evaluating a fixed deterministic set
of up to 32 target groups for each validation kind. The two kinds receive equal
weight. The earliest checkpoint attaining the minimum held-out velocity MSE is
selected; step archives, `best.pt`, `last.pt`, and receipts are saved.

Velocity validation is a training diagnostic, not a challenge score. After
checkpoint selection is frozen, the trainer evaluates the selected model with
20-step Euler rollouts on the same fixed public validation groups. It compares
modeled-panel centroid MSE and MMD with sampled controls, reports domain-clamp
fractions separately for whole-target and whole-context validation, and writes
`best_rollout_diagnostics.json`. These endpoint diagnostics cannot change the
selected checkpoint. The training summary marks public diagnostic evaluation
as performed and official challenge metrics as unevaluated. Because these
groups also informed checkpoint selection and contain normalized abundances,
the diagnostics do not establish independent raw-count challenge performance.

The generator samples 400 released controls without replacement for each
context/target, integrates the learned velocity with 20 Euler steps by default,
and predicts the selected 512 genes. A fixed decoder clips to the mathematical
log-CP10k domain, inverse-normalizes using each source control's shared-axis
library, and stochastically rounds to integer counts. The other 18,021 genes
retain the sampled control's counts exactly, including the 456 genes absent
from the support axis. There is no hard-coded target knockdown, hand-assigned
response direction, or target-specific manual adjustment. New expression can
appear in source-zero entries of the 512 modeled genes.

This decoder does not preserve the original complete library size exactly;
the generation receipt records input/output library ratios and clamp counts.
Substantial clamping or library inflation is a failure signal to investigate,
not proof of a strong perturbation effect. Output is streamed to sparse H5AD.

## Command recipes

These are recipes for a fresh run, not instructions to duplicate an existing
job chain. Inspect current Slurm jobs and output receipts first. Preparation
refuses to overwrite its three outputs; training requires a fresh run directory
unless `--resume` is explicitly given.

The cache was prepared on the BioHPC compute server with:

```bash
cd /fs/cbsuvlaminck3/workdir/yz3482/virtual-cell-challenge-2026
.venv-state/bin/python scripts/prepare_lingshu_scdfm_data.py \
  --gene-count 512 --max-cells-per-target 64 --seed 20260909 \
  --output-npz artifacts/lingshu_scdfm/public_cache_v1.npz \
  --output-json artifacts/lingshu_scdfm/public_cache_v1.json \
  --output-genes-csv artifacts/lingshu_scdfm/embedding_genes_v1.csv
```

For AIDA setup/download/extraction, the existing Slurm entry points are
`slurm/aida_setup_lingshu_scdfm_v1.sbatch`,
`slurm/aida_download_lingshu_v1.sbatch`, and
`slurm/aida_h100_lingshu_embeddings_v1.sbatch`, in that dependency order.
The CPU test launcher is `slurm/aida_test_lingshu_scdfm_v1.sbatch`.
After its checks and embedding extraction succeed, the fit and generation
launchers are `slurm/aida_h100_train_lingshu_scdfm_v1.sbatch` and
`slurm/aida_h100_generate_lingshu_scdfm_v1.sbatch`. They use one H100 each,
with 2-hour/64-GB and 4-hour/96-GB limits respectively. Chain them with Slurm
`afterok` dependencies so an upstream failure prevents downstream execution.
The CPU launcher `slurm/aida_package_lingshu_scdfm_v1.sbatch` follows successful
generation, runs the official dry run and packaging, and writes
`artifacts/lingshu_scdfm/v1/prediction.vcc`. It does not upload or submit an
entry. The final stage, `slurm/aida_submit_lingshu_scdfm_v1.sbatch`, performs the
registered preupload audit and one guarded submission after packaging succeeds.
This dependent stage can continue without an open SSH connection; queued status
alone does not mean an entry has been uploaded or scored.
The extraction launcher runs the following **inside its H100 allocation**:

```bash
cd /home/fs01/yl4259/yl/virtual-cell-challenge-2026
srun .venv-lingshu-scdfm/bin/python scripts/extract_lingshu_gene_embeddings.py \
  --model-dir models/Lingshu-7B \
  --genes-csv artifacts/lingshu_scdfm/embedding_genes_v1.csv \
  --output-npz artifacts/lingshu_scdfm/v1/lingshu_embeddings.npz \
  --output-json artifacts/lingshu_scdfm/v1/lingshu_embeddings.json \
  --batch-size 8 --require-h100
```

After authenticated embeddings exist, the model commands below likewise belong
inside an H100 Slurm allocation. The run/output paths are explicit examples.
Set `PYTHONDONTWRITEBYTECODE=1` and `PYTHONNOUSERSITE=1` in the job environment.

```bash
srun .venv-lingshu-scdfm/bin/python scripts/train_lingshu_scdfm.py \
  --cache artifacts/lingshu_scdfm/public_cache_v1.npz \
  --embeddings artifacts/lingshu_scdfm/v1/lingshu_embeddings.npz \
  --embedding-receipt artifacts/lingshu_scdfm/v1/lingshu_embeddings.json \
  --source external/scdfm_lingshu_2cf6bca1 \
  --output-dir artifacts/lingshu_scdfm/v1/fit \
  --steps 2000 --validate-every 200 --batch-size 16 --seed 20260909

srun .venv-lingshu-scdfm/bin/python scripts/generate_lingshu_scdfm.py \
  --checkpoint artifacts/lingshu_scdfm/v1/fit/best.pt \
  --training-summary artifacts/lingshu_scdfm/v1/fit/training_summary.json \
  --embeddings artifacts/lingshu_scdfm/v1/lingshu_embeddings.npz \
  --source external/scdfm_lingshu_2cf6bca1 \
  --controls A=dataset/controls/context_A.h5ad \
  --controls B=dataset/controls/context_B.h5ad \
  --controls C=dataset/controls/context_C.h5ad \
  --gene-axis dataset/controls/gene_names.csv \
  --targets dataset/controls/pert_counts.csv \
  --output artifacts/lingshu_scdfm/v1/prediction.h5ad \
  --cells-per-target 400 --batch-size 64 --ode-steps 20 --seed 20260909
```

Use the isolated VCC CLI environment on a BioHPC compute server or a Slurm CPU
allocation for packaging. After returning the prediction and its receipts to
the main BioHPC directory:

```bash
cd /fs/cbsuvlaminck3/workdir/yz3482/virtual-cell-challenge-2026
.venv-vcc/bin/vcc prep artifacts/lingshu_scdfm/v1/prediction.h5ad \
  -g dataset/controls/gene_names.csv --perts dataset/controls/pert_counts.csv \
  --dry-run --json
.venv-vcc/bin/vcc prep artifacts/lingshu_scdfm/v1/prediction.h5ad \
  -g dataset/controls/gene_names.csv --perts dataset/controls/pert_counts.csv \
  -o artifacts/lingshu_scdfm/v1/prediction.vcc --json
```

The [official format](https://vcc-cli-wiki.virtualcellchallenge.org/) is one
360,000-by-18,533 raw-count file covering A/B/C, with exactly 400 cells for each
of 300 targets in each context. It must contain no non-targeting rows, no
negative/fractional/nonfinite values, no cell exceeding 1,000,000 counts, and
at most 4,750,000,000 stored matrix entries. The official CSVs have headers.
Passing these checks establishes format correctness, not model quality.

The user has authorized one first submission for this route. The queued workflow
uses `scripts/audit_lingshu_scdfm_submission.py --run-dir
artifacts/lingshu_scdfm/v1` before upload. It hashes and cross-checks the selected
checkpoint, training/embedding/generation receipts, prediction, package, and
official prep results; rejects synthetic or partial-test outputs; and writes
`audit.json` with a pass/fail status. The preregistered empirical safeguards
require strictly better centroid MSE and MMD than sampled controls in **both**
public validation kinds, high-domain clamping at most 1%, and all 900 group
library-size ratios in `[0.5, 2.0]`. Low-domain clamping is reported without a
rejection threshold. These safeguards do not prove challenge-score improvement
and do not alter any predictions.

The submission launcher stops on an audit failure. On a pass it invokes
`vcc submit` exactly once with default quota and in-flight checks enabled,
saves `submission.json` with the entry ID, and then waits for status into
`submission_status.json`. It has no new-entry retry loop. An interrupted upload
requires resuming the recorded entry, not repeating the chain. The authorized
token is supplied through the transient Slurm job environment, excluded from
the audit process, and never written in the launcher or documentation.
Submission has a twice-per-day UTC quota and a one-in-flight limit.
Predictions must come from models; public experimental
responses may train models but cannot be inserted as submitted cells or manually
used to revise them. See the [rules](https://virtualcellchallenge.org/rules) and
[data-use FAQ](https://virtualcellchallenge.org/faq).

## Next fitting and post-training experiments

1. **Extend endpoint validation before increasing scale.** Review the existing
   selected-checkpoint rollout diagnostics, then freeze an outer
   target/context test split independent of checkpoint selection. Generate
   complete trajectories and compare modeled-gene pseudobulk effects, effect
   correlation, magnitude, and control/target discrimination with control-only
   and target-agnostic baselines. Current log-normalized support inputs cannot
   establish exact raw-count DE behavior; add an independently permitted raw
   count evaluation source for the official evaluator. Keep each metric's
   validation role distinct from training velocity MSE.

2. **Test what Lingshu contributes.** Hold data, decoder, seed, and training
   budget fixed while comparing true Lingshu features with shuffled features,
   a constant condition, and permitted sequence-based features. A trained
   one-hot condition is a useful seen-target control but cannot represent unseen
   targets directly. Compare full-prompt pooling with gene-token pooling and
   context-free gene descriptions on held-out targets. Frozen embedding
   distinctness alone does not establish useful response information.

3. **Expand target and context coverage.** The current fit has one source
   context and only 146 target identities. Add broader licensed CRISPRi data,
   with complete target or feature-cluster holdouts and entire context/donor
   holdouts. Keep raw-count and normalized-abundance sources explicitly tagged;
   do not treat normalized support matrices as integer observations. Regenerate
   preprocessing within each fold. Include matched batching and sampling
   ablations before pooling incompatible protocols.

4. **Expand the response head and calibrate the decoder.** Compare 512, 1,024,
   and broader gene sets using training-only feature selection and the same
   shared normalization axis. Attention memory grows with gene count, so check
   H100 peak memory before larger runs. Investigate an independently trained
   count decoder or calibrated response residual; tune magnitude, dispersion,
   and source-zero activation on public holdouts rather than challenge scores.
   Record clamp fractions and library-ratio changes alongside performance.

5. **Ablate objective and numerical choices.** Compare CFM alone with the MMD
   term, noise levels, and 10/20/40-step Euler integration. The source treated
   cells and controls are unpaired; compare simple random control conditioning
   with a justified coupling method. Ensure any coupling or correlation graph
   is fitted on training rows only. Require improvements over a zero-velocity
   model in both held-out validation kinds, then check generated endpoints.

6. **Use additional H100s for independent runs first.** The implemented trainer
   is single-GPU. Run predeclared independent seeds or ablations in a Slurm array
   and compare held-out outcomes before defining a frozen ensemble rule. Merely
   requesting multiple GPUs does not make this code distributed. Resume binds
   the data, embeddings, architecture, and optimizer settings; changing the
   training recipe belongs to a new run with explicit provenance.

7. **Post-train only after a response signal is demonstrated.** A small
   low-rank update of Lingshu or a response-aware projection can be assessed
   after frozen-feature ablations show its value. Keep an untouched outer test
   split because adapting language features can overfit 146 training targets.
   Prefer calibration and broader public supervision before reinforcement
   learning against sparse leaderboard feedback. None of these future
   experiments is implemented or completed merely by being listed here.
