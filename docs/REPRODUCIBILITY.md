# Reproducibility

## Environment separation

The working cluster uses three isolated environments:

- Python 3.10 with CUDA PyTorch for model fitting and generation;
- Python 3.11 with `arc-state==0.11.3`, CUDA PyTorch, `cell-load`, and Lightning
  for STATE training and inference;
- Python 3.11 with `vcc-cli==0.1.0` for official validation, packaging, and submission.

Recorded package versions are in `requirements/`. Install the CUDA build appropriate for the target cluster rather than assuming a generic PyPI wheel matches the H100 driver.

## Required local files

Data is not stored in Git. The expected layout is:

```text
dataset/
  controls.zip
  controls/
    context_A.h5ad
    context_B.h5ad
    context_C.h5ad
    gene_names.csv
    pert_counts.csv
    manifest.json
  state_support/
    extracted/
      competition_train.h5
      k562.h5
      k562_gwps.h5
      rpe1.h5
      jurkat.h5
      hepg2.h5
      ESM2_pert_features.pt
  raw/
    replogle_2022/
      K562_gwps_raw_bulk_01.h5ad
      K562_essential_raw_bulk_01.h5ad
      rpe1_raw_bulk_01.h5ad
    replogle_nadig/
      GSE264667_hepg2_raw_singlecell_01.h5ad
      GSE264667_jurkat_raw_singlecell_01.h5ad
    feng2026/
      feng24_preprocessed.h5ad
```

Download the controls with the authenticated VCC CLI. Only use public or proprietary support data that the team has the legal right to use.

The STATE source reference used during the baseline audit was:

```bash
git clone https://github.com/ArcInstitute/state.git external/state
git -C external/state checkout 9bbfe78a434a55205e4de834e1ea99f85f7a3add
```

`external/` remains untracked.

Install STATE in an isolated Python 3.11 environment:

```bash
python3.11 -m venv .venv-state
.venv-state/bin/pip install -e external/state
```

The cluster compute image provides the Python runtime but not `Python.h`. The
matching Rocky Linux `python3.11-devel` RPM may be extracted without root under
`.venv-state-headers/`; the Slurm scripts expose that local include directory
through `CPATH`. No system package is modified.

## Local variables

Set these values for the current cluster before submitting jobs:

```bash
export VCC_PROJECT_DIR="$(pwd)"
export VCC_MODEL_PYTHON=/absolute/path/to/cuda/python
export VCC_CLI="$VCC_PROJECT_DIR/.venv-vcc/bin/vcc"
```

Slurm account, partition, QoS, memory, and time directives are site-specific and may be overridden with `sbatch` flags.

## Stage 1: audit controls

```bash
"$VCC_MODEL_PYTHON" scripts/audit_controls.py \
  --archive dataset/controls.zip \
  --data-dir dataset/controls \
  --output artifacts/control_audit.json \
  --deep-hash
```

Review the report before any model fit. Context labels must remain attached to their original files throughout the pipeline.

## Stage 2: build the public-effect cache

```bash
"$VCC_MODEL_PYTHON" scripts/build_public_effect_cache.py \
  --support-dir dataset/state_support/extracted \
  --controls-dir dataset/controls \
  --output-npz artifacts/bayes_public_effects_v1.npz \
  --output-json artifacts/bayes_public_effects_v1.json
```

The cache builder harmonizes public abundance matrices to log1p(CP10K), uses batch-matched controls with shrinkage, and records current controls separately. It does not use hidden challenge perturbation labels.

## Stage 3: fit the H100 prior

```bash
fit_job=$(sbatch --parsable --export=ALL \
  slurm/h100_fit_bayesian_prior_v0.sbatch)
```

The production script requires CUDA, verifies that the allocated device is an H100, and rejects the ESM transfer path unless its promotion criteria are met.

## Stage 4: generate and package

```bash
sbatch --dependency="afterok:$fit_job" --export=ALL \
  slurm/cpu_generate_package_bayesian_v0.sbatch
```

This job generates the complete sparse H5AD, asserts internal scientific and schema checks, runs official `vcc prep --dry-run`, builds the `.vcc`, validates the container, and records checksums. It refuses to overwrite an existing production artifact.

## STATE production path

The official support archive has inconsistent `uns/log1p` markers across files.
Create tiny external-link wrappers instead of modifying or duplicating the
official matrices:

```bash
.venv-state/bin/python scripts/prepare_state_support_links.py
```

Run the 20-step H100 smoke test first:

```bash
state_smoke_job=$(sbatch --parsable slurm/h100_train_state_smoke_v0.sbatch)
```

The production job stages the six support H5 files on node-local storage,
recreates the metadata wrappers there, and trains `state_sm` for 20,000 steps:

```bash
state_train_job=$(sbatch --parsable \
  --dependency="afterok:$state_smoke_job" \
  slurm/h100_train_state_sm_v0.sbatch)
```

The completed production training run was H100 job `859832`. It trained
162,626,484 parameters for 20,000 steps and exited successfully. Its recorded
validation losses include:

| Step | Validation loss |
|---:|---:|
| 12,000 | `1.737823367` |
| 14,000 | `1.812705994` |
| 16,000 | `1.666636586` |
| 18,000 | `1.735803962` |
| 20,000 | `1.701111555` |

The current STATE callback can save before validation at the same 2,000-step
boundary, which makes its automatic `best.ckpt` lag the validation result.
Preserve `last.ckpt` as `checkpoints/stepNNNNN.ckpt` immediately after every
validation. Then select the exact validation-aligned archive before inference:

```bash
.venv-state/bin/python scripts/select_state_checkpoint.py \
  --run-dir artifacts/state_runs/state_sm_20k_v0
```

This selected `checkpoints/step16000.ckpt`, the minimum-loss archived
checkpoint. The selection manifest records the rule, every validation result,
the metrics-file hash, checkpoint size, and checkpoint SHA256. Do not infer
with the automatic `best.ckpt` without checking that manifest.

Run bounded-memory inference on exact 128-cell sets and then package only if all
prior-structure gates pass:

```bash
state_infer_job=$(sbatch --parsable \
  --dependency="afterok:$state_train_job" \
  slurm/h100_infer_state_20k_best_v0.sbatch)

sbatch --dependency="afterok:$state_infer_job" \
  slurm/cpu_generate_package_state_20k_best_v0.sbatch
```

The STATE adapter applies `log1p` directly to the raw support-axis control
counts, matching the released training path: the official support matrices are
already log transformed but are not uniformly CP10K-scaled. It checks all 300
ESM embeddings, maps 18,077 shared genes back to the official axis, assigns zero
modeled delta to 456 challenge-only genes, and writes the same sparse effect-prior
contract consumed by the raw-count generator. Target and model-control outputs are
aggregated as arithmetic pseudobulks over the 18,077 shared genes, normalized to
the official 50,000-count bulk scale, and then differenced in log1p space. The
STATE prior carries this normalization mask, and the package interprets those
effects against the same shared-gene 50,000-count baseline. The Bayesian CP10K
prior keeps its 10,000-count, all-gene default. Never submit STATE's continuous
normalized output directly.

H100 job `860433` completed this effect-prior route for all 900 groups using the
selected step-16,000 checkpoint. It is retained as a diagnostic and fallback;
the direct paired-residual count adapter below is the preferred production
route.

## Direct paired-residual STATE count adapter

The direct adapter avoids treating STATE's absolute prediction as calibrated
raw counts. For every context-target group it:

1. selects real challenge control cells with deterministic stratification;
2. maps raw counts onto the 18,080-gene support axis and applies elementwise
   `log1p`;
3. runs target and non-targeting embeddings on the identical cell chunk;
4. subtracts the paired predictions cell by cell;
5. smoothly bounds non-target modeled effects as
   `0.6 * tanh(delta / 0.6)` and forces the target-gene fold to `0.20`;
6. exponentiates the bounded log effect only at source-observed shared genes;
7. copies all 456 challenge-only gene counts exactly; and
8. uses deterministic largest remainder to restore the exact source-cell
   library size.

The source-zero policy is deliberate: genes absent from a source control cell
remain absent after transformation. It preserves sparsity and prevents a dense
STATE output from inventing counts, but it also prevents genuine de novo gene
activation. Largest-remainder integerization avoids extra multinomial noise,
while exact library preservation means the model can change composition but not
cell depth. The `0.60` tanh bound is a conservative calibration choice, not a
learned biological constant. These are material modeling limitations and must
be revisited with permitted held-out evidence.

The raw-log1p input policy matches one of the two representations present in
the released training mixture. `competition_train.h5`, the largest matrix, is
log1p of integer raw counts; the other five matrices are continuous normalized
log1p profiles with approximately 13,000--14,400 implied counts per cell. The
checkpoint saw the stored matrices without another normalization pass, so
neither forced CP10K nor forced CP14K is uniquely faithful to training. The
adapter therefore keeps elementwise `log1p(raw)` for this candidate and records
that choice in provenance. A future normalization change requires a paired
held-out comparison rather than an assumed convention.

Run the bounded H100 target-limit smoke before production:

```bash
sbatch slurm/h100_generate_state_direct_counts_smoke_v0.sbatch
```

A minimal CPU plumbing smoke can be reproduced without claiming GPU parity:

```bash
.venv-state/bin/python scripts/generate_state_direct_counts.py \
  --model-dir artifacts/state_runs/state_sm_20k_v0 \
  --checkpoint selected.ckpt \
  --selection-json artifacts/state_runs/state_sm_20k_v0/checkpoints/selected_checkpoint.json \
  --controls-dir dataset/controls \
  --support-genes dataset/state_support/extracted/gene_names.csv \
  --output-h5ad artifacts/state_direct_counts_cpu_smoke_v0.h5ad \
  --output-json artifacts/state_direct_counts_cpu_smoke_v0.json \
  --device cpu --model-chunk-size 1 --cells-per-group 1 --target-limit 1 \
  --effect-scale 1.0 --effect-clip 0.60 --effect-bounding tanh \
  --target-strategy force --target-remaining-fraction 0.20 \
  --max-genes-per-cell 5900 --integerization largest-remainder \
  --groups-per-shard 1 --seed 20260831
```

The completed CPU smoke covered one target in A, B, and C. It produced a
`3 x 18,533` `int32` CSR matrix with 12,768 stored nonzeros. All internal schema
checks passed; source library sizes and all current-only counts were preserved
exactly. This deliberately tiny run does not satisfy the full official
360,000-cell contract and does not replace the H100 smoke or VCC dry run.

The production run used the following dependency chain so packaging could only
start after successful H100 generation:

```bash
state_direct_job=$(sbatch --parsable \
  slurm/h100_generate_state_direct_counts_v0.sbatch)

sbatch --dependency="afterok:$state_direct_job" \
  slurm/cpu_package_state_direct_counts_v0.sbatch
```

The production wrapper uses 400 cells per group in chunks no larger than 128,
all 300 targets in A/B/C, the step-16,000 selection manifest, tanh bound `0.60`,
forced target remaining fraction `0.20`, largest-remainder integerization, and
on-disk H5AD concatenation. The package wrapper independently checks provenance,
shape, groups, integer counts, scientific invariants, official target order,
`vcc prep --require-counts --dry-run`, and the generated container.

Production generation job `860523` and packaging job `860524` completed
successfully. The full output contained 360,000 cells, 18,533 genes, 900 groups,
400 cells per group, and 1,920,065,097 stored nonzeros. Exact source libraries,
all 456 challenge-only gene counts, and all target-knockdown gates passed. The
official dry run and package validation also passed. Submission entry
`JbDxq7SJV2wI0DWlIREn` was published on the validation panel; its complete
sanitized receipt is tracked in `results/state_direct_v0/submission.json`.

## Cross-context scorer-aware candidate

The replacement candidate separates the shared perturbation response from the
target residual. Build the corrected K562-to-RPE1 route with complete ESM2
cluster holdout and a full-axis fallback:

```bash
.venv-state/bin/python scripts/calibrate_cross_context_effects.py \
  artifacts/v2/k562_gwps_effect_atlas.npz \
  artifacts/v2/k562_essential_effect_atlas.npz \
  artifacts/v2/rpe1_effect_atlas.npz \
  artifacts/bayesian_prior_h100_v0.npz \
  dataset/controls/pert_counts.csv \
  artifacts/v2/cross_context_effect_prior_v3.npz \
  artifacts/v2/cross_context_effect_route_v3.npz \
  artifacts/v2/cross_context_effect_prior_v3.json \
  --challenge-genes dataset/controls/gene_names.csv \
  --esm2-embeddings dataset/state_support/extracted/ESM2_pert_features.pt
```

The selected ridge, target-residual, and route scales are `0.01`, `1.0`, and
`1.0`. The held-cluster internal proxy is `0.215397588`. The calibrator fits
every common response and projection inside the training fold. It merges
direct and fallback information at target-gene resolution and applies
cell-count reliability only to measured coordinates. Do not reuse v1 or v2
artifacts produced before these leakage and axis-alignment fixes.

Convert the full-axis response into a model-derived expected-count pseudobulk:

```bash
.venv-state/bin/python scripts/build_context_pseudobulk_prior.py \
  artifacts/v2/cross_context_effect_prior_v3.npz \
  dataset/controls \
  dataset/controls/pert_counts.csv \
  artifacts/v2/cross_context_pseudobulk_v3.npz \
  artifacts/v2/cross_context_pseudobulk_v3.json \
  --effect-clip 1.0
```

The pseudobulk contract is `3 x 300 x 18,533`, contains expected raw counts per
cell, preserves each context's mean control depth, and records that no measured
perturbed profile was inserted. Its SHA256 is
`b892bf83d6e26f1d86781a4385cf7389ad8f344308cc16c319da1afb741b887c`.

Run a minimal CPU structure smoke before allocating a GPU:

```bash
.venv-state/bin/python scripts/generate_state_scorer_aware_counts.py \
  --model-dir artifacts/state_runs/state_sm_20k_v0 \
  --checkpoint checkpoints/selected.ckpt \
  --selection-json artifacts/state_runs/state_sm_20k_v0/checkpoints/selected_checkpoint.json \
  --controls-dir dataset/controls \
  --support-genes dataset/state_support/extracted/gene_names.csv \
  --output-h5ad artifacts/v2/cross_context_state_v3_smoke.h5ad \
  --output-json artifacts/v2/cross_context_state_v3_smoke.json \
  --device cpu --model-chunk-size 2 --cells-per-group 2 --target-limit 1 \
  --state-effect-weight 0.05 --state-effect-clip 0.30 \
  --response-npz artifacts/v2/cross_context_pseudobulk_v3.npz \
  --common-response-weight 1.0 --learned-response-weight 1.0 \
  --pseudobulk-npz artifacts/v2/cross_context_pseudobulk_v3.npz \
  --pseudobulk-blend-weight 0.60 --zero-induction-scale 1.0 \
  --pseudobulk-depth-policy source-library --target-policy off \
  --count-emission stochastic-round --combined-effect-clip 1.0 \
  --groups-per-shard 3
```

The smoke produces six cells over all 18,533 genes, with one target in each
context. It verifies finite non-negative integer counts, sparse structure,
context labels, and response provenance; it is not a full official contract.

After all unit tests pass, generate and package the 360,000-cell artifact with
an explicit dependency:

```bash
cross_context_job=$(sbatch --parsable \
  slurm/h100_generate_cross_context_v2.sbatch)

sbatch --dependency="afterok:$cross_context_job" \
  slurm/cpu_package_cross_context_v2.sbatch
```

By default, the H100 wrapper consumes the corrected v3 pseudobulk. It accepts
only the explicitly versioned `v3` and `v4` candidates, uses target policy
`off`, and refuses to overwrite an existing production artifact. The CPU job
rechecks the generation manifest and SHA256 before running official VCC dry-run,
package, and container validation. Neither successful local proxy metrics nor
the CPU smoke authorizes a submission by itself.

The production generator also records exact count mass before and after the
5,900-gene sparsity cap. It rejects the artifact if dropped mass exceeds 10%
or absolute source-to-output library drift exceeds 15%, evaluated both overall
and separately in A, B, and C. The six-cell production-policy smoke measured
5.70% overall dropped mass and 5.83% library drift; its worst-context values
were 6.88% and 6.99%. Packaging re-hashes the response/pseudobulk, generator,
both gene axes, target axis, and all three controls against the generation
manifest before invoking the official CLI.

An additional one-H100 job trains an ESM2-conditioned K562 signature candidate:

```bash
signature_job=$(sbatch --parsable \
  slurm/h100_train_k562_signature_v2.sbatch)
```

The trainer keeps the 9,522 of 9,675 K562 targets with available ESM2 vectors
and records all 153 excluded training names in the NPZ, checkpoint, and JSON.
All 300 requested challenge targets must have ESM2 vectors; any missing
prediction embedding remains a hard failure rather than a zero-filled feature.

Its output is not automatically used by the v3 production path. It must first
pass whole-cluster validation and be overlaid onto the full-axis fallback by an
audited coordinate-aligned rebuild; the derived route and pseudobulk must then
receive new versioned hashes. The complete v4 dependency chain is:

```bash
prior_v4_job=$(sbatch --parsable \
  --dependency="afterok:$signature_job" \
  slurm/cpu_build_cross_context_prior_v4.sbatch)

generation_v4_job=$(sbatch --parsable \
  --dependency="afterok:$prior_v4_job" \
  --export=ALL,VCC_CANDIDATE_VERSION=v4 \
  slurm/h100_generate_cross_context_v2.sbatch)

sbatch \
  --dependency="afterok:$generation_v4_job" \
  --export=ALL,VCC_CANDIDATE_VERSION=v4 \
  slurm/cpu_package_cross_context_v2.sbatch
```

The v4 builder treats modeled zero effects as authoritative inside the
signature's named target-gene mask and proves that every other fallback value
is byte-identical. The generator and package wrappers accept only candidate
versions `v3` or `v4`, preventing an arbitrary environment value from changing
the production paths.

Build the raw HepG2 and Jurkat response atlases as independent research jobs:

```bash
sbatch slurm/h100_build_hepg2_raw_effect_atlas_v1.sbatch
sbatch slurm/h100_build_jurkat_raw_effect_atlas_v1.sbatch
```

These jobs may run concurrently. Their outputs are not automatically included
in v3 or v4; a future control-conditioned multi-route candidate must receive a
new version, held-context validation, and an independent pseudobulk hash.

## HepG2 zero-shot proxy

The released HepG2 support matrix was held out of training and used as a
zero-shot diagnostic. Prepare the evaluation H5AD and run the pinned direct
STATE inference plus current `cell-eval2` wrappers with:

```bash
.venv-state/bin/python scripts/prepare_state_holdout_h5ad.py \
  dataset/state_support/extracted/hepg2.h5 \
  dataset/state_support/hepg2_holdout.h5ad
sbatch slurm/h100_infer_state_hepg2_holdout_v0.sbatch
sbatch slurm/cpu_score_state_hepg2_celleval2_v0.sbatch
```

H100 inference job `860448` completed on 9,386 cells: 4,976 non-targeting
controls and 4,410 treated cells across 68 perturbations. Scoring job `860455`
used the current evaluator with `--input-type lognorm`. Mean proxy metrics were
direction fidelity yield `0.496288`, direction reach `0.250694`, LFC NMAE
`0.968758`, significant-gene Jaccard `0.002487`, unbiased expression distance
`0.002238`, unbiased expression MSE `0.000466`, capped unbiased expression MSE
`0.001821`, real-mass ratio `0.011470`, and PDS cosine `0.635865`.

These are raw log-normalized-space diagnostics, not normalized VCC leaderboard
scores. The public HepG2 matrix does not contain recoverable raw integer counts,
so it cannot reproduce the official counts-space evaluation. It also evaluates
the upstream direct inference output rather than the paired raw-count adapter.
Use it only as evidence about zero-shot behavior and failure modes, not as a
submission-score estimate.

## Pre-submission review

Verify all of the following:

- exact `360000 × 18533` shape;
- exactly 900 context-target groups and 400 cells per group;
- exact official target and context sets;
- raw finite non-negative integer counts;
- no submitted control cells;
- density and per-cell library limits;
- nonzero realized delta for every group;
- target-knockdown and direction checks;
- official dry-run success;
- package validation and stable SHA256;
- recorded commit, seed, environment, data hashes, and job identifiers.

## Submission safety

Authenticate without placing a token on the command line:

```bash
"$VCC_CLI" login --token-stdin
"$VCC_CLI" whoami --json
```

Submission is intentionally manual and limited to the designated submitter:

```bash
"$VCC_CLI" submit artifacts/candidate.vcc \
  --model-name "reviewed model name" \
  --description "reviewed description" \
  --json
```

Use the exact flags supported by the installed CLI version. Never use `--skip-limit-check`. Save the entry ID immediately and monitor the same entry with `vcc status`. If an upload is interrupted, use `--resume` rather than creating a new entry.

## Source checks

Run before every push:

```bash
python -m compileall -q scripts presentation
bash -n scripts/*.sh slurm/*.sbatch
git diff --check
git status --short
```

Inspect the staged file list and run a secret scanner before the initial push and before every release tag.
