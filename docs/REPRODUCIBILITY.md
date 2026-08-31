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
recreates the metadata wrappers there, trains `state_sm` for 20,000 steps, and
selects `best.ckpt` by the held-out HepG2 validation loss:

```bash
state_train_job=$(sbatch --parsable \
  --dependency="afterok:$state_smoke_job" \
  slurm/h100_train_state_sm_v0.sbatch)
```

Run bounded-memory inference on exact 128-cell sets and then package only if all
prior-structure gates pass:

```bash
state_infer_job=$(sbatch --parsable \
  --dependency="afterok:$state_train_job" \
  slurm/h100_infer_state_20k_best_v0.sbatch)

sbatch --dependency="afterok:$state_infer_job" \
  slurm/cpu_generate_package_state_20k_best_v0.sbatch
```

The STATE adapter performs support-axis CP10K/log1p normalization, checks all
300 ESM embeddings, maps 18,077 shared genes back to the official axis, assigns
zero modeled delta to 456 challenge-only genes, and writes the same sparse
effect-prior contract consumed by the raw-count generator. Never submit STATE's
continuous normalized output directly.

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
