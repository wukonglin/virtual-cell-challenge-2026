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
  --checkpoint step16000.ckpt \
  --checkpoint-expected-sha256 e407e51855792d0dc85483a977f63497d9ac9a6893db360d840c7d9ce556780d \
  --perturbation-map-expected-sha256 6ab08c62dd117012b386051a9f375505abcd17b88ed76f0ad232199624291170 \
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
  --checkpoint checkpoints/step16000.ckpt \
  --checkpoint-expected-sha256 e407e51855792d0dc85483a977f63497d9ac9a6893db360d840c7d9ce556780d \
  --perturbation-map-expected-sha256 6ab08c62dd117012b386051a9f375505abcd17b88ed76f0ad232199624291170 \
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
versions `v3`, `v4`, or `v5`, preventing an arbitrary environment value from
changing the production paths. V5 intentionally reuses the immutable v4
response artifact and changes only the recorded count-adapter calibration.

### Executed v4 candidate run on 2026-08-31

The complete v4 chain was generated and validated offline on AIDA before one
validation-round submission. H100 signature job `860790` completed in 1 minute
57 seconds on an NVIDIA H100 80GB. It retained 9,522 of 9,675 K562 targets with ESM2 vectors,
covered all 300 requested challenge targets, and selected the rank objective at
epoch 4 with common scale `1.25`, residual scale `0.50`, and whole-cluster proxy
`0.1006820524`. The signature NPZ SHA256 is
`ef4e0f384ef67dbb9cb910bfbcd3d7a7f6c1b54c4840a1292917f1a9ad2068e6`.

CPU prior job `860800` completed in 1 minute 33 seconds and passed the overlay,
axis, fallback-preservation, calibration, and pseudobulk hash gates. Its v4
pseudobulk SHA256 is
`7b08f8852daa662c1769f2eb00f6915ebf0c16a3332d6ca118c755ce98077971`.

Single-H100 generation job `860803` completed in 21 minutes 27 seconds with
exit code `0:0`. The resulting H5AD has shape `360,000 x 18,533`, 900 required
groups, and `2,051,087,553` nonzero entries, which is `96,396,094` below the
signed int32 limit. All generation checks passed. Overall dropped count mass
was `0.0592241884` and absolute library drift was `0.0594704175`; the respective
context values were A `0.0688672081/0.0689821933`, B
`0.0471537802/0.0477338097`, and C `0.0610039876/0.0610712342`. The H5AD is
`16,437,781,680` bytes with SHA256
`4d7de29483387b277cf9e5162889f755b751f8e3a1f44b33b283c7a4de88e3b8`.

CPU package job `860804` completed in 6 minutes 47 seconds with exit code
`0:0`. Official VCC dry-run, real packaging, and independent container
validation all passed; the sole container member is `pred.h5ad.zst`. The final
offline package is `2,802,677,760` bytes with SHA256
`5f4b999cefe5b66c990e274beeb1fd41f6757b8328145bc934714b597757801a`.
All nine entries in `cross_context_state_v4_SHA256SUMS` were independently
verified. These checks establish format and production integrity, not model
superiority.

The package was submitted once as entry `jUKnSMYeYJaOCyOXJ4mc`, model name
`V4`, at `2026-09-01T03:30:59Z`. It was published without an upload, format, or
scoring error at overall score `-0.0501571428` and rank 334 at receipt. Scaled
components were PDS `0.0309523426`, expression MSE `0`, LFC NMAE
`-0.2933153459`, direction fidelity `-0.0441675259`, direction reach
`0.0027474151`, and significance Jaccard `0.0028402571`.

Relative to entry `JbDxq7SJV2wI0DWlIREn` (`STATE prediction`, overall
`-0.0059348511`), v4 improved scaled PDS by `0.0364660289`, fidelity by
`0.0025677290`, and Jaccard by `0.0329601634`, but regressed LFC NMAE by
`0.2638889184` and direction reach by `0.0734387530`. Raw capped expression MSE
rose from `4.7389056308` to `30.1804215609`, while raw LFC NMAE rose from
`1.0176913088` to `1.1800212368`. This is an effect-amplitude and expression
calibration failure, not a packaging failure. V4 is therefore not promoted;
future candidates must preserve its PDS/Jaccard gains while applying a much
stronger amplitude shrinkage and an exact-counts calibration gate.

### V5 amplitude-rescue candidate

The v4 response audit identified a post-processing failure rather than an
insufficiently large STATE model. Mapping effects from `log1p(CP50000)` back
to raw-count log fold increased response RMS from `0.02224` to `0.13621`; the
median per-target amplification was `7.03x`. The resulting response was 86.5%
dense. With a pseudobulk blend of `0.60` and zero-induction scale of `1.0`, v4
created 3.644 billion positive expectations at source-zero coordinates. The
5,900-gene cap then removed 442.088 million counts. Expected mass before the
cap differed from source mass by only 0.027%, so the failure was broad profile
redistribution and cap loss, not an exploding total library.

A held-HepG2 68-perturbation effect-space sweep placed the best common response
scale near `0.10` (proxy MSE `0.992089` at `0.10` versus `1.429704` at `1.0`;
proxy NMAE `0.998411` versus `1.201409`). Positive scalar shrinkage preserved
the local PDS and direction/top-set proxies. This is a directional diagnostic,
not an official score: it does not reproduce the raw-count scorer's jackknife,
Wilcoxon gates, or panel aggregation.

The versioned v5 configuration is therefore:

```text
state_effect_weight       0.01
state_effect_clip         0.15
common_response_weight    0.10
learned_response_weight   0.22
combined_effect_clip      0.35
pseudobulk_blend_weight   0.00
zero_induction_scale      0.00
target_policy             off
count_emission            stochastic-round
```

This keeps a larger centered target-specific residual than common response,
while disabling the absolute pseudobulk branch that caused dense zero-gene
activation. It also tightens generation gates to at most 3% dropped count mass
per context, 2.5% overall, and 5% absolute library drift per context. The
nonzero base-cap floor measured directly from released controls is about
2.47%, 1.02%, and 2.12% in A, B, and C.

Run the single-H100 technical smoke first:

```bash
mkdir -p logs artifacts/v2/smoke
sbatch slurm/h100_generate_cross_context_v5_smoke.sbatch
```

The smoke covers 8 targets, all three contexts, and 32 cells per group. It
requires an H100 at runtime, zero induced source-zero entries, fewer than 0.1%
combined-effect clip events, exact mass accounting, and the tightened loss and
library gates. It is not an accuracy benchmark and is never submitted.

Job `861094` completed this smoke on one NVIDIA H100 80GB in 19 seconds with
exit code `0:0`. The result contained 768 cells, 24 groups, and 4,181,409
nonzero entries. No source-zero entries were induced and no combined effects
were clipped. Overall dropped mass and absolute library drift were 1.788% and
1.801%; A/B/C dropped mass was 2.396%/0.923%/2.010%. Every smoke gate passed.
The sanitized receipt is `results/v5/h100_smoke.json`.

Only after those gates pass, generate the complete candidate:

```bash
generation_v5_job=$(sbatch --parsable \
  --export=ALL,VCC_CANDIDATE_VERSION=v5 \
  slurm/h100_generate_cross_context_v2.sbatch)

sbatch \
  --dependency="afterok:$generation_v5_job" \
  --export=ALL,VCC_CANDIDATE_VERSION=v5 \
  slurm/cpu_package_cross_context_v2.sbatch
```

Packaging and official submission remain separate decisions. A successful
technical smoke does not establish that v5 beats the published STATE entry.

Full generation job `861096` subsequently completed on one NVIDIA H100 80GB in
18 minutes 31 seconds with exit code `0:0`. The exact official-size output has
shape `360,000 x 18,533`, 900 groups, and `1,943,174,479` nonzero entries. It
contains no induced source-zero entries, and only 12 of 6.672 billion
combined-effect values reached the final clip. Overall dropped count
mass was `0.0186387016` and absolute library drift was `0.0188558471`; the
respective context values were A `0.0244272073/0.0245881851`, B
`0.0100426109/0.0104316759`, and C `0.0209856337/0.0210982552`. The H5AD is
15,574,412,552 bytes with SHA256
`831ab2587231a2b40f2bc604bd2ae63f252f52b1c7e2eee99071063ec316ea8e`.

CPU packaging job `861111` completed in 5 minutes 5 seconds with exit code
`0:0`. The official dry-run, real package, independent container validation,
and all nine checksum-manifest entries passed. The sole container member is
`pred.h5ad.zst`. The VCC package is 3,122,759,680 bytes with SHA256
`c1717f35d4a99ef1535e09f118a1bb86bfc8232b796f7a67d69dfc6a13d6671e`.
The sanitized full receipt is `results/v5/full_candidate.json`. This candidate
has not been submitted; format integrity and improved count calibration do not
by themselves establish leaderboard superiority.

### Authenticated V5.1 public validation

V5.1 is a separate public-data validation lane, not a retrospective claim
about the unsubmitted V5 official package. It evaluates a paired-count STATE
anchor and a power-zero public perturbation residual at alpha `0.10`, with a
forced model-space target remaining fraction of `0.20`. Each context contains
all 300 panel targets and 400 generated cells per target. HepG2 is the primary
zero-shot context because it was excluded from STATE training; Jurkat is an
in-distribution non-harm context.

The scoring receipts pin cell-eval2 commit
`5e64833518a6603a0301cbe28185d49c30f4a986`, declared version `0.16.0`, and
pdex `0.3.0`. Validation reconstructs expression MSE as the official ratio of
the complete numerator and denominator sums, authenticates the matched NMAE
omission subset, and retains model-dependent NaN masks for direction metrics.
The candidate and anchor truth views are byte-identical. Held-target count
matrices are also exact; small held-panel MSE changes can still arise from the
official panel-wide predicted-control correction and are not output drift.

Against the matched STATE anchor, the candidate improved all six oriented
aggregate metrics in both contexts. The HepG2 all-target changes were PDS
`+0.0505685619`, MSE `+0.0096956673`, NMAE `+0.0059183518`, fidelity
`+0.0097314642`, reach `+0.0252442968`, and Jaccard `+0.0023731408`.
The corresponding direct-target changes were `+0.0568186088`,
`+0.0105761023`, `+0.0065284912`, `+0.0109713804`, `+0.0285686486`, and
`+0.0026664503`. All six direct-target one-sided 90% lower bounds were positive
in 10,000 paired bootstrap resamples with seed `20260901`. Jurkat likewise
improved all six aggregate metrics, so the primary, non-harm, and joint gates
passed.

The small sanitized record is
[`results/public_v51/validation_decision.json`](../results/public_v51/validation_decision.json).
Large predictions, scorer caches, truth views, and detailed receipts remain
untracked under `artifacts/public_v51`. This result makes V5.1 the incumbent
for subsequent public-data ablations; it is not an official VCC submission or
leaderboard score.

### V6 sequential ablations

V6 keeps each candidate isolated under `artifacts/public_v6`, authenticates an
immutable pre-inference spec, and refuses scoring unless generation verification
passes. The first P2 candidate changed only residual allocation: reliability
power `0.5`, alpha `0.1248983248`, and total residual energy matched to the
V5.1 power-zero alpha-`0.10` incumbent. It completed authenticated generation
and HepG2 scoring, but failed promotion. Relative to V5.1, its all-target
oriented changes were PDS `+0.0003121516`, MSE `-0.0002796614`, NMAE
`+0.0000391370`, fidelity `-0.0016294632`, reach `-0.0002250083`, and Jaccard
`-0.0000113371`. The two-sided 90% paired-bootstrap MSE interval was
`[-0.0004250866, -0.0001381588]`; P2 therefore failed both primary point gates
and was not promoted to Jurkat.

The sanitized P2 record is
[`results/public_v6/p2_reliability_decision.json`](../results/public_v6/p2_reliability_decision.json),
and the executable contract is documented in
[`PUBLIC_V6_ALPHA_CALIBRATION.md`](PUBLIC_V6_ALPHA_CALIBRATION.md).

The pre-registered P3 step changed target force while returning to the
validated power-zero residual. It used exactly two new HepG2 arms at a
model-space target remaining fraction of `0.40`: STATE-only alpha `0.00` and
residual alpha `0.10`. The existing V5.1 anchor and incumbent were the matched
`0.20` arms. Commands, causal contrasts, held-target invariants, and the
deterministic promotion rule were pre-registered in
[`PUBLIC_V6_P3_FACTORIAL_DECISION.md`](PUBLIC_V6_P3_FACTORIAL_DECISION.md).
Selection also requires the
[`four-arm provenance contract`](PUBLIC_V6_P3_FACTOR_CONTRACT.md) and the
[`held-target count-identity gate`](PUBLIC_V6_P3_COUNT_IDENTITY.md).

Generation jobs `861527` and `861528` ran sequentially on one NVIDIA H100
80GB and completed in 4 minutes 1 second and 4 minutes 0 seconds. Scoring jobs
`861529` and `861530` completed in 4 minutes 56 seconds and 5 minutes 22
seconds. All four jobs exited `0:0`. Both strict-v2 generation receipts and
both pinned cell-eval2 scoring receipts passed. The 33 held-target groups were
exactly count-identical across the alpha contrast: zero differing rows and
entries across 13,200 cells and 9,623 model genes.

The immutable four-arm factor receipt passed, including shared input hashes,
axes, annotations, and sampled source-control pairing. The factorial analyzer
used 10,000 common paired-target bootstrap draws per cohort and metric. Arm C
failed against B on all six all-target and direct-target oriented metrics.
Arm D was nearly neutral against B: all-target deltas were PDS `-0.00002230`,
MSE `+0.00000151`, NMAE `-0.00000188`, fidelity `-0.00006192`, reach
`+0.00005446`, and Jaccard `-0.00000498`. It failed the all, direct, and
held-target point gates. D did pass against C, confirming that the residual
remained beneficial, but neither new arm passed the incumbent. The
deterministic selection therefore retained B. No Jurkat P3 run or official
submission was made, and no VCC quota was consumed. The sanitized result is
[`results/public_v6/p3_target_force_decision.json`](../results/public_v6/p3_target_force_decision.json).

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
