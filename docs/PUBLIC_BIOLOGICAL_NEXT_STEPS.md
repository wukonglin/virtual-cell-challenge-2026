# Proposed public-data conditioning experiment

Status: proposal only; **not preregistered, implemented, or launched**. The three
v5 conditioning arms failed their screen. More GPU time alone is not evidence
that their target features will become useful. This handoff proposes a larger
public-target screen before another flow-model training job.

## Existing inputs and metadata-only inventory

Project-relative roots: `dataset/raw/replogle_2022/` (raw), `artifacts/v2/` (atlases).

| Context/source | Raw filename | Atlas filename | Atlas targets | After V7 exclusions |
| --- | --- | --- | ---: | ---: |
| K562 GWPS | `K562_gwps_raw_bulk_01.h5ad` | `k562_gwps_effect_atlas.npz` | 9,675 | 8,959 |
| K562 essential | `K562_essential_raw_bulk_01.h5ad` | `k562_essential_effect_atlas.npz` | 1,971 | 1,379 |
| RPE1 | `rpe1_raw_bulk_01.h5ad` | `rpe1_effect_atlas.npz` | 2,016 | 1,386 |

Each atlas has an adjacent same-stem JSON provenance receipt. Authenticate the
raw and derived files against those receipts and `docs/DATA_PROVENANCE.md`
before use; this inventory is not a fresh content-hash verification.

Counts above were calculated from target/gene metadata, not expression values.
The paired K562-essential/RPE1 target set contains 1,676 targets, or 1,174 after
V7 exclusions. Those sources share 7,225 genes; all three share 7,078 genes.
GWPS and essential are two K562 assays, **not two independent cell contexts**.

## Freeze the contract before expression reads

- Bind `artifacts/scdfm/v7/splits/jurkat_non_harm.json` and exclude all 741
  `partition.global_exclusion_targets` from every fitted stage and selection.
- Decide and record whether to additionally exclude the original 37 K562
  development targets in `artifacts/lingshu_scdfm/public_cache_v1.json`.
  Applying both denylists leaves 8,929 GWPS, 1,372 K562-essential, and 1,381 RPE1
  atlas targets. Do not choose this policy after seeing effects.
- Freeze source identity, eligible target list, embedding coverage, gene axis,
  metadata-only cluster construction, seeds, folds, model grid, and gates.
- Build authorized source-row indices from target/count/guide metadata first.
  Read only allowed treated rows and selected non-targeting `core_control` rows.
  Assert the exclusions before accessing `X`, including during QC/inspection.
- The existing bulk builder scans all expression rows before masking. Do not
  run `scripts/build_bulk_effect_atlas.py` unchanged for this contract. Loading
  an NPZ `effects` member also materializes the complete array before slicing;
  use a reviewed row-filter-first raw-H5AD consumer for strict read isolation.

## Stage 1: proposed 1,024-target CPU pilot

Select 1,024 eligible GWPS targets deterministically using names, counts, and
embedding coverage only, balanced across registered embedding clusters. The
selection must not depend on effect magnitude. Missing-feature handling and
the exact selected list must be frozen before response extraction.

Compare optimized constant, ESM2-conditioned regularized adapter, and three
shuffled-feature controls on identical targets, output genes, and folds. Use
nested whole-cluster target CV; fit feature transforms, effect bases, means,
regularization, and gains only inside the corresponding training folds.
Register a small bounded candidate grid and retain the existing conditioning
screen's improvement/uncertainty requirements rather than relaxing them.
Report per-repeat and per-target errors. Preregister paired whole-cluster
bootstrap intervals so correlated members are not counted as independent
targets; average repeated predictions before resampling clusters.

Existing frozen Lingshu embeddings at
`artifacts/lingshu_scdfm/v1/lingshu_embeddings.npz` cover only 410 eligible GWPS,
36 K562-essential, and 43 RPE1 targets after V7 exclusions, before the optional
37-target denylist. Compare Lingshu and ESM2 only on a matched covered subset;
a broader Lingshu arm requires separately authenticated feature extraction.

## Stage 2: conditional cross-context calibration

Only if Stage 1 passes, register expansion to the eligible GWPS set and a
separate paired K562-to-RPE1 experiment. Exclude complete target clusters across
both sources before fitting. Separate embedding-only target extrapolation from
same-target transfer using measured public K562 effects as model inputs.
RPE1 calibration with other RPE1 targets is not unseen-context evaluation.

`scripts/train_target_signature_model.py` provides fold-local representation
functions; `scripts/calibrate_cross_context_effects.py` provides transfer
functions. Their full CLIs are not authorized consumers of this new proposal:
they lack its exclusions and include full-data refits/challenge-output steps.

## Representation, tracking, and compute boundaries

The existing atlases contain log1p(CP50k of pooled pseudobulk abundance) minus
matched controls, not mean per-cell log1p(CP10k). Pool guide/promoter means using
represented cell counts and retain the recorded global-control fallback.
A common-axis CP10k rebuild must normalize treated and control profiles
separately before subtraction; never rescale a signed effect directly.
Pseudobulk expectations cannot train or validate a raw-count emitter or prove
single-cell variance, covariance, MMD, or full-gene generation quality.

Use W&B `validation` for the screen, `pretrain` for any later broader fit, and
`posttrain` for separately registered calibration, with shared experiment
grouping and parent-checkpoint hashes where applicable. Log allowlisted steps,
losses, aggregate diagnostics, provenance hashes, and decisions; keep raw data
and checkpoints local. Offline recording is valid; cloud sync is not assumed.

Run the pilot on authorized CPU compute, never a login/head node. Request one
Anvil H100 only after scientific gates and an explicit GPU workload justify it;
profile throughput/memory before requesting multiple GPUs. Use authorized
allocation/QoS and ordinary user priority, never a privileged priority claim.
These reused public sources support development evidence, not an untouched
independent test or a challenge-score guarantee. No submission is proposed.
