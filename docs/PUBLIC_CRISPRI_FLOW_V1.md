# GenePT/GO + control-context flow: implementation status

Latest continuation: [completed public GPU flow pretraining v10](PUBLIC_AUXILIARY_V10_RESULTS_20260910.md).
Six true/shuffled/constant GO arms completed3,600 actual AdamW steps on the
local RTX2080Ti with3,606 independently replayed offline W&B events. The cache
contains62,055 unique cells,7,097 measured normalization genes and482 modeled
genes on the exact VCC-compatible axis. Source-local held-target MSE remains
4.8331%/9.0513% worse than controls(K562/Jurkat), despite modest GO benefits
over negative-feature arms. No score improvement is established. The user's
exploratory-upload authorization supersedes older mandatory1% promotion gates
for submission, but native counts, exact axes/quotas, provenance and official
CLI validation remain mandatory. Count-safe inference and official packaging
have now completed, and VCC entry `0ivHnDQV8wi5WQH14o2b` was uploaded with MD5
verification. VCC published the result at19:48:37UTC:overall **-0.06402253**,
rank589 at retrieval, weakest scaled component FID **-0.28615245** (DE direction
fidelity, not a distribution distance). Upload and
scoring succeeded, but improvement is not established. The panel/anchor and
exact six metric values are recorded in the v10 report; no duplicate was sent.
See the v10 report and the tracked
[sanitized server receipt](../results/go_context_flow_v10/submission.json) for
evidence; detailed local receipts remain outside Git.
GenePT numeric conversion and source-fit-only conditioning artifacts have now
[completed and replayed](GENEPT_NUMERIC_V11_RESULTS_20260910.md). Raw coverage
is300/300official names; eight ambiguous duplicate-vector names are quarantined,
leaving297/300usable official GenePT vectors plus unchanged GO.555 related tests
passed. No new predictive-model fit used these features yet; the scored model
is still GO-only. No Feng/PromoterAI post-training has run.

Earlier continuation: [completed metadata-only auxiliary v9 registration](PUBLIC_AUXILIARY_V9_RESULTS_20260910.md).
The source-specific candidate plan now seals 487 extra targets, 5,075 groups
and 57,207 treated cells, excluding all56 panel and772 protected targets.
Only fit-reference16/context32 pools are included; tuning/sham remain excluded.
Real annotation registration and separate reconstruction passed with zero X/NPZ
read attempts.341 focused tests passed. No new expression materialization,
training job, GPU allocation, cloud download or VCC submission was launched.
The next implementation needs a separately contracted deduplicated cache and
source-specific feature/training registration; promoter/Feng remain gated.

Earlier continuation: [completed expanded-cohort validation v8.1](PUBLIC_EXPANDED_V81_RESULTS_20260910.md).
All 192 two-head fits, eight controls and 207 offline W&B events completed and
were independently replayed. True-GO MSE is worse than matched controls by
0.699664% / 0.828936% (Jurkat/K562). Integrity/safety passed; conditioning and
distribution failed. No promotion, flow/Feng training or VCC submission.
The next bounded step is a separately registered metadata-only auxiliary-target
plan preserving source, panel-target and control-role holdouts.

Earlier running snapshot: [expanded-cohort validation v8.1 monitoring](PUBLIC_EXPANDED_V81_STATUS_20260910.md).
The eight-thread CPU run has restarted on the BioHPC compute host after 2,767
related regression tests, 50 logging-recovery tests and 58 monitor tests passed. The frozen design
has 192 two-head fits and eight control references, with per-fit artifacts and
offline W&B tracking. The first v8 attempt stopped on W&B's debug-log symlink
before any learned fit or tracking event; its partial files are preserved.
The new guard handles only this SDK link as metadata without following its
target. Actual pinned-SDK initialization smoke passed, with no model fitting.
Execution contract: 24d4581d9637cf082a028938fe425e515bf3b9a86a8cd4c6c08659419615cab8.
Status at 16:06 UTC: control and context-only arms completed and replayed;
additive GO/context fitting is in progress. Training PID 2078078; bounded
60-second managed monitor PID 2093590 (replacing the stopped detached launch).
No final scientific result or progression decision.
The monitor writes local verification/readiness files, not online notifications,
and never starts training or submissions. A metadata-only future audit identifies
487 extra eligible targets; they remain outside this run's admission contract.
No GPU, new download, GCP transfer, promoter/Feng training or submission.

Earlier continuation: [completed replication-expanded public cache v7](PUBLIC_REPLICATED_V7_RESULTS_20260910.md).
The same 56 targets now have 1,767 matched groups and 22,213 retained treated
cells, versus 398 groups and 4,083 treated cells in v2. All 398 original groups
are preserved exactly; an independent real-cache audit confirmed float32 bit
equality. The 772 target exclusions and disjoint control roles are unchanged.
Related regression: 2,568 tests across 47 modules. Cache: 54.99 MiB compressed;
materialization peak RAM: 1.04 GiB. No GPU, GCP transfer or new download.
This is data preparation, not a new model-quality result. The separately
registered expanded-cohort training runner with per-fold/candidate artifacts
has now been implemented and launched as v8 above. Promoter/Feng/submission
remain gated.

Earlier continuation: [completed nested public-only regularization experiment v6](PUBLIC_NESTED_V6_RESULTS_20260910.md).
Reviewed and fixed numerical validation, typed role/provenance checks and saved
artifact replay in new versioned code; frozen v5 results remain unchanged.
All 192 two-head fits and eight control references completed. True GO selects
lambda10 in every direction; gains remain 0.02879% / 0.02242% (Jurkat/K562).
Conditioning/distribution gates fail and safety passes. W&B:207 offline events,
zero errors. Related regression:2,454 tests across45 modules.
Authenticated one-shot readiness is valid; promoter/Feng/submission remain false.
Its proposed replication-expanded cache for the same 56 targets has now been
registered, materialized and verified as v7 above, without weakening the existing
cell-count filters or 772 exclusions. Expanded-cohort model fitting is still
pending. No new download, GPU allocation or submission.

Earlier continuation: [completed GO-by-control-context interaction experiment v5](PUBLIC_TARGET_CONTEXT_V5_RESULTS_20260910.md).
Seven arms completed: 48 two-head kernel fits plus eight control references.
True GO improves primary MSE over controls only 0.02879% / 0.02242%
(Jurkat/K562), and the interaction is slightly worse than additive GO in K562.
MMD and batch-centroid gates fail; safety passes. The new training diagnostics
show strong shrinkage (effective df0.270–0.324) and nearly unchanged positive
counts, not proof that lower regularization would generalize better.
W&B: 63 offline events, zero errors; 1,787 related tests pass.
The one-shot authenticated v5 readiness report is valid and all progression
branches remain false. No promoter/Feng training, count emitter or submission.
Its proposed inner public regularization experiment with disjoint reference/
tuning control rows has now been executed as v6 above.
The existing 24-hour watcher remains pinned to v4 only.

Earlier continuation: [completed sparse two-part output validation and readiness monitoring](PUBLIC_SPARSE_HURDLE_V4_RESULTS_20260910.md).
All six fixed arms completed on the BioHPC compute host. True-GO outputs retain
54.49% / 54.05% exact zeros, avoiding the v3 sparsity defect, but primary gains
over controls are only 0.02894% / 0.02225%. MMD is slightly worse than controls;
conditioning and distribution gates fail, while safety passes. No promotion,
promoter/Feng training or submission. W&B: 54 offline events, zero errors;
1,410 related tests pass. A 24-hour, five-minute local readiness watcher is
running on the pinned v4 evidence (PID1789427; deadline2026-09-11 05:50UTC).
It records status only, not automatic training/submission or chat notifications;
a future candidate requires new verified pins. Next design: target/context
interactions and training-only shrinkage/quantization assessment.

Earlier continuation: [completed joint target/source validation v3](PUBLIC_JOINT_TARGET_SOURCE_V3_RESULTS_20260910.md).
Four target folds and two source directions completed for six fixed arms
(40 ridge fits plus eight control references). True GO improves target-pooled
MSE by only 0.1809% / 0.2744% versus controls, and 0.00459% / 0.00624% versus
context-only. It narrowly beats all three shuffles but fails the frozen 1%
quality margins in both sources. Predictions remain nonnegative but lose too
many exact zeros. W&B: 54 offline events, zero errors; 1,088 related tests pass.
No promotion, Feng/promoter training, new download, GPU use or submission.
Next design: sparsity-aware control-anchored outputs and target/context
interactions, under a new public-only registration rather than tuning v3.

Earlier continuation: [completed sample-size-matched public reliability audit](PUBLIC_EFFECT_RELIABILITY_V1_RESULTS_20260909.md).
Both sources show positive within-/between-batch reproducible components above
matched-null comparisons, but normalized cell-half agreement is weak (~0.046).
No model was fitted. The next bounded model test should jointly hold out targets
and sources to avoid shared-reference noise leakage. W&B: 35 offline events,
zero tracking errors; related regression: 875 tests passed. The v2 model gate
remains failed, and promoter/Feng scale-up remains deferred.

Earlier continuation: [completed public control-anchor v2 validation](PUBLIC_CONTROL_ANCHOR_V2_RESULTS_20260909.md).
The new cohort has 56 common K562/Jurkat targets and disjoint context/reference/sham
controls. All four fixed arms completed with nonnegative predictions and offline
W&B. True GO improves control MSE by only 0.108% / 0.146% and loses to shuffled GO
in both source holdouts. The registered quality gate failed; promoter ablations
and Feng post-training remain deferred. Related regression: 815 tests passed.

Earlier diagnostic continuation: [completed replay and conservative baselines](PUBLIC_FLOW_DIAGNOSIS_20260909.md).
The H100 pilot reproduced correctly but transferred wrong-direction expression
shifts. Fixed shrinkage improves this small H1 development MSE only 0.159% over
controls, without convincing GO-versus-shuffled benefit. No checkpoint is
promoted; no Feng post-training or new challenge submission was performed.

Earlier continuation: [the registered real-data GO/H1-development pilot](PUBLIC_FLOW_PILOT_20260909.md)
records completed Replogle/Feng acquisitions, the new row-authorized consumer,
GO coverage, bounded Anvil execution and remaining post-training gates. The
sections below preserve the earlier prototype-stage record.

2026-09-09. **Prototype components and synthetic tests, not a trained public
model, registered complete experiment, score improvement, or submission.**

## Implemented components

- `scripts/public_flow_conditioning.py`: source-keyed feature tables, explicit
  GenePT/GO availability masks, train-fold-only normalization/PCA, deterministic
  shuffle/zero/train-constant controls, and strictly matched control summaries.
  Source hashes are supplied by an authenticated data loader; the primitives do
  not independently authenticate arbitrary caller data or read files.
- `scripts/public_crispri_flow.py`: a small PyTorch velocity-field prototype
  initialized from sampled controls and conditioned on the evolving cell state,
  continuous target features, pooled control context, time and observed-gene
  mask. It supplies a linear-path unpaired population flow loss and Heun rollout.
  Its initial zero head reproduces controls; this is not a trained anti-collapse
  result. Outputs are continuous expression, **not integer UMI counts**.
  An independent review removed exact-source-cell conditioning (a point-mass
  conditional-source shortcut) and inference-only noise. Training and rollout
  now use the same sampled-control base; no claim of solved mode collapse follows.
- `scripts/wandb_training.py`: strict allowlist extended for `genept`, `go`,
  `genept_go` arms and their eight-digit seeded shuffles; five provenance hashes;
  scoped variance, covariance rank, library, zero and duplicate diagnostics.
  No paths, credentials, cell IDs, matrices or arbitrary metric names are added.

Missing-modality indicators must be appended to transformed target features and
retained in every comparison. Unobserved expression coordinates remain marked;
their zero storage placeholders are not observed biological zeros. Context
summaries reject treated cells and mismatched donor/assay/batch strata instead
of silently falling back to another population. Shuffled/constant comparisons
are feature-value tests with the same availability masks.

## Acquisitions and cost boundary

Google OAuth works and `vcc-dataset` reports billing enabled. After the user
confirmed active Arc Marketplace enrollment and zero prior monthly use, the
seven processed H1 files were downloaded and checksum-verified on 2026-09-09:
**34,362,325,889 bytes**, using `--billing-project vcc-dataset`. The 42 FASTQ files
(5.81 TB) were excluded. See [the completed acquisition record](GCP_H1_SETUP.md).
No GPU job or new paid cloud service was started. Account-wide credits have not
been independently verified; local transfer accounting is not a Google hard cap.

The official GO human GAF and matching ontology were downloaded through public
HTTPS, without a GCP billing project or paid embedding API, into
`dataset/reference/go/2026-08-05/`: **44,337,039 bytes total**. The acquisition
receipt records sizes, local SHA-256, gzip integrity and matching ontology
headers. The release bundle date is 2026-08-05; both files reference ontology
2026-07-26. Do not silently substitute a newer rolling release.
[GO downloads/version guidance](https://geneontology.org/docs/download-go-annotations/),
[license](https://geneontology.org/docs/go-citation-policy/).

GenePT acquisition is currently **blocked by Zenodo timeouts/HTTP 504**, with no
archive or verified receipt published. Live DataCite metadata confirms CC BY 4.0,
but current Zenodo file metadata and ZIP headers could not be retrieved.
The publisher-linked GenePT release is DOI `10.5281/zenodo.10833191`, archive
`GenePT_emebdding_v2.zip` (expected 574,395,233 bytes, MD5
`3f6ce4317e3a0091978ae5cb8fbf05a3`). Do not count it as acquired unless a local
verified acquisition receipt exists. Do not execute or blindly unpickle its
legacy embedding files. The remaining feature-loader work includes safe array
conversion, stable gene/protein mapping, duplicate/alias handling, GO `NOT` and
obsolete-term handling, a frozen evidence/ancestor policy, and coverage audits.
[Official GenePT release link](https://github.com/yiqunchen/GenePT).

## Data eligibility discovered in the metadata-only audit

**H1 has already appeared in this project's old support-data/model lineage.**
`dataset/state_support/extracted/competition_train.h5` represents an H1 training
subset (221,273 cells, 150 targets in existing metadata). Its stored expression
is documented as log1p(integer counts). Downloaded full H1 requires cell/guide/
target overlap checks; combining both copies would double-count an experiment.
Reusing H1-trained weights and calling H1 an untouched recipient would be wrong.

- Existing raw-count fit source: `dataset/raw/replogle_nadig/GSE264667_jurkat_raw_singlecell_01.h5ad`
  (9,366,490,264 bytes; historical V7 gate identifies 189,593 allowed rows including
  12,013 controls, and 8,882 measured genes). This is not a full challenge gene
  axis. File contents were not freshly hashed or inspected in this audit.
- Existing raw HepG2 must remain excluded from fitting under the preserved V7
  whole-context protocol. K562/RPE1 support matrices and Replogle `raw_bulk`
  population means cannot be silently repurposed as raw single-cell counts.
- Exclude all **741** global V7 target-cluster members, not only the 300 directly
  scored targets. V5 exposed 14 excluded members: do not reuse its fitted
  transforms/checkpoints under a clean V7 label. Old RPE1/Jurkat/HepG2 development
  results are not a newly untouched test.

Preserve these existing seals unchanged:

| Existing contract | SHA-256 verified from local JSON |
| --- | --- |
| `artifacts/scdfm/v7/splits/jurkat_non_harm.json` | `3114e59b5430d140cfeeaccd8b2f6974ceb1c14ba254ec41597057ce7a70c439` |
| `artifacts/scdfm/v7/training/jurkat_training_preflight.json` | `b5dfc61625625e8b70f126e9eb563f12f62ec629e8a138ef6045688c0f233ba1` |
| `artifacts/scdfm/v7/training/portable_training_gate_lock.json` | `f9c5b8675a1a46c561c056574009722d1faa4bfaf8b4e12e7d7017d3c5a9628d` |

The old portable gate supports its seven historical sources and dense-row
consumer. It does not automatically authorize a new H1 raw file or a CSR
loader. The old bulk builders read excluded expression before masking and are
not acceptable consumers for the new route's read-isolation contract.

## Remaining work before a public fit

1. H1 acquisition and its object/generation/size/checksum manifest are complete.
   Authenticate the files against `vcc_2025_h1/acquisition_complete.json` before
   creating any derived dataset. Finish public feature-artifact acquisition
   separately; no paid embedding calls are needed.
2. Register a **new** source/row/split contract before expression inspection:
   lineage/deduplication, all target exclusions, optional prior 37-target
   development denylist, matching units, stable gene axis, feature coverage,
   aliases, normalization, seeds and independent context/target-family folds.
3. Implement and test a row-authorized dense/CSR loader. Select allowed rows
   from metadata before touching X/layers. Fit every learned transform only on
   the applicable training fold. No recipient-treated or 2026 challenge-treated
   cells may enter pretraining, context construction, or post-training.
4. Run a bounded public CPU baseline/flow smoke test with balanced conditions,
   true/shuffled/constant feature arms, matched controls and distributional
   diagnostics. Then profile one allocated Anvil H100 before a larger run.
5. Public-only post-training requires a separately bound stage/split and an
   authenticated parent checkpoint; W&B's hash flag alone does not verify a
   checkpoint. Continuous-expression success does not establish a count decoder
   or full-gene submission's validity. Do not promote or submit on unit tests.

## Tracking contract

Use existing `run_training_with_wandb.py` with `--stage pretrain` or `posttrain`,
`--group public-crispri-flow-v1`, a fresh tracking directory and fresh summary
paths. Offline recording is available; online sync has not been configured.

Trainer flags recognized for provenance are `--split-sha256`,
`--contract-sha256`, `--source-manifest-sha256`, `--target-features-sha256`, and
`--context-transform-sha256`. They record validated hash strings, not evidence
that the trainer actually authenticated a file; the loader must do that check.

Emit JSON scalar step/loss/CFM/gradient/learning-rate records. Add aggregate
`kind_metrics.context` / `kind_metrics.target` fields `variance_ratio`,
`covariance_effective_rank`, `library_size_error`, `zero_fraction`, and
`duplicate_fraction`, alongside existing centroid/MMD diagnostics. Their
definitions, gene masks and scale must be frozen in the experiment contract;
names alone are not metric implementations or validated scientific gates.

No production public pretraining/post-training, H100 job, W&B cloud upload,
challenge generation or submission was started during this implementation.

## Verification completed

**183 tests passed in 8.04 seconds**, with CUDA hidden and synthetic inputs only:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  .venv-state/bin/python -m pytest -q \
  tests/test_public_flow_conditioning.py \
  tests/test_public_crispri_flow.py \
  tests/test_public_crispri_flow_distribution.py \
  tests/test_wandb_training.py tests/test_wandb_public_flow.py \
  tests/test_vcc_gcloud.py
```

The distribution regressions include active-head masked-state invariance,
masked-endpoint overflow prevention, equal per-cell loss weighting, analytic
Heun convergence, and restoration of model mode when nonfinite rollouts fail.
These establish implementation behavior, not biological transfer performance.
