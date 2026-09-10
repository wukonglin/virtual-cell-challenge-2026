# Public GO/control-context flow pilot — 2026-09-09

This is a bounded engineering pilot, not a full-gene submission or evidence of
improved challenge scores. GenePT is still unavailable from its previously
timing-out publisher endpoint; this first real-data arm uses GO, not GenePT/GO
fusion. The model is the small project-native continuous-expression flow MLP,
not an unchanged CellFlow/scDFM model or the proposed full-gene U-Net hybrid.

## Acquired public data

All five approved Figshare raw files completed, with exact byte size and
publisher MD5 verified before publication, and local SHA-256 recorded:

| Source | Bytes | Local receipt |
| --- | ---: | --- |
| Replogle K562-essential + RPE1 raw cells | 19,362,753,211 | `dataset/raw/replogle_2022_singlecell/acquisition_complete.json` |
| Feng fitness + nonfitness + targeted RNA counts | 7,675,835,277 | `dataset/raw/feng_rna_counts/acquisition_complete.json` |

Total: **27,038,588,488 bytes**. No failed/repeated transfers, GCP billing project,
GWPS expansion, or raw FASTQ downloads were used for this acquisition. The earlier
H1 GCP transfer remains 34,362,325,889 bytes; local accounting does not establish
an account-wide hard spending cap or verify credited invoice charges.

Primary manifests: [Replogle Figshare v1](https://api.figshare.com/v2/articles/20029387/versions/1)
(CC BY 4.0), [Feng Figshare v2](https://api.figshare.com/v2/articles/27989294/versions/2)
(MIT). Companion Feng metadata and header-only join receipts are in
`dataset/raw/feng_rna_counts/metadata/`.

## Frozen first pilot

While RPE1 was still downloading, the plan was explicitly amended **before any
new expression read** to the `h1_dev` profile. RPE1 subsequently completed; that
does not change the registered split:

- Fit: freshly acquired Replogle K562 and existing authenticated public Nadig
  Jurkat raw cells. No prior model weights are reused.
- Development only: newly downloaded H1 training-file cells, never fitted in
  this pilot. H1 appeared in older project lineage, so this is not a newly
  untouched benchmark. Its ordered cell/guide/target/batch/gene axes exactly
  match the old support copy, which is not combined with it.
- All 741 V7 excluded target-cluster members plus the prior 37 development
  targets are denied: **772 unique target labels**. Existing V7 seals are
  unchanged. HepG2 and 2026 challenge-treated cells remain excluded from fitting.
- Rows and exact source/batch-matched NTC groups were selected using annotations
  before accessing expression. Three full source hashes and the frozen contract
  are authenticated before selected dense/CSR reads. External HDF links and
  virtual/external-storage datasets are rejected.
- Raw-count integer validation passed on the admitted H1/K562/Jurkat values.
  Float dtype alone was not treated as proof of count representation.
- Normalize over **7,006 shared uniquely named measured genes**, CP10k then
  log1p; model a fixed hash-selected **512-gene** subset. Duplicate symbols are
  excluded, not silently resolved to the first row. No response-based HVG fit.
- Train: 29 groups, 303 treated samples, 1,856 matched-control sample entries.
  Development: 27 groups, 592 treated samples, 1,728 control sample entries.
  Control entries may be reused across targets sharing a batch; these are not
  counts of independent unique cells. Unique raw rows read were 847 K562,
  928 Jurkat and 1,936 H1.
- GO: all 16 registered targets covered; 275 direct GO terms. Exact symbols,
  primary active terms, train-target-only vocabulary; no IEA/ND/NOT/obsolete/
  alternate terms or ancestor propagation. No paid embedding API calls.

Frozen artifacts are in `artifacts/public_flow/h1_dev_pilot_v1/`:

| Artifact | SHA-256 |
| --- | --- |
| `sources.json` | `aa836a654761b3c1639526f0d757aea202afb0fddf1ffd94bd873c3b2bb74fbb` |
| `contract.json` | `64214578c0b20837f29a374fc719e6cc3edfc644d0e184d4ff4cd915110ee33c` |
| `cache.npz` | `408dda72ff7f7135d92ec824940ba2116966adc0ef5aa651869e98d03f45d52c` |
| `go/features.npz` | `b478be572e57a05e74a0833a9a7124cedb48d38c4a19b5a609b0d39fd672fd15` |

## Execution and W&B

**Completed job: `20539599`.** Slurm accounting confirms `COMPLETED`, exit `0:0`,
elapsed `00:01:12`, node `h017`. The real CUDA probe recorded an NVIDIA H100
80GB HBM3, PyTorch 2.7.1+cu126. All three 200-step arms completed with 202 W&B
events each and zero tracking errors. These are offline runs, not cloud-synced.
Local copies of summaries, weights and receipts under
`artifacts/public_flow/h1_dev_pilot_v1/anvil_20539599/` match the remote hashes.

Technical success is not scientific success: every trained arm loses to unchanged
controls on both centroid MSE and the fixed biased MMD-squared diagnostic in all
27 development groups. No checkpoint has been promoted or submitted.

| Arm | Centroid MSE | MMD squared | Negative fraction |
| --- | ---: | ---: | ---: |
| Unchanged controls | 0.01134486 | 0.01086383 | 0 |
| True GO | 0.02511518 | 0.02371001 | 0.08799 |
| Constant GO | 0.03225175 | 0.03034355 | 0.10188 |
| Shuffled GO | 0.02416125 | 0.02281305 | 0.08423 |

True GO is better than constant but worse than shuffled GO, so this pilot does
not establish biological conditioning value. True-GO predicted/treated variance
ratio averages 0.97893 with group range 0.76126–1.33203; these values alone do not
support calling the failure strong mode collapse. The follow-up protocol is in
`docs/PUBLIC_FLOW_DIAGNOSIS_20260909.md` and focuses on harmful displacement and
conservative public-only residual learning. These 512-gene log-expression
metrics are not challenge leaderboard metrics.

The staged Anvil release is
`/anvil/scratch/x-yzhang142/virtual-cell-challenge-2026/public_flow_releases/h1_dev_v1_20260909`.
All 15 staged files were SHA-256 matched to BioHPC before job submission.
`slurm/anvil_public_flow_pilot.sbatch` requests one H100 in `ai`, authorized `ai`
QoS, 8 CPUs, 32 GB RAM and **20 minutes maximum**, with no requeue. This is normal
authorized scheduler priority, not an administrative priority override.

Three sequential arms: true GO, train-constant GO, seeded-shuffled GO. Each uses
200 AdamW steps, batch 64, width 128, two layers, learning rate 0.0003,
weight decay 0.0001 and clipped gradient norm 1. Seeds and minibatch/time/control
sampling match across arms. The final checkpoint is evaluated once with 16-step
Heun integration, without clipping or checkpoint selection on H1.

W&B uses separate pinned tracking environments and separate offline runs per
arm. Scalar training steps and `validation/context/*` diagnostics are recorded;
no cloud sync, raw data, cell IDs or credentials are uploaded. Completion requires
valid model summaries/checksums and successful W&B receipts, not exit zero alone.
Setup and training refuse login-node execution, including inherited Slurm vars.

Entry points: `prepare_public_flow_pilot.py`, `build_go_target_features.py`,
`train_public_flow_pilot.py`, and `run_public_flow_pilot_suite.py` in `scripts/`.
The final combined regression run passed **472 tests in 8.88 seconds**. All of
these tests use synthetic data.

## Public-only post-training: next admission gate

Feng raw counts are acquired, but **post-training has not been run**. The new
`scripts/feng_authorized_csv.py` has 51 synthetic tests and separates authenticated
metadata/header preparation from count parsing. It scans compressed source
bytes but never numeric-parses unselected cell columns. It is not yet connected
to a registered real post-training cache or a parent-checkpoint loader.

The metadata audit found important requirements:

- Never attach metadata by position. Fitness/nonfitness RNA IDs require the
  exact `MP-` prefix mapping and both axes are reordered.
- Targeted `P1_I1_<barcode>` maps to `PC-P1-D3_I1_<barcode>` under its explicit
  source rule. One RNA column, `P2_I48_CAACCTCAGCCCAGCT-1`, has no metadata and
  must remain explicitly excluded. Do not infer its perturbation or control role.
- `unassigned` is not NTC. Fitness/nonfitness have only 36/12 explicitly labeled
  NTCs; targeted has 8,241. At >=8 control and >=8 treated cells per exact
  target/Cell_Line/Batch stratum, targeted has 183 eligible strata across 84
  targets before feature coverage, role selection and sampling limits.
- Cell_Line prefix before `_` is donor per the author processing code; do not
  equate every cell-line label with an independent donor. Targeted covers ten
  donors. Freeze donor/line/batch roles before fitting or selecting checkpoints.

Before post-training: bind the exact RNA/header/metadata/cell/gene allowlists;
preserve the 772 exclusions and public-only restriction; align the measured-gene
axis and normalization with the parent; authenticate both parent weights and
conditioning transforms; register held-out donor/context evaluation. The current
trainer intentionally rejects resume/posttrain rather than silently starting new
weights. Continue only after the conditioning ablations and distributional
diagnostics justify a candidate. No count decoder, full-axis challenge output,
or challenge submission is established by this pilot.
