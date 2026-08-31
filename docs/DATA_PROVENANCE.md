# Data Provenance and Measured-Data Firewall

## Purpose

This document records the origin, integrity, intended use, and redistribution
status of local data used by the project. Large data files and generated model
artifacts are excluded from Git. A filename in this document is not permission
to redistribute the underlying file.

## Challenge inputs

The 2026 validation controls are stored under `dataset/controls/` and were
downloaded from the Virtual Cell Challenge service. They contain only released
non-targeting control cells, the official 18,533-gene axis, and the official
300-target validation panel. These files may condition predictions and local
format checks. They do not contain measured post-perturbation challenge cells.

The Arc competition support set is stored under `dataset/state_support/`. It
contains the published training resources and ESM2 gene representations used
by the released STATE workflow. Its upstream terms, rather than this
repository's license, govern use and redistribution.

## Public perturbation sources

### Replogle et al. 2022

Source dataset:

- Figshare article: <https://doi.org/10.25452/figshare.plus.20029387.v1>
- Associated publication: <https://doi.org/10.1016/j.cell.2022.05.013>
- Figshare compilation license: CC BY 4.0

Locally verified files:

| Local file | Bytes | Upstream MD5 | Local SHA-256 | Role |
|---|---:|---|---|---|
| `dataset/raw/replogle_2022/K562_gwps_raw_bulk_01.h5ad` | 374,587,922 | `4570b53c9d62ff6df281e622f0350060` | `7cec96b3b76169abbf6b6ab9d10bf00d71d942d89e63292351f745e130b154db` | Direct same-target K562 response bank and challenge-panel response estimate |
| `dataset/raw/replogle_2022/K562_essential_raw_bulk_01.h5ad` | 79,766,954 | `8321d5d3ffc99db2a5c71edca4189735` | `80de95e54fcbca0e0537d569b43ec92fde6bd0482801505504baebe3118dcadf` | Source side of paired cross-context response calibration |
| `dataset/raw/replogle_2022/rpe1_raw_bulk_01.h5ad` | 95,350,546 | `74765fa87635467a869ea972356ae0e7` | `603c655f1cfa41d649baf3ae63fca224cc11f297e40d4ed59d390b1e8d2e2db2` | Recipient side of paired cross-context response calibration |

The files called `raw_bulk` contain per-population mean raw UMI abundance, not
integer single-cell matrices. They must be treated as pseudobulk expectations.
They must not be passed to code that requires integer single-cell observations.

The K562 genome-wide file has 11,258 populations by 8,248 genes, 9,867 unique
target labels, and 585 non-targeting populations. It directly covers 272 of the
300 released challenge targets and 7,681 symbols on the challenge gene axis.
These overlap counts are computed locally from gene symbols and are not claims
made by the upstream authors.

K562-essential and RPE1 contain 2,055 paired target labels across 7,225 shared
gene symbols. They are used to estimate how much of a K562 target residual is
stable in a second cell line. They are not used as if RPE1 were one of the
anonymous challenge contexts.

### Replogle-Nadig harmonized sources

Local files under `dataset/raw/replogle_nadig/` are retained for target and
context holdout experiments. In particular,
`GSE264667_hepg2_raw_singlecell_01.h5ad` contains raw integer single-cell counts
and is suitable for testing count emission and differential-expression behavior.
`replogle.h5ad` is an integrated normalized matrix and is not a substitute for
raw counts. Every derived artifact must record which representation was used.

The official GEO supplementary directory is
<https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/>. The
locally verified Jurkat file is:

| Local file | Bytes | Local SHA-256 | Role |
|---|---:|---|---|
| `dataset/raw/replogle_nadig/GSE264667_jurkat_raw_singlecell_01.h5ad` | 9,366,490,264 | `ffbe15f2c8f7ffcfd7b0ba9e6937d4ebc2d03b0179fa8234648a59bcb82c04a3` | Raw-count cross-cell-line response and count-decoder training |

The Jurkat AnnData matrix has 262,956 cells by 8,882 selected genes, 55
`gem_group` batches, 2,394 target labels, and 12,013 non-targeting cells.
Sampled nonzero values are finite non-negative integers. Matrix row totals are
slightly below the `UMI_count` metadata because the matrix contains a selected
gene axis rather than every counted feature. It overlaps 8,284 challenge-axis
genes but none of the 300 validation targets, so it is useful for held-context
response geometry, dispersion, and count emission, not direct panel-target
supervision. The GEO deposit and associated publication terms must be reviewed
before redistribution; public availability is not itself a repository license.

### Feng et al. multi-iPSC CRISPRi atlas

Sources and terms:

- Upstream count deposit: <https://doi.org/10.6084/m9.figshare.27989294.v2>
- Upstream study: <https://doi.org/10.1016/j.xgen.2025.101076>
- Preprocessed compilation:
  <https://huggingface.co/datasets/altoslabs/primeflow-vcc-datasets>
- Upstream deposit license: MIT
- Compilation license: CC BY 4.0 for its curation and preprocessing only

Locally verified files:

| Local file | Bytes | Local SHA-256 | Role |
|---|---:|---|---|
| `dataset/raw/feng2026/feng24_preprocessed.h5ad.gz` | 10,252,894,481 | `40a6c426bad1e8353c5ee8e72d2622bb4c7728d52162866481f0669644048bac` | Immutable downloaded archive |
| `dataset/raw/feng2026/feng24_preprocessed.h5ad` | 54,995,018,506 | `a7261df9fb26c57217c6313e85d9f6833ae29613462e9c7df78f86cfe147bc12` | Normalized-log1p response and representation training input |
| `dataset/raw/feng2026/LICENSE_feng24_MIT.txt` | 1,460 | `36491445e0d025b932512dcc31b0d228752fbde72f73ff556908a7be6ab17f9d` | Archived upstream notice from the compilation |

The AnnData matrix has 850,726 cells by 36,518 genes, 286 batches, 45
cell-line labels from 31 donors, 6,699 perturbation conditions, and 8,289
control cells. All cells are iPSCs collected at the same reported time point.
It contains 182 of the 300 validation targets as perturbation conditions and
18,120 of the 18,533 challenge genes on its expression axis. These are local
symbol-overlap calculations, not upstream performance claims.

`X` is a backed float32 CSR matrix with no raw slot and no count layer. Sampled
nonzero values are fractional, and inverse-log-transformed rows have an almost
fixed total while the raw-QC `total_counts` field varies. Therefore `X` is a
fixed-total normalized-log1p representation, not raw UMI counts. It must not
train the integer-count emitter or exact count-space DE statistics. It may
train a response or representation head only with an explicit dataset
transform/token. Evaluation folds must hold out complete target or ESM2
clusters and complete donor/cell-line groups before fitted preprocessing; a
`condition_line` group may not cross folds.

The upstream publication, MIT notice, compilation, and modification notice
must be cited together. The compilation's CC BY 4.0 label does not replace the
terms of the underlying dataset.

## Model-code license boundary

The PRiMeFlow source checkout is stored only under the Git-ignored `external/`
directory for configuration and license review. Its Academic Research License
contains public-availability and license-back requirements for derivatives.
No PRiMeFlow source is copied into this private repository. Any project-native
flow implementation must be independently written from general published flow
matching methods and reviewed before use.

## Measured-data firewall

The team reports receiving an official clarification that newly generated
experiments for released final contexts and targets may be used solely as
training or fine-tuning data when submitted cells are generated by an ML model
and no measured profile is inserted directly or manually used to revise a
prediction. The original written clarification must be archived by the team;
this document is not a substitute for the official message.

The implementation enforces the following policy:

1. Raw measured perturbation matrices are immutable inputs with recorded hashes.
2. Measured post-perturbation rows are never copied, resampled, or manually
   selected into a submission.
3. A submitted cell must be a function of a versioned model, released control
   input, target identifier, configuration, and recorded random seed.
4. Training, validation, and final generation are separate commands and
   directories. The generation command cannot accept a measured perturbed-cell
   matrix as a sampling source.
5. Pseudobulk priors are model parameters, not submission rows. The count
   renderer generates new integer cells and records the model and prior hashes.
6. Human inspection may reject an artifact for safety or schema errors, but may
   not manually edit target-specific expression values.
7. Every submitted artifact receives an immutable manifest containing code,
   data, model, configuration, and output hashes.

## Required pre-promotion checks

- Verify the upstream license and citation for every new dataset.
- Verify byte size and an upstream checksum when one is published.
- Inspect expression scale, sparsity, gene identifiers, target identifiers,
  controls, batches, and replicate structure before fitting.
- Fit all learned preprocessing inside each held-target or held-context fold.
- Report direct target overlap separately from embedding-based extrapolation.
- Run the complete scorer-aligned validation suite before allocating multi-GPU
  training or a submission quota.
