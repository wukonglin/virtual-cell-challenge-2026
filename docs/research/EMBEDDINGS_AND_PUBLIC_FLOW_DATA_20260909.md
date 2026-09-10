# Embeddings and public data for context-aware cell-flow training

Date: 2026-09-09. Status: evidence-backed research recommendation, **not a new
training run, frozen protocol, download receipt, or promotion decision**.
The v3/v4/v5 results and existing data exclusions remain unchanged.

## Decision

Lingshu is optional, not a dependency. Prioritize **GenePT gene-function features,
versioned GO features, and representations learned from public perturbation
responses**. Retain protein-sequence ESM2 as a baseline. Encode recipient context
separately from its control cells, initially comparing a simple control-summary
baseline against frozen STATE SE-600M. Test fusion rather than assuming it helps.

The organizer describes the 2026 task as transfer into contexts whose treated
cells are unavailable, and permits training on public data. Therefore public
same-target responses in other contexts are useful information, not automatically
forbidden target leakage. A target-held-out K562 screen tests a different,
secondary axis; it cannot decide all cross-context strategies. Challenge-treated
cells remain excluded. [Official 2026 task](https://arcinstitute.org/news/virtual-cell-challenge-2026)

Published benchmarking found cross-system perturbation pretraining useful within
its tested tasks, while several larger models did not consistently beat simple
baselines. This motivates response-informed features and strong baselines, not a
claim that any model will win this challenge. [Ahlmann-Eltze, Huber and Anders,
Nature Methods 2025](https://www.nature.com/articles/s41592-025-02772-6)

## 1. Embedding shortlist

| Candidate | Role and reason to test | Access and important qualification |
| --- | --- | --- |
| **GenePT** | Target-gene function, using descriptions rather than gene-symbol-only prompts. First new target-feature arm. | Released NCBI/UniProt summaries and precomputed vectors; no paid API or GPU needed to reuse them. Zenodo data CC BY 4.0. Coverage/aliases still need inspection; this is not a ready perturbation-response model. [Repository](https://github.com/yiqunchen/GenePT), [release](https://doi.org/10.5281/zenodo.10833191). |
| **GO features / GEARS-style graph** | Target function and related-gene structure. Start with sparse, versioned annotation features; add a learned graph only if useful. | GO data CC BY 4.0; GEARS code MIT. Missing annotations must be explicit. GEARS itself does not supply cross-cell-type prediction as-is. [GEARS](https://github.com/snap-stanford/GEARS), [GO citation/version policy](https://geneontology.org/docs/go-citation-policy/). |
| **ESM2-650M** | Auditable target-protein sequence reference, not presumed better than v5's Arc-supplied vectors. | `facebook/esm2_t33_650M_UR50D`, MIT, 33 layers and 1,280 hidden features. Requires protein/isoform mapping and a long-sequence policy; noncoding targets need another branch. [Model card](https://huggingface.co/facebook/esm2_t33_650M_UR50D), [extraction code](https://github.com/facebookresearch/esm/blob/main/scripts/extract.py). |
| **Public-response representation** | Learn target-response programs from allowed public CRISPRi profiles across contexts; compare same-target measured-source transfer separately from unseen-target extrapolation. | Fit only on each experiment's training sources. Never reuse a representation trained on a reserved recipient's responses. The supporting benchmark is narrow, not a general performance guarantee. [Study](https://www.nature.com/articles/s41592-025-02772-6). |
| **STATE SE-600M** | Frozen control-cell embeddings and a pooled control-context summary; separate from target identity. Compare with control pseudobulk/PCA. | Public model, 2,048-dimensional cell representation; model has noncommercial terms. Bind checkpoint, config, protein table and code. No assumption that good cell embeddings predict responses. [Model](https://huggingface.co/arcinstitute/SE-600M), [terms](https://huggingface.co/arcinstitute/SE-600M/blob/main/MODEL_LICENSE.md). |
| **UCE or scGPT** | Secondary control-state alternatives if STATE does not improve public context transfer. Not an initial sweep of every foundation model. | UCE expects counts and has incompatible 4-/33-layer embedding spaces; its repo is MIT. scGPT needs its matched vocabulary, checkpoint and preprocessing; weight terms/hashes remain to verify. [UCE](https://github.com/snap-stanford/UCE), [scGPT model zoo](https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo). |

Useful immutable metadata verified without downloading weights:

- ESM2-650M revision: `08e4846e537177426273712802403f7ba8261b6c`.
- STATE SE-600M revision: `5a9a80f44f7ce32ce57059933ef0d735d7c10ce5`.
- GEARS code revision: `f374e43e197b295016d80395d7a54ddb81cc6769`.
- GenePT release file `GenePT_emebdding_v2.zip`: 574,395,233 bytes; publisher MD5
  `3f6ce4317e3a0091978ae5cb8fbf05a3`. Authenticate before use and do not blindly
  deserialize legacy pickle dictionaries. Data rights do not imply a license
  for every notebook in the source repository. Released vectors are reproducible
  artifacts; their hosted upstream embedding models do not have downloadable weights.
- ESM2's extraction script defaults to truncating at 1,022 residues. Freeze
  sequence release, stable gene IDs, isoform mapping, pooling and long-protein
  handling explicitly. A newly extracted 1,280-D table is a new feature lineage,
  not an authenticated reconstruction of the opaque 5,120-D Arc table.

The GenePT paper is a gene/cell embedding study, not proof of CRISPRi generation
quality. Its published version is Chen and Zou, Nature Biomedical Engineering,
online December 2024 / volume 9 (2025), DOI `10.1038/s41551-024-01284-6`.
[Published paper](https://www.nature.com/articles/s41551-024-01284-6)
GenePert is a separate regularized response-prediction application, not a flow
model; its README's modality labels must be checked against original studies.
[GenePert](https://github.com/zou-group/GenePert)

## 2. Public dataset priorities

| Priority/source | Intervention and useful scope | Representation, access and intended role |
| --- | --- | --- |
| **Core: Replogle 2022** | Human CRISPRi, genome-wide K562 plus essential-gene K562/RPE1; broad target supervision. | Official Figshare provides source data under CC BY 4.0. Our local `raw_bulk` files are population means, useful for response embeddings but **not individual-cell flow targets**. Obtain/verify raw single-cell counterparts for distribution training. [Deposit](https://doi.org/10.25452/figshare.plus.20029387.v1), [Cell study](https://doi.org/10.1016/j.cell.2022.05.013). |
| **Core: Nadig / GSE264667** | Human CRISPRi; adds Jurkat and HepG2 to the K562/RPE1 comparison. | Raw single-cell H5AD files for Jurkat/HepG2 already exist locally. Suitable for matched-control transport and count calibration after exclusions. GEO availability is not a blanket redistribution license; retain original attribution/terms. [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667), [supplementary files](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE264nnn/GSE264667/suppl/). |
| **High-value addition: full VCC 2025 H1 hESC** | CRISPRi, roughly 300,000 cells and 300 targets; another biological context. The old train/validation/test responses are now public. | Arc documents count H5ADs and CC0. Do not confuse these with still-hidden 2026 treated cells. Current official access uses Marketplace/Requester Pays; billing authorization is unresolved. [Dataset layout](https://raw.githubusercontent.com/ArcInstitute/arc-virtual-cell-atlas/main/virtual-cell-challenge/README.md), [Arc license/release](https://arcinstitute.org/tools/virtualcellatlas). |
| **High-value addition: Feng multi-iPSC** | CRISPRi across donor/cell-line backgrounds; useful donor transfer and a broad gene axis. Primary methods identify dCas9-KRAB-MeCP2, resolving the earlier modality uncertainty. | Local curated file has 850,726 cells, 45 line labels and 31 donors; these are compilation counts, not a count of independent tissues or a sum of publication screens. It is normalized log1p, not counts. Upstream count deposit is MIT; compilation attribution also applies. [Study](https://doi.org/10.1016/j.xgen.2025.101076), [full-text record](https://pmc.ncbi.nlm.nih.gov/articles/PMC12903452/), [count deposit](https://doi.org/10.6084/m9.figshare.27989294.v2). |
| **Next-scale addition: KOLF2.1J atlas** | Human iPSC CRISPRi; 11,692 perturbed genes and over 2.5M cells, published July 2026. A potentially valuable extra source/context after overlap checks. | CC BY 4.0 Figshare deposit; full QC AnnData is about 189GB. A roughly 4.64GB chromatin-modifier QC subset can support a narrow pilot, not an unbiased genome-wide benchmark. Inspect X/layers before calling them raw counts; avoid selecting the outcome-filtered Strong Perturbations subset for validation. [Paper](https://www.nature.com/articles/s41587-026-03199-w), [deposit](https://doi.org/10.25452/figshare.plus.27261219.v1), [author portal](https://y-doctor.github.io/KOLF2.1J_Perturbation_Cell_Atlas/). |
| **Small genetic controls: Adamson / Norman** | Adamson supplies CRISPRi stress-response tasks; Norman supplies CRISPRa single/combinatorial perturbations. | Useful method/sampler checks, not new independent contexts if both are K562. Keep CRISPRa distinct from knockdown. Obtain original counts when testing distributions. [Adamson GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546), [Norman GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344). |
| **Optional: Parse 10M PBMC** | Approximately 9.7M cells, 12 donors, 90 cytokines plus PBS; donor/context and heterogeneous-population learning. | Publisher 41GB H5AD is separate from pseudobulk outputs; CC BY-NC 4.0. Cytokines are not CRISPRi. CellFlow's convenient processed example is restricted to 2,000 HVGs. Start with selected donors/conditions, not the complete experiment. [Publisher](https://www.parsebiosciences.com/datasets/10-million-human-pbmcs-in-a-single-experiment/), [count workflow](https://cdn.parsebiosciences.com/gigalab/10m/Parse_10M_PBMC_cytokines_dask_workflow.html). |
| **Optional later: Tahoe-100M** | Chemical perturbations in 50 cancer cell lines; broad context coverage rather than direct genetic knockdown supervision. | Publisher HF release exposes about 95.6M raw-count rows in Parquet, CC0, roughly 338GB of expression shards. Use selected conditions and plate-matched DMSO controls; initial gene/expression CLS markers are not counts. Streaming can still transfer many bytes. [Publisher card](https://huggingface.co/datasets/tahoebio/Tahoe-100M). |

Catalogs such as scPerturb/PerturBase are discovery and harmonization resources,
not independent new experiments to add on top of the original studies. Match
accession, donor, guide and cell identifiers across copies before combining.
Likewise, raw Feng data and its PRiMeFlow compilation are overlapping sources,
not separate pretraining and test cohorts. [Compilation](https://huggingface.co/datasets/altoslabs/primeflow-vcc-datasets)

### Practical first acquisition batch (proposed, not downloaded)

The official file listings expose manageable raw-count additions. Start here
before Tahoe or the complete KOLF2.1J corpus; reuse existing Nadig files.

| File | Publisher-listed bytes | Direct source |
| --- | ---: | --- |
| Replogle K562-essential raw single-cell H5AD | 10,661,879,995 | [Figshare file 35773219](https://ndownloader.figshare.com/files/35773219) |
| Replogle RPE1 raw single-cell H5AD | 8,700,873,216 | [Figshare file 35775606](https://ndownloader.figshare.com/files/35775606) |
| Feng genome-wide fitness RNA counts CSV.gz | 1,445,924,890 | [Figshare file 51059483](https://ndownloader.figshare.com/files/51059483) |
| Feng genome-wide nonfitness RNA counts CSV.gz | 1,748,940,996 | [Figshare file 51059558](https://ndownloader.figshare.com/files/51059558) |
| Feng targeted-screen RNA counts CSV.gz | 4,480,969,391 | [Figshare file 51072902](https://ndownloader.figshare.com/files/51072902) |

The Feng files are **RNA-UMI**, not merely guide-UMI counts. Retrieve the linked
cell/guide metadata as well and distinguish no-guide cells from verified NTCs.
Compressed transfer sizes are not estimates of conversion RAM or expanded disk
requirements. Replogle's genome-wide raw single-cell file is an additional
65,830,941,948 bytes, so defer it until the smaller multi-context pilot works.
[GWPS file](https://ndownloader.figshare.com/files/35775507)

Publisher MD5s for the three Replogle files (K562-essential, RPE1, GWPS):
`4f1122ce1c7f13299a68df6459a266d3`, `6a2a9d0d2bf4ec147f4d1104043b268c`,
`887e3e6a8c8df6eadf7a3030a53c9546`. Hash actual retrieved bytes before deriving
training material; also record SHA-256 and the exact source-listing version.

### Access boundary

Arc's legacy GCS buckets have been retired. Its current atlas requires a
subscribed Google Cloud billing project for the advertised monthly allowance;
ordinary requester-pays access can incur charges. The relevant prefix is
`gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/`, **not** the
entire atlas bucket. No project was subscribed, no billing identity used, and
no transfer started. Prefer already available files or publisher-hosted public
downloads while that access remains unresolved.
[Current official access notice](https://raw.githubusercontent.com/ArcInstitute/arc-virtual-cell-atlas/main/README.md)

### Local starting points (not freshly re-hashed in this research turn)

- `dataset/raw/replogle_2022/`: K562 GWPS, K562-essential and RPE1 pseudobulk files.
- `dataset/raw/replogle_nadig/GSE264667_jurkat_raw_singlecell_01.h5ad` and
  `GSE264667_hepg2_raw_singlecell_01.h5ad`: real single-cell sources.
- `dataset/raw/feng2026/feng24_preprocessed.h5ad`: broad normalized response data;
  do not round or invert-normalize it into fabricated UMI counts.
- Registered local hashes, observed-gene overlap, and original notices are in
  `docs/DATA_PROVENANCE.md`. File sizes/counts here are inventory, not fresh QC.
- No additional full 2025 H1, Parse or Tahoe corpus was identified by the bounded
  filename/documentation search. This is not an exhaustive filesystem proof.

## 3. Recommended pretraining/post-training sequence

This is a proposed experiment design, not a claim about a released checkpoint.

1. **Register sources and splits first.** Primary evaluation: hold out a complete
   public recipient context's treated cells and infer its context only from
   controls. Separately test unseen target families, and both axes together where
   coverage permits. Record whether a target's measured response exists in other
   training contexts. Keep donor/batch/guide units intact and audit pretrained
   model/data exposure. Reused HepG2/Jurkat/RPE1 results remain development data.
2. **Compare inexpensive response heads.** Use matched-target GenePT, GO and ESM2
   arms, plus a public-response-transfer arm with its distinct information budget.
   Evaluate target features and recipient-control features separately before
   fusion. Include no-change, trained mean effect and shuffled-feature baselines;
   fit PCA/scales/decoders inside training folds. Do not require success on the
   old K562-only unseen-target task before considering a different registered
   cross-context hypothesis.
3. **Pretrain a small context-aware flow.** Use actual control/treated cells from
   allowed public CRISPRi sources, with cell-level state plus pooled context,
   target features, intervention type and assay information. Balance sources and
   targets so GWPS does not dominate simply by size. These are unpaired population
   samples; optimal-transport matching is not a measured same-cell before/after
   pair. Do not erase the control branch or train only a deterministic mean shift.
4. **Post-train on public data only.** Initially freeze feature encoders and
   calibrate transport/decoder/library behavior on allowed public training
   contexts. Use low-rate unfreezing only as a registered ablation. Count heads
   require real counts, an explicit gene/library axis, and missing-gene masks.
   A normalized-expression head may use Feng's local representation but must not
   be mislabeled a count generator. No recipient treated cells for a claimed
   unseen-context test may enter this stage.
5. **Gate generation, not just mean fit.** Require useful context transfer and
   target discrimination alongside distribution checks: variance ratios,
   covariance/effective rank, MMD or energy distance, cell-state proportions,
   library sizes, zeros, clipping and duplicate samples. Report missing output
   genes; 512/2,000-HVG success is not full-axis challenge readiness. Preserve
   existing public quality gates; independently calibrate new ones before use.

For the first flow prototype, use **CellFlow-style continuous perturbation and
control-context conditioning**, not an unchanged scDFM/PRiMeFlow checkpoint.
CellFlow's PBMC example uses protein-based cytokine features and donor-control
summaries, but its showcased split tests new combinations, not fully unseen
donors and cytokines simultaneously. Its evaluation PCA fitted on all cells
must not be copied into a locked test. A PRiMeFlow-inspired full-gene stochastic
decoder is a later hypothesis, subject to the recorded source-license boundary.
[CellFlow example](https://cellflow.readthedocs.io/en/latest/notebooks/100_pbmc.html),
[PRiMeFlow paper](https://arxiv.org/abs/2604.13986),
[local code-license boundary](../DATA_PROVENANCE.md#model-code-license-boundary)

The older v5 lineage used 14 members of V7-excluded clusters. Do not reuse that
lineage and claim V7's reserved protocol is clean. A new route preserving that
protocol must enforce the complete global exclusions before expression reads
and fitting. The old bulk builder reads full chunks before masking and is not
an authorized implementation of that new read-isolation contract.
[Prior audit](../PUBLIC_BIOLOGICAL_NEXT_STEPS.md)

## 4. Compute and recording

Precomputed GenePT/GO comparisons and metadata/QC need CPU compute. One allocated
Anvil H100 is a reasonable *starting proposal* for frozen cell embeddings or a
small flow pilot, not a measured runtime guarantee. Profile before multiple
H100s; no work, installs, tests or preprocessing on a login/head node. No new
Slurm job was submitted for this research request.

Use the existing W&B wrapper for each new `pretrain`, `posttrain` and `validation`
stage; offline until an authorized destination is configured. Bind data/model
hashes, stage lineage, folds and feature coverage in local receipts. Emit losses,
learning rate, validation and collapse diagnostics; add tests before extending
the logger's allowlist. Raw data, identifiers and weights stay out of telemetry.
Do not relabel previous completed runs or upload all old offline directories.

## 5. Search and evidence notes

Used the academic-search multi-source workflow. Specialized MCP literature
connectors were unavailable; the supplied fallback searched OpenAlex metadata,
and primary publications, official repositories, release APIs, GEO/Figshare and
publisher dataset cards were checked independently. PubMed, CrossRef and arXiv
connectivity checks passed. Discovery queries covered cell-flow perturbation
models, GenePT and multi-donor CRISPRi; irrelevant broad-query results were
discarded. GenePT's preprint and published study are not counted as independent
evidence. No ranking was based on citation counts or repository popularity.

Some PMC/GEO web views presented a browser challenge, and one Zenodo web view
timed out; indexed primary text and official metadata endpoints supplied the
specific claims above. Those access limitations are not evidence that files
were downloaded or their contents newly validated. This is a targeted shortlist,
not a systematic review or a prediction of leaderboard gains.
