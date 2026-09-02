# Next-Route Primary-Source Audit

Access date for every external source below: **2026-09-02** (America/New_York project time).
Repository state at audit time: branch `experiment/v7-scdfm-gamma1`, commit
`fbb12a06ae1755d799c8961c67792e81a2f14601`, clean worktree, 324 passing tests.

## Method

Nine external primary-source clusters and seven local read-only audits were run in
parallel. Every research cluster was then handed to an independent adversary whose
instructions were to refute it, to re-fetch every source, and to re-check every commit
sha, revision sha, sha256, licence identifier, parameter count, date and numeric
hyperparameter against the live source. Where the researcher and the adversary
disagree, this document follows the adversary and says so inline.

Label key:

- `PRIMARY-SOURCE FACT` — fetched and read during this audit **and** independently
  re-confirmed by the adversary;
- `LOCAL REPOSITORY FACT` — established by read-only inspection of this checkout;
- `INFERENCE` — a conclusion drawn from the above;
- `UNRESOLVED` — not established. No unresolved field is filled with a guess.

Sources that could not be read are recorded as failures rather than paraphrased.
Notable fetch failures: OpenReview `QSGanMEcUV` returned HTTP 403 with a JavaScript
challenge shell; `cell.com` returned a Cloudflare interstitial; `virtualcellchallenge.org`
serves a client-rendered shell, so its content was recovered from the same-origin
Next.js RSC flight payload and served chunk bundles rather than from the rendered page.

---

## 1. What the competition actually scores

This section is load-bearing for every route decision that follows, and it corrects
two errors carried in this repository's own prose.

`PRIMARY-SOURCE FACT` — The validation contract on the live site matches this
repository's recorded contract exactly: three anonymised contexts, 300 perturbations
shared across all three, 400 cells per perturbation, 360,000 cells, 18,533 genes,
non-negative finite whole numbers, and zero `non-targeting` rows in the artifact.

`PRIMARY-SOURCE FACT` — **A/B/C is only the leaderboard round.** Prizes are decided
solely on a second, unseen round in contexts D/E/F. The final-test perturbation panel
and the D/E/F non-targeting control profiles are released **2026-10-22**; the final
entry is due **2026-11-05 23:59 UTC**; no leaderboard is displayed between those dates;
only the last final entry counts. Quota is two scored submissions per team per day,
renewing at midnight UTC, with one submission in flight at a time. `vcc cancel` is free.

`PRIMARY-SOURCE FACT` — **`fid` is not Fréchet Inception Distance.** The string
"Fréchet" does not occur anywhere on the site. The six scored components are:

| Leaderboard name | cell-eval2 raw key | What it measures |
|---|---|---|
| `pds` | `pds_cosine` | Perturbation discrimination: per-target cosine ranking of the predicted pseudobulk effect against every target's real effect |
| `mse` | `expr_mse_unbiased_capped_norm` | Expression accuracy: one panel-level noise-corrected ratio, with no per-perturbation value |
| `nmae` | `de_wilcoxon_lfc_nmae` | DE log-fold-change accuracy |
| `fid` | `de_wilcoxon_direction_fidelity_yield_raw` | DE **direction fidelity**: of the genes the prediction calls significant, the share moving the reference's direction, scaled by yield |
| `reach` | `de_wilcoxon_direction_reach_raw` | DE direction reach: directional-purity depth `k*/n_real` where `k*` is the deepest prefix holding precision ≥ 0.9 |
| `jac` | `de_wilcoxon_sig_jaccard` | DE significance-set overlap |

`PRIMARY-SOURCE FACT` — **None of the six is a distribution-to-distribution
divergence.** All six reduce a group's 400 cells either to a count-summed pseudobulk
profile or to a two-sided Wilcoxon rank-sum DE table. Effort aimed at generative
realism per se is not scored.

`PRIMARY-SOURCE FACT` — Normalisation is reference scaling `s = (u - b) / (r - b)`,
per metric per context, where `b` is the **cell-context mean baseline** — the same
average profile predicted for every perturbation — and `r` is a real half-replicate
anchor averaged over five disjoint splits. The overall score is an unweighted mean over
six metrics and three contexts. **A scaled score of 0 therefore means "no better than
predicting the context mean for every target".** Only the scaled values are
higher-is-better and comparable; raw `mse` and raw `nmae` are lower-is-better.

`PRIMARY-SOURCE FACT` — Published per-context anchors for `pds_cosine`: `b = 0.500`,
`r = 0.927–0.984`. For the other five, `b`/`r` are: `expr_mse_unbiased_capped_norm`
0.986–0.992 / 0.028–0.045; `direction_fidelity` 0.505–0.522 / 0.795–0.832;
`direction_reach` 0.047–0.097 / 0.958–0.978; `sig_jaccard` 0.021–0.037 / 0.375–0.423;
`lfc_nmae` 1.0009–1.0017 / 0.369–0.431.

`PRIMARY-SOURCE FACT` — Differential expression is two-sided Wilcoxon rank-sum on
counts normalised per cell to 1e6, Benjamini–Hochberg corrected within each
perturbation at alpha 0.05, after a 5 CPM low-expression gate read from the
**reference's** control cells only. The perturbed gene's own row is excluded from all
six metrics, and `pds_cosine` additionally removes **all 300 panel target genes** from
every distance before any distance is computed.

`PRIMARY-SOURCE FACT` — External data is broadly permitted: "As long as you have the
appropriate permissions to use the data, you may use any data to improve the
performance of your predictions." There is no rule phrased as "no challenge
perturbation labels in training", because none are released; the actual rule is scoped
to the entry ("your predictions must be generated solely by one or more machine
learning models you use"), and explicitly allows literature and laboratory results to
*train* a model.

`PRIMARY-SOURCE FACT` — Rejection conditions beyond the shape contract: exactly 400
cells per perturbation, exactly the `pert_counts.csv` set, gene symbols not construct
ids, raw integer counts, no cell above 1,000,000 total counts, at most 400,000 cells,
at most 4,750,000,000 **stored** entries in `.X` (a dense array is 6,671,880,000 =
1.40× the cap), and zero `non-targeting` rows.

`PRIMARY-SOURCE FACT` — Current scorer: `ArcInstitute/cell-eval2`, MIT, one tag
`v0.16.0` published 2026-08-20, main HEAD `5e64833518a6603a0301cbe28185d49c30f4a986`
— which is the commit this repository already pins. The vendored
`docs/vcc2026_metrics/vcc2026-metrics-brief.md` is byte-identical to upstream at that
commit (18,162 bytes). `vcc-cli` 0.2.0 was published to PyPI on 2026-09-01; the wiki
instructs re-running `vcc skill install` after every CLI upgrade.

`PRIMARY-SOURCE FACT` — The in-force leaderboard anchor set is
`vcc2026-valA-r4+vcc2026-valB-r4+vcc2026-valC-r4` on panel `vcc2026-val-1`, on all 50
live rows. The r3→r4 move is a **version re-stamp**, not a re-measurement: the release
notes state "Nothing about the scoring rule changed — rule_version stays 3 and the rule
digest is bit-identical". The section-8 `b`/`r` tables are therefore in force.

`INFERENCE` — Two site statements about Arc's STATE stand in tension and are not
reconciled anywhere in the published text. The Evaluation page's Resources card says
STATE "is intended as a starting benchmark and is not able to win the competition";
the FAQ says "Any entrant may use Arc's State and/or Stack model code in order to
participate in the Challenge (and such use will be considered a Non-Commercial Purpose
solely to the extent of your use for participation in the Challenge)".

**RESOLVED, 2026-09-02, by the team's own enquiry to the organiser: STATE may be used in
the competition.** The parenthetical on the Evaluation page describes Arc's own benchmark
entry, not a restriction on entrants who build on STATE. This closes the eligibility
question and unblocks the STATE-anchored route. Two narrower items remain open and are
Arc licensing matters rather than organiser matters: whether "use as a prior" under Arc
`MODEL_LICENSE` §1.2 makes a submission a Derivative Work, and the §1.4 Commercial Entity
determination. The FAQ's own carve-out — participation counts as a Non-Commercial Purpose
— addresses most of the second. Record the organiser's answer in writing when available;
this entry rests on the team's report of the exchange, not on a document this audit read.

---

## 2. Method matrix

Nineteen registered columns, one row per method. `UNR` = UNRESOLVED.

### 2.1 scDFM

| Column | Value |
|---|---|
| Primary paper | arXiv:2602.07103v1, 2026-02-06, only version, CC BY 4.0. "ICLR 2026 poster" rests solely on the author-supplied arXiv comments field and the repo README; OpenReview `QSGanMEcUV` was unreachable (HTTP 403) — `UNRESOLVED` |
| Repository / pin | `AI4Science-WestlakeU/scDFM`, main HEAD `2cf6bca1f044e74c4e1dc586892c0495880cf125`, tree `a04130b07020505a609158cbe31e9e65083c0d79`; no tags, no releases. The local pin **is** the upstream tip, byte-exact across all 64 blobs |
| Checkpoint | `scDFM_ckpts.zip`, Google Drive, 5.5 GB; no version, no upstream checksum, no model card |
| Code licence | MIT, `LICENSE` sha256 `dedf7a6ce9cb9f1062d6f387a6a164320a442d61591627f81ce6aedc7c8e2fdb`; scope is the 64 repo files only |
| Model licence | `UNRESOLVED` — no licence accompanies the Drive weights |
| Dataset licence | `UNRESOLVED` — Figshare/Drive mirrors, no terms stated |
| Perturbation mechanism | **CRISPR activation** on Norman/K562, plus small-molecule drugs on ComboSciPlex/A549. **Zero CRISPRi anywhere** |
| Training contexts | One cell line per model, trained separately: K562 and A549. No cell-line covariate; Appendix A.6 concedes the exclusion of multi-cell-line resources |
| Target encoding | Perturbation embedding **shares parameters with the gene-identity embedding**; no sequence, ontology or network features |
| Context encoding | The raw control cell's log1p vector, tokenised per gene, injected as key/value of cross-differential attention at every layer; no learned context token |
| Output space | log1p(CP10k) continuous. Training space ≈5,029 genes; a **fresh random 1,000-gene subset per step**; evaluation on 1,000 HVG selected on `adata_test`; output `clamp(min=0)`. A full-vocabulary head is "not used in our experiments" |
| Objective | `L = L_CFM + 0.5·L_MMD`; unbiased multi-kernel Gaussian-RBF MMD², bandwidths `σ = √(s·m)` with `s ∈ {0.5,1,2,4}`, median heuristic recomputed per step; computed on one-step endpoints, not an ODE rollout |
| Demonstrated zero-shot axis | **Unseen CRISPRa target identity and unseen dual combinations, within K562**; plus held-out drug pairs within A549. Each inside a single seen context. **No cross-cell-context axis**, conceded in A.6 |
| Unsupported VCC assumptions | Raw integer counts; 18,533-gene axis; CRISPRi; unseen contexts; 400 cells per group (`test()` hardcodes N=128); out-of-vocabulary targets (`Vocab.encode` raises `KeyError`) |
| Required adaptation | Full-axis decoder; count emitter; retrain on CRISPRi (the gene-tied embedding carries a knock-**up** sign prior); replace the target encoding with continuous features; build a genuine validation split and checkpoint rule (neither exists upstream); rebuild the co-expression graph inside each fold |
| Local reproduction | Vendored read-only at the pin in two paths, MIT present. The released `holdout/fold0` checkpoint strict-loads on one H100 (job `861799`, 8 s): 219 tensors, 60,359,169 parameters, no missing or unexpected keys. **No quality evaluation has been run** |
| Leakage risks | See §3 — the upstream validation split is constructed from the test split |
| Compute / storage | Compact registered configuration is one H100; the released checkpoint archive is 5.9 GB locally |
| Recommendation | **residual, conditional on evidence** — see the decision memo |

Paper-versus-code reconciliation, all `PRIMARY-SOURCE FACT`:

- The paper reports batch size 96, 100,000 steps, width 512, four layers, eight heads.
  `run.sh` carries `--steps=200000`. The two disagree and neither was adopted silently.
- The paper never defines the source distribution `q₀`. The code resolves it to a
  standard Gaussian, so the flow starts from **pure noise**, with the control entering
  only as conditioning. The alternative log-normal-Poisson source is dead code
  (`utils.py:54` guard is always true). A claimed `per_cell_L = 1e4` is **not** in
  force: `config_flow.py` sets `poisson_target_sum = -1`.
- Per-layer conditioning is **fourfold**, not threefold: perturbation MLP adapter,
  residual self-differential attention, residual cross-differential attention against
  the control, plus adaLN-Zero modulation by the time embedding at every layer.
- The live training sampler is `PerturbationDataset.__getitem__` (`data.py:374-398`),
  **not** the vendored CellFlow `TrainSampler` in `_dataloader.py`, which `data.py`
  never imports. One perturbation is drawn uniformly per step; treated and control
  cells are drawn with replacement, controls from a single fixed global pool. The port's
  operative conclusion — one condition per step, unpaired, population-versus-population
  MMD — survives; the mechanism differs from the paper's description.
- There is **no `holdout` value of `split_method` anywhere in the code**. Which value
  produced the released `ckpts/holdout/` directory is `UNRESOLVED`.

### 2.2 Arc Institute STATE — State Transition

| Column | Value |
|---|---|
| Primary paper | Adduri et al., bioRxiv `10.1101/2025.06.26.661135` v2, 2025-07-10. **Still a preprint**, no peer-reviewed venue. Neither v1 nor v2 abstract contains the phrase "zero-shot" |
| Repository / pin | `ArcInstitute/state`, main HEAD `9bbfe78a434a55205e4de834e1ea99f85f7a3add`, 2026-07-24, GPG-verified. The local vendored copy is byte-identical |
| Checkpoints | HF `ST-HVG-Replogle` rev `bb6a9562cbbf1fd152df14cc53b4cc7517c77175`; `ST-SE-Replogle` rev `e324967ff4cea5ec199e29bcbb5c1f00e5b9d69c`. 195 files each = zeroshot/fewshot × {hepg2, jurkat, k562, rpe1}. **No model card on any Replogle repo** — README returns 404 |
| Code licence | CC BY-NC-SA 4.0, `LICENSE` sha256 `e66c269d4819aaab34b49ef5220c4ddab6756f21bb5180761a4eb8561f2b7bbd`, 20,850 bytes. GitHub's API classifies the repo as `NOASSERTION` |
| Model licence | Arc State Model **Non-Commercial** Licence + acceptable-use policy. §1.2 defines Derivative Work to **expressly include "use as a prior"**; §1.4 says sponsorship by a Commercial Entity defeats non-commercial status. **Outputs are carved out** — the AUP states "you are free to use any outputs you create for any purpose" |
| Dataset licence | `UNRESOLVED` — Replogle-Nadig, Tahoe and the VCC support set terms are not published in the repositories read |
| Perturbation mechanism | CRISPRi knockdown, gene-level |
| Training contexts | Replogle K562/RPE1/Jurkat/HepG2, plus k562_gwps, Tahoe, Parse. **Context is not an embedded covariate** — it enters only through the sampled real control cells and a batch embedding |
| Target encoding | **Released checkpoints use one-hot.** `pert_rep: onehot`, `perturbation_features_file: null` in all four released zero-shot repositories. `pert_onehot_map.pt` is literally `np.eye(2024)` with 2,023 symbols plus `non-targeting`, and **0 of the 300 VCC targets** appear in it, exactly or case-insensitively. The framework does support `perturbation_features_file` with continuous features — the path Arc's own VCC recipe uses with `ESM2_pert_features.pt` |
| Context encoding | Basal control cells sampled from the target context, plus a learned batch encoder over `batch_dim = 56`. A closed four-way cell-type one-hot `{hepg2, jurkat, k562, rpe1}` exists and has no row for A/B/C |
| Output space | Released ST reads out 2,000 HVG, log-normalised. Only 1,877 of those 2,000 are on the VCC axis, and only 57 of the 300 VCC targets appear in the readout. `output_space: all` writes the full support axis; outputs are float32 clipped in place to `[0.0, 14.0]` with **no rounding**. A counts head exists (`FinetuneVCICountsDecoder`) but is disabled by default and is itself continuous (`relu`) |
| Objective | geomloss `SamplesLoss(loss='energy', blur=0.05)` over cell sets; unmatched random basal pairing; no likelihood, no count head |
| Demonstrated zero-shot axis | **Unseen cell context with SEEN perturbations**, with the held-out context's own controls available. This is the only one of the six methods whose demonstrated axis matches what the organiser says is being tested |
| Unsupported VCC assumptions | Unseen target identity under the released checkpoints; raw integer counts; the 18,533-gene axis |
| Required adaptation | Continuous target features with held-target validation; gene-axis projection; a count renderer |
| Local reproduction | Trained locally as `state_sm` (job `859832`), 162,626,484 parameters, 20,000 steps, checkpoint selected at step 16,000 on validation loss 1.666636586 |
| Leakage risks | The published zero-shot setting still trains on the same perturbations in other contexts; the on-target knockdown-efficacy filter in the Replogle preprocessing reads perturbed outcomes |
| Compute / storage | One H100 for training and generation; 2.86 GB for SE-600M if used |
| Recommendation | **primary** — it is the only method demonstrating the organiser's axis, and it is already the local incumbent |

### 2.3 Arc Institute STATE — State Embedding (SE-600M)

`PRIMARY-SOURCE FACT` — HF `SE-600M` rev `5a9a80f44f7ce32ce57059933ef0d735d7c10ce5`;
**715,062,442** float32 parameters over 238 tensors, despite the name. Not
perturbation-trained: masked-expression self-supervision over 36,238,464 cells from
14,420 datasets. Genes are tokenised by ESM2 protein embeddings;
`pe_embedding.weight` has shape `[19790, 5120]`.

`PRIMARY-SOURCE FACT` — **`SE-600M/protein_embeddings.pt` has lfs sha256
`a210e1cc7901513999b2bca3836ba9e2f203cd008be4e9a9d6412a2267de9748`, size 410,886,729
bytes — byte-identical to this repository's `ESM2_pert_features.pt`.** Forty sampled
rows of `pe_embedding.weight`, fetched by byte range, are sha256-identical to
individual vectors inside the local file.

`INFERENCE` — This establishes the **distribution** lineage of the local artifact but
**not** its upstream model lineage. SE-600M's config names only an internal path
(`Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt`, `num: 19790`, `size: 5120`).
A 5,120-dimensional embedding is consistent with `esm2_t48_15B_UR50D`, but that is
inference from dimensionality alone. **Gate 1b stays closed**: upstream ESM2 model id,
revision, weights digest, protein-sequence source, isoform mapping and pooling rule
remain `UNRESOLVED`, exactly as
`configs/scdfm/vcc2026_v7_gamma1.toml [target_conditioning_provenance]` records.

`PRIMARY-SOURCE FACT` — Arc's own shipped evaluation CSVs do not support putting SE in
front of the transition model: ST-HVG beats ST-SE on DE ranking in Jurkat (23 of 26
statistics), RPE1 (24 of 26) and K562 (about 22 of 26). HepG2 reverses to 13–12–1 with
SE winning all ten `precision_at_k`. Do not repeat the "HVG wins 6 of 8 on HepG2"
framing — it selects a favourable subset.

### 2.4 PertMind

`PRIMARY-SOURCE FACT` — arXiv:2608.16419 v1 2026-08-17; **v2 2026-08-22 WITHDRAWN**,
comment verbatim: "Withdrawn because one or more co-authors do not consent to the
public posting of this work on arXiv. No replacement version will be submitted."
The v1 HTML and PDF (8,664,424 bytes) remain retrievable; v2 returns 404 and states
"No license for this version due to withdrawn".

`PRIMARY-SOURCE FACT` — `shapsider/PertMind` @ `500da7b6ad3acf3854c55854cb1ce7a6679e6099`
contains **7 blobs, landing content only, zero source code**, contradicting the paper's
own statement that source code is available there. The runnable boundary is
HF `tzcfly/PertMind` rev `14a19cc33cd8c00ee84d251ae0e714f51c361487`, whose shipped
`LICENSE` is **altered Apache text** (10,235 bytes; §8 reads "in tort," where canonical
reads "in tort (including negligence),"). The base model `Qwen/Qwen3-4B-Base`
rev `906bfd4b4dc7f14ee4320094d8b41684abff8539` is Apache-2.0 and ungated. The retrieval
knowledge graph is **not released**, so no reported number is reproducible.

`PRIMARY-SOURCE FACT` — Output is a reasoning trajectory plus per-pathway direction
statements plus exactly one ternary target-gene label (`Up`, `Down`, `No`) with no
uncertain class. Supervision is Tahoe-100M pseudobulk DESeq2-style statistics,
thresholded (`Up` if padj ≤ 0.05 and log2FC ≥ 0.585; `Down` if padj ≤ 0.05 and
log2FC ≤ −0.585; `No` if |log2FC| < 0.20 and padj ≥ 0.50 and baseMean ≥ 10; otherwise
excluded). Training is GRPO on a scalar reward. There is no likelihood, no density and
no per-cell sampling.

`PRIMARY-SOURCE FACT`, adversary correction — the claim that PertMind evaluates no
CRISPRi anywhere is **not supportable**: §3.3, §4.5 and Fig. 6b report zero-shot task
transfer onto genetic (CRISPRa/CRISPRi, Perturb-seq) reverse-ranking tasks without
task-specific post-training. Its *forward* supervision remains small-molecule only.

### 2.5 PerturbDiff

`PRIMARY-SOURCE FACT` — arXiv:2602.19685v1, 2026-02-23, only version.
`DeepGraphLearning/PerturbDiff` main HEAD `31be89cb8cce5dddef0782bac29d5c748ce55931`;
MIT, but the `LICENSE` was committed only on 2026-08-22. HF
`katarinayuan/PerturbDiff_release_ckpt` rev `c33e578bf50445ce2516198a97bbef787bde6425`
declares `license: None` with no model card — `UNRESOLVED`.

`PRIMARY-SOURCE FACT` — Output is `normalize_total` → `log1p` → **divide by 10** into
roughly [0,1) with a `relu` head over 2,000 top HVG. Appendix A.2.1 is *titled* "Raw
Count Preprocessing" but no raw-count model exists. The released `cov_encoding` is a
closed one-hot `nn.Embedding(num_pert+1)`; ESM2 and GenePT branches exist in code but
no released config or checkpoint uses them.

`PRIMARY-SOURCE FACT` — Demonstrated axis is the unseen **(cell context × perturbation)
combination** under partial coverage: the test split is the *intersection* of held-out
context and held-out perturbations, and A.2.3 states "30% of the perturbations from
these held-out donors were moved into the training set". **No unseen-target-identity
result is reported anywhere.** Its Replogle preprocessing inherits STATE's on-target
knockdown-efficacy filter, which reads perturbed outcomes and cannot be applied to
challenge targets.

`PRIMARY-SOURCE FACT`, adversary correction — the loss is not a duplicate of STATE's:
the paper derives a squared RKHS distance between kernel mean embeddings, and the
released code implements geomloss `SamplesLoss(loss='energy', blur=0.05)` at weight
1.0 — an **energy** distance, not a multi-scale RBF MMD at 0.5. Reject PerturbDiff on
its contract mismatches, not on a false duplication claim.

### 2.6 Flow Matching and Multisample Flow Matching

`PRIMARY-SOURCE FACT` — Lipman et al. arXiv:2210.02747v2; Pooladian et al. PMLR v202
`pooladian23a`. Neither paper names an official repository in the fetched text.

`PRIMARY-SOURCE FACT`, adversary correction — **Flow Matching contains no "MMD"
string at all** (zero hits for "MMD", "maximum mean discrepancy", "energy distance" and
"discrepancy" across three independent PDF renderings). Multisample FM's sole hit is
the "MMD GAN" bibliography entry. Neither paper offers a training-time distributional
loss; Multisample FM's machinery is OT/Sinkhorn **coupling**.

`PRIMARY-SOURCE FACT` — Multisample FM's converged Joint-CFM variance upper bounds
(Figure 2 caption): CondOT 10.72, Stable 1.60, Heuristic 1.56, BatchEOT 0.57, BatchOT
0.24, at 0.8%–3.9% runtime overhead. Its marginal guarantee (Lemma 4.1) covers the
**unconditional** pair, not per-(context, target) conditionals.

`INFERENCE` — Minibatch-OT coupling is a cheap in-family ablation, not a route. Any
claim that this project's `ode_steps = 20` "sits at the CondOT break-even point" is
numerology: the phrase appears nowhere in either paper, and this project has no
measured step-count/quality curve at all.

---

## 3. Upstream leakage audit (scDFM)

All `PRIMARY-SOURCE FACT` unless marked.

- Issue #7's allegation that validation data enters a training sampler is
  **substantiated in the direction that matters**: in the norman branch, the validation
  sampler is constructed from the test split (`data.py:255-256`). An upstream selection
  path must therefore never be copied.
- Six further defects were found at the pinned commit that are unreported upstream:
  test-side HVG selection (`data.py:210-211`, `85-86`); full-`adata` training HVG
  (`data.py:98`, `66`); an inert `--infer_top_gene` because `run.py:257` does not
  forward it; a truthiness bug at `data.py:119` that makes `split_method='additive'`
  vacuously true; the paper's claimed train/validation/test split absent from code with
  five folds rather than four; and the paper describing "most highly expressed"
  evaluation genes where the code selects highly variable ones.
- No maintainer has responded to issue #7 in the 36 days to the access date.
- File digests re-verified independently: `data.py` sha256
  `8824dd6cd3891dd2738b8905b9b1386c392645a997975e0bb3c54ccdbeb36a45`; `run.py` sha256
  `f9ff9465a35e5fcd4854e1a0f3c42480c4b192a84d25856d8b0442208960772e`.

`INFERENCE` — Concepts from scDFM may be reimplemented locally. Its splits, sampler,
selection path, evaluation loop and released checkpoint must not be adopted.

---

## 4. STATE VCC notebook archival

`LOCAL REPOSITORY FACT` — The Colab URL returns HTTP 200 but a JavaScript-only shell
with no extractable text; the notebook is **not** in the `ArcInstitute/state` git tree
at `9bbfe78a` (full recursive tree enumerated, `truncated=false`, zero `.ipynb` paths).
A copy exists locally as an **untracked** file `external/state/VCC_2026_STATE_colab.ipynb`,
714,487 bytes.

`PRIMARY-SOURCE FACT` — **That local notebook is the 2025 challenge, not 2026.** Its
markdown says the data "is from H1 cells", it downloads `competition_support_set`, and
it embeds an image named `Screenshot 2025-07-07`. The "2026" in the filename is local
naming. Its conclusions must not be transferred to the 2026 contract without
re-derivation.

`UNRESOLVED` — No stable Drive revision could be obtained. The reproducible authority
remains the pinned STATE source commit `9bbfe78a434a55205e4de834e1ea99f85f7a3add` and
the pinned Hugging Face checkpoint revisions recorded above.

Notable content of the local notebook, `LOCAL REPOSITORY FACT`: it records global step
23,184 (epoch 108 × 213 + 180) against `max_steps = 40000`, i.e. 58.0% of the budget;
`cell_set_len` 128; and it enables no counts decoder. Two quotes it carries —
"STATE is meant for context generalization, not perturbation generalization" and
"Note that we are now generalizing across both contexts and perturbations (not just
contexts)" — exist only in this untracked local file and must be labelled accordingly.

---

## 5. Local dataset and 300-target coverage

`LOCAL REPOSITORY FACT` — Panel identity: `dataset/controls/pert_counts.csv` sha256
`f57edd7b912ebd718efc7ee9d0f334772513e7cc418d133ce525470e373b3276`, 300 unique symbols;
`dataset/controls/gene_names.csv` sha256
`25bfa66715e186bebabce7ac788bbcea47e2bf59ca70be1f8f3a06f2f0e47201`, 18,533 unique
symbols, all 300 targets present on the axis; `manifest.json` sha256
`2c2b5d425df817ca35e9edf08282d12494ce42913ba824375bf0f98a1cc676f5`.

| Question | Answer |
|---|---|
| Challenge targets with **any** same-target evidence locally | 282 of 300 |
| Challenge targets with **CRISPRi single-cell** same-target evidence | **17 of 300** |
| Challenge targets with same-target evidence on the official 18,533-gene axis | **0 of 300** |
| Carrier of the 282 figure | A K562 GWPS **pseudobulk** matrix (267–272) plus a log-normalised iPSC atlas (182) |
| Local perturbed sources providing raw integer counts on the 18,533 axis | **none**; only the controls-only `context_{A,B,C}.h5ad` sit on that axis |

`LOCAL REPOSITORY FACT` — **The entire V7 evaluation apparatus is disjoint from the
challenge panel.** The V7 non-harm split's 2,339-target clustering universe, its 741
globally sealed targets, its 300 scored held targets and its 1,598 allowed targets each
have **zero** intersection with the VCC 300. The HepG2 P4 panel's 300 targets likewise
have zero overlap, and `dataset/public_v51/split_manifest.csv` is this project's own
v5.1 proxy panel drawn from Nadig Jurkat, not a prior official VCC panel.

`INFERENCE` — No V7 or public-lane held-out score is an estimate of leaderboard
performance on any of the 300 challenge targets. Section 7 of the decision memo shows
this empirically.

`LOCAL REPOSITORY FACT` — Licence and provenance gaps that must be closed before the
affected data is used: the Arc competition support set (including
`competition_train.h5` and `ESM2_pert_features.pt`) names no licence at all;
GSE264667 Jurkat and HepG2 terms are recorded as "must be reviewed"; the three scDFM
archives carry no archive-level licence; `dataset/state_pretrained/` is registered
nowhere. Six registered files under `dataset/raw/replogle_2022/` and
`dataset/raw/feng2026/` have their sha256 recorded only as Markdown prose with no
machine-readable receipt and no authenticator. `GSE264667_hepg2_raw_singlecell_01.h5ad`
— the **sealed primary zero-shot validation context** — has no size and no sha256 in
`docs/DATA_PROVENANCE.md`; its only recorded digest lives in a Git-ignored file.

`LOCAL REPOSITORY FACT` — Three large local files are registered in no document, no
config and no receipt: `dataset/raw/replogle_nadig/replogle.h5ad` (22.3 GB);
`dataset/state_support/template_extract/competition_support_set/competition_val_template.h5ad`
(7.16 GB, whose filename suggests challenge validation material); and
`dataset/state_pretrained/st-se-replogle-full-d1441f5/`.

---

## 6. Local code audit findings that bear on the route

All `LOCAL REPOSITORY FACT`.

**The V7 continuous anchor is not merely missing; it is not admissible as specified.**

- No producer exists for either `vcc-state-anchor-provenance-v2` or the anchor array.
  The schema string appears only in the consumer (`scripts/scdfm_centered_residual.py:32`,
  `:509`, `:632`), one unit-test fixture and prose.
- No continuous per-cell array exists anywhere on disk. There is not one project `.npy`
  file in the repository; every candidate `.npz` under `artifacts/` is a group-level
  signed log-fold delta with no 400-cell axis.
- The per-cell continuous tensor **is computed on the H100 and then deleted in-process**:
  `scripts/generate_state_direct_counts.py:450-452` deletes `control_prediction` and
  `target_prediction`, returning only the signed delta;
  `scripts/generate_native_state_anchor_counts.py:711` deletes `final_delta`. No CLI
  flag emits a continuous array. Recovering it needs edits to two files **and** a new
  H100 inference run.
- **Axis-disjoint panels.** The HepG2 P4 panel has zero target overlap with the
  challenge panel, 9,623 genes of which only 9,024 are on the 18,533 axis, and one
  context versus three. A HepG2 anchor fails six separate assertions in
  `scdfm_centered_residual.py` under `configs/scdfm/vcc2026_v7_gamma1.toml`. A separate
  panel-bound config and one-context manifest must be authored and does not exist.
- **The non-negativity gate fails at every registered non-zero weight.** Measured on a
  real sealed group: 57.24% of a `library_normalized_log1p` anchor is exactly 0.0, and
  a zero-mean centred residual drives 28.62% of candidate entries negative at
  w = 0.05, 0.10 and 0.20. `scdfm_centered_residual.py:656-657` rejects that.
- **Memory ceiling.** The compositor upcasts to float64 and holds four to five copies
  live, measured at about 48 bytes per element. The registered `[900, 400, 18533]`
  shape needs roughly 320 GB peak against 250 GB total and 138 GB available.
- `cell_axis_identity_sha256` and `latent_axis_identity_sha256` have **no definition
  anywhere in the repository** — only "64 hex, not all zeros, equal between anchor and
  flow". A convention must be fixed by fiat and reproduced bit-for-bit by any future
  flow emitter.
- A **Route-A derived anchor is buildable on CPU and is provably lossless**: on group 0
  of the sealed prediction, `expm1(log1p(X·1e4/lib))·(lib/1e4)` reproduces the integer
  counts with max absolute error 9.09e-13 and exact integer rounding. Cost: one
  streaming pass over 3.97 GB, output 4,619,040,000 bytes.
- **The registered producer scripts have drifted.** `spec.json` pins
  `0b0d81e9…`/`4b2e1419…`/`36f7dfca…`; the working tree now holds
  `fc18e5dd…`/`eedecb14…`/`861609ad…`, changed by commit `d5644ac`. The registered
  blobs are recoverable read-only via
  `git show af26a35b2d083dcf1600eb0f826be81011b8fcb4:<path>`.

**The portable training gate authorises less than its name suggests.**

- It is a data-**read** gate, not an operation gate: it authorises 3 of its 7
  registered protected stages, binds no gene axis, no seeds, no checkpoint rule and no
  ESM2 conditioning artifact.
- `competition_train.h5` stores `X` as a CSR group; the gate accepts only a dense
  `h5py.Dataset`, so the largest registered training source (206,492 allowed rows) is
  unreadable. Compact training cannot proceed until an authenticated sparse read path
  exists.
- The receipt field `trainer_introduced_no_split_builder_or_kmeans` is delta-scoped and
  was **falsified on CPU**: it reported `True` with `sklearn.cluster` loaded.
- `AuthorizedH5Source` is an unfrozen dataclass, so an in-process caller can read
  evaluation-only HepG2 `X` in two lines.
- Repository-wide, all seven protected stages are reachable without the gate: only
  three files import it, while `scripts/fit_bayesian_prior_h100.py`,
  `scripts/train_target_signature_model.py` and `scripts/select_state_checkpoint.py`
  run ungated.
- **HepG2 evaluation data has already entered a fitted stage outside the gate**:
  `artifacts/bayes_public_effects_v1.npz` carries 68 HepG2-derived rows of a
  `(590, 18080)` effects matrix that `fit_bayesian_prior_h100.py` KMeans-clusters and
  ridge-tunes. Any lineage from that artifact into a V7 submission must be traced and
  severed, or the HepG2 zero-shot claim is compromised.
- The lock schema hard-requires `production_training_ready == false`, so it can never
  be edited to authorise training; a new
  `artifacts/scdfm/v7/training/compact_training_activation_v1.json` must bind it by
  digest rather than replace it.

**Unenforceable promotion rules.** V7 promotion rule 2 ("non-inferior under the
registered margin") and the stop rule ("a material MSE/PDS regression") reference
thresholds that exist nowhere numerically. The only executable threshold in the tree is
`compare_public_v6_sequential.py --absolute-tolerance` defaulting to 1e-12. This
conflicts with the project's own rule forbidding undefined words such as "materially"
in a promotion decision.

**Stale binding.** `results/scdfm_v7/vcc_compatibility.json` and
`dataset/scdfm/receipts/vcc_compatibility_config_bound_v2.json` bind config sha256
`257b28a8…`, a superseded revision; the live trust root is `f4f20f3b…`. By the
receipt's own contract (`adapter_config_sha256_required = true`,
`legacy_unbound_adapter_receipts_accepted = false`) it is currently invalid.

---

## 7. Cluster and repository state

`LOCAL REPOSITORY FACT` — All four V7 trust roots hash-match exactly:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `configs/scdfm/vcc2026_v7_gamma1.toml` | 11,883 | `f4f20f3b86002d74e7f3d44837eedd5c7dd270dc3f9ab5adae052484fad3bf6a` |
| `artifacts/scdfm/v7/splits/jurkat_non_harm.json` | 260,829 | `3114e59b5430d140cfeeaccd8b2f6974ceb1c14ba254ec41597057ce7a70c439` |
| `artifacts/scdfm/v7/training/jurkat_training_preflight.json` | 8,271 | `b5dfc61625625e8b70f126e9eb563f12f62ec629e8a138ef6045688c0f233ba1` |
| `artifacts/scdfm/v7/training/portable_training_gate_lock.json` | 7,406 | `f9c5b8675a1a46c561c056574009722d1faa4bfaf8b4e12e7d7017d3c5a9628d` |

The P4 identities also match: decision `cd6618f38f0b93c8e7b3c63a8a2c1a31b87360ead5cc91b3f7e43898c100d312`,
contract `c0cff8a0c616edf523bc7c59807231dc3309cc187958a6a63b6983125bfa2fb2`,
prediction `5687bb747516cf6c82345d468390b7203a4328cc75588297d6e5c2fdb099364e`.

`LOCAL REPOSITORY FACT` — `origin` is unchanged at
`https://github.com/wukonglin/virtual-cell-challenge-2026.git`; no alternate remote has
been added; `/home/fs01/yl4259/yl/Virtual_cell_jwsup` does not exist and was not
created. The recorded mirror-point table lists `experiment/v7-scdfm-gamma1` at
`be930949`; both `origin` and the local branch now serve `fbb12a06`. The table should be
re-baselined.

**Owner decision, 2026-09-02: `jwsup` retains `write` on the mirror.** The exposure the
audit recorded is therefore accepted, not remediated: that account can force-move any
branch tip on the mirror, including `submission/state-weights-v1`. The compensating
control is detection rather than prevention — install an `origin`-versus-mirror
`git ls-remote | diff` tripwire so an unauthorised write is visible, and keep publishing
this project's authoritative history to `origin` only.

`LOCAL REPOSITORY FACT` — Verified H100 evidence is confined to job `862102`:
node `c0002`, `COMPLETED`, exit `0:0`, 14 seconds, 189,593 allowed Jurkat rows,
73,363 excluded, two deterministic optimizer replicas with identical model-state sha256
`e6d33e1431c49acbadd909df2e769a371266ccef1c4828098ffcc6c0f9b5c685`. This is an
adapter smoke. It is not compact training, not biological evidence, and not submission
authorisation.

---

## 8. Open items

| # | Item | Status |
|---|---|---|
| 1 | Prize eligibility of a STATE-derived entry, given "not able to win the competition" versus the FAQ's permission | **RESOLVED 2026-09-02** — the team asked the organiser; STATE may be used. Archive the written reply |
| 2 | Whether a prize-bearing entry is a Non-Commercial Purpose under Arc MODEL_LICENSE §1.4, and whether "use as a prior" under §1.2 makes it a Derivative Work | `UNRESOLVED` — an Arc licensing question, not an organiser one; needs counsel |
| 3 | Upstream ESM2 lineage for the 5,120-dimensional target features | `UNRESOLVED` — never released; Gate 1b stays closed |
| 4 | Feng atlas perturbation modality (CRISPRi / CRISPRa / KO) | `UNRESOLVED` — if not CRISPRi, the same-modality union falls from 282 to 100 |
| 5 | scDFM `split_method` value that produced the released `ckpts/holdout/` | `UNRESOLVED` |
| 6 | scDFM ICLR 2026 status | `UNRESOLVED` — OpenReview unreachable |
| 7 | Numerical -r4 versus -r3 anchor identity | `UNRESOLVED` by independent measurement; asserted upstream by digest comparison |
| 8 | Licence terms of the Arc competition support set | `UNRESOLVED` |

---

Companion document: [`NEXT_ROUTE_DECISION.md`](NEXT_ROUTE_DECISION.md).
