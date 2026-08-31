# VCC 2026 Research Review and Model V2 Design

**Research cutoff:** 2026-08-31
**Purpose:** define a scorer-aligned, evidence-backed modeling program for the
2026 Virtual Cell Challenge.
**Primary decision:** preserve a shared perturbation response for expression
accuracy, learn a target-specific downstream residual for discrimination, and
use a count decoder or conditional flow only after the perturbation-level mean
has been validated.

## Evidence labels

This document deliberately separates facts from design judgments.

- **[OFFICIAL]** is stated by the challenge, its public API, or the pinned
  official scorer.
- **[PUBLICATION]** is reported by a paper, model card, or upstream repository.
  A preprint claim is not treated as independent validation.
- **[PROJECT]** is a reproducible observation from this repository or its local
  data audit.
- **[INFERENCE]** is our interpretation of the evidence.
- **[PROPOSAL]** is a model or experiment that still requires sealed-fold
  validation.

The main official sources are:

- Challenge: <https://virtualcellchallenge.org/>
- Data: <https://virtualcellchallenge.org/data>
- Evaluation: <https://virtualcellchallenge.org/evaluation>
- FAQ, including external-data and STATE-use rules:
  <https://virtualcellchallenge.org/faq>
- Public leaderboard API: <https://virtualcellchallenge.org/api/leaderboard>
- CLI guide: <https://vcc-cli-wiki.virtualcellchallenge.org/>
- Official scorer: <https://github.com/ArcInstitute/cell-eval2>
- Pinned metric specification used in this audit:
  <https://github.com/ArcInstitute/cell-eval2/blob/5e64833518a6603a0301cbe28185d49c30f4a986/docs/vcc2026_metrics/vcc2026-metrics.md>

The local scorer checkout was commit
`5e64833518a6603a0301cbe28185d49c30f4a986`. The public leaderboard snapshot
used below contained 517 latest-team entries on panel `vcc2026-val-1` with
anchor set `vcc2026-valA-r4+vcc2026-valB-r4+vcc2026-valC-r4`. The HTTP response
date was 2026-08-31 22:48:59 UTC; the downloaded JSON was 630,583 bytes with
SHA256
`cd6cb74c0adf1355851cdf3d08abb39a44136249eab8f8316fbaa9cfba088ff1`.
Because the API is live, later requests will not necessarily reproduce the same
order or hash.

## 1. The task, stated precisely

### 1.1 Inputs and required output

**[OFFICIAL]** There is no challenge-specific perturbation training set for
2026. For each anonymous context A, B, and C in validation, the team receives
non-targeting control cells and the 300 CRISPRi target genes to predict. The
final round uses three different contexts D, E, and F and a new panel. The
model must produce a post-perturbation single-cell expression distribution for
every context-target pair.

The upload contract is exact:

- one AnnData file covering all three contexts;
- exactly 300 listed targets per context and no extra target;
- exactly 400 predicted cells per context-target group;
- exactly 18,533 genes in the supplied order;
- `.obs["target_gene"]` contains gene symbols;
- no uploaded non-targeting rows;
- `.X` contains finite, non-negative, integral raw counts;
- no cell total exceeds one million counts;
- the matrix must be sparse and contain at most 4.75 billion stored entries.

The platform appends the held-out measured control cells before scoring. Thus,
the released controls are model inputs, not rows to copy into the upload.

### 1.2 What the six metrics actually measure

**[OFFICIAL]** Every context is scored separately. Let `u` be a model's raw
metric, `b` the hidden context-mean perturbation baseline, and `r` the
split-half experimental replicate. The published component is

\[
s = \frac{u-b}{r-b}.
\]

Zero therefore means the mean-perturbation baseline, not the non-targeting
control. One means split-half experimental reproducibility. The context score
is the unweighted mean of six scaled components, and the leaderboard score is
the context average.

| Component | Official calculation | Modeling implication |
|---|---|---|
| PDS cosine | Rank the cosine distance between each predicted downstream effect and all 300 measured effects after group-sum normalization to 50,000 and `log1p` | A common effect cannot identify targets. The target-specific downstream pattern matters more than the knocked-down transcript. |
| Expression MSE | Panel-wide ratio of jackknife-corrected squared profile error to the measured perturbation-to-control distance | Group-summed expression and effect magnitude must be right. Cell noise cannot compensate for a wrong centroid. |
| Direction fidelity | Correct-sign fraction of predicted significant genes, with a penalty for insufficient call coverage | Sign and DE call count must both be calibrated. |
| Direction reach | Deepest prefix of reference-significant genes whose predicted directions remain at least 90% correct | The ranking of high-confidence directions matters, not only average sign. |
| Significant-set Jaccard | Intersection over union of predicted and measured significant genes | Both false positive and false negative calls matter. |
| LFC NMAE | Absolute log-fold-change error over the reference-significant set, normalized by measured absolute effect | Effect strength must be calibrated on the responding genes. |

The two profile metrics aggregate raw counts within a perturbation, normalize
the group sum to 50,000, and apply `log1p`. The four DE metrics compare each
predicted group against the measured control using per-cell CPM normalized to
one million, a two-sided Wilcoxon rank-sum test, a control-only expression
filter greater than 5 CPM, and Benjamini-Hochberg FDR below 0.05 within each
perturbation. NMAE requires at least ten reference-significant genes.

Raw expression MSE and raw NMAE are lower-is-better, while their scaled
components are higher-is-better. Scaled expression MSE is clamped to `[0, 1]`;
scaled NMAE has a lower floor of `-6`. The other four components are not
clipped.

### 1.3 The target-gene exclusion changes the optimal design

**[OFFICIAL]** The perturbed gene's own coordinate is excluded from all six
metrics. PDS is stricter: it removes the complete set of 300 panel target-gene
coordinates from every comparison.

The group profile is normalized before the excluded coordinates are removed.
Consequently, changing the knocked-down transcript can still alter the
normalization denominator and indirectly move every scored gene.

**[PROJECT]** The current direct-count STATE adapter forces the target to a
fixed 20% remaining fraction and redistributes counts to preserve each source
library. That operation cannot directly earn any of the six metrics, while the
redistributed mass can worsen expression MSE and downstream effects.

**[PROPOSAL]** Stop forcing on-target counts in the scored adapter. Leave the
target coordinate to the ordinary decoder, mask it from scorer-aligned losses,
and evaluate target knockdown separately as a biological QC measure. Never
move mass from the excluded target into scored genes merely to preserve the
original library. Library depth should be modeled explicitly.

### 1.4 Why shared response and target residual must be separate

The scaled zero anchor is the hidden mean perturbation response. A model needs
to preserve broad programs common to many perturbations—stress, proliferation,
cell-cycle, and technical response—to remain near that expression surface.
Target discrimination, however, requires deviations from that surface.

**[PROJECT]** The current paired adapter computes `STATE(target) - STATE(NTC)`
on the same basal cells. This is useful for cancelling model bias, but it can
also remove the shared perturbation response. Its official raw expression MSE
was `4.738906`, far beyond the approximately one no-skill point, while PDS was
`0.497369`, essentially chance. Increasing the same architecture's size does
not directly repair this decomposition error.

**[PUBLICATION]** Recent response-decomposition work explicitly separates a
global response, cell-line response, conserved target response, and
context-by-target interaction. In a balanced tensor of 638 perturbations in
four cell lines, it attributes 27.9% of total response variance to the global
plus cell-line template, 29.4% to the conserved target component, 23.5% to the
context-target interaction, and 19.2% to noise. Its key limitation is equally
important: controls preferentially identify the template, while the
interaction cannot generally be inferred without target-context information.
Biological priors aligned to response geometry are most useful for the
conserved target component:
<https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1> and
<https://github.com/xinyizhanglab/perturbation-decomposition>.

**[PROPOSAL]** Use the same decomposition as the organizing model:

\[
\widehat{\Delta}_{c,p,g}
= \underbrace{\widehat{\mu}_{g}+\widehat{\alpha}_{c,g}}_{\text{shared context template}}
+ \underbrace{\widehat{\beta}_{p,g}}_{\text{conserved target residual}}
+ \underbrace{\widehat{\gamma}_{c,p,g}}_{\text{context-target interaction}}.
\]

The shared term primarily protects expression MSE. The target residual drives
PDS, NMAE, and direction reach. The interaction is a high-uncertainty term,
not a free source of transfer: absent same-target source responses or other
permitted target-context evidence, it should be strongly shrunk toward zero.
Any whitening, Gram decorrelation, or target-contrastive normalization must
operate on `beta` or a reliability-gated `gamma`, never on the shared template.

## 2. Public leaderboard evidence

### 2.1 Top-12 snapshot

**[OFFICIAL]** The table reports scaled components from the public API. Rank is
the array position because the API does not provide a separate rank field.
Model names and descriptions are participant-supplied and are not verified
configurations.

| Rank | Team | Participant model label | Overall | sPDS | sMSE | sNMAE | sFID | sReach | sJAC |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Aginglab.com | GeroAI_v12 | 0.21122 | 0.70892 | 0.16932 | 0.18378 | 0.00296 | 0.20659 | -0.00424 |
| 2 | Jurassic Park | jp26 | 0.19145 | 0.71352 | 0.14998 | 0.12543 | -0.01999 | 0.17307 | 0.00671 |
| 3 | cqawesome | m0831b | 0.19054 | 0.77626 | 0.17979 | 0.11216 | -0.08676 | 0.17810 | -0.01633 |
| 4 | We527 | STATE baseline | 0.18942 | 0.70603 | 0.14993 | 0.13745 | -0.01248 | 0.15162 | 0.00399 |
| 5 | Happy | HP-023 | 0.18601 | 0.73564 | 0.11595 | 0.14380 | -0.02293 | 0.13291 | 0.01072 |
| 6 | Vivia lnc | m86 | 0.18468 | 0.72626 | 0.11050 | 0.13290 | -0.01386 | 0.16299 | -0.01071 |
| 7 | pop | toy model | 0.18354 | 0.77191 | 0.00000 | 0.12577 | -0.00955 | 0.21850 | -0.00542 |
| 8 | Leo | (CPA+2jnevl) | 0.18050 | 0.70665 | 0.12503 | 0.12712 | -0.02018 | 0.14401 | 0.00037 |
| 9 | Haystack | GSA-016 | 0.17987 | 0.66203 | 0.16510 | 0.10575 | -0.01962 | 0.18656 | -0.02063 |
| 10 | UM | BoomLi | 0.17612 | 0.71760 | 0.00000 | 0.13121 | -0.00877 | 0.23290 | -0.01621 |
| 11 | MIT | m89 | 0.17503 | 0.72598 | 0.10371 | 0.09694 | -0.01756 | 0.15223 | -0.01111 |
| 12 | Ahmet Dedeler | The Fern Filed An Appeal (restored) | 0.17409 | 0.74587 | 0.12006 | 0.09424 | -0.03040 | 0.11537 | -0.00062 |

Only one of the top 20 entries had a non-empty description, and only 13 of the
top 100 did. Therefore, `STATE baseline` at rank 4 proves only that an artifact
with that participant-supplied name scored `0.18942`; it does not establish
which checkpoint, data, count adapter, or training recipe was used.

### 2.2 What the visible score geometry says

**[PROJECT]** In the top 20, 15 entries retained positive scaled MSE; their
mean overall score was `0.18098`, compared with `0.17479` for the five
zero-MSE entries. Within the top 50, the 23 positive-MSE entries averaged
`0.17041` overall, compared with `0.15591` for the 27 zero-MSE entries. Some
models can sacrifice MSE and compensate
with PDS or reach, but the strongest region generally rewards keeping both
target specificity and the shared expression surface. The leading model is
balanced rather than the winner of every component.

FID and Jaccard are close to zero for most current leaders, whereas PDS, MSE,
NMAE, and reach show more visible spread. This is a snapshot observation, not
evidence that FID or Jaccard should be ignored: the overall score weights all
six equally, and the final panel may have different difficulty.

### 2.3 The a15/a19 controlled lesson

The rank-12 description reports an exact restore of artifact a15, SHA256
`55f7bd41bfc4d751bcc125d42e25b40c7aed474589257c6db4b575afe26aac59`.
The API independently confirms the restored overall score `0.1740865087` and
its current component values. The same participant self-reports that a19 used
a whitener ridge with `gamma=0.80` and `lambda=0.007` and scored `0.1605691`.

| Scaled component change, a15 to a19 | Change |
|---|---:|
| MSE | `0.120057 -> 0.024149` (`-0.095908`) |
| PDS | `-0.00118` |
| NMAE | `+0.00845` |
| Fidelity | `+0.00400` |
| Reach | `+0.00297` |
| Jaccard | `+0.00057` |

MSE alone changed the overall by approximately `-0.095908 / 6 = -0.015985`.
The other five components recovered only about `+0.00247`, yielding the
reported net regression near `-0.01352`.

**[INFERENCE]** The failure was not that whitening is always harmful. It was
that an operation intended to improve target-specific geometry changed the
common expression surface, while the local screen omitted the official MSE.
The operational rule is strict: screen every candidate with all six official
metrics, and constrain geometry-changing operations to the target residual.

### 2.4 Other useful—but self-reported—leaderboard clues

These descriptions are hypotheses and engineering clues, not reproducible
evidence.

- Rank 22 `vomics-v5`, overall `0.16610`, reports that cross-line consensus was
  useful for direction but not for identifying the DE set.
- Rank 45 `Fusion v6 whitened + calibrated emission`, overall `0.14008`, reports
  multi-source K562, HCT116, and primary-CD4 fusion, common-response centering,
  residual Gram decorrelation, and library-preserving count transport.
- Rank 109 `Sidechain SER-4afn`, overall `0.10777`, reports cross-line transfer
  from K562, H1, colon, and kidney genome-wide screens.
- Several entries independently report 271 or 272 of the 300 validation
  targets covered by full Replogle K562 genome-wide CRISPRi. Their simple raw
  transfer scores are not competitive, but the coverage makes this dataset a
  high-priority target-residual source.
- Rank 170 `hrblab-stage8-gamma-poisson-emission`, overall `0.08137`, reports
  that replacing bootstrap controls with a context-fitted gamma-Poisson
  emitter improved several local metrics with a fixed mean model. This
  supports treating emission as a separate calibrated module.
- Rank 374 `PrimeFlow-context-s6000-e10`, overall `-0.13259`, reports EMA and
  Euler-10 inference. One weak entry does not invalidate flow matching; it
  shows that a generator cannot rescue a poorly transferred mean response.

Leaderboard iteration is not a substitute for validation. The top 20 had a
mean of 14.4 submissions and a median of 13 in this snapshot. The public API
shows only each team's latest submission, descriptions are self-reported, and
the final uses new contexts and a new panel. Hyperparameters must therefore be
frozen using public-data double holdouts, not fitted to repeated leaderboard
feedback.

### 2.5 Implemented response-head candidate

**[PROJECT]** The downloaded K562 genome-wide pseudobulk atlas contains 9,675
targets after a 30-cell reliability filter. It directly covers 267 of the 300
validation targets and 7,681 challenge-axis genes. K562-essential and RPE1
provide 1,676 paired targets for learning a source-to-recipient residual route.

The implemented calibrator subtracts a train-fold common response, fits source
and recipient residual PCA spaces, learns a weighted ridge route, and selects
ridge and amplitude values on whole held-out ESM2 clusters. The selected
configuration is ridge `0.01`, target-residual scale `1.0`, and route scale
`1.0`. Its internal six-member proxy is `0.21540`, versus `0.09530` for the
best no-route setting in the same split. This proxy is not an official VCC
score: it is useful evidence that the cross-line route carries target-specific
signal, not evidence that the production count artifact will score `0.215`.

Two independent-review failures were fixed before promotion. First, common
responses were originally estimated before the ESM2-cluster split, leaking
held-recipient labels into preprocessing; they are now fitted on the training
fold only, with a counterfactual leakage test. Second, a target-level merge
discarded full-axis fallback values for genes absent from K562; merging and
cell-count reliability are now applied at target-by-gene coordinate level.
The corrected full-axis effect prior has SHA256
`41e910252fdfcdabd6193274c86dfd0582f3388c70acd7c833df1626235bbe14`.
The context-conditioned expected-count prior has SHA256
`b892bf83d6e26f1d86781a4385cf7389ad8f344308cc16c319da1afb741b887c`.

The remaining 33 targets currently use the earlier ESM2/Bayesian fallback.
The next one-H100 job trains a K562-wide ESM2 residual head, and its predictions
will overlay only the target-gene coordinates it actually models; all other
coordinates retain the full-axis fallback. A flow is deliberately not yet the
production generator. It will be evaluated only as a zero-mean residual around
this frozen pseudobulk so that heterogeneity cannot erase the newly protected
expression mean.

**[PROJECT / EXPLORATORY]** A control-only route diagnostic compared A/B/C
against K562, RPE1, HepG2, and Jurkat after per-profile log-CP10K normalization
on 6,203 common genes. A was most correlated with Jurkat (`0.519` Pearson), B
with RPE1 (`0.353`), and C was nearly tied between RPE1 (`0.382`) and HepG2
(`0.379`). A was also the most distinct challenge control: A-C correlation was
`0.488`, versus A-B `0.685` and B-C `0.674`. These values do not identify the
anonymous cell lines; protocol and gene-panel effects are confounded with cell
identity. They justify testing a control-conditioned multi-route ensemble once
raw HepG2 and Jurkat atlases are built, with route weights selected only on
held public contexts. They do not justify manually assigning a hidden context
or replacing a prediction with a nearest public profile.

## 3. Model and publication review

### 3.1 Comparison matrix

| Method | Evidence and core idea | Best use here | Main limitation for VCC 2026 | Status and license caution |
|---|---|---|---|---|
| STATE | Set-to-set state transition conditioned on cell state and perturbation; trained at very large perturbation scale and optionally uses a shared State Embedding | A target- and context-conditioned response expert; frozen or lightly adapted representations; comparison baseline | Published zero-shot context tests are leave-one-context-out within a dataset, not arbitrary cross-dataset transfer; a larger ST does not fix mean/residual decomposition or count calibration | BioRxiv preprint; source checkout `9bbfe78a`. Source and weights have separate terms; the official VCC FAQ supplies a competition-specific use interpretation described in Section 8 |
| Stack | 217M-parameter tabular-attention encoder-decoder trained on about 150M cells; learns from prompt and query cells in context | In-context same-perturbation transport when prompt responses exist in other contexts; context representation | It needs informative prompt examples; it is not automatically a label-to-response model for an unseen target with no response example | BioRxiv preprint; source checkout `cacc2e4b`. Code is CC BY-NC-SA 4.0 and weights are non-commercial |
| PRiMeFlow | Conditional flow matching in full gene-expression space with a U-Net velocity field; strong distributional and DEG-recall results; basis of the 2025 VCC Generalist Prize model | Architecture reference, reproducible 2025 training scaffold, and curated public-data manifest | Its result used a different 2025 task; flow does not supply missing target biology in the 2026 zero-shot task, and the published default pretraining is multi-node | 2026 arXiv preprint. Custom academic non-commercial license has unusually broad derivative, public-release, and license-back clauses; do not copy code into this private repository without review |
| PerturbDiff | Functional diffusion over distributions represented in a Hilbert space, intended to represent latent population-level variability | Optional distribution-level ensemble or research benchmark after the mean head works | Direct performance on the exact six VCC 2026 metrics and zero-shot CRISPRi target transfer is unestablished; released data/checkpoint footprint is very large | 2026 arXiv preprint; code is MIT |
| scDFM | Conditional distributional flow matching with an MMD objective and a PAD-Transformer gene graph | Compact flow prototype and graph-conditioned objective reference | Its released Norman/ComboSciPlex setup tests unseen perturbations or combinations, not VCC's double-unseen target and cell-line regime; it uses a reduced gene axis and discrete IDs | ICLR 2026; MIT |
| GARM | Five decoders combine pointwise MAE with efficient pairwise gene-ranking and perturbation-ranking objectives | Loss design for target residual, PDS-like discrimination, and magnitude; simple strong pseudobulk baseline | Its four-screen cross-dataset experiment uses a 6,641-gene pseudobulk axis, not a raw-count distribution or the VCC six-metric contract | October 2025 non-peer-reviewed preprint; code is BSD 3-Clause |
| Response decomposition | Separates global, cell-line, conserved-target, and context-target components in a balanced four-line tensor | Lightweight baseline, variance audit, and an explicit uncertainty prior for the interaction term | Controls and generic target priors do not identify the context-target interaction; it must be shrunk or supported by query-specific routes | July 2026 preprint; reference implementation is MIT |
| Stable-Shift | Fits a low-rank response basis and predicts unseen-target coordinates from STRING, control expression, network, and GO features | Direct template for biologically structured unseen-target residual prediction | Initial evidence is mainly K562 and preprint-scale; sparse network neighborhoods and gene-space accuracy remain limitations | 2026 arXiv preprint. Public repository displayed no license in this audit, so code reuse rights must not be assumed |
| PerturbMap | Recipient-local low-rank base plus source-to-recipient ridge experts for the same perturbation, weighted by route reliability on training anchors | Transfer full K562 or other same-target responses into anonymous VCC contexts; especially valuable if new exact-target experiments are allowed for training | Requires the query perturbation to be measured in at least one source context, predicts a condition-level mean, and was tested on a melanoma cohort rather than VCC | July 2026 arXiv preprint; no official code repository was found in this audit |
| Tahoe-x1 | 70M, 1.3B, and 3B perturbation-trained single-cell foundation models; masked-expression objective over 266M profiles including Tahoe-100M | Frozen context/cell and gene representations; a candidate State Embedding replacement; decoder pretraining | Pretraining is cancer/drug-oriented and the largest model sees at most 2,048 genes per sequence, far short of the 18,533-gene output; representation scale is not response skill | 2025 bioRxiv preprint; code and model weights are Apache 2.0 |
| X-Cell / X-Atlas | Diffusion language model with ESM2, STRING, GenePT, DepMap, JUMP, and scGPT priors; paper reports 25.6M perturbed cells in 16 contexts | Future pretrained comparator and source of multimodal target-feature designs | On 2026-08-31 the official repository says weights and inference code are coming soon, and the dataset page is only a placeholder; it is not executable evidence today | March 2026 preprint; repository and placeholder dataset are CC BY-NC-SA 4.0 |

### 3.2 STATE

Sources:

- Paper: <https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2>
- Repository: <https://github.com/ArcInstitute/state>
- Example released genetic checkpoints:
  <https://huggingface.co/arcinstitute/ST-SE-Replogle>

**[PUBLICATION]** STATE combines molecular perturbation features, individual
cell representations, and population-level set transitions. The authors report
training the transition model on more than 100 million perturbed cells across
about 70 contexts, while the State Embedding model was trained on 167 million
observational cells. The published gains are largest in the higher-data drug
and cytokine regimes; genetic screens remain a more data-limited regime.
Its reported zero-shot context protocol leaves one context out within a
dataset; that is meaningful evidence, but it is weaker than transfer into a
new anonymous dataset with different assay and cell-line effects.

**[PROJECT]** A real 162.6M-parameter `state_sm` trained for 20,000 steps on the
current six-file support set improved the team's score over its Bayesian
baseline, but remained below the official zero anchor. The support set has only
197 unique target labels and 17 direct overlaps with the 300 challenge targets.
The adapter cannot activate source-zero genes, mixes log-normalization regimes,
and removes the shared target-versus-NTC response.

**[INFERENCE]** The next STATE experiment should not begin with `state_lg`.
First, expose separate absolute/shared and target-residual heads, remove forced
target redistribution, and validate count-space calibration. A larger model is
justified only if the corrected small model improves double-held-out PDS and
MSE together.

### 3.3 Stack

Sources:

- Paper: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12803207/>
- Repository: <https://github.com/ArcInstitute/stack>
- Aligned checkpoint: <https://huggingface.co/arcinstitute/Stack-Large-Aligned>

**[PUBLICATION]** Stack alternates attention across cells and genes and learns
to answer query cells from prompt cells at inference. The aligned model card
reports 217M parameters, pretraining on approximately 150M scBaseCount cells,
and alignment on CELLxGENE 45M plus Parse 10M PBMC.

**[INFERENCE]** Stack is strongest when a measured response to the same target
can be supplied in a source context and the query is an unperturbed cell in a
recipient context. It should become a route expert next to PerturbMap, not the
only zero-shot target encoder. Full K562 GWPS coverage would make this much
more useful than the current 17-target support overlap.

### 3.4 PRiMeFlow and general flow matching

Sources:

- Flow Matching: <https://arxiv.org/abs/2210.02747>
- Multisample Flow Matching:
  <https://proceedings.mlr.press/v202/pooladian23a.html>
- PRiMeFlow paper: <https://arxiv.org/abs/2604.13986>
- PRiMeFlow repository: <https://github.com/altoslabs/primeflow>
- Preprocessed public-data collection:
  <https://huggingface.co/datasets/altoslabs/primeflow-vcc-datasets>
- PRiMeFlow license:
  <https://github.com/altoslabs/primeflow/blob/main/LICENSE.md>

**[PUBLICATION]** Flow Matching trains a continuous normalizing flow by
regressing a conditional vector field without simulating the ODE during
training. Optimal-transport or multisample couplings can create straighter
paths and reduce sampling cost. PRiMeFlow applies this end-to-end in expression
space with a U-Net and reports that full-gene modeling improves distributional
fit and DEG recall over PCA-space ablations. Its paper also shows that stronger
classifier-free guidance can trade distributional fit against DEG recall.

**[PUBLICATION]** The repository now includes the complete VCC 2025
pretraining and fine-tuning scaffold, distributed training, an ESM2 target
feature path, and a public-data manifest spanning VCC 2025 H1, Replogle 2020
and 2022, Nadig, McFaline-Figueroa/Jiang, Feng 2026, and Zhu 2025. The combined
preprocessed release is approximately 244 GB compressed, of which the 12 CD4T
files account for about 198 GB. Its default reproduction uses 12 GPUs across
two nodes and a long two-stage schedule. These are operational assets, but the
2025 schedule and target split are not evidence that the same hyperparameters
transfer to VCC 2026. The Hugging Face wrapper is marked CC BY 4.0; every
component dataset still requires its own upstream license and transformation
audit.

**[INFERENCE]** Flow is a distribution model, not a source of target identity.
If the conditional mean is wrong, a realistic cloud of cells remains wrong.
The safest use is a zero-mean residual flow around an independently validated
pseudobulk mean. This makes the signal model responsible for MSE and effect
magnitude, while the flow learns heterogeneity, responder fractions, and
correlated deviations relevant to Wilcoxon calls.

### 3.5 PerturbDiff and scDFM

Sources:

- PerturbDiff paper: <https://arxiv.org/abs/2602.19685>
- PerturbDiff code: <https://github.com/DeepGraphLearning/PerturbDiff>
- scDFM paper: <https://openreview.net/forum?id=QSGanMEcUV>
- scDFM code: <https://github.com/AI4Science-WestlakeU/scDFM>

**[PUBLICATION]** Both methods take distribution prediction seriously rather
than treating cells as paired before/after observations. PerturbDiff diffuses
distribution representations; scDFM learns a conditional flow from controls
to perturbed populations using an MMD objective and PAD-Transformer gene
graph. PerturbDiff publishes a roughly 16.3 GB checkpoint collection at
<https://huggingface.co/katarinayuan/PerturbDiff_release_ckpt>; its full
processed training release is much larger. scDFM releases Norman and
ComboSciPlex pipelines, which test unseen perturbations or combinations but
not a simultaneously unseen cell line and target.

**[INFERENCE]** Their objectives are useful references, but neither should be
dropped into production unchanged. VCC requires raw counts on the full gene
axis, a continuous representation for unseen target genes, exact group size,
and an unusual six-metric target exclusion. We should port the distributional
idea into the scorer-aware decomposition rather than porting a benchmark
pipeline wholesale.

### 3.6 GARM

Sources:

- Paper: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12621900/>
- Code: <https://github.com/Jerby-Lab/GARM>

**[PUBLICATION]** GARM trains multiple decoders for pointwise accuracy,
gene-ranking, and perturbation-ranking, then reconciles them. Its pairwise
losses are computed in linear rather than explicit quadratic cost. Across
HepG2, Jurkat, K562, and RPE1 CRISPRi screens containing 2,042-2,373
perturbations each, the paper reports better perturbation-ranking than GEARS,
scGPT, GenePert, and coexpression baselines. The cross-screen analysis is on
6,641 common genes and log1p-TPM pseudobulk. GARM is not uniformly best on
pointwise MSE or gene ranking, and the posted manuscript is a non-peer-reviewed
preprint.

**[INFERENCE]** This directly addresses the distinction between predicting a
generic response and distinguishing targets. We should use a GARM-style
pairwise residual loss in the pseudobulk signal model, but retain pointwise
profile and LFC losses so target ranking cannot destroy MSE, as a19 did.

### 3.7 Stable-Shift

Sources:

- Paper: <https://arxiv.org/abs/2606.24940>
- Repository: <https://github.com/Sajib-006/PerturbGraph>

**[PUBLICATION]** Stable-Shift learns a low-rank perturbation-response basis and
predicts an unseen target's basis coordinates from STRING interactions,
network structure, baseline-control statistics, and GO annotations. The paper
reports mean cosine `0.592` versus `0.569` for GEARS on its K562 benchmark, with
the same ordering across several ablations.

**[INFERENCE]** This is one of the closest methods to the missing VCC module:
predicting downstream target identity for unseen genes. It should be
implemented first as ridge, partial least squares, and MLP baselines before a
GCN. The biological features must be projected into the response geometry
using training targets only, with target clusters held out to prevent close
homolog leakage.

### 3.8 PerturbMap

Source: <https://arxiv.org/abs/2607.28090>.

**[PUBLICATION]** PerturbMap transports the same target's measured response
from source contexts through source-to-recipient ridge experts. Route weights
are estimated using disjoint training anchors rather than the held response.
The paper reports a 4.1% full-effect MSE improvement over a recipient-local
low-rank base on Perturb-CITE-seq melanoma.

**[INFERENCE]** This is immediately relevant once full GWPS is available and
even more relevant if new exact-target experiments are used as training data.
The route expert should propose `beta + gamma`; a conservative recipient-local
base remains available when route reliability is low. No raw source response
should ever be copied directly into a submission. Because the method predicts
a condition-level mean and needs the query target in a source context, it
cannot help targets absent from every source and still needs the VCC count
decoder.

### 3.9 Tahoe-x1

Sources:

- Paper: <https://www.biorxiv.org/content/10.1101/2025.10.23.683759v1>
- Repository: <https://github.com/tahoebio/tahoe-x1>
- Checkpoints: <https://huggingface.co/tahoebio/Tahoe-x1>

**[PUBLICATION]** Tahoe-x1 is a family of models up to 3B parameters trained on
266M single-cell profiles, including Tahoe-100M. It jointly represents genes,
cells, and drug tokens and reports state-of-the-art results on cancer-relevant
representation and held-context tasks. The 70M model uses a maximum context
length of 1,024 tokens, the 1.3B and 3B variants use 2,048, and the reported 3B
training used 128 GPUs. Thus no released variant directly emits all 18,533 VCC
genes in one sequence.

**[INFERENCE]** A frozen intermediate-layer embedding is worth comparing with
STATE SE, Stack, PCA, and control pseudobulk as the context descriptor `h_c`.
It should not be called a direct CRISPRi predictor without an explicit,
compatible transition head. Public users have reported uncertainty about
matching released Tahoe-x1 embedding widths to released STATE checkpoints:
<https://github.com/tahoebio/tahoe-x1/issues/79>.

### 3.10 X-Cell and X-Atlas: useful design, unavailable artifact

Sources:

- Paper: <https://www.biorxiv.org/content/10.64898/2026.03.18.712807v1>
- Repository: <https://github.com/xaira-therapeutics/x-cell>
- Dataset page: <https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Pisces>

**[PUBLICATION]** X-Cell is a diffusion language model with multimodal target
priors including ESM2, STRING, GenePT, DepMap, JUMP, and scGPT. The paper
reports an X-Atlas/Pisces corpus of 25.6 million perturbed cells, 16 contexts,
and seven genome-scale screens, with models up to 4.9B parameters.

**[PROJECT]** At the 2026-08-31 cutoff, the official repository states that
weights and inference code are coming soon, and the Hugging Face dataset is a
small placeholder rather than the reported cells. Therefore X-Cell is a
design reference and release watchlist, not a runnable baseline or available
training set. The visible repositories are CC BY-NC-SA 4.0.

### 3.11 Benchmark warnings that constrain the design

- Systema shows that average perturbation effects and systematic variation can
  make a model look predictive without recovering target-specific biology:
  <https://www.nature.com/articles/s41587-025-02777-8>.
- PerturBench reports mode collapse, recommends rank metrics alongside RMSE,
  and finds simple architectures competitive:
  <https://proceedings.neurips.cc/paper_files/paper/2025/hash/8aee537279a66ced96319dfca3c00002-Abstract-Datasets_and_Benchmarks_Track.html>.
- A Nature Methods benchmark found that then-current deep models did not beat
  simple linear baselines under its unseen-perturbation protocols:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC12328236/>.

These results do not prove that deep models cannot work. They require every
deep candidate to beat context mean, target mean, same-target transport,
low-rank ridge, and MLP baselines under identical double-held-out splits.

## 4. Data acquisition priorities

### 4.1 Priority order

| Priority | Data or feature source | Why it matters | Required controls |
|---:|---|---|---|
| 0 | Replogle K562 genome-wide CRISPRi: paper/data portal <https://pmc.ncbi.nlm.nih.gov/articles/PMC9380471/>, Figshare <https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387>, and BioProject <https://www.ncbi.nlm.nih.gov/bioproject/PRJNA831566> | More than 2.5M cells targeting all expressed genes; multiple participant descriptions claim 271-272/300 VCC targets overlap, making this the highest-value same-target response source | Verify overlap independently. Use raw single-cell day-8 K562 GWPS for count modeling; the normalized bulk H5AD is gemgroup-normalized pseudobulk and is suitable only for the response head. Audit guide aggregation, gene mapping, time point, and license |
| 0 | Response-decomposition processed data and baselines, <https://github.com/xinyizhanglab/perturbation-decomposition> | Approximately 90 MB of processed pseudobulk, DepMap features, and precomputed ridge/MLP/STATE/MORPH comparisons provide a fast variance and split audit before large training | Treat processed matrices as signal-head data, not raw-count decoder data; reproduce double holdouts and pin the MIT-licensed code revision |
| 1 | Raw HepG2 and Jurkat GSE264667, <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667>, plus K562/RPE1 screens and the Arc mirror <https://huggingface.co/datasets/arcinstitute/Replogle-Nadig-Preprint> | Four CRISPRi contexts with roughly two thousand targets each enable target/context decomposition, route fitting, and real count validation | Preserve raw counts; GSE264667 uses dual-guide designs and mostly common-essential targets, so aggregate guides at gene level and record design compatibility. Split target and context before preprocessing |
| 1 | PRiMeFlow VCC public corpus, <https://huggingface.co/datasets/altoslabs/primeflow-vcc-datasets> | Ready-made manifest and preprocessing for VCC 2025 H1 plus multiple Replogle, Nadig, McFaline-Figueroa/Jiang, Feng, Zhu, and CD4T sources | The compressed release is about 244 GB and the wrapper's CC BY 4.0 tag does not replace upstream component licenses. Audit raw/normalized status and transformations before reuse |
| 2 | VCC 2025 H1 and other compatible public CRISPRi screens | Protocol-relevant perturbation effects and additional contexts | Keep 2025 and 2026 panels separated; audit protocol, target, and batch differences |
| 3 | STRING, GO, DepMap coessentiality/dependency, TF networks, pathways, and ESM2 | Continuous features for challenge targets absent from measured screens | Record versions and licenses; fit all feature projections inside each outer fold |
| 4 | Stack-Large-Aligned, STATE SE/ST, Tahoe-x1, and PerturbDiff checkpoints | Context, gene, or distribution representations as frozen comparator encoders | Pin revisions and model licenses; never assume source-code terms cover weights. Start with smaller frozen variants before allocating multi-GPU compute |
| 5 | Tahoe-100M and Parse 10M PBMC | Large-scale pretraining for cell variability, depth, and decoder robustness | Drug/cytokine domains do not directly supervise CRISPRi target residuals; use as auxiliary data, not truth for target identity |
| Watch | X-Atlas/Pisces, <https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Pisces> | The paper reports a broad genome-scale resource that could materially improve target/context coverage | No reported cells were downloadable at the cutoff; do not schedule it as available data. Re-audit release completeness and CC BY-NC-SA terms when artifacts appear |
| 6 | Perturb-Sapiens and other model-generated atlases | Potential regularization and infrastructure smoke tests | Mark as synthetic; prevent a model from being evaluated mainly against another model's artifacts |

**[PUBLICATION]** Storage planning must distinguish a catalog from the files
actually needed. The Replogle Figshare collection is approximately 159.7 GB
and includes raw and normalized single-cell and bulk artifacts; select the
K562 GWPS day-8 raw single-cell file plus metadata rather than downloading by
filename guess. GSE264667 exposes approximately 5.2 GB HepG2 and 8.7 GB Jurkat
H5AD files, with a roughly 15.2 GB raw archive. Record server-provided checksums
before extraction.

**[PROJECT]** The current six-file support cache is a narrow subset. Its 197
unique public targets and 17 direct challenge overlaps must not be mistaken for
the coverage of the full genome-wide resources.

**[PROJECT]** The verified Feng file is 850,726 cells by 36,518 genes across 45
cell-line labels and 31 donors. It contains 6,699 perturbations, including 182
of the 300 validation targets, and overlaps 18,120 challenge genes. Its matrix
is fixed-total normalized-log1p with no raw-count layer. It can therefore train
an explicitly transformed response/representation head, with complete
target-cluster and donor/cell-line holdouts, but cannot train or validate the
integer-count emitter. Exact hashes and license boundaries are recorded in
`docs/DATA_PROVENANCE.md`.

### 4.2 Raw-count and provenance contract

Every imported dataset must have a manifest containing:

- source URL, accession, download timestamp, immutable hash, and license;
- assay, CRISPR modality, cell line, time point, chemistry, and guide field;
- whether `.X` is raw counts, normalized counts, or log-transformed values;
- target-symbol mapping and duplicate/ambiguous target handling;
- gene-axis mapping to the official 18,533 genes;
- train, validation, and test membership assigned before learned preprocessing;
- per-target cell counts, control count, median depth, and knockdown QC.

Mixed raw-log1p and normalized-log1p matrices must not be fed to the same
encoder without an explicit dataset transform or dataset token. Count-decoder
training and exact local scoring require integer raw counts; normalized files
can still train the pseudobulk signal head after their limitations are marked.
In particular, the Replogle `K562_gwps_normalized_bulk_01.h5ad` artifact is not
a substitute for the corresponding raw single-cell GWPS file when training or
validating an integer-count emitter.

### 4.3 Newly generated experiments

**[OFFICIAL]** The public FAQ permits training or fine-tuning with other data,
including proprietary data, when the entrant has the necessary rights; the
Challenge does not require access to that data. It also says code is not
automatically required with every entry, but organizers may request
verification information and prize finalists must publish a description of
how the entry was created: <https://virtualcellchallenge.org/faq>.

**[PROJECT / TEAM-PROVIDED CLARIFICATION]** The team separately reports that
the organizer confirmed newly generated experiments for exact final contexts
and targets may be used as training or fine-tuning data, provided every
submitted cell is generated by an ML model and no measured profile is directly
inserted or used for manual profile revision. This more specific clarification
should be archived with the official correspondence because its exact wording
is not independently visible on the public FAQ.

If such experiments are conducted, enforce a technical firewall:

1. Register targets, contexts, replicates, and train/validation allocation
   before opening expression matrices.
2. Store measurements outside Git with immutable manifests and role-based
   access.
3. Permit measurements to enter only automated fit and model-selection jobs.
4. Prohibit nearest-neighbor copying, per-target manual edits, measured-cell
   resampling into output, or post-hoc replacement of predicted profiles.
5. Generate all 400 submitted cells from frozen model weights and a recorded
   random seed.
6. Retain untouched experimental replicates for audit and generalization
   estimates.
7. Record a lineage from every submission cell to code, weights, configuration,
   control input, and RNG—not to a measured post-perturbation row.
8. Preserve enough automated training and generation evidence to support an
   organizer verification request and the finalist public-method description,
   while respecting permitted confidentiality of proprietary source data.

This route is scientifically powerful because exact-target measurements can
train `beta_p` and `gamma_cp`. It does not remove the need for a generative
model or permit direct insertion of measured profiles.

## 5. Proposed model: shared response, target residual, and count distribution

### 5.1 Stage A: context and target representations

For context `c`, construct `h_c` only from its released controls:

- log-CPM pseudobulk and per-gene dispersion/zero fraction;
- PCA or a small learned control encoder;
- frozen STATE SE, Stack, and Tahoe-x1 representations as separate ablations;
- cell-cycle, stress, and pathway scores;
- dataset/protocol tokens during public-data training.

For target `p`, construct `z_p` from:

- direct same-target CRISPRi signatures where available;
- ESM2 protein-sequence embedding;
- STRING and regulatory-network neighborhoods;
- GO and pathway annotations;
- DepMap coessentiality and dependency profiles;
- basal expression and control-state accessibility proxies.

ESM2 sources are the primary paper
<https://www.science.org/doi/10.1126/science.ade2574>, repository
<https://github.com/facebookresearch/esm>, and 650M checkpoint card
<https://huggingface.co/facebook/esm2_t33_650M_UR50D>. ESM2 is useful because
it provides a continuous evolutionary, structural, and functional prior for an
unseen protein-coding target where a one-hot ID provides no transfer.
PRiMeFlow's released VCC feature path is useful architecture precedent, but it
is not proof that sequence similarity is causally aligned with CRISPRi
response. Protein sequence does not directly encode regulatory direction,
target efficiency, context rewiring, off-target effects, noncoding targets, or
isoform-specific consequences. The ESM2 vector must be projected into response
space together with DepMap, network, and measured-signature features, using
training targets only inside each fold.

### 5.2 Stage B: scorer-aligned pseudobulk signal model

Fit the response decomposition in a variance-stabilized pseudobulk space.

1. Estimate training-context templates from training targets, then learn
   `T_c = t(h_c)` so a completely held-out context receives its template from
   released controls alone. If permitted exact-context experiments later enter
   training, they are handled by the same fitted route rather than by a manual
   replacement.
2. Learn a low-rank response basis `U` from template-removed effects.
3. Predict conserved target coordinates `a_p` from `z_p` using ridge, partial
   least squares, MLP, and graph models.
4. Treat interaction coordinates `q_cp` as uncertain. Fit a low-capacity FiLM
   or bilinear prior from `h_c` and `z_p`, but shrink it to zero unless it
   improves double-held-out folds; controls alone do not identify this term.
5. Add PerturbMap-style source-to-recipient proposals only when the same target
   has been measured in another accepted context. Estimate route reliability
   and interaction strength from disjoint training anchors; exact-context
   automated training measurements, if permitted, provide the strongest
   target-context evidence.
6. Learn an effect-strength scalar and gene-wise shrinkage from training data;
   never hand-tune them against the challenge leaderboard.

A compact parameterization is

\[
\widehat{\Delta}_{c,p}
= \widehat{T}_c
+ U\,f_{\theta}(z_p)
+ \rho_{c,p}\,U\,g_{\phi}(h_c,z_p)
+ \sum_s w_{s\rightarrow c,p}\,R_{s\rightarrow c}(\Delta_{s,p}).
\]

Here `rho_cp` is a calibrated interaction-reliability gate and defaults toward
zero without query-specific evidence. The route sum is zero when no accepted
same-target source exists. Whitening and decorrelation apply only after
subtracting `T_c`.

### 5.3 Stage C: multi-objective losses

Train the signal model with complementary losses rather than a single energy or
MSE objective:

\[
\mathcal{L}_{signal}
= \lambda_{profile}\mathcal{L}_{profile}
+ \lambda_{lfc}\mathcal{L}_{lfc}
+ \lambda_{pair}\mathcal{L}_{GARM}
+ \lambda_{contrast}\mathcal{L}_{target}
+ \lambda_{sign}\mathcal{L}_{direction}
+ \lambda_{sparse}\mathcal{L}_{DE-cardinality}.
\]

- `L_profile` matches the official group-sum, 50,000-normalized `log1p`
  profile and protects MSE.
- `L_lfc` is robust absolute error on held-reference responding genes and
  calibrates NMAE.
- `L_GARM` preserves pairwise target and gene ordering.
- `L_target` is a panel-contrastive objective on downstream genes with all
  panel-target coordinates masked, approximating PDS.
- `L_direction` penalizes high-confidence sign errors.
- `L_DE-cardinality` calibrates response prevalence and avoids calling every
  gene or no genes.

Exact Wilcoxon/BH metrics are non-differentiable. They remain model-selection
criteria, while the differentiable terms above are training surrogates. Loss
weights must be selected on nested public-data folds using the exact six-metric
pipeline.

### 5.4 Stage D: count decoder

The signal model determines the desired context-target group mean. The decoder
must produce 400 integer-count cells with realistic depth, dispersion,
zero-inflation, and correlated programs without moving that mean arbitrarily.

Start with two transparent baselines:

1. **Gamma-Poisson / negative binomial:** fit gene-wise or low-rank dispersion
   from the context controls, condition the mean on the predicted response, and
   sample context-calibrated library sizes.
2. **Dirichlet-multinomial:** sample a correlated composition around the
   predicted mean and then draw integer counts at the sampled depth.

The decoder must allow a gene absent in a particular source cell to become
expressed. It must also model depth independently rather than restoring an
arbitrary source library after target knockdown. Any deterministic balancing or
largest-remainder step must be a fixed model-output transform applied to all
groups, not a hidden-target-specific manual edit.

### 5.5 Stage E: conditional residual flow

Promote a flow only after the signal and count baselines pass sealed-fold
gates. Let `m_cp` be the fixed mean prediction and `r_i` a cell residual. Train
a conditional vector field

\[
\frac{d r_i(t)}{dt}
= v_{\psi}\!\left(r_i(t), t, h_c, z_p, m_{c,p}\right),
\qquad
r_i(1) = r_i(0) + \int_0^1 v_{\psi}(\cdot)\,dt.
\]

Operational constraints:

- center the 400 generated residuals to zero in the decoder's mean space so
  the flow cannot silently destroy the validated pseudobulk;
- condition on both controls and target biology;
- compare independent Gaussian coupling, control-to-perturbed minibatch OT,
  and multisample coupling as explicit ablations;
- begin in a 256-512-dimensional response latent, then test full-gene U-Net
  flow only if the latent model is beneficial;
- compare Euler 10/20/50 and adaptive ODE solvers;
- tune classifier-free guidance on all six metrics because higher guidance may
  improve target separation while harming expression fit;
- decode to raw counts through the same calibrated count head used by the
  non-flow baseline.

This design gives the flow a narrow, testable responsibility: improve
cell-level distribution and DE calls without changing the validated response
centroid.

### 5.6 Scorer-to-module responsibility map

| Current failure | Primary module | Required evidence before promotion |
|---|---|---|
| Effect strength and NMAE | Shared/residual mean head plus learned strength scalar | Better held-context and held-target NMAE without lower PDS or MSE |
| Expression error | Shared template, conservative residual shrinkage, depth-aware count decoder | Positive scaled-like MSE on every outer context and no a19-style tradeoff |
| Target discrimination | Same-target GWPS transport, biological target features, pairwise and contrastive losses | PDS clearly above 0.5 on double-held-out panels after masking all panel targets |
| DE set and direction | Response ranking, responder prevalence, dispersion and residual generator | Higher fidelity, reach, and Jaccard under exact Wilcoxon/BH evaluation |
| Source-zero activation | Count decoder or full-gene generator | Improved responding-gene coverage without excess false positives |
| Context transfer | Control representation and route-reliability interaction head | Improvement on entirely held-out cell lines, not only held targets in known lines |

## 6. Validation protocol and promotion gates

### 6.1 Double-out-of-distribution splits

Construct nested outer folds that match the final task:

- **held target:** no post-perturbation data for the target in the recipient;
- **held context:** no perturbation data from the recipient cell line;
- **double holdout:** neither target nor recipient context appears as a fitted
  target-context pair;
- **target-family holdout:** related genes or network-neighbor clusters are held
  together to prevent trivial feature leakage.

Fit normalization, PCA, response bases, network projections, route weights,
and calibration inside each outer fold. Select hyperparameters only with an
inner split. A same-target route may use a source response only if that source
is allowed by the simulated test scenario; its reliability must be learned on
other targets, never from the held recipient response. When no such source
exists, the route is exactly zero and the interaction gate is evaluated in its
conservative regime.

### 6.2 Local pseudo-challenge construction

For every outer test context:

1. Select exactly 300 held targets wherever the source screen supports it, so
   PDS has the same rank geometry as the challenge.
2. Downsample or generate exactly 400 cells per target.
3. Downsample the reference near the official 20,000 median UMI depth when the
   source depth permits it; never invent additional measured reads merely to
   reach that depth.
4. Append the held measured controls only inside the evaluator.
5. Build context-mean baseline and split-half replicate anchors from the held
   reference.
6. Run the pinned `cell-eval2` VCC 2026 implementation on raw counts.
7. Record raw and scaled six-member results per context and per seed.

The evaluation must mask each target gene for all six metrics and the whole
panel target set for PDS, exactly as production does.

### 6.3 Mandatory baselines

Every candidate is compared against:

- control/no-effect;
- context mean perturbation response;
- target mean across source contexts;
- full-K562 same-target raw and calibrated transfer;
- response-decomposition ridge and MLP with interaction shrunk to zero;
- low-rank ridge and partial least squares;
- MLP with identical inputs;
- corrected `state_sm` absolute/shared plus residual output;
- gamma-Poisson decoder with the same mean head.

### 6.4 Promotion gates

A candidate is promoted only when:

- the overall six-member score improves in most outer folds and the worst
  context does not regress materially;
- the gain appears in at least three fixed seeds;
- PDS improves above its no-information point while expression MSE remains
  protected;
- no component improvement is explained by including excluded target
  coordinates or by a format artifact;
- the exact output contract and sparse-count QC pass;
- configuration, commit, data hashes, hardware, runtime, and rollback artifact
  are recorded.

No model is selected from a two-surface proxy. The a15/a19 result makes a full
six-metric gate non-negotiable.

## 7. Compute plan

Multiple H100s are useful only after the data and objective are correct.

### Phase 0: scorer and data audit, CPU or one H100

- Download and validate full K562 GWPS and full raw multi-line CRISPRi data.
- Build the double-held-out count-space evaluator.
- Remove forced target knockdown and test absolute/shared versus paired
  residual STATE adapters.
- Establish ridge, MLP, same-target transfer, and gamma-Poisson baselines.

**Gate:** repeatable improvement over the current artifact on all six local
metrics, especially PDS and MSE.

### Phase 1: structured signal model, one H100

- Fit 256-512-dimensional response bases.
- Train GARM-style multi-objective MLP and graph ablations.
- Add route-reliable same-target transport.
- Compare frozen PCA, STATE SE, Stack, and Tahoe-x1 context encoders.

**Gate:** improvement on double-held-out targets and contexts across three
seeds.

### Phase 2: residual flow smoke, one H100

- Train a compact latent conditional flow with a frozen mean head.
- Compare against gamma-Poisson and Dirichlet-multinomial decoders.
- Measure whether flow improves DE components without moving MSE.

**Gate:** distributional gain in most folds with no centroid regression.

### Phase 3: four-H100 DDP

Use four H100s for a larger flow, `state_lg`, or full-gene U-Net only after a
smaller version passes Phase 2. Use BF16, deterministic data manifests, fixed
seeds, resumable checkpoints, and per-rank shard validation. Scaling is for
throughput and capacity, not for searching around a broken objective.

## 8. License and governance matrix

This is an engineering audit, not legal advice. A prize competition, an
academic institution, model outputs, and a private repository may be treated
differently by individual licenses. Obtain institutional review before using
non-commercial assets in a final submission.

Primary license files:

- Official VCC FAQ interpretation of STATE and external-data use:
  <https://virtualcellchallenge.org/faq>
- STATE source and model terms:
  <https://github.com/ArcInstitute/state/blob/main/LICENSE>,
  <https://github.com/ArcInstitute/state/blob/main/MODEL_LICENSE.md>, and
  <https://github.com/ArcInstitute/state/blob/main/MODEL_ACCEPTABLE_USE_POLICY.md>
- Stack source and model terms:
  <https://github.com/ArcInstitute/stack/blob/main/LICENSE>,
  <https://github.com/ArcInstitute/stack/blob/main/MODEL_LICENSE.md>, and
  <https://github.com/ArcInstitute/stack/blob/main/MODEL_ACCEPTABLE_USE_POLICY.md>
- PRiMeFlow: <https://github.com/altoslabs/primeflow/blob/main/LICENSE.md>
- PerturbDiff:
  <https://github.com/DeepGraphLearning/PerturbDiff/blob/main/LICENSE>
- scDFM: <https://github.com/AI4Science-WestlakeU/scDFM/blob/main/LICENSE>
- GARM: <https://github.com/Jerby-Lab/GARM/blob/master/LICENSE>
- Tahoe-x1: <https://github.com/tahoebio/tahoe-x1/blob/main/LICENSE>
- Response decomposition:
  <https://github.com/xinyizhanglab/perturbation-decomposition/blob/main/LICENSE>
- ESM: <https://github.com/facebookresearch/esm/blob/main/LICENSE>
- X-Cell: <https://github.com/xaira-therapeutics/x-cell>

| Asset | Observed license status on 2026-08-31 | Repository policy |
|---|---|---|
| STATE GitHub source | CC BY-NC-SA 4.0; the official VCC FAQ says participation-only use of STATE code by any entrant is a Non-Commercial Purpose, subject to the license | Pin the source revision; preserve attribution and ShareAlike obligations; keep the official FAQ interpretation with the run manifest |
| STATE weights | Arc State Model Non-Commercial License plus Acceptable Use Policy; the FAQ says non-commercial entrants may use released checkpoints, while commercial entrants may request a commercial license or 90-day trial | Record the entrant's status, exact checkpoint terms, and any Arc authorization; do not infer that the code exception automatically changes model-weight terms |
| Stack GitHub source | CC BY-NC-SA 4.0 | Same separation and attribution rule |
| Stack weights | Arc non-commercial model license/AUP | Review separately from source code; the STATE-specific FAQ text should not be generalized to Stack |
| PRiMeFlow | Custom Academic Research License, non-commercial; it requires public derivatives, notice to the licensor, and a broad license-back covering derivatives, results, and outputs | Do not copy or modify its code in this private repository without explicit review; use the paper, scripts, and data manifest only under an approved plan |
| PRiMeFlow Hugging Face corpus | Wrapper metadata says CC BY 4.0 | Treat this only as the wrapper status; audit every upstream dataset, processed artifact, and redistribution right before training |
| PerturbDiff | MIT | May integrate with notice and source pinning |
| scDFM | MIT | May adapt with notice; still audit data/checkpoint licenses separately |
| GARM | BSD 3-Clause | May adapt with notice and non-endorsement conditions |
| Response-decomposition reference implementation | MIT | Suitable for an independently pinned lightweight baseline; audit the bundled data sources separately |
| Stable-Shift / PerturbGraph | No license file displayed in the audited repository | Treat as all-rights-reserved for code; reimplement paper concepts or obtain permission |
| PerturbMap | No official implementation found | Use the paper's method description; do not copy unknown code |
| Tahoe-x1 code and weights | Apache 2.0 | Preserve license, notices, and changed-file markings |
| ESM code and ESM2 checkpoint card | MIT | Preserve notices and checkpoint provenance; protein-sequence input sources still require provenance |
| X-Cell repository and X-Atlas placeholder | CC BY-NC-SA 4.0 | Watch for a real release, then re-audit artifact-specific terms before downloading or integrating it |

Dataset terms are independent of model and code terms. No dataset enters
training until its own license and permitted contest use are recorded. The
FAQ's broad permission to use external data does not grant rights that the
team does not already hold.

## 9. Immediate implementation sequence

1. Acquire full K562 GWPS raw counts and mapping; verify the claimed
   approximately 272/300 challenge-target overlap independently.
2. Reproduce the lightweight response-decomposition ridge/MLP audit and
   quantify how much conserved-target signal transfers when `gamma=0`.
3. Acquire full raw HepG2, Jurkat, K562, and RPE1 screens and replace the narrow
   197-target support subset for signal-model training.
4. Build exact raw-count double-held-out evaluation with all six metrics and
   official exclusions.
5. Modify the STATE adapter to emit an absolute/shared response plus a
   separately shrinkable target residual; remove forced target-count
   redistribution.
6. Fit low-rank ridge, GARM-style MLP, and Stable-Shift-style target-feature
   baselines.
7. Add same-target route transport and reliability gating; keep the
   context-target interaction near zero when target-context evidence is absent.
8. Calibrate gamma-Poisson and Dirichlet-multinomial count decoders.
9. Only then train a zero-mean conditional residual flow and compare it against
   the matched non-flow decoder.
10. Use four H100s only after a one-H100 model clears the sealed-fold gates.
11. Freeze one scientifically conservative candidate and one higher-variance
    candidate before any further leaderboard submission.

## 10. Decision summary

The most defensible aggressive system is not a monolithic large STATE,
diffusion model, or flow. It is a modular predictor:

1. a shared context perturbation template that protects expression MSE;
2. a biologically structured, same-target-informed residual that distinguishes
   the 300 interventions;
3. a strongly regularized context-target interaction that activates only when
   route reliability or permitted target-context evidence supports it;
4. a count decoder that reproduces depth, dispersion, zeros, and DE power;
5. an optional conditional residual flow that adds heterogeneity without
   changing the validated pseudobulk mean.

This architecture directly addresses the three unresolved failures:

- **effect intensity** is handled by the signal head and learned shrinkage;
- **expression error** is protected by the shared template and mean-constrained
  decoder;
- **target discrimination** is handled by full-GWPS target evidence,
  biological features, and pairwise/contrastive residual losses.

The leaderboard supports the same conclusion: high PDS alone is not enough,
MSE cannot be omitted from local selection, and a realistic emitter cannot
rescue the wrong perturbation-level mean. Flow matching is promising, but it
belongs after—not instead of—the target-specific response model.
