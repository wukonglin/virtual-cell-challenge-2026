# Virtual Cell Challenge 2026

Private team repository for reproducible zero-shot CRISPRi perturbation-response modeling in the [2026 Virtual Cell Challenge](https://virtualcellchallenge.org/).

## Objective

For each anonymous cell context and CRISPRi target, predict a distribution of post-perturbation single-cell raw counts. The validation contract is:

- three anonymous contexts: `A`, `B`, and `C`;
- 300 target genes per context;
- exactly 400 predicted cells per context-target group;
- 18,533 genes in the official order;
- one sparse raw-count submission with 360,000 cells in total.

The evaluation cell lines have no released perturbation labels. Models must therefore learn perturbation effects from permitted external data and adapt them to each released non-targeting-control population.

## Tracked baselines

The first submitted model is a conservative context-weighted Bayesian public-prior baseline. It combines:

1. batch-matched effects from public Perturb-seq support data;
2. context weights estimated from the released basal control profiles;
3. directly observed public effects for 17 of the 300 validation targets;
4. real control cells as anchors for generating sparse integer counts.

ESM2 target transfer was evaluated but rejected because held-target cross-validation was worse than a generic response. The production baseline therefore does not claim successful ESM2 zero-shot transfer.

The official validation submission was published on 2026-08-26:

| Field | Value |
|---|---:|
| Entry | `4gqHnJ61sXnaMRRcLPTc` |
| Overall score | `-0.0205876393` |
| Rank at publication | `173` |
| Panel | `vcc2026-val-1` |
| Anchor set | `vcc2026-valA-r4+vcc2026-valB-r4+vcc2026-valC-r4` |

Ranks change as teams submit. Scores should only be compared when the partition, panel, and anchor set match. See [the baseline report](docs/BASELINE_RESULTS.md) for the six component scores and scientific interpretation.

The second tracked model is a real STATE adaptation rather than an ESM-weighted
Bayesian proxy. H100 training job `859832` completed 20,000 optimizer steps for
the 162,626,484-parameter `state_sm` architecture. It uses:

1. all six official STATE support matrices and the 5,120-dimensional ESM2
   perturbation embeddings;
2. HepG2 sealed as a zero-shot validation context;
3. node-local HDF5 staging and 16 loader workers on one H100;
4. strict 300-of-300 target-embedding coverage checks;
5. an explicit validation-aligned checkpoint archive because the upstream
   callback can save weights before validation at the same step boundary;
6. a bounded-memory adapter from the 18,080-gene STATE axis to the official
   18,533-gene VCC axis.

The minimum recorded validation loss was `1.666636586` at step 16,000, so
`step16000.ckpt` is the selected checkpoint; steps 18,000 and 20,000 were worse.
An earlier effect-prior inference on H100 job `860433` completed all 900
context-target pairs, but that pseudobulk bridge is diagnostic rather than the
preferred count-generation path.

The preferred adapter runs target and non-targeting STATE predictions on the
same raw-log1p control-cell chunks and transfers only their paired cellwise
residuals. Non-target modeled residuals are smoothly bounded as
`0.6 * tanh(delta / 0.6)`. The transformed expectations reweight only genes
already observed in each source cell, so a source-zero gene remains zero. A
deterministic largest-remainder conversion restores the exact source-cell
library size, and the 456 challenge-only genes are copied exactly. This design
preserves real A/B/C sparsity and depth while avoiding direct conversion of
STATE's uncalibrated absolute output into counts.

A production-shape-per-group CPU smoke used 400 cells for one target in each
context and produced a `1,200 x 18,533` canonical `int32` CSR matrix with exact
per-cell libraries, exact challenge-only counts, unique observations, no
controls, and no failed internal checks. Full H100 generation job `860523`
then completed the official `360,000 x 18,533` contract with 900 context-target
groups and 1,920,065,097 stored nonzeros. Packaging job `860524` passed the
official counts-preserving dry run, target verification, and container
validation. The submitted container SHA256 is
`a286243bab905c12d84cb83e5460ca18ab8cc0931624c3368ec98fbc9919e4d6`.
The pinned 2026 STATE repository calls this model
`state_sm`; the saved output in the older official notebook has the same
128-cell, 672-hidden, four-layer architecture even though its source cell says
`model=state`.

The STATE direct-count validation submission was published on 2026-08-31:

| Field | Value |
|---|---:|
| Entry | `JbDxq7SJV2wI0DWlIREn` |
| Status | `published` |
| Overall score | `-0.0059348511` |
| Rank at publication | `278` |
| Partition | `val` |
| Panel | `vcc2026-val-1` |
| Anchor set | `vcc2026-valA-r4+vcc2026-valB-r4+vcc2026-valC-r4` |
| Submitted model name | `STATE prediction` |
| Submitted description | `mom~mom~` |

This improves the overall score over the first Bayesian submission on the same
panel and anchor set, but it remains below the official context-mean baseline
of zero. The receipt and component-level diagnosis are recorded in
[the baseline report](docs/BASELINE_RESULTS.md).

## Cross-context candidates

The cross-context system is deliberately modular. It addresses the initial
STATE adapter's three main errors: a missing shared
perturbation response, weak target discrimination, and a count emitter that
could not activate source-zero genes.

1. A K562 genome-wide CRISPRi atlas supplies same-target response evidence for
   267 of the 300 validation targets after a 30-cell reliability filter.
2. A K562-essential to RPE1 residual ridge route transfers target-specific
   response geometry while leaving the shared expression surface unchanged.
3. Whole-ESM2-cluster holdout selects route and amplitude hyperparameters.
   The corrected internal six-member proxy is `0.215398`, compared with
   `0.095302` for the no-route control in the same split. These are local proxy
   values, not leaderboard scores.
4. Direct and fallback effects are merged per target-gene coordinate. Missing
   K562 genes retain a full-axis fallback instead of being silently zeroed.
5. The scorer-aware generator keeps target forcing off, does not redistribute
   target counts, preserves a model-derived absolute pseudobulk mean, and emits
   independent sparse integer expectations with deterministic stochastic
   rounding.

The v4 ESM2 head retains 9,522 of 9,675 reliable K562 training targets and
records the 153 missing-training-embedding exclusions. Coverage of all 300
requested challenge targets remains mandatory.

Before packaging, the generator enforces exact count-mass accounting and
overall/per-context caps on sparsification loss and library drift. Packaging
then re-hashes every response, axis, control, and implementation input against
the generation manifest before running the official VCC checks.

The corrected full-axis effect prior is
`41e910252fdfcdabd6193274c86dfd0582f3388c70acd7c833df1626235bbe14`;
the derived context pseudobulk is
`b892bf83d6e26f1d86781a4385cf7389ad8f344308cc16c319da1afb741b887c`.
A CPU structural smoke passed with no failed checks. V4 subsequently passed
full H100 generation and official packaging, but its validation score regressed
to `-0.0501571428`: PDS and Jaccard improved while raw expression MSE rose from
`4.73891` to `30.18042` and LFC NMAE rose from `1.01769` to `1.18002`.

V5 is an amplitude-rescue calibration, not a larger model. It disables the
dense absolute-pseudobulk/zero-induction branch, shrinks common response to
`0.10`, retains centered target-specific response at `0.22`, and tightens
count-mass gates. Its single-H100 smoke is versioned separately and cannot be
submitted. Its full official-shape generation and package also passed every
format and integrity gate, but no V5 leaderboard submission has been made.
Those facts are historical and distinct from the public-data V5.1 validation
lane described below.

## Authenticated public V5.1 and V6 validation

V5.1 adds a power-zero public perturbation residual at alpha `0.10` to the
paired-count STATE anchor while retaining a forced model-space target fold of
`0.20`. It was evaluated on complete 300-target HepG2 and Jurkat panels with
400 generated cells per target. HepG2 is the primary zero-shot context;
Jurkat, which was present in STATE training, is only an in-distribution
non-harm check. Every score directory was authenticated before comparison
with cell-eval2 commit `5e64833518a6603a0301cbe28185d49c30f4a986`, declared
version `0.16.0`, and pdex `0.3.0`.

Relative to the matched STATE anchor, V5.1 improved all six oriented aggregate
metrics in both contexts. On HepG2, the all-target improvements were PDS
`+0.050569`, official ratio-of-sums MSE `+0.009696`, NMAE `+0.005918`,
direction fidelity `+0.009731`, direction reach `+0.025245`, and Jaccard
`+0.002373`. All six direct-target one-sided 90% paired-bootstrap lower bounds
were positive across 10,000 resamples. The primary zero-shot, Jurkat non-harm,
and joint promotion gates therefore passed. The sanitized decision is
[tracked here](results/public_v51/validation_decision.json); it establishes a
validated public-data incumbent, not an official leaderboard score.

The first V6 P2 experiment redistributed the same residual energy with
reliability power `0.5` and alpha `0.1248983248`. Against the V5.1 incumbent on
HepG2 it changed all-target PDS by `+0.000312`, but regressed official MSE by
`-0.000280`, fidelity by `-0.001629`, reach by `-0.000225`, and Jaccard by
`-0.000011`; the adverse MSE interval excluded zero. P2 failed the all-target
and direct-target gates and was not run on Jurkat. See the
[P2 decision](results/public_v6/p2_reliability_decision.json) and the
[authenticated V6 workflow](docs/PUBLIC_V6_ALPHA_CALIBRATION.md).

P3 then evaluated target remaining fractions `0.20` and `0.40` in a matched
two-by-two design with residual alphas `0.00` and `0.10`. Both new HepG2 arms
completed strict single-H100 generation, authenticated cell-eval2 scoring,
held-target count-identity checks, and a four-arm paired-bootstrap analysis.
The residual remained strongly beneficial at both target-force levels, while
changing the target remaining fraction was nearly neutral. Neither new arm
passed the deployment gate against V5.1: the alpha-`0.10` arm at fraction
`0.40` changed all-target PDS by `-0.000022`, MSE by `+0.000002`, NMAE by
`-0.000002`, fidelity by `-0.000062`, reach by `+0.000054`, and Jaccard by
`-0.000005`. V5.1 remains the incumbent; no Jurkat P3 run or official
submission was made. See the
[P3 decision](results/public_v6/p3_target_force_decision.json).

The downloaded Feng multi-iPSC atlas adds 850,726 normalized-log1p cells,
6,699 perturbations, and 182 direct validation-target overlaps for future
representation training. It has no raw-count layer and is therefore excluded
from count-emitter training. A separate GEO Jurkat file adds 262,956 raw-count
cells, 2,394 target labels, and 12,013 controls for held-context and dispersion
training, although it has no direct validation-target overlap. See
[data provenance](docs/DATA_PROVENANCE.md) and the
[model-v2 research review](docs/RESEARCH_AND_MODEL_V2.md) for the complete
split, license, and flow-matching analysis.

## Repository layout

```text
configs/       STATE support-set splits used by local and node-local runs
scripts/       Data audits, STATE adapters, prior fitting, QC, and generation
slurm/         H100 training/inference, CPU packaging, and smoke jobs
docs/          Project plan, baseline report, and reproduction instructions
requirements/  Recorded software environments for modeling, CLI, and slides
results/       Small sanitized metric snapshots only
presentation/  English strategy deck, notes, builders, and validators
dataset/       Local data mount point; data files are never tracked
artifacts/     Local model/submission outputs; generated files are never tracked
logs/          Local scheduler logs; generated files are never tracked
```

## Reproduction overview

1. Install the model environment and a separate Python 3.11 VCC CLI environment.
2. Download the official controls through the VCC CLI.
3. Place permitted public support data under `dataset/state_support/extracted/`.
4. Build the public perturbation-effect cache.
5. Fit the prior on one allocated H100.
6. Generate the 360,000-cell prediction on an allocated H100 node and package
   it on a high-memory CPU node.
7. Require all internal scientific checks and the official `vcc prep --dry-run` checks to pass.
8. Package, checksum, review, and submit exactly one approved candidate.

Exact commands and expected paths are documented in [Reproducibility](docs/REPRODUCIBILITY.md).

## Development priorities

The immediate goal is not a larger generator. It is stronger target-specific biology:

1. build sealed leave-one-target and leave-one-context public validation folds;
2. implement scorer-aligned local proxy metrics and DE calibration;
3. improve target representations and cross-context response transfer;
4. calibrate effect magnitude, dispersion, and significant-gene cardinality;
5. evaluate residual flow or diffusion only after it beats the matched statistical sampler;
6. keep reinforcement learning offline and low priority because leaderboard feedback is sparse and delayed.

See [Project Plan](docs/PROJECT_PLAN.md) for milestones and team ownership.

## Data and security policy

- Do not commit official challenge data, public support matrices, model weights, `.h5ad`, `.vcc`, logs, credentials, or API tokens.
- Do not redistribute data unless its license explicitly permits redistribution.
- Use `vcc login --token-stdin`; never place a token in a notebook, script, shell history, issue, or pull request.
- Every submission requires a human review of its commit, configuration, hashes, scientific QC, and official dry-run receipt.

All repository prose, code comments, commit messages, issues, and pull requests must be written in English. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Upstream references

- [Virtual Cell Challenge](https://virtualcellchallenge.org/)
- [VCC CLI guide](https://vcc-cli-wiki.virtualcellchallenge.org/)
- [Arc Institute STATE](https://github.com/ArcInstitute/state), audited at commit `9bbfe78a434a55205e4de834e1ea99f85f7a3add`
- [Arc Institute cell-eval2](https://github.com/ArcInstitute/cell-eval2)

This repository is private competition work. No public license is granted.
