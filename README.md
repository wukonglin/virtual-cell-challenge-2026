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
Bayesian proxy. It trains the 162-million-parameter `state_sm` architecture for
20,000 optimizer steps with:

1. all six official STATE support matrices and the 5,120-dimensional ESM2
   perturbation embeddings;
2. HepG2 sealed as a zero-shot context for checkpoint selection;
3. node-local HDF5 staging and 16 loader workers on one H100;
4. strict 300-of-300 target-embedding coverage checks;
5. a bounded-memory adapter from the 18,080-gene STATE axis to the official
   18,533-gene VCC axis;
6. real A/B/C control cells as raw-count anchors after STATE inference.

The adapter retains at most 161 modeled effects per context-target group and
keeps all 456 challenge-only genes at their matched-control values. Label-free
response-collapse diagnostics and the official VCC dry run must pass before a
package can be submitted. The pinned 2026 STATE repository calls this model
`state_sm`; the saved output in the older official notebook has the same
128-cell, 672-hidden, four-layer architecture even though its source cell says
`model=state`.

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
6. Generate the 360,000-cell prediction on a high-memory CPU node.
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
- [Arc Institute cell-eval](https://github.com/ArcInstitute/cell-eval)

This repository is private competition work. No public license is granted.
