# Project Plan

## Scientific objective

Predict target-specific CRISPRi response distributions in unseen cellular contexts from basal control cells, target identity, and permitted public perturbation data. The model must jointly recover the response centroid, effect direction and magnitude, differential-expression set, and realistic cell-to-cell variability.

## Modeling principles

1. **Target identity comes first.** A context-generic response cannot improve perturbation discrimination.
2. **Evaluation must be out of distribution.** Hyperparameters are selected on sealed held-target and held-context folds.
3. **Mean and dispersion are separate problems.** The response centroid is fit before cell-level variance and zero inflation are calibrated.
4. **Complex generators must earn deployment.** Flow or diffusion is promoted only when it beats a matched count sampler on multiple folds and the worst context.
5. **Leaderboard feedback is not a training environment.** Reinforcement learning is limited to offline, constraint-aware search after a validated supervised model exists.

## Milestones

### M0: Reproducible baseline and repository hardening

Status: completed.

- Audit the official control contract and H100 execution path.
- Fit the conservative public-prior baseline.
- Generate and package a valid 360,000-cell submission.
- Record the published six-metric result.
- Establish an English-only private repository with secret and data exclusions.

### M1: Sealed public validation

- Build leave-one-context-out and target-cluster-held-out splits.
- Refit all learned preprocessing inside each outer fold.
- Implement scorer-aligned pseudobulk and differential-expression proxies.
- Record fold-level and worst-context metrics with uncertainty.

Exit criterion: every promotion decision is supported by sealed-fold evidence rather than training reconstruction or leaderboard feedback.

### M2: Strong statistical transfer

- Balance public targets and merge replicate panels before estimating priors.
- Learn target-specific response factors from sequence, pathway, network, and perturbation evidence.
- Calibrate context adaptation without guessing anonymous cell-line identities.
- Fit effect magnitude and significant-gene cardinality per held-out context.

Exit criterion: clear improvement over the generic baseline in target discrimination, fold-change accuracy, and direction fidelity on most folds without harming the worst context.

### M3: STATE baseline and adaptation

- Audit checkpoint gene and target vocabulary coverage before inference.
- Reproduce a public-data STATE baseline under the same sealed splits.
- Fine-tune only when target/context transfer is measurable and leakage-free.
- Compare compute, calibration, and component metrics against the statistical model.

Exit criterion: STATE must beat the best statistical baseline on aggregate and worst-context criteria.

### M4: Conditional distribution modeling

- Use real control cells as anchors.
- Compare negative-binomial, Dirichlet-multinomial, residual-flow, and diffusion samplers with the same mean head.
- Calibrate responder fraction, dispersion, zero fraction, and library size.

Exit criterion: a generative sampler improves DE overlap or direction metrics without degrading centroid metrics.

### M5: Ensemble and final-round freeze

- Construct a small, ablated ensemble from independently validated models.
- Freeze environments, configurations, seeds, and submission scripts before final controls arrive.
- Run the final D/E/F pipeline without reusing A/B/C labels or assumptions.
- Preserve one reviewed fallback candidate.

## Three-person ownership

### Integration and submission owner

- repository integration and release tags;
- cluster and packaging infrastructure;
- final schema, checksum, and quota checks;
- single-authorized submission execution.

### Public-data and transfer owner

- dataset provenance and licenses;
- statistical baselines and STATE experiments;
- target representations and cross-context transfer;
- leakage-safe split implementation.

### Evaluation and generation owner

- local proxy metrics and error analysis;
- count-distribution calibration;
- residual flow or diffusion experiments;
- independent candidate QC.

Ownership rotates for reviews so no subsystem has a single reviewer.

## Decision log rules

Every promoted model must record:

- hypothesis and rejected alternatives;
- data and split definitions;
- configuration, seed, commit, and input hashes;
- hardware and runtime;
- per-fold component metrics and uncertainty;
- known limitations and rollback target.
