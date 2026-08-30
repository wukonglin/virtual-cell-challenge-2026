# VCC 2026 — English Speaker Notes

Companion notes for `VCC_2026_Three_Person_Team_Strategy_EN.pptx`.
All factual competition statements use the 30 August 2026 evidence snapshot.

## Slide 01 — Virtual Cell Challenge 2026

Open with the decision problem, not the architecture. The current near-zero score is a normalized benchmark result, not an accuracy percentage. This deck explains the biological task, exact evaluation, and a three-person operating plan.

## Slide 02 — The decision in 60 seconds

State the four strategic calls. The team should agree on these before discussing implementation details.

## Slide 03 — Road map

Use this map to orient a mixed biology/ML audience. The deck moves from assay to score to system to execution.

## Slide 04 — The challenge in one sentence

Define one condition as a question: given all non-targeting controls for an unseen context and one target gene, generate 400 post-CRISPRi cells over the fixed gene axis. There are 900 such questions in a round.

## Slide 05 — What CRISPR interference changes

CRISPRi uses a catalytically inactive Cas9 recruited to a target locus, commonly with a repressor domain, to suppress transcription. It is a knockdown rather than a DNA-cutting knockout. The challenge reports greater than 80% on-target knockdown for scored targets.

## Slide 06 — What Perturb-seq measures

Perturb-seq links a guide identity to a whole-transcriptome readout in each cell. Pooling provides scale; single-cell measurements reveal heterogeneous responses that pseudobulk alone cannot represent.

## Slide 07 — How to read a single-cell count matrix

Explain rows, columns, counts, and zeros. A zero can reflect true absence, low abundance, or finite UMI sampling. The matrix is sparse in storage even though this dataset has substantial per-cell coverage.

## Slide 08 — Why 400 cells per perturbation matter

There is no one-to-one pairing between predicted and measured cells. Four hundred cells determine group means, variance, zero fractions, and Wilcoxon power. Repeating one centroid 400 times can distort differential-expression calls.

## Slide 09 — Control cells define the context

The anonymous label is not the context representation. The control population contains basal state, cell-state composition, depth, covariance, and technical noise. Preserve A/B/C labels exactly; swapping labels can look like a weak model rather than a file error.

## Slide 10 — The causal quantity of interest

For each target we observe controls but not the matched counterfactual in challenge contexts. Model the response as a delta from the context-specific control profile. Separating baseline and effect improves transfer and calibration.

## Slide 11 — Zero-shot means multiple extrapolations

Final evaluation changes both cell contexts and target panel. In addition, public training data may differ in assay, time point, perturbation modality, and sequencing depth. Use target representations and context representations that extrapolate.

## Slide 12 — Validation is not the final test

Keep development and final evaluation conceptually separate. Final controls and a different target panel arrive October 22. The architecture and decision rule should be frozen before that date.

## Slide 13 — The exact prediction contract

The upload contains exactly 360,000 predicted cells: 300 targets times 400 cells times 3 contexts. All 18,533 genes are present. One sparse .vcc file is required and controls must not be included.

## Slide 14 — Downloaded controls: measured facts

Summarize what was actually downloaded and audited. Each context contains 18,400 NTC cells: 46 guide IDs times 400 cells. All numeric and axis checks passed. Context B is somewhat sparser and shallower than A/C.

## Slide 15 — The three contexts are genuinely different

The control centroids are not interchangeable. Pairwise cosine similarities in a smoke subset range from 0.721 to 0.808. This is evidence that context conditioning matters, not an attempt to identify the anonymous lines.

## Slide 16 — The end-to-end H100 smoke test passed

Report the actual infrastructure result. Job 792580 completed on an H100 80GB. It loaded real controls, ran BF16 forward/backward/update, built a full sparse 360,000-cell artifact, passed the official dry run, and packaged a .vcc file.

## Slide 17 — What the smoke test did not prove

Be explicit: the smoke output repeats a control-derived null population across all targets. It is named DO_NOT_SUBMIT. It proves mechanics, not target-specific biological prediction or leaderboard performance.

## Slide 18 — The score is a ruler—not a percentage

The normalized score uses two anchors per context and metric: zero is a context-mean perturbation-response baseline; one is split-half experimental reproducibility. Negative means worse than baseline. The current −0.0206 is not −2.06% accuracy.

## Slide 19 — Six metrics = six failure modes

Introduce the six metrics as distinct diagnostic lenses. PDS and MSE mainly evaluate group-level effects; the four DE metrics also depend on cell-level dispersion because significance is estimated from 400 cells.

## Slide 20 — PDS: make each target response identifiable

PDS compares each predicted perturbation delta with the panel of measured deltas and asks where the correct target ranks by cosine distance. A generic stress response shared across targets will not discriminate perturbations. All panel targets are excluded from the feature axis for PDS.

## Slide 21 — Expression MSE: put the centroid in the right place

Expression MSE focuses on normalized group-level profiles with sampling-noise correction and a real-effect normalization. It rewards centroid and magnitude accuracy. It does not by itself guarantee calibrated cell-to-cell variability.

## Slide 22 — Four DE metrics probe different mistakes

The four DE metrics are complementary. Fidelity balances sign correctness with coverage; reach measures a high-confidence prefix; Jaccard compares significant sets; NMAE measures fold-change magnitude on reference-significant genes.

## Slide 23 — The metrics pull in different directions

Explain the central multi-objective tension. Amplifying deltas can help reach but hurt MSE/NMAE. Shrinking everything protects MSE but destroys discrimination. Excess variance hides DE; insufficient variance over-calls it. Calibration and ensembling are first-class components.

## Slide 24 — A scoring-aware hierarchy of needs

This hierarchy prevents premature optimization. File validity is foundational. Target-specific mean effects come next. Direction and magnitude follow, then cell-level calibration, and only then expensive generative or RL refinements.

## Slide 25 — Baselines before deep models

Before deep models, build a baseline ladder with increasing biological specificity. Each rung isolates value added. A sophisticated model that cannot beat similarity-weighted low-rank transfer on held-out contexts should not be promoted.

## Slide 26 — A scientific decomposition of the response

Use a decomposed response: context baseline plus a global target effect, a context-target interaction, and stochastic residual. Low rank and shrinkage make the interaction estimable. External data provide perturbation supervision; challenge controls provide context.

## Slide 27 — External data are the training set

The challenge controls contain no matched perturbations, so supervised signal must come from licensed external data. Build a provenance registry, harmonize gene symbols and modality, and measure transfer under held-out cell-line splits. Source diversity matters more than raw cell count alone.

## Slide 28 — Recommended hybrid system

This is the recommended system. A context encoder summarizes NTC cells. A perturbation encoder represents unseen genes using multi-view priors. A mean-effect predictor produces a calibrated delta. A count generator samples 400 cells anchored to real controls. The local scorer calibrates and blends candidates.

## Slide 29 — Context encoder: summarize a population

The context encoder should accept a set or distribution of cells and be insensitive to their order. Start with robust pseudobulk and module summaries; then compare learned set encoders. Include nuisance statistics explicitly so biology is not confounded with depth or sparsity.

## Slide 30 — Perturbation encoder: represent an unseen gene

The final target panel changes, so the perturbation encoder must generalize by biology. Combine learned perturbation signatures with priors such as protein/gene embeddings, pathways, regulatory graphs, and basal target expression. Missing priors should reduce confidence rather than produce arbitrary effects.

## Slide 31 — Mean-effect predictor: optimize the signal first

Predict the perturbation effect in a low-dimensional response basis, then decode to genes. Train with pseudobulk and direction-aware losses. Use shrinkage and calibrated interaction terms to avoid overfitting sparse context-target evidence.

## Slide 32 — Generate raw counts without losing the mean

Convert the mean profile into 400 raw-count cells. A pragmatic first choice is control-anchored negative-binomial or Dirichlet-multinomial sampling: sample a real control state, apply a positive mean transformation, draw a library size, then draw counts with calibrated dispersion.

## Slide 33 — Where diffusion or flow matching helps

Diffusion or flow matching is useful for the residual distribution after a reliable mean effect exists. Work in a compact latent or residual space, condition on context and target, and reconstruct counts through a likelihood-aware decoder. Compare against a simpler NB/DM sampler.

## Slide 34 — Where reinforcement learning helps—and why it is optional

RL is not the core method because there is no sequential environment and no challenge ground truth for A–F. The action space is enormous and the reward is a small set of panel-coupled metrics. If used, RL should update a small adapter against exact offline metrics on public held-out contexts with KL and diversity constraints.

## Slide 35 — LOCO validation is the scientific backbone

Random cell splits leak cell-line identity and greatly overstate transfer. Use leave-one-cell-line-out outer folds and target-held-out inner folds. The exact metric emulator must be applied to generated 400-cell groups, not only model-space losses.

## Slide 36 — Ablations and decision gates

Pre-register ablations so each experiment has a hypothesis, target metrics, and stop rule. This table is a suggested initial sequence. A component is promoted only if it improves multiple held-out contexts without catastrophic metric regressions.

## Slide 37 — Three owners, one integrated system

Assign three durable ownership lanes. P1 owns biological transfer and scientific model selection. P2 owns generative architecture and count calibration. P3 owns scorer parity, cluster operations, artifacts, and submissions. Every critical workflow has a backup.

## Slide 38 — Decision ownership: compact RACI

Use a small RACI table for the critical workstreams. The purpose is not bureaucracy: it ensures that public-data rights, metric parity, model architecture, and final upload each have exactly one accountable owner.

## Slide 39 — Timeline to the final submission

The calendar is built backward from October 22 and November 5. The key management choice is to freeze architecture before final controls arrive. Final phase should be adaptation, calibration, generation, and QC—not research redesign.

## Slide 40 — The first 14 days

This is the concrete first two-week plan. It starts by retrieving the current metric breakdown and reproducing the existing artifact, then locks data provenance and LOCO evaluation, and ends with a trusted baseline plus a single reproducible full-generation rehearsal.

## Slide 41 — Submission protocol: learn without overfitting

Treat every leaderboard submission as an expensive scientific experiment. Keep immutable artifacts and SHA-256 hashes, use paired comparisons, record the panel and anchor stamp, and retain the prior champion. Two submissions per day is a ceiling, not a target.

## Slide 42 — Risk register: protect the result

Review these risks weekly. The most dangerous failures are not always modeling failures: a context swap, scorer mismatch, invalid raw counts, or a late upload can erase months of scientific work.

## Slide 43 — Final-phase 48-hour playbook

The final 48-hour process is a controlled deployment. Verify checksums and labels, generate three pre-frozen candidates, run structural and distributional QC, apply the predeclared selection rule, and submit with recovery margin. Do not infer hidden cell-line identities.

## Slide 44 — Decisions to make today

End the discussion with decisions, not broad agreement. Assign names to P1/P2/P3, define the first baseline and data sources, choose a metric-emulator owner, set compute budget, and freeze the next review date.

## Slide 45 — The first build we should ship

This is the recommended minimum viable scientific system. It is intentionally simpler than a full diffusion model. It provides a trustworthy baseline, a biologically meaningful effect predictor, calibrated raw-count cells, and a complete audit trail.

## Slide 46 — Primary sources and local evidence

Use the primary sources for rules and biology. The deck distinguishes official facts, measured local evidence, and team recommendations. Access dates are 30 August 2026 for web documentation.

## Suggested presenter split

- P1 (Biological Modeling): slides 1–15, 18–27, 44–46.
- P2 (Generative & Optimization): slides 28–34 and model questions.
- P3 (Evaluation, MLOps & Competition): slides 16–17, 35–43.
- For a 20-minute talk, use slides 1, 2, 4, 5, 8, 11–13, 16–19, 23, 28, 33–35, 37, 39–41, 45.
