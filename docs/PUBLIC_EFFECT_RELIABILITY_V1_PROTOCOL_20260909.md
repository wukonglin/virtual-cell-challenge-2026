# Public effect reliability: fixed cache-only diagnostic protocol

Written after the exposed v2 source-holdout screen failed, before executing these
new numerical diagnostics. This is retrospective engineering on an existing public
cohort, not a new untouched test. No model fitting, feature selection, checkpoint
selection, promoter ablation, Feng post-training or challenge submission occurs.

## Inputs and fixed roles

Reuse only the authenticated `control_anchor_v2_20260909` cache, its row/source
contracts and protected v1/v7 seals: 56 common K562/Jurkat targets, 398 matched
target/batch groups, 512 outputs normalized over 7,563 shared measured genes.
Preserve all 772 exclusions. Do not reopen raw H5AD expression or introduce H1,
RPE1, HepG2, challenge-treated or Feng observations. Pin code/input hashes before
decoding the cache for this analysis. Keep all previous source files/artifacts
unchanged and write a fresh reliability directory.

In this model-free diagnostic, the previous 32-cell context pool becomes the
NTC pseudo-treated pool. The disjoint 32-cell anchor/reference pool stays the
comparison reference. No context encoder is evaluated or fitted. The old eight-cell
sham pool is not used in numerical comparisons. This explicit role change is not
an independent new dataset and does not retroactively alter the v2 model protocol.

## Sample-size-matched null and aggregation

Use exactly 32 deterministic seeds, integers 20260920 through 20260951 inclusive.
For every seed and source/batch, generate one fixed row-identity permutation of
the pseudo-treated NTC pool and reuse it across every target in that batch.
The first n cells form the null outcome for a group with n real treated cells
(8 <= n <= 32). Compare both real and null means against the same 32-cell reference
mean. No replacement, synthetic cells, extra counts or expression-based selection.

For each group, delta is the outcome mean minus its exact matched reference mean.
Report mean squared delta across genes. For each target/source, additionally
average its batch deltas equally first, then square across genes. This target-level
summary differs from averaging squared per-batch effects; do not conflate them.
Report real and null versions side by side, with equal batches within target,
equal targets within source, and equal sources in global summaries. Preserve
per-target results and the variable treated population sizes.

Reusing a source/batch reference induces dependence across its target groups.
Every corresponding null calculation must retain that same dependency structure.
The 32 seeded permutations reuse the same cells and are not 32 biological
replicates. At n=32 the full pseudo-treated mean is identical across seeds.
Seed means/ranges are sensitivity summaries, not confidence intervals.

## Within-batch and between-batch agreement

For each seed/group, split real treated cells into disjoint floor(n/2) and
ceil(n/2) halves by a source/batch/target-specific row permutation. Split reference
controls into disjoint 16/16 halves by one source/batch-global permutation, reused
across targets. Use the same two reference halves for the corresponding null
comparison. Divide the selected n pseudo-treated NTCs into matching floor/ceil
halves. Context-derived NTC and anchor/reference identities remain disjoint.

For real and null halves separately, calculate mean gene-wise cross-product of
the two half effects, keeping negative values. Also report normalized agreement
`2 * dot / (energy_A + energy_B)`, null when the denominator is zero. This is a
descriptive bounded agreement statistic, not Pearson correlation, ICC, a formal
reliability coefficient or an independent-replicate significance test. Only under
appropriate sampling assumptions does cross-product suppress independent
measurement noise; it does not prove causal perturbation effects.
Also report mean squared disagreement between the two half effects, with the same
matched null statistic. Disagreement alone is not evidence that biology is absent.

For each target/source, calculate the average cross-product over unordered pairs
of distinct matched batches, using full group deltas and the corresponding null
deltas. Different batches use different physical control pools. Positive values
suggest a reproducible component but may include common shifts or systematic
confounding; they do not establish target-specific semantic learning. Keep negative
values and all targets rather than filtering for apparent success.

Do not subtract null errors and label the result a true biological variance or
an unbiased noise-corrected challenge score. Do not use cell/group independence
assumptions to attach p-values or confidence intervals. This audit provides no
automatic pass/promotion decision and does not revise the failed v2 gate.
Even a size-matched real-minus-null energy difference includes treated/control
variance differences divided by sample size, not just squared mean effects.

## Execution and interpretation

Run with bounded CPU threads on a BioHPC compute host, never a login/head node.
No GPU allocation, network acquisition, GCP request or paid embedding API is needed.
Use an opt-in diagnostic tracker with explicit scalar allowlists and W&B offline
validation records. Do not modify the frozen v2 tracking/model modules. Record
source-level and global summaries, seeds and artifact/provenance hashes locally;
never upload raw cells, cell identifiers or model artifacts.

Use the results to distinguish measurement precision limitations from inadequate
predictive modeling. A later model-calibration or held-target experiment needs
its own frozen split, train-only transforms and comparator/gating rules. None is
implicitly launched or selected by this diagnostic.
