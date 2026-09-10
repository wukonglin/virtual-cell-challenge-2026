# Public-source control-anchor validation v2

## Protocol, frozen before new expression reads

This is a new, bounded engineering experiment following the failed H1 flow pilot
and inconclusive v1 residual ablations. It does not replace any v1 checkpoint or
claim an untouched project-level test. K562 and Jurkat have prior project exposure.
No H1, RPE1, HepG2, challenge-2026-treated or Feng expression enters this round.
All 772 protected target exclusions and the three V7 admission seals remain intact.

Select a common K562/Jurkat target cohort from authenticated source metadata only:
at most 64 targets, at least two eligible exact source/batch strata per target in
each source, at most four strata per target/source, 8--32 treated cells per stratum.
Hash-rank eligible targets, batches and cell identities without expression feedback.
The exact admitted cohort and row identities must be fixed in the row contract
before loading expression. Metadata initially found 56 shared replicated targets;
the final control-pool eligibility check determines the frozen count.

Use disjoint pools of 32 context-estimation controls, 32 prediction-anchor controls
and eight NTC-sham-outcome controls per eligible source/batch (72 total). Metadata
confirmed that this requirement retains 56 targets. Membership is fixed once per
source/batch, reused consistently across target groups, and reported with unique
cell counts. Context summaries must not include anchor or sham cells. All controls
must be explicitly labeled non-targeting, never unassigned guides. A genuine
disjoint NTC-to-NTC sham estimates sampling discrepancy; a hard null gate proves
identity by construction, not that a null response has been learned.

Inside each fold, use the training source's anchor/reference pool to estimate
`treated mean - reference-control mean`; keep its context-estimation pool separate.
Subtracting the context mean from the response while also using it as a feature
would introduce a spurious negative association through shared sampling noise.
The held-out source's anchor pool is prediction/evaluation-only and never enters
that fold's fit. Pool roles rotate with the whole-source cross-validation folds.

Normalize integer raw counts to CP10k over the uniquely measured common gene axis
of K562 and Jurkat, then log1p; model a fixed hash-ranked 512-gene subset. This is a
new normalization/output contract, not the old 7,006/512 axis. Never zero-fill
unmeasured genes or compare raw metric values across incompatible representations.

Evaluate both whole-source holdouts: fit K562, report Jurkat; fit Jurkat, report
K562. Each fold uses only its training-source treated cells and the other source's
allowed control context at prediction time. This tests same-target source/context
transfer; assay and cell-line changes are confounded. It does not test unseen-target
semantics. Fit all feature/context transforms inside each training fold. Use direct
GO annotations from the previously pinned public release and the opt-in stable-v2
constant-column handling. GenePT, ESM, Lingshu and promoter features are not inputs.

Four fixed arms: unchanged controls, context-only ridge, true-GO-plus-context ridge,
and fixed-shuffled-GO-plus-context ridge. Fixed lambda=10, gain=1, penalized intercept,
context scale floor=0.1 and block-width normalization; source, then target, then
batch balanced fitting/reporting. No source-specific hyperparameter search or
adjustment after seeing held-out outcomes. This deliberately avoids calibrating
and evaluating on the same two source holdouts. A later strength search would need
a separate registered nested design or another public validation source.

For each anchor cell, the predictor is explicitly:

`prediction = max(0, anchor_control + predicted_delta)`.

Zero gain, zero delta and explicitly flagged NTCs preserve controls exactly.
Projection is part of the registered predictor, not a silent cleanup of old output.
Retain paired unprojected predictions and report negative frequency, mean negative
magnitude, negative RMS, projection fraction, mean/max correction and centroid
distortion. Report projected and raw centroid MSE and fixed biased RBF MMD squared,
zero fraction, variance and shift direction. These are continuous nonnegative
log-expression outputs, not integer counts, a calibrated count likelihood or a
validated full-gene distribution generator.

Aggregate batches equally within targets and targets equally within each source;
give the two held-out sources equal weight. Reused controls do not justify treating
all groups/cells as independent replicates. Do not attach independent-cell p-values
or confidence intervals. Report per-source and per-target results, cohort sizes,
unique control counts and the two-source limitation.

Predeclared conservative engineering screen: true GO must improve projected
centroid MSE by at least 1% relative to both unchanged controls and shuffled GO in
each source holdout, beat context-only MSE in each, and not worsen projected MMD
relative to any of those three comparators. All outputs must be finite/nonnegative,
with exact zero/null identity. This threshold is an effect-size screen, not a
statistical significance test or evidence of unseen-target semantics. Even a pass
does not automatically authorize checkpoint promotion, challenge submission,
promoter ablation or Feng post-training; inspect the complete report first.

Run small closed-form fits on a BioHPC compute host with bounded CPU threads and
CUDA disabled. Do not consume Anvil H100 allocation for this diagnostic or compute
on a login/head node. Record distinct W&B offline validation runs, fold-level
metrics and code/input/output hashes; verify receipts and zero tracking errors.
No downloads, GCP requests, paid embedding calls or new cloud spending are needed.

The small eight-cell sham pool is noisy and does not establish null calibration
precision. Results will be written separately so these protocol bytes stay frozen.
