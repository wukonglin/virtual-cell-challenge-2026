# Lingshu conditioning v4: public-data adapter screen

## Scope and motivation

This is a new **research-only conditioning adapter**, not a replacement scDFM
submission and not evidence of improved challenge scores. The earlier Anvil
job 20532815 ran successfully on H100 node h001: 38 tests passed without skips,
and true/shuffled/constant fits each completed 4,000 steps. Its final scientific
gate returned exit 1 because true features failed the fixed public centroid
MSE and MMD comparisons. All checkpoints and historical gates are preserved.

The v2/v3 local wrapper deliberately feeds residual expression and a zero
control branch to the official PAD. Its predicted pre-clamp shift is therefore
context-independent. Control-cell variability survives, but learned
context/state-specific response and perturbation-induced covariance changes
cannot be represented. This is a limitation of our wrapper, not a proof that
the upstream scDFM architecture is incapable of context conditioning.

The existing 3,584-to-128-to-128 projection has over 475,000 parameters before
PAD, but training contains only 146 distinct target identities. Earlier
training-only ridge probes found very small, uncertain feature-specific signal.
The v4 screen tests a much smaller fitted target-to-response map before another
flow-training allocation is justified.

## Implementation

`scripts/lingshu_conditioning_adapter.py` provides a separately implemented
adapter:

1. Frozen genuine Lingshu symbol-prompt features, L2-normalized per gene.
2. Training-fold feature centering, unit mean squared **row** norm, and PCA.
   No whitening of tiny-variance directions.
3. Ridge regression to centered training-fold effect PCs.
4. An independently tuned constant mean-effect branch and a shrunk
   target-specific residual branch. A zero residual is an exact fallback.

The fitted response latent is accessible through `condition(features)` for
future conditioning experiments. `predict_effect(features)` reconstructs all
512 pilot genes for scoring, including error in discarded effect directions.
It does not generate individual cells or represent cell context.

Input ranks are 8/32, effect ranks 4/16, and penalties 0.1/1/10. Mean gains are
0/0.5/1; residual gains are 0/0.25/1. The exact constant candidate is included
once, leaving 25 conditional candidates. Ridge solves
`X_pc.T @ X_pc / n_train + penalty * I` with total input variance at most one.
Element-RMS scaling would silently weaken these penalties by approximately
3,584-fold; the row-energy normalization avoids that mismatch.

## Validation contract

`scripts/validate_lingshu_conditioning.py` authenticates the existing expanded
public cache, frozen embeddings, and original extraction receipt against their
registered SHA-256 digests. It writes `contract.json` **before fitting** and
refuses to overwrite an existing output directory.

- Expression access: only `train_x`, `train_control`, `train_targets`, and
  `train_contexts`, plus the non-expression ordered `gene_names` metadata.
  No `val_*` arrays or challenge expression are materialized.
- Data: 28,579 K562 training cells, 146 targets, 512 output genes. Each target
  effect uses all cached training cells and matched controls for that target.
- Three fixed outer seeds, five target-held-out folds per repeat, three inner
  folds. Every PCA, regression, gain and hyperparameter selection stays inside
  the outer-training data; PCA/regression also stay inside inner-training folds.
- True features and three fixed shuffled-feature associations use the same
  feature pool, folds and tuning budget. Optimized constant and zero effects
  are separate controls. Feature shuffling never relabels expression rows.
- A mean gain is selected on inner folds once and shared by every feature arm
  within that outer fold. Conditional candidates are then selected on the same
  inner folds; no outer-target effects influence selection.
- Error is reconstructed 512-gene mean squared group-effect error, equally
  weighted by target. Repeated out-of-fold losses are averaged per target
  before exploratory paired bootstrap intervals; 438 repeated observations
  are not treated as 438 independent targets.

The early-compute screen requires at least 1% improvement over the optimized
constant, improvement over constant and each shuffle in every repeat, paired
exploratory interval upper bounds below zero for those comparisons, and lower
aggregate error than the zero-effect control. Missing controls or nonfinite
errors are rejected. Passing would authorize consideration of a **separate**
context/distribution experiment, not generation or submission. Existing public
MSE/MMD/clipping gates are unchanged.

The final all-training-target fit is exported as `adapter.research-only.npz`
even if the screen fails, so the result can be inspected and reproduced. It
has no deployment permission, binds the ordered gene axis and input/code
hashes, and requires its authenticated result/contract sidecars. It must not
be loaded into a submission route as if it were a passing checkpoint.

## Limits

The 512-gene axis was selected on original training targets before these CV
folds. Cached control cells may recur across target groups, and biological
pathways violate simple target independence. These are exploratory development
diagnostics, not an independent final test or a permutation significance test.
K562-only fitting cannot establish unseen-context generalization. Group means
cannot validate variance, covariance, MMD, perturbation discrimination, count
calibration or full-gene output. No claim about challenge scores follows.

## Other flow directions considered

| Direction | Evidence and decision |
| --- | --- |
| scDFM | The released genetic checkpoint is not a ready-made cross-context, full-axis VCC model. Preserve it as a controlled backbone reference; do not label the architecture categorically unsuitable. |
| CellFlow | Its official tutorial separates perturbation features from control-derived donor features. This is a useful conditioning design reference, not evidence that it will score best here. The reported local mode-collapse pilot was not found in the bounded repository/artifact search; its failure remains user-reported pending receipts. |
| PRiMeFlow | The paper supports full-expression U-Net output and classifier-free guidance. Its ESM2 option initializes expression-gene tokens, not unseen perturbation/context conditions. The 12-GPU, 14-day example is a requested ceiling, not measured runtime. No source is copied across the existing project license boundary. |
| Hybrid | A full-axis stochastic generator with separate continuous target and control-derived context branches is a plausible, unimplemented hypothesis. It requires public multi-context training and held-context/held-target validation, not just a larger K562-only fit. |

Primary references checked 2026-09-09:

- [scDFM paper](https://arxiv.org/html/2602.07103v1).
- [CellFlow official donor-conditioning tutorial](https://cellflow.readthedocs.io/en/latest/notebooks/100_pbmc.html).
- [PRiMeFlow paper](https://arxiv.org/html/2604.13986v1).
- [PRiMeFlow expression-gene embedding implementation](https://github.com/altoslabs/primeflow/blob/main/src/primeflow/modelcore/nn/unet.py).
- [PRiMeFlow example allocation](https://github.com/altoslabs/primeflow/blob/main/scripts/vcc_pretrain.sh).
- [PRiMeFlow license](https://github.com/altoslabs/primeflow/blob/main/LICENSE.md)
  and this project's [recorded code boundary](DATA_PROVENANCE.md#model-code-license-boundary).

Future progression: first establish target-conditioning value. Then fit a
separate control-derived context branch on multiple public training contexts,
keeping an untouched context and target panel. Only then profile a full-axis
stochastic generator on one allocated H100 and increase GPU count if measured
throughput warrants it. Track variance ratios, covariance/effective rank,
MMD, target discrimination and clipping in addition to centroid accuracy.
No hyperparameter changes may be chosen by repeatedly inspecting a purported
final held context.

## Reproduction

Run on the authorized BioHPC compute server, not an Anvil/AIDA head node:

```bash
.venv-state/bin/python -m pytest -q tests/test_lingshu_conditioning_adapter.py tests/test_diagnose_lingshu_training_signal.py
.venv-state/bin/python scripts/validate_lingshu_conditioning.py \
  --cache artifacts/lingshu_scdfm/public_cache_expanded_train_v3.npz \
  --embeddings artifacts/lingshu_scdfm/v1/lingshu_embeddings.npz \
  --embedding-receipt artifacts/lingshu_scdfm/v1/lingshu_embeddings.json \
  --output-dir artifacts/lingshu_scdfm/conditioning_v4
```

This small matrix experiment uses four CPU threads and no GPU allocation.
Unlike the historical training launcher, it reports execution completion and
scientific eligibility separately: exit zero does not mean the screen passed.
Read `result.json:decision.conditioning_screen_passed` explicitly.

## Completed result: 2026-09-09

The four-thread BioHPC compute run completed in 24.62 seconds, after 71 tests
passed (65 new adapter/screen tests plus six previous train-only probe tests).
No H100 allocation or challenge upload was used. Artifacts are under
`artifacts/lingshu_scdfm/conditioning_v4/` and remain excluded from Git.

| Arm | Repeated target-held-out MSE (lower is better) |
| --- | ---: |
| True Lingshu low-rank adapter | 0.01305664663 |
| Shuffled features, seed 20260920 | 0.01305623559 |
| Shuffled features, seed 20260921 | 0.01305301655 |
| Shuffled features, seed 20260922 | 0.01305632967 |
| Optimized constant mean effect | 0.01305293320 |
| Zero effect | 0.01332761458 |

The screen **failed**. True features are 0.02845% worse than the optimized
constant and slightly worse than all three shuffled arms in aggregate.
True-minus-constant MSE is 0.0000037134, with exploratory paired target
bootstrap interval [-0.0000162230, 0.0000240252]. True beats zero effect by
2.033%, but the optimized constant improves still more; this is not evidence
of Lingshu-specific value.

The final training-only selection chose mean gain 0.5 and **residual gain 0**,
turning off target conditioning. This research export is therefore a constant
fit, not a Lingshu-conditioned submission candidate. It is preserved as a
diagnostic result and must not be relabeled or deployed as the requested model.
No public quality gate was weakened, no old experiment was rerun, and no score
improvement is claimed.

Recorded identities:

- Contract SHA-256: `ca8c4a19fbd5120dfd15abeaec9320541f6046e17321b8207f4eb6b702ff27ff`.
- Research export SHA-256: `1f648a9952bd1b2bffd249d6df35636a232adb46f15b1335159a7542bf88d588`.
- Adapter source SHA-256: `6b133099d3b941862529b6b903d79fefd5576e5b9267d79117cdff32d5c709c3`.
- Validator source SHA-256: `8379542df0cefaf972291157342488f84abac6f4965e0aad0d270285234cd806`.

Decision: do not connect this failed target branch to a larger generator.
The next hypothesis should compare biologically grounded protein/function
features and their incremental combination with Lingshu, using the same strict
negative controls. Freeze versioned curated function text, symbol/isoform
mapping, missing-feature handling, and prompts without response labels. Include
function-text/Lingshu-only, sequence-only, fused, shuffled and constant arms;
the fused arm must beat sequence-only before claiming added Lingshu value.
Audit the project's earlier split/selection history before designating any new
target or whole context as untouched confirmation data.
This would be a **new registered experiment**, not a retune
of the completed v4 screen. If useful conditioning emerges, train a separate
control-derived context branch on multiple public contexts before a full-axis
stochastic-flow pilot. The present outcome does not show that every possible
Lingshu representation is useless; it shows that these frozen symbol-prompt
features do not justify scaling this training route.
