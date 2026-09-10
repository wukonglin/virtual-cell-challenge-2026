# Controlled control-anchored Lingshu/scDFM experiment

Plan frozen before v2 fitting on 2026-09-09. V1's full candidate passed official
format validation but failed all four public model/control quality comparisons;
it was not submitted. Its checkpoints and receipts remain intact.

## Evidence and hypothesis

Randomly paired K562 single-cell differences have average MSE approximately
0.925, while target-group mean effects have MSE approximately 0.0217. Single-cell
variation is much larger than the target-mean supervision signal. K562 versus
RPE1 control-centroid distance is approximately 0.3722, comparable in scale to
v1's failed RPE1 output error of 0.3375. A target-blind Gaussian velocity
predictor fitted only on K562 training means/variances also obtains low
teacher-forced validation error. These observations support, but do not alone
prove, source-context drift and a denoising shortcut.

The next model therefore learns context-agnostic log-expression effects and
adds them to recipient controls. It does not claim to learn context-dependent
response modulation. The original public cache, 512-gene panel, 18,077-gene
normalization axis, target split, and entire RPE1 holdout stay unchanged.

## Registered recipe and ablations

- Use the pinned, unmodified official scDFM PAD through a new wrapper. Give PAD
  residual state `x_t - control` and a zero control branch. This makes the
  learned velocity invariant to shifts in recipient baseline expression.
- Supervise each K562 training target with its measured group mean expression
  minus its sampled-control group mean, using training cache rows only. Do not
  use raw randomly paired individual-cell differences as the effect label.
- Start the residual flow at zero with no initial noise. Use MSE conditional
  flow matching. Exactly half of each training batch has time zero, forcing
  direct conditioning-to-effect supervision; the other half has uniform time.
  This reduces the residual interpolation shortcut `velocity = residual/time`.
- Center frozen Lingshu vectors using unique training-target features and
  divide by a train-fitted scalar RMS scale. Store the transform with the model.
- Compare three predeclared feature arms: `true`, `shuffled`, and `constant`.
  Use a reproducible feature permutation for the shuffled arm; the constant
  arm receives zero transformed features. No generated gene annotations or
  hand-assigned knockdowns are introduced.
- Use 4,000 steps, batch 32 distinct training target groups, seed 20260909,
  width 128, four heads, two layers, and evaluate every 400 steps. Reuse the
  H100 environment; each fit is single-GPU. The launch job runs the three
  arms sequentially within one H100 allocation.
- Select the earliest minimum equal-kind mean error of actual 20-step,
  zero-start group-effect rollouts. Selection uses all cached cells to estimate
  each selected validation group's measured mean effect, not teacher-forced
  interpolants containing treated endpoints.
- Freeze that checkpoint, then separately compute the existing 16-cell,
  32-groups-per-kind rollout diagnostics against unchanged controls. These
  final diagnostics do not change the selected checkpoint. The same public
  development groups inform selection and diagnostics, so this is not an
  independent test set or a claim of unbiased generalization performance.

All v2 checkpoints use a distinct model schema and explicit method metadata.
They must not be passed to the v1 generation program as if interchangeable.
The shared training-summary format is retained only for diagnostic tooling.

## Quality checks and scope

`scripts/check_lingshu_scdfm_public_quality.py` runs after fitting, before any
full challenge generation. Both public holdout kinds must strictly improve
centroid MSE and signed unbiased MMD over unchanged controls, with high-domain
clipping at most 1%. These numerical criteria are unchanged from v1. The full
generated-artifact and official-format audit remains necessary before upload.

A learned training-only global mean effect already gives small improvements
over unchanged controls in exploratory public diagnostics. Consequently, a
passing gate alone does not show that Lingshu contributes useful target
information. Report true-feature performance alongside both feature ablations;
do not claim a Lingshu-specific gain without that evidence.

This launch is a bounded training/diagnostic experiment, not an automatic
submission chain. No 360,000-cell v2 generation or upload will be queued before
its actual public results and feature comparisons are reviewed. Challenge-treated
cells remain unseen. Future stronger validation requires a new independent
public context/target test source, without relabeling these development holdouts
as untouched after iterative recipe changes.

Runtime root: `/home/fs01/yl4259/yl/virtual-cell-challenge-2026`.
Planned outputs: `artifacts/lingshu_scdfm/effect_v2/{true,shuffled,constant}/fit/`.

## Launch record

On 2026-09-09, AIDA CPU integration job **890073** completed successfully:
**16 tests passed**, none skipped. This includes the authentic pinned PAD
gradient/control-anchoring test and the early quality-preflight checks.

H100 job **890074** was submitted with an `afterok` dependency on that test.
Launcher: `slurm/aida_h100_lingshu_scdfm_effect_v2.sbatch`; one H100, eight CPUs,
32 GB host RAM, one-hour limit, three sequential arms. No challenge generation
or submission is included. Completion and quality must be read from the actual
per-arm receipts; a submitted job is not a training result.

The actual H100 start was **2026-09-09 05:00:50 EDT**, on `c0002`; the earlier
scheduler estimate was superseded. All three training steps completed with
exit code zero (1:54, 1:52, and 1:52). The 5:38 allocation ended `FAILED 1:0`
because the final public-quality check failed, not because of GPU, SSH, or
sandbox access. Slurm accounting records one allocated `gres/gpu:h100`, and
the job records NVIDIA H100 80GB HBM3, 81,559 MiB, driver 580.167.08.

CPU comparison **890077** completed successfully. Its report status means
the arms were comparable, not that their biological-quality gates passed.
All three arms failed the overall public-quality gate. No v2 challenge
generation or upload was performed.

| Arm | RPE1 centroid MSE | RPE1 MMD2 | K562 target MSE | K562 target MMD2 |
| --- | ---: | ---: | ---: | ---: |
| Unchanged controls | 0.056355 | 0.013098 | 0.060319 | 0.001588 |
| True Lingshu | 0.070421 | 0.025603 | 0.086802 | 0.018253 |
| Shuffled Lingshu | 0.066936 | 0.022996 | 0.083991 | 0.017013 |
| Constant features | 0.054673 | 0.012686 | 0.061082 | 0.002772 |

Lower is better for these fixed development diagnostics. The true-feature
arm was worse than both feature controls on all four comparisons, so a
Lingshu-specific benefit is not established. The constant arm improved the
RPE1 comparisons but failed both K562 target comparisons. These are not
official challenge scores.

`scripts/compare_lingshu_scdfm_effect_v2.py` and its CPU launcher compare the
three completed arms without choosing a new checkpoint or altering their
predictions. The report retains per-arm gate failures and compares true features
with both feature controls. A favorable comparison remains development-set
evidence, not proof of independent generalization or higher challenge scores.

The next bounded test is documented in [LINGSHU_SCDFM_EXPANDED_V3.md](LINGSHU_SCDFM_EXPANDED_V3.md):
expand only public K562 training rows/control pools, keeping validation arrays,
gene axes, model recipe, and quality criteria fixed. The v2 artifacts remain
unchanged under `artifacts/lingshu_scdfm/effect_v2/`.
