# Public control-anchor v2: completed, quality gate not passed

The four-arm, two-source validation completed on the BioHPC compute host.
The nonnegative/control-identity requirements passed, but true GO did not beat
shuffled GO in either source. This is improved validation infrastructure, not an
established challenge-score improvement. No model was promoted or submitted;
promoter ablations and Feng post-training remain deferred.

## Frozen scope and cohort

The pre-expression [protocol](PUBLIC_CONTROL_ANCHOR_V2_20260909.md) remains
unchanged. K562 and Jurkat each have the same 56 targets, with 2--4 exact matched
batches per target/source. Whole-source folds use fixed lambda=10 and gain=1,
without tuning on held-source outcomes. These sources have prior project exposure;
this is same-target source/context transfer, not unseen-target evaluation.

| Admitted population | K562 | Jurkat | Total |
| --- | ---: | ---: | ---: |
| Matched target/batch groups | 207 | 191 | 398 |
| Distinct source/batches | 46 | 51 | 97 |
| Unique treated cells | 2,019 | 2,064 | 4,083 |
| Unique context controls | 1,472 | 1,632 | 3,104 |
| Unique anchor/reference controls | 1,472 | 1,632 | 3,104 |
| Unique sham controls | 368 | 408 | 776 |

There are 11,067 unique admitted cells. Controls are reused across target groups;
the 12,736 context, 12,736 anchor and 3,184 sham cache entries are not independent
cells. Three disjoint pools contain 32 context, 32 anchor/reference and eight sham
NTCs per source/batch. Within a fold, training effects subtract the independent
training reference mean; the context mean is not subtracted from its own response.
This removes the shared-noise coupling identified before fitting. Held-source
anchor controls and treated outcomes never enter that fold's fitting calculation.

Raw counts were authenticated and checked as nonnegative integers before CP10k
normalization/log1p over 7,563 jointly measured genes and a fixed 512-gene output
subset. This differs from v1's normalization/output contract: do not compare its
raw metric values directly with the earlier H1 pilot. No H1, RPE1, HepG2,
challenge-2026-treated or Feng expression was opened for this round. All 772 target
exclusions and protected v1/v7 seals remain intact.

GO uses the already downloaded, pinned release: 489 direct terms and 53/56 covered
targets. C9orf16, RGPD6 and SMN2 remain explicitly missing, without alias guesses.
Stable-v2 transforms are fitted inside each training fold. GenePT, ESM, Lingshu
and promoter features are not inputs to these four arms.

## Actual results

Values below are post-projection centroid MSE in continuous log1p-CP10k space,
macro-averaged equally over batches within target, then targets within source.
They are not challenge leaderboard scores or count-space metrics.

| Arm | Held-out Jurkat MSE | Held-out K562 MSE |
| --- | ---: | ---: |
| Unchanged controls | 0.026396218 | 0.020587578 |
| Context only | 0.026369987 | 0.020559768 |
| True GO + context | 0.026367812 | 0.020557562 |
| Shuffled GO + context | 0.026367374 | 0.020557070 |

True GO reduces error against controls by **0.1076% in Jurkat and 0.1458% in K562**,
below the preregistered 1% engineering threshold. It also loses slightly to shuffled
GO in both folds. True-GO MMD squared is 0.023893113 / 0.019068447 (Jurkat/K562),
versus shuffled 0.023892766 / 0.019068027. The required noninferiority to all three
comparators therefore fails as well. True GO beats context-only MSE, but that is
not sufficient to pass the fixed gate.

At target-macro level, true GO wins against controls for 42/56 Jurkat and 37/56
K562 targets; shuffled GO wins for 43/56 and 37/56. These are descriptive counts,
not independent-replicate significance tests. The suite's equal-source true-GO
MSE is 0.023462687, versus 0.023491898 for controls and 0.023462222 for shuffled GO.

## What the nonnegative mapping fixes, and what it does not

The registered predictor is `max(0, anchor_control + predicted_delta)`.
Independent checks of every saved arm/fold verified finite, nonnegative outputs,
exact agreement with positive-part projection of the saved raw values, exact
control-arm identity, and correct separate anchor/treated row mappings. The model
also verifies explicit NTC and zero-delta identity. These are construction and
serialization checks, not evidence of a learned null response.

True GO still has 26.34% / 27.92% negative entries before projection. Their mean
negative magnitudes are small (0.001053 / 0.001218), and the equal-source average
absolute projection correction is 0.000310 in log-expression units. Saving both
raw and projected diagnostics prevents the nonnegative wrapper from hiding this.
The equal-source raw MSE is 0.023466894; projected MSE is 0.023462687.

Nonnegativity alone does not validate sparsity. True-GO exact-zero fractions fall
from the control baseline's 54.49% / 54.05% to 26.34% / 27.92%: small positive
offsets turn many control zeros into tiny positive values. These remain continuous
log-expression predictions, not integer counts or a calibrated count generator.
No silent rounding, library renormalization or count sampling was performed.

True-GO projected shift RMS is 0.001369 / 0.001780, versus observed sampled shifts
of 0.159489 / 0.142378; mean shift cosine is only 0.0617 / 0.0578. The current
fixed, strongly shrunk predictor mostly preserves controls. Its slight numerical
gains do not demonstrate meaningful target-feature use. Observed shifts themselves
contain sampling noise, so their size is not a clean estimate of biological effect.

Disjoint NTC-sham centroid errors are 0.025624522 / 0.020002683, comparable in scale
to control-versus-treated errors. This flags substantial finite-cell variability,
but is not an unbiased noise-floor correction: sham pools have eight cells whereas
treated group sizes vary from eight to 32, and the summaries weight batches/groups
differently. Do not subtract sham MSE or claim a precise fraction of biological
signal from these aggregates. No independent-cell confidence intervals were used.

## Tracking and reproducibility

All four runs use W&B 0.19.11 **offline**, stage `validation`, with three events
each (two fold reports and the final summary), zero tracking errors and verified
output hashes. No cloud sync is claimed. Closed-form ridge fits do not have a
long optimizer-step history and are not resumed flow/Feng post-training.

| Arm | W&B run ID |
| --- | --- |
| Control | `1ec15af12b7c` |
| Context | `0d7940c3d7f3` |
| True GO | `d18edc2c062b` |
| Shuffled GO | `1eb4f988e1d9` |

Artifacts are under `artifacts/public_flow/control_anchor_v2_20260909/`, including
the row contract, cache, GO receipt, diagnostic contract and
`validation/{suite.json,runs/,tracking/}`. All previous artifacts are preserved.

| Artifact | SHA-256 |
| --- | --- |
| Protocol | `85522cd08deba55b6364e1e44be54003407e44e5df50e830197bbf3425ae0e08` |
| Row contract | `a94565b29d14d98588e8dbf58ad25481e1ebc5e20b209db643667d8a5344bbf1` |
| Cache | `13ed1ea60e0dff4d516098e9ef5c4d0520c7847ce2bdfc733082f645a5ee7186` |
| GO features | `834a1e5ff61df85029f37d4c5a67c1102fd4d4967af9068116b21e80a6930b55` |
| Diagnostic contract | `0490a7af920de63a9e81c70d93171fe70253304bbad33edf03756d50695fbc29` |
| Completed suite | `c6eaae863a037bf0826d3f0384eeb781e4c4be29c1e094e66d5f9a25b917044f` |

Final related synthetic regression: **815 passed in 13.88 seconds** across 21
modules. A separate real-artifact check authenticated
the outputs and verified projection, finiteness, nonnegativity and unpaired row
alignment. The first GO command stopped before feature creation because of an
incorrect receipt filename; the corrected existing `acquisition.json` matched
the previously pinned SHA. No source fallback or download was used.

## Next engineering gate

Keep promoter ablations and Feng post-training off this checkpoint. The next
registered experiment should address public effect-estimate precision and show
conditioning value beyond shuffled features before increasing model scale:

1. Estimate response reliability with size-matched disjoint NTC pools and
   target-level batch-aware aggregation; keep this separate from any fitted model.
2. If testing residual strength, register an honest inner calibration/outer
   evaluation split instead of selecting from the two reported source holdouts.
   Check projected sparsity/mean effects alongside MSE, not nonnegativity alone.
3. Add a distinct held-target audit for GO semantic generalization; same-target
   cross-source results cannot establish this claim. Refit transforms/vocabulary
   only within the allowed training targets for that audit.
4. Only after a defensible public validation gain, register output-axis-aligned
   promoter ablations and an authenticated, compatible parent for Feng post-training.

This round used bounded CPU work on a BioHPC compute host, not an HPC head node.
It made no new downloads or GCP requests and consumed no H100 jobs. No Git push
or external challenge submission was performed.
