# Public effect reliability: completed cache-only audit

The public cohort contains a reproducible response component in both sources,
but individual split-half estimates remain noisy. This supports a small next
unseen-target/source modeling experiment, not a claim of GO predictability,
challenge-score improvement or readiness for large flow/Feng training. The failed
v2 model-quality gate remains unchanged; no model was fitted or promoted here.

## What was executed

The [fixed protocol](PUBLIC_EFFECT_RELIABILITY_V1_PROTOCOL_20260909.md) was
hash-registered before the new statistics were computed. Only the previously
authenticated v2 cache was used: 56 common targets, 398 matched groups, 4,083
treated cells, 3,104 reference controls and 3,104 disjoint former context controls.
The latter became pseudo-treated NTCs explicitly for this model-free audit.
There are 97 distinct source/batches; repeated cache control entries are not
independent cells. All 772 exclusions and previous protected artifacts were retained.

For each of 32 fixed seeds, null outcomes match each group's actual 8--32 treated
cells. Control permutations are shared consistently within a source/batch.
Treated and null halves have matching floor/ceil sizes; reference halves are
disjoint 16/16 cells. Target-level summaries average matched batch delta vectors
before squaring. No model outputs, learned embeddings or context encoder were
used. The loader integrity-validated all cache arrays; the old eight-cell sham
pool and cached context summaries did not enter these statistics.

## Results

These are descriptive gene-averaged products/energies in the existing continuous
log1p-CP10k representation, not leaderboard scores. Cross-dot is the mean product
of independently estimated effect vectors; positive values indicate agreement.
Source aggregates weight targets equally and batches equally within target.

| Statistic | Jurkat real | Jurkat matched null | K562 real | K562 matched null |
| --- | ---: | ---: | ---: | ---: |
| Group squared effect | 0.026396218 | 0.022014568 | 0.020587578 | 0.018125167 |
| Batch-pooled target squared effect | 0.010447376 | 0.007261555 | 0.007270141 | 0.005211159 |
| Within-batch half cross-dot | 0.003071585 | -0.000213745 | 0.002071880 | 0.000082410 |
| Between-batch cross-dot | 0.002665350 | 0.000037647 | 0.001904489 | -0.000018026 |
| Normalized half agreement | 0.046824 | -0.005758 | 0.046168 | 0.002883 |

Real within-batch half cross-dot remains positive across all registered cell
partitions: 0.002692--0.003483 in Jurkat and 0.001870--0.002333 in K562. Corresponding
matched-null ranges are -0.001201--0.000586 and -0.000546--0.000699. Real between-batch
cross-dot also exceeds the maximum matched-null partition value in each source
(null maxima 0.000320 and 0.000221). The real between-batch statistic uses all cells
and is invariant across these resamples; it is not 32 repeated confirmations.

At the individual target level, 39/56 targets in each source have positive
between-batch cross-dot above their mean matched-null counterpart. These are
descriptive counts, not selected training/evaluation target lists. No targets were
removed based on the results.

A post-hoc concentration check of the registered per-target statistics finds
median signed real-minus-null between-batch cross-dots of 0.000749 in Jurkat and
0.000792 in K562. The five largest target contributions account for 63.7% / 55.1%
of the signed total excess, so the mean should not be presented as uniform
reliability across all targets. This descriptive check did not filter any target.
Batch-pooled half agreement rises to 0.1154 / 0.1224. Null squared-effect energy
falls by 67.0% / 71.2% under target-level pooling; this changes the estimand and
reduces sampling variability, not a model's prediction error on the same task.

## Interpretation and limitations

The earlier eight-cell null comparison understated how important sample-size
matching is. With matched sizes and independent half references, the data show
more reproducible structure than those earlier raw error comparisons alone could
establish. However, half agreement around 0.046 is weak: precise effect estimation
remains difficult at mostly 8--11 treated cells per group. Agreement is
`2*dot/(energy_A+energy_B)`, averaged over defined values—not Pearson correlation,
ICC, a formal reliability coefficient or an explained-signal percentage.

The real-minus-null squared-effect difference is not pure biological signal:
treated/control variance differences divided by sample size also contribute.
Reproducible effects can include common shifts and systematic confounding; this
audit cannot establish target-specific semantics or cross-source prediction.
All 32 seeds reuse the same finite cells. Their ranges are sensitivity summaries,
not confidence intervals or independent biological replicates. No p-values,
outcome-based filtering, model selection or revised v2 pass decision were used.

## Next bounded modeling experiment

A metadata-only review found all 56 targets connected through shared reference
control batches within each source. Ordinary target-only cross-validation on
these responses would therefore share reference sampling noise between fitting
and evaluation. Before the next fit, register a joint target/source holdout:

1. Four deterministic target folds, each holding out 14 targets across both sources.
2. For each fold, fit the other 42 targets in one source and evaluate the held 14
   only in the opposite source; then reverse the source direction.
3. Freeze lambda=10 and gain=1 without tuning this exposed cohort. Compare unchanged
   controls, context only, true GO and three fixed shuffled-GO comparators:
   eight folds/directions times six arms, or 48 small closed-form fits.
4. Rebuild GO vocabulary and stable transforms from fitting targets only. Keep
   C9orf16, RGPD6 and SMN2 explicitly missing rather than dropping them. Hash-based
   target folds are not GO-family-disjoint folds.
5. Implement this in a new version: the sealed v2 decoder intentionally rejects
   unseen targets. Report exploratory joint target/source generalization, not an
   untouched final test. Keep nonnegative/sparsity and control-baseline checks.

This experiment is recommended but **not executed by this audit**. Any subsequent
strength tuning requires a separate inner calibration design and independently
referenced outcomes. Promoter features, a larger flow model, H100 allocation and
Feng post-training remain deferred pending a meaningful model-quality gain.

## Verification and tracking

The implementation underwent an independent code review and synthetic checks for
matching sizes, odd splits, disjoint references, repeated control pools, negative
cross-dots, constant full-32 null means, vector-before-square aggregation and
report/provenance integrity. The related regression suite passed **875 tests in
14.91 seconds**, including 60 tests for the new statistic/tracker/runner.

W&B 0.19.11 recorded the audit offline as validation run `d36b21333a1a`: 32 global
partition events, two source summaries and one global summary, **35 events total**,
zero tracking errors. Analysis events are not optimizer steps. No cloud sync or
raw-artifact upload is claimed. The old tracking implementation remains unchanged.

Artifacts: `artifacts/public_flow/effect_reliability_v1_20260909/contract.json` and
`run/{report.json,complete.json,tracking/}`. Report values and completion hashes
were independently authenticated after execution.
The observed per-source group energies reproduce the previous v2 control errors
exactly. The original v2 diagnostic code/input bindings and its failed quality
decision were rechecked unchanged after this audit.

| Artifact | SHA-256 |
| --- | --- |
| Protocol | `37fd830c6e4611eda2ed33d10394fc4760072970d6fade809fab614ac61a08d1` |
| Statistics implementation | `20d77f32852a1387bfbea500cf5f38b8db9859ccf25afa9ae3db3a5a52f7d850` |
| Diagnostic contract | `387de23fbe86ec960b5c4a85e2a6890d791fdc54391a51a58c736a79c08cc410` |
| Report | `2e6a8923a8da2b3a791b7c12853e384d128e4a4cd91814245e00274e41c01468` |
| Tracking receipt | `06b9f76ae22035b1f80625ed9abcca81892660e5d616324ceae2da12b4df3b3a` |
| Completion receipt | `5a32d137a14b2f8bc8a2c09860008fbfe41e53bec49addb49c8dc908dbf8e0b5` |

No new raw expression, H1/RPE1/HepG2/challenge/Feng source reads, downloads,
GCP requests, GPU jobs, training/post-training, challenge submission or Git push
occurred in this continuation. Existing user changes were preserved.
