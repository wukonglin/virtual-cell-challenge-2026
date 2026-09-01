# STATE Weights V1 Submission Lane

## Purpose

This branch preserves one authenticated STATE-weight benchmark for the current
VCC team without changing the validated V5.1/V6 work on `main`. The submitted
candidate is the previously generated paired-residual STATE count model. It is
not an unmodified Hugging Face checkpoint and it is not the diagnostic
absolute-pseudobulk STATE effect-prior candidate.

The model name discloses STATE on the leaderboard. The underlying source and
model terms require attribution to Adduri et al., *Predicting cellular
responses to perturbation across diverse contexts with State*.

## Why the released Replogle weights are not submitted directly

Arc publishes several STATE checkpoints on Hugging Face. The closest
full-expression genetic checkpoint inspected for this lane was
[`arcinstitute/st-se-replogle-full`](https://huggingface.co/arcinstitute/st-se-replogle-full)
at revision `d1441f5587ace12c46247b3703c7d25f24f8abb6`.

Its configuration uses `pert_rep: onehot`. A restricted, weights-only audit of
the official `hepg2_0.99/pert_onehot_map.pt` found 2,024 checkpoint labels and
zero overlap with the 300 VCC 2026 validation targets. Direct inference would
therefore have no learned condition vector for any requested target. Replacing
the 2025 input AnnData with 2026 controls cannot repair that missing condition
vocabulary. The sanitized hashes and coverage counts are recorded in the
[checkpoint coverage receipt](../results/state_weights_v1/checkpoint_coverage.json).

The official VCC STATE notebook instead trains a competition-specific model
with the support set and 5,120-dimensional ESM2 perturbation features. The
candidate in this branch follows that model family. A future pretrained-weight
experiment must initialize compatible backbone tensors, rebuild the
perturbation encoder for ESM2, and fine-tune under target- and context-held-out
validation before it is eligible for submission.

## Frozen candidate

| Field | Value |
|---|---|
| STATE source revision | `9bbfe78a434a55205e4de834e1ea99f85f7a3add` |
| Architecture | `state_sm`, 162,626,484 parameters |
| Training job | `859832`, one NVIDIA H100 80GB |
| Training steps | 20,000 |
| Selected checkpoint | step 16,000 |
| Selected validation loss | `1.66663658618927` |
| Checkpoint SHA256 | `e407e51855792d0dc85483a977f63497d9ac9a6893db360d840c7d9ce556780d` |
| Prediction generation job | `860523`, one NVIDIA H100 80GB |
| Package job | `860524` |
| Prediction shape | `360000 x 18533` |
| Context-target groups | 900 |
| Cells per group | 400 |
| Stored nonzeros | 1,920,065,097 |
| VCC container SHA256 | `a286243bab905c12d84cb83e5460ca18ab8cc0931624c3368ec98fbc9919e4d6` |

For each sampled challenge control cell, the generator runs the selected model
with the requested ESM2 target feature and with the non-targeting feature on
the same input. It transfers only the paired cellwise residual, smoothly bounds
non-target effects, preserves challenge-only genes, and restores the exact
source library size with deterministic largest-remainder integerization.

## Authentication gates

The frozen prediction and package passed all of the following before upload:

- exact `360000 x 18533` shape and official gene order;
- exactly 300 targets per context and 400 cells per context-target group;
- sparse non-negative `int32` raw counts with no control rows;
- exact source-cell library sizes and exact challenge-only counts;
- official target verification and counts-preserving `vcc prep --dry-run`;
- container validation with the pinned VCC CLI `0.1.0`;
- exact package SHA256 re-verification before submission.

Large checkpoints, H5AD files, and VCC containers remain ignored. Only the
code, exact revisions, hashes, configuration, and sanitized server receipt are
versioned.

## Official result

The authenticated container was submitted on 2026-09-01 as `STATE weights v1`
with entry `OYBJARG6NeWPgONA3AXl`. The server published the entry on validation
panel `vcc2026-val-1` with overall score `-0.005934851126896183` and receipt
rank 298. The six normalized component scores were:

| Component | Score |
|---|---:|
| PDS | `-0.005513686296451654` |
| Expression MSE | `0.0` |
| LFC NMAE | `-0.029426427423495067` |
| Direction fidelity | `-0.046735254825253474` |
| Direction reach | `0.07618616804190716` |
| Significance Jaccard | `-0.030119906258084073` |

The container is byte-identical to the earlier `STATE prediction` artifact.
The official component scores are correspondingly identical up to serialized
floating-point precision; only the volatile leaderboard rank changed. This
submission is therefore a reproducibility checkpoint, not a new performance
claim. The complete sanitized response is stored in the
[submission receipt](../results/state_weights_v1/submission.json).
