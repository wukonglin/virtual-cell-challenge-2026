# V6 P3 target-force factorial decision

## Decision

P3 completes a matched two-by-two experiment. It requires exactly two new
HepG2 predictions, not four:

| Arm | Forced model-space target fold | Power-zero residual alpha | Artifact |
| --- | ---: | ---: | --- |
| A | 0.20 | 0.00 | existing V5.1 STATE anchor |
| B | 0.20 | 0.10 | existing V5.1 incumbent |
| C | 0.40 | 0.00 | new P3 STATE-only arm |
| D | 0.40 | 0.10 | new P3 residual arm |

The exact new tags are `p3_tf040_state_a000` for C and
`p3_tf040_p0_a010` for D. Arm B remains the incumbent until a new arm passes
the pre-registered incremental gate.

The existing A and B predictions are valid matched controls. Their generation
reports have the same controls, frozen panel, checkpoint, checkpoint-selection
receipt, STATE support axis, perturbation map, model configuration, generator
SHA-256, seed, cell count, decoder, clipping, integerization, and target-force
setting. Their observation and variable axes are identical, including the
sampled source-control cell for every generated row. The only generator
configuration difference is residual alpha. Regenerating A and B would add
compute without adding an experimental factor.

## Scientific interpretation

`target_policy=force` overwrites the target coordinate with
`log(target_remaining_fraction)` after combining and clipping the STATE and
residual effects. It is a force operation, not a cap: both weak and strong
model-predicted target effects are replaced. The configured value is a
pre-compositional model-space fold. It is not the realized target-count ratio
after the shared-gene composition is renormalized and integerized to preserve
the source library exactly.

This distinction matters. With a configured fold of 0.20, the existing HepG2
STATE anchor has a median realized target-count ratio of approximately 0.0993.
Changing the configured fold to 0.40 is therefore a local weakening ablation;
it must not be described as guaranteeing 40% realized counts. It is
biologically motivated by the challenge-panel target effects in the two public
source atlases: the median `exp(target effect)` is approximately 0.4160 in the
K562 essential atlas and 0.4322 in the RPE1 atlas on the available target and
gene intersections. These are log1p-pseudobulk effect summaries rather than
direct single-cell efficiency measurements, so they motivate the direction of
the ablation but do not calibrate its realized count ratio.

The preceding reliability-power-0.5 candidate does not advance. Relative to B
on all 300 HepG2 targets, it changes oriented metrics by approximately
`+0.000312` PDS, `-0.000280` MSE, `+0.000039` NMAE, `-0.001629` fidelity,
`-0.000225` reach, and `-0.000011` Jaccard. It fails the all-target and
direct-target gates, with MSE showing a strict adverse 90% paired-bootstrap
interval. P3 consequently changes one independent structural factor while
returning to the proven power-zero residual.

## Generation and scoring

Submit only C and D on HepG2. Planning authenticates the declared residual for
both arms. Generation of C must report `enabled=false`, `artifact_read=false`,
and no residual artifact in generation provenance, so alpha zero remains an
artifact-unread inference path.

```bash
sbatch --parsable \
  --export=ALL,VCC_CONTEXT=HepG2,VCC_OUTPUT_TAG=p3_tf040_state_a000,VCC_RESIDUAL_ALPHA=0.00,VCC_TARGET_REMAINING_FRACTION=0.40,VCC_RESIDUAL_NPZ=artifacts/public_v51/hepg2_state_residual_top100_v1.npz,VCC_RESIDUAL_JSON=artifacts/public_v51/hepg2_state_residual_top100_v1.json \
  slurm/h100_generate_public_candidate_v6.sbatch

sbatch --parsable \
  --export=ALL,VCC_CONTEXT=HepG2,VCC_OUTPUT_TAG=p3_tf040_p0_a010,VCC_RESIDUAL_ALPHA=0.10,VCC_TARGET_REMAINING_FRACTION=0.40,VCC_RESIDUAL_NPZ=artifacts/public_v51/hepg2_state_residual_top100_v1.npz,VCC_RESIDUAL_JSON=artifacts/public_v51/hepg2_state_residual_top100_v1.json \
  slurm/h100_generate_public_candidate_v6.sbatch
```

After each generation job succeeds, score the corresponding immutable output:

```bash
sbatch --parsable \
  --dependency=afterok:C_GENERATION_JOB_ID \
  --export=ALL,VCC_CONTEXT=HepG2,VCC_OUTPUT_TAG=p3_tf040_state_a000 \
  slurm/cpu_score_public_candidate_v6.sbatch

sbatch --parsable \
  --dependency=afterok:D_GENERATION_JOB_ID \
  --export=ALL,VCC_CONTEXT=HepG2,VCC_OUTPUT_TAG=p3_tf040_p0_a010 \
  slurm/cpu_score_public_candidate_v6.sbatch
```

Do not score a prediction unless `generation_verified.json` passes. Each score
must also end with a passing `scoring_verified.json` before comparison.

### Failed-run recovery

Candidate and scoring directories are immutable by design. A failed job can
therefore leave a planned spec, partial prediction, or incomplete scoring
views that deliberately block reuse of the registered tag. First confirm the
job is no longer running with `sacct`/`squeue`; never merge or repair files in
place. Quarantine the entire failed directory with one recoverable rename,
record the failed Slurm job ID, and then rerun the same registered tag:

```bash
mkdir -p artifacts/public_v6/quarantine
mv artifacts/public_v6/candidates/hepg2/REGISTERED_TAG \
  artifacts/public_v6/quarantine/REGISTERED_TAG.generation-failed-JOB_ID

# Use this second move only when an incomplete scoring directory exists.
mv artifacts/public_v6/scoring/hepg2/REGISTERED_TAG \
  artifacts/public_v6/quarantine/REGISTERED_TAG.scoring-failed-JOB_ID
```

Do not quarantine a directory from a running or successfully verified job.
The quarantine path must not already exist. Retain it until the replacement
run passes and its receipt hashes have been independently checked.

## Required contrasts

Use the same authenticated scorer results and target manifest for every
contrast. The four causal contrasts are:

1. `C - A`: target-force effect without the residual.
2. `D - B`: target-force effect with the residual; this is the primary
   incremental P3 contrast.
3. `B - A`: residual effect at target fold 0.20; already measured by V5.1.
4. `D - C`: residual effect at target fold 0.40.

The descriptive interaction is `(D - C) - (B - A)`, equivalently
`(D - B) - (C - A)`, calculated separately for each oriented metric. Because
official MSE is a ratio of sums, the interaction is a difference between four
official aggregate ratios, not an average of per-target normalized MSE rows.
Do not claim an interaction confidence interval unless all four arms are
bootstrapped with the same target resampling indices.

Two deployment comparisons are also required:

5. `C - B`: replace the incumbent with weaker target force and no residual.
6. `D - B`: replace the incumbent with weaker target force while retaining the
   residual.

Run the primary deployment comparisons as follows:

```bash
.venv-state/bin/python scripts/compare_public_v6_sequential.py \
  --manifest dataset/public_v51/split_manifest.csv \
  --context HepG2 \
  --role primary \
  --baseline-label v51_alpha010 \
  --baseline-results artifacts/public_v51/scoring/hepg2/v51_alpha010_cell_eval2/results.csv \
  --baseline-scoring-receipt artifacts/public_v51/scoring/hepg2/scoring_receipt.json \
  --baseline-receipt-role v51_alpha010 \
  --candidate-label p3_tf040_state_a000 \
  --candidate-results artifacts/public_v6/scoring/hepg2/p3_tf040_state_a000/cell_eval2/results.csv \
  --candidate-scoring-receipt artifacts/public_v6/scoring/hepg2/p3_tf040_state_a000/scoring_verified.json \
  --candidate-candidate-spec artifacts/public_v6/candidates/hepg2/p3_tf040_state_a000/spec.json \
  --output-json artifacts/public_v6/comparisons/hepg2_p3_tf040_state_a000_vs_v51.json

.venv-state/bin/python scripts/compare_public_v6_sequential.py \
  --manifest dataset/public_v51/split_manifest.csv \
  --context HepG2 \
  --role primary \
  --baseline-label v51_alpha010 \
  --baseline-results artifacts/public_v51/scoring/hepg2/v51_alpha010_cell_eval2/results.csv \
  --baseline-scoring-receipt artifacts/public_v51/scoring/hepg2/scoring_receipt.json \
  --baseline-receipt-role v51_alpha010 \
  --candidate-label p3_tf040_p0_a010 \
  --candidate-results artifacts/public_v6/scoring/hepg2/p3_tf040_p0_a010/cell_eval2/results.csv \
  --candidate-scoring-receipt artifacts/public_v6/scoring/hepg2/p3_tf040_p0_a010/scoring_verified.json \
  --candidate-candidate-spec artifacts/public_v6/candidates/hepg2/p3_tf040_p0_a010/spec.json \
  --output-json artifacts/public_v6/comparisons/hepg2_p3_tf040_p0_a010_vs_v51.json
```

Run D versus C with the same command shape and `--role primary` if both new
arms pass against B. Run C versus A as a diagnostic mechanistic contrast;
neither mechanistic contrast may replace the deployment comparisons.

## Promotion rule

For a candidate to pass against B, both the all-target and direct-target
cohorts must preserve official ratio-of-sums MSE, matched-subset NMAE, and
direction reach, while strictly improving PDS or Jaccard. Every evaluable held
target metric must be non-regressing. The default absolute tolerance remains
`1e-12`. Paired 10,000-draw bootstrap intervals are evidence summaries and do
not override a failed point-estimate gate.

Selection is deterministic:

1. If neither C nor D passes against B, retain B.
2. If exactly one of C or D passes against B, select that arm.
3. If both pass against B, select D only if D also passes against C; otherwise
   select the simpler C arm.
4. Generate the selected configuration for Jurkat only after HepG2 promotion,
   and retain it only if the pre-registered Jurkat non-harm gate passes.

The 33 held-target residual rows must be exact zeros. Consequently, C and D
must be count-identical within held-target groups; any difference is a hard
generation failure. A 0.20-versus-0.40 comparison is expected to change held
groups because target force applies to every panel target. Even when held
counts are identical within an alpha contrast, official held metrics can move
through the scorer's panel-wide predicted-control correction; the count-level
identity check remains the leakage and implementation invariant. The exact
command and strict-v2 provenance requirements are documented in
[`PUBLIC_V6_P3_COUNT_IDENTITY.md`](PUBLIC_V6_P3_COUNT_IDENTITY.md).

## Four-arm factorial analyzer

First build the mandatory immutable four-arm factor receipt described in
[`PUBLIC_V6_P3_FACTOR_CONTRACT.md`](PUBLIC_V6_P3_FACTOR_CONTRACT.md). After all
four scorer bundles are authenticated, run the dedicated analyzer.
It re-hashes both V5.1 output roles and both V6 `scoring_verified.json`
receipts, their scorer outputs, candidate specs, generation/view sidecars, and
the exact route manifest. Before any metric analysis or selection, it also
re-authenticates the factor receipt against the exact manifest, shared A/B
V5.1 receipt, and C/D strict specs. The factor receipt binds the four immutable
prediction hashes and its previously computed four-way axes/source-pairing and
C/D held-count gates. The analyzer does not re-scan the four large generation
predictions. It re-hashes V5.1 views reports and the immutable V6 prediction
and truth scoring views, but never loads scoring-view expression matrices.

```bash
.venv-state/bin/python scripts/analyze_public_v6_p3_factorial.py \
  --manifest dataset/public_v51/split_manifest.csv \
  --context HepG2 \
  --factor-contract-receipt artifacts/public_v6/comparisons/hepg2_p3_factor_contract.json \
  --arm-a-label anchor \
  --arm-a-results artifacts/public_v51/scoring/hepg2/anchor_cell_eval2/results.csv \
  --arm-a-scoring-receipt artifacts/public_v51/scoring/hepg2/scoring_receipt.json \
  --arm-a-receipt-role anchor \
  --arm-b-label v51_alpha010 \
  --arm-b-results artifacts/public_v51/scoring/hepg2/v51_alpha010_cell_eval2/results.csv \
  --arm-b-scoring-receipt artifacts/public_v51/scoring/hepg2/scoring_receipt.json \
  --arm-b-receipt-role v51_alpha010 \
  --arm-c-label p3_tf040_state_a000 \
  --arm-c-results artifacts/public_v6/scoring/hepg2/p3_tf040_state_a000/cell_eval2/results.csv \
  --arm-c-scoring-receipt artifacts/public_v6/scoring/hepg2/p3_tf040_state_a000/scoring_verified.json \
  --arm-c-candidate-spec artifacts/public_v6/candidates/hepg2/p3_tf040_state_a000/spec.json \
  --arm-d-label p3_tf040_p0_a010 \
  --arm-d-results artifacts/public_v6/scoring/hepg2/p3_tf040_p0_a010/cell_eval2/results.csv \
  --arm-d-scoring-receipt artifacts/public_v6/scoring/hepg2/p3_tf040_p0_a010/scoring_verified.json \
  --arm-d-candidate-spec artifacts/public_v6/candidates/hepg2/p3_tf040_p0_a010/spec.json \
  --output-json artifacts/public_v6/comparisons/hepg2_p3_factorial.json
```

For every metric and each of the all, direct, and held-target cohorts, the
report contains `C-A`, `D-B`, `B-A`, `D-C`, and `(D-C)-(B-A)`. A positive
oriented value is favorable. Expression MSE is reconstructed independently in
each arm as the official ratio of component sums; no nonexistent per-target
normalized-MSE row is synthesized. NMAE is evaluated only on the exact subset
present in all four arms, and any subset drift is a hard error. Direction
metrics retain the official finite-mean behavior and include four-arm plus
pairwise finite-mask diagnostics.

One accepted bootstrap index matrix is shared by A, B, C, and D for a given
cohort and metric. The same matrix is reused for every contrast, which makes
the interaction confidence interval valid. The report records a SHA-256 of
each accepted index matrix so this invariant is auditable. There are 10,000
paired-target draws with seed `20260901`; invalid direction-metric draws that
contain no finite value for at least one arm are rejected jointly before any
contrast is calculated.

The analyzer also recomputes the existing primary deployment gate for `C` vs
`B`, `D` vs `B`, and `D` vs `C`, then applies the pre-registered deterministic
selection rule above. Bootstrap intervals are evidence summaries only and
cannot override a failed point-estimate gate. Selection of C or D remains
provisional until the selected configuration passes the Jurkat non-harm gate.
