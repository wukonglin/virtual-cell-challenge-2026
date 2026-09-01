# V6 P4 STATE-anchor amplitude experiment

## Decision

P4 is a pre-registered, one-factor ablation of the paired STATE anchor. It
does not change the validated V5.1 public residual, target-force policy, count
decoder, sampled controls, or scorer. The factor is the raw STATE effect
weight, denoted `gamma`, with the authenticated set:

```text
gamma = 1.00    new strict-v2 control arm
gamma = 0.75    new P4 arm
gamma = 0.50    new P4 arm
```

The three new immutable tags are:

```text
p4_g100_p0_a010
p4_g075_p0_a010
p4_g050_p0_a010
```

The gamma `1.00` arm is regenerated under the same strict-v2 contract as the
two experimental arms. This avoids treating the less-complete historical
V5.1 receipt as proof of a one-factor contrast. The existing `v51_alpha010`
candidate remains the historical scientific reference, but P4 model selection
uses `p4_g100_p0_a010` as its mechanically matched control. P4 creates no new
residual artifact and every arm uses the exact V5.1 power-zero Top-100 residual.

## Scientific hypothesis

V5.1 established that the centered K562/RPE1 residual at alpha `0.10`
improves all six authenticated public-validation metrics relative to the
paired STATE anchor. P2 showed that target-level count-reliability shrinkage
misallocates residual energy, and P3 showed that changing the forced target
fold from `0.20` to `0.40` is nearly neutral. Neither experiment tested the
amplitude of the STATE anchor itself.

The generator computes the bounded STATE contribution as:

```text
bounded_state = 0.60 * tanh(gamma * raw_state_delta / 0.60)
```

It then adds `0.10 * public_residual`, applies the final symmetric `0.65`
clip, and sets the model-space target coordinate to `log(0.20)`. Gamma is
therefore a pre-tanh raw STATE amplitude, not a linear multiplier on the final
effect. The model-space target override is independent of gamma, but exact
library-size integerization can still change the realized target count ratio
when non-target coordinates change. P4 records and evaluates that downstream
effect rather than claiming that emitted target counts are invariant.

## Fixed configuration

Every arm must retain:

| Field | Locked value |
|---|---:|
| Context for primary screen | `HepG2` |
| Public residual alpha | `0.10` |
| Public residual reliability power | `0.0` |
| Shared public response weight | `0.0` |
| Target remaining fraction | `0.20` |
| STATE bound | `0.60 tanh` |
| Combined effect clip | `0.65` |
| Cells per target | `400` |
| Maximum genes per cell | `5900` |
| Integerization | `largest-remainder` |
| Seed | `20260901` |

The controls, panel, residual NPZ and sidecar, checkpoint, checkpoint-selection
receipt, support axis, STATE auxiliary files, isolated runtime tree, generator
and imported helpers must also be byte-identical across arms. The strict V6
spec records gamma and verification rejects any generation-report mismatch.
The factor validator additionally pins their SHA-256 values to the frozen V5.1
HepG2 artifacts and STATE runtime; three mutually consistent substitutions do
not qualify as P4.

## Generation

Generate the strict control and both experimental HepG2 arms, retaining the
real Slurm job identifiers:

```bash
g100_job="$(sbatch --parsable \
  --export=ALL,VCC_CONTEXT=HepG2,VCC_OUTPUT_TAG=p4_g100_p0_a010,VCC_STATE_EFFECT_WEIGHT=1.00,VCC_RESIDUAL_ALPHA=0.10,VCC_TARGET_REMAINING_FRACTION=0.20,VCC_RESIDUAL_NPZ=artifacts/public_v51/hepg2_state_residual_top100_v1.npz,VCC_RESIDUAL_JSON=artifacts/public_v51/hepg2_state_residual_top100_v1.json \
  slurm/h100_generate_public_candidate_v6.sbatch)"

g075_job="$(sbatch --parsable \
  --export=ALL,VCC_CONTEXT=HepG2,VCC_OUTPUT_TAG=p4_g075_p0_a010,VCC_STATE_EFFECT_WEIGHT=0.75,VCC_RESIDUAL_ALPHA=0.10,VCC_TARGET_REMAINING_FRACTION=0.20,VCC_RESIDUAL_NPZ=artifacts/public_v51/hepg2_state_residual_top100_v1.npz,VCC_RESIDUAL_JSON=artifacts/public_v51/hepg2_state_residual_top100_v1.json \
  slurm/h100_generate_public_candidate_v6.sbatch)"

g050_job="$(sbatch --parsable \
  --export=ALL,VCC_CONTEXT=HepG2,VCC_OUTPUT_TAG=p4_g050_p0_a010,VCC_STATE_EFFECT_WEIGHT=0.50,VCC_RESIDUAL_ALPHA=0.10,VCC_TARGET_REMAINING_FRACTION=0.20,VCC_RESIDUAL_NPZ=artifacts/public_v51/hepg2_state_residual_top100_v1.npz,VCC_RESIDUAL_JSON=artifacts/public_v51/hepg2_state_residual_top100_v1.json \
  slurm/h100_generate_public_candidate_v6.sbatch)"

g100_job="${g100_job%%;*}"
g075_job="${g075_job%%;*}"
g050_job="${g050_job%%;*}"
```

After all three generation jobs pass, build the immutable factor receipt. The
builder re-hashes every prediction and generator input, pins the V5.1 inputs,
compares raw-inference summaries and source-control identities, verifies exact H5AD
metadata pairing, and streams generated counts to prove that every gamma pair
has a realized nonzero contrast. It proves that the generation factor is
isolated; scorer identity is authenticated later by each pairwise comparison:

```bash
factor_job="$(sbatch --parsable \
  --dependency="afterok:${g100_job}:${g075_job}:${g050_job}" \
  slurm/cpu_build_public_v6_p4_factor_contract.sbatch)"
factor_job="${factor_job%%;*}"
```

Only the passing generation-factor receipt authorizes scoring. Capture all
scorer job identifiers for monitoring, and do not start comparisons until all
three jobs have completed successfully:

```bash
g100_score_job="$(sbatch --parsable \
  --dependency="afterok:${factor_job}" \
  --export=ALL,VCC_CONTEXT=HepG2,VCC_OUTPUT_TAG=p4_g100_p0_a010 \
  slurm/cpu_score_public_candidate_v6.sbatch)"

g075_score_job="$(sbatch --parsable \
  --dependency="afterok:${factor_job}" \
  --export=ALL,VCC_CONTEXT=HepG2,VCC_OUTPUT_TAG=p4_g075_p0_a010 \
  slurm/cpu_score_public_candidate_v6.sbatch)"

g050_score_job="$(sbatch --parsable \
  --dependency="afterok:${factor_job}" \
  --export=ALL,VCC_CONTEXT=HepG2,VCC_OUTPUT_TAG=p4_g050_p0_a010 \
  slurm/cpu_score_public_candidate_v6.sbatch)"

g100_score_job="${g100_score_job%%;*}"
g075_score_job="${g075_score_job%%;*}"
g050_score_job="${g050_score_job%%;*}"

squeue --jobs="${g100_score_job},${g075_score_job},${g050_score_job}"
```

## Comparison and deterministic selection

Use `compare_public_v6_sequential.py` independently for each experimental arm
against the authenticated gamma `1.00` control. The `--factor-contract` option
re-authenticates all three original predictions, inputs, specs, and generation
receipts before it reads scorer outputs. The existing primary gate
requires both all-target and direct-target cohorts to preserve MSE, NMAE, and
direction reach while strictly improving PDS or Jaccard. Every evaluable
held-target metric must also be non-regressing. Paired-target bootstrap
intervals are evidence summaries and cannot override this point gate.

After all three score jobs finish, run the authenticated comparison once for
each experimental tag:

```bash
for TAG in p4_g075_p0_a010 p4_g050_p0_a010; do
  .venv-state/bin/python scripts/compare_public_v6_sequential.py \
    --manifest dataset/public_v51/split_manifest.csv \
    --context HepG2 \
    --role primary \
    --baseline-label p4_g100_p0_a010 \
    --baseline-results artifacts/public_v6/scoring/hepg2/p4_g100_p0_a010/cell_eval2/results.csv \
    --baseline-scoring-receipt artifacts/public_v6/scoring/hepg2/p4_g100_p0_a010/scoring_verified.json \
    --baseline-candidate-spec artifacts/public_v6/candidates/hepg2/p4_g100_p0_a010/spec.json \
    --candidate-label "$TAG" \
    --candidate-results "artifacts/public_v6/scoring/hepg2/$TAG/cell_eval2/results.csv" \
    --candidate-scoring-receipt "artifacts/public_v6/scoring/hepg2/$TAG/scoring_verified.json" \
    --candidate-candidate-spec "artifacts/public_v6/candidates/hepg2/$TAG/spec.json" \
    --factor-contract artifacts/public_v6/contracts/hepg2_p4_state_gamma.json \
    --output-json "artifacts/public_v6/comparisons/hepg2_${TAG}_vs_g100.json"
done
```

Select deterministically:

1. retain gamma `1.00` if neither new arm passes against the incumbent;
2. select the sole passing arm if exactly one passes; or
3. if both pass, select gamma `0.50` only if it also passes the same primary
   gate against gamma `0.75`; otherwise select gamma `0.75`.

When both arms pass, authenticate the incremental comparison directly:

```bash
.venv-state/bin/python scripts/compare_public_v6_sequential.py \
  --manifest dataset/public_v51/split_manifest.csv \
  --context HepG2 \
  --role primary \
  --baseline-label p4_g075_p0_a010 \
  --baseline-results artifacts/public_v6/scoring/hepg2/p4_g075_p0_a010/cell_eval2/results.csv \
  --baseline-scoring-receipt artifacts/public_v6/scoring/hepg2/p4_g075_p0_a010/scoring_verified.json \
  --baseline-candidate-spec artifacts/public_v6/candidates/hepg2/p4_g075_p0_a010/spec.json \
  --candidate-label p4_g050_p0_a010 \
  --candidate-results artifacts/public_v6/scoring/hepg2/p4_g050_p0_a010/cell_eval2/results.csv \
  --candidate-scoring-receipt artifacts/public_v6/scoring/hepg2/p4_g050_p0_a010/scoring_verified.json \
  --candidate-candidate-spec artifacts/public_v6/candidates/hepg2/p4_g050_p0_a010/spec.json \
  --factor-contract artifacts/public_v6/contracts/hepg2_p4_state_gamma.json \
  --output-json artifacts/public_v6/comparisons/hepg2_p4_g050_vs_g075.json
```

This rule favors the smaller departure from the incumbent unless the stronger
shrinkage demonstrates an incremental benefit. Do not use the P3 factorial
analyzer because P4 has a different causal factor and no target-force
interaction.

Build the immutable selection receipt after the two primary comparisons. If
both primary `experiment_gate.passed` values are `true`, first create the
incremental comparison above and include its optional argument. If zero or one
primary arm passes, leave `incremental_args` empty; supplying an unnecessary
incremental comparison fails closed.

```bash
incremental_args=()

# Use this line only when both primary arms passed.
# incremental_args=(--g050-vs-g075 artifacts/public_v6/comparisons/hepg2_p4_g050_vs_g075.json)

.venv-state/bin/python scripts/select_public_v6_p4_state_gamma.py build \
  --factor-contract artifacts/public_v6/contracts/hepg2_p4_state_gamma.json \
  --g075-vs-g100 artifacts/public_v6/comparisons/hepg2_p4_g075_p0_a010_vs_g100.json \
  --g050-vs-g100 artifacts/public_v6/comparisons/hepg2_p4_g050_p0_a010_vs_g100.json \
  "${incremental_args[@]}" \
  --output-json artifacts/public_v6/decisions/hepg2_p4_state_gamma.json

.venv-state/bin/python scripts/select_public_v6_p4_state_gamma.py validate \
  --receipt artifacts/public_v6/decisions/hepg2_p4_state_gamma.json
```

The selector re-authenticates the factor receipt, the current manifest,
comparison JSON files, results CSVs, and scoring receipts. It independently
reproduces every point gate and requires repeated appearances of each arm to
reuse byte-identical scoring bundles and sidecars. Factor authentication
streams generated prediction counts but never reads treated truth; the
selector does not read prediction matrices outside that validator.

This P4 receipt is deliberately restricted to the HepG2 primary screen. A
later Jurkat non-harm check must regenerate a strict gamma `1.00` control and
the selected gamma under a new matched factor receipt; comparing only the
selected arm with the legacy V5.1 file is not sufficient. P4 does not authorize
an official VCC submission. Promotion requires the complete authenticated
public-validation sequence.

## Deferred reliability work

The next residual experiment should add a target-by-gene K562/RPE1 direction
consensus gate while keeping all 33 held-target residual rows at exact zero.
That gate is a public-validation diagnostic, not yet an official-panel feature:
the K562/RPE1 direct-target atlas and the official 300-target panel do not have
the same target coverage. Deployment therefore requires a separate K562-GWPS
feature-parity benchmark and an explicit missing-feature route; public gains
alone cannot authorize transfer to the official A/B/C contexts.

Nonzero ESM2 fallback requires a new residual schema and separate direct and
fallback alpha contracts. It must not be introduced by weakening the existing
V5.1 held-row firewall or mixed into this gamma experiment.
