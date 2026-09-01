# Public V6 candidate orchestration

This lane evaluates one public-validation candidate at a time without reading
or writing any V5.1 generation or scoring path. Its default root is
`artifacts/public_v6`, while the running V5.1 workflow remains under
`artifacts/public_v51`.

## Contract

Generation has two mandatory contract boundaries:

1. `public_candidate_v6_contract.py plan` authenticates the controls-only role,
   frozen panel and routing masks, residual sidecar and SHA-256, residual named
   axes, zero held-target rows, STATE support axis, validation-selected
   checkpoint, generator script, the executable
   `generate_state_direct_counts.py` and `infer_state_effect_prior.py` helper
   modules, `pert_onehot_map.pt`, and `var_dims.pkl`. It also binds
   `config.yaml` when that optional STATE model config is present. The helper
   paths must be those exact module names beside the top-level generator, so a
   receipt cannot bind unused decoy files. `--state-source-dir
   external/state_runtime_9bbfe78a`
   additionally binds the runtime repository HEAD and tree, requires a clean
   tracked worktree, and rejects every untracked or ignored file under `src/`.
   Create the detached runtime checkout with `git -C external/state worktree
   add --detach ../state_runtime_9bbfe78a
   9bbfe78a434a55205e4de834e1ea99f85f7a3add`. The Slurm wrapper prepends its
   `src/` to `PYTHONPATH` and exports `PYTHONDONTWRITEBYTECODE=1`.
   It writes an immutable v2 `spec.json` before inference begins.
2. `public_candidate_v6_contract.py verify` authenticates the generated H5AD,
   generation report, every code/model descriptor above, locked decoder
   configuration, full 300-target shape, exact libraries, and protected native
   genes. It re-hashes the strict code/model inputs from their bound paths and
   records those post-generation descriptors under
   `provenance.authenticated_strict_inputs` in the immutable verification
   receipt. The byte-identical historical top-level generator therefore stays
   unchanged; helper attestation is performed by the external strict wrapper.
   The
   scoring job refuses to open scorer-only truth unless this receipt has
   passed.

New candidates use `vcc-public-candidate-spec-v2` and carry the explicit
`vcc-public-generator-provenance-v2` strict contract. Missing code/model
descriptors, a changed SHA-256, or a mismatch in optional `config.yaml`
presence fails closed. The completed P2 candidate predates this boundary and
retains a v1 spec and receipt. It may be inspected only with the explicit
`verify --allow-legacy-v1-read-only` compatibility mode and without
`--output-json`; that mode cannot mint a new verification receipt and is not
valid for P3.

Binding the imported helper modules is mandatory because the repository may
have unrelated or uncommitted changes: a HEAD revision alone does not prove
which executable helper bytes inference imported. Replacing either helper
after planning causes strict verification to fail before prediction parsing.

Candidate tags must match `[a-z0-9][a-z0-9._-]{0,63}`. Existing specs,
predictions, receipts, score directories, and caches are never overwritten.

The historical locked P2/P3 configuration is:

```text
STATE weight             1.0
STATE bound              0.60 tanh
combined effect clip     0.65
target remaining fraction 0.20 by default; 0.40 only for P3
cells per target         400
integerization           largest remainder
maximum genes per cell   5900
seed                     20260901
```

Only the residual artifact, residual alpha, context, output tag, and the
pre-registered target remaining fraction are candidate parameters. The target
fraction accepts exactly `0.20` or `0.40`; every other value fails before the
spec is written. Omitting it preserves the `0.20` default. The selected value
is bound independently in `spec.json`, `generation.json`, and
`generation_verified.json`.

P4 additionally exposes the raw STATE effect weight as one authenticated
factor with the pre-registered set `0.50`, `0.75`, and `1.00`; omission keeps
the historical `1.00` default. No other generator parameter is opened.

## P3 matched target-force ablation

P3 asks whether the fixed 80% target knockdown is too strong for the public
CRISPRi panel. The configured fraction is the forced model-space target fold
before count composition and integerization, not the realized count ratio. Use
a pre-registered 2-by-2 design so target-force and residual effects are
identifiable. Run it only after the preceding residual candidate has a complete
authenticated score. For each context, all four arms must use identical
controls, panel, checkpoint, residual NPZ and sidecar, decoder, seed, and cell
count:

```text
STATE-only control       alpha 0.00    target remaining fraction 0.20
STATE-only P3            alpha 0.00    target remaining fraction 0.40
residual control         alpha 0.10    target remaining fraction 0.20
residual P3              alpha 0.10    target remaining fraction 0.40
```

Alpha zero is explicitly supported for the STATE-only arms. Planning still
authenticates and binds the declared residual NPZ and sidecar, while generation
must report that the residual values were not read. Negative or non-finite
alphas fail before a spec can be written.

Use distinct output tags and isolated output directories. The two new HepG2
`0.40` arms are planned as follows (the declared residual remains authenticated
but unread in the alpha-zero arm):

```bash
sbatch --parsable \
  --export=ALL,VCC_CONTEXT=HepG2,VCC_OUTPUT_TAG=p3_tf040_state_a000,VCC_RESIDUAL_ALPHA=0.00,VCC_TARGET_REMAINING_FRACTION=0.40,VCC_RESIDUAL_NPZ=artifacts/public_v51/hepg2_state_residual_top100_v1.npz,VCC_RESIDUAL_JSON=artifacts/public_v51/hepg2_state_residual_top100_v1.json \
  slurm/h100_generate_public_candidate_v6.sbatch

sbatch --parsable \
  --export=ALL,VCC_CONTEXT=HepG2,VCC_OUTPUT_TAG=p3_tf040_p0_a010,VCC_RESIDUAL_ALPHA=0.10,VCC_TARGET_REMAINING_FRACTION=0.40,VCC_RESIDUAL_NPZ=artifacts/public_v51/hepg2_state_residual_top100_v1.npz,VCC_RESIDUAL_JSON=artifacts/public_v51/hepg2_state_residual_top100_v1.json \
  slurm/h100_generate_public_candidate_v6.sbatch
```

Name the arms `A=(0.20, alpha 0)`, `B=(0.20, alpha 0.10)`,
`C=(0.40, alpha 0)`, and `D=(0.40, alpha 0.10)`. Each matched `0.20` arm must
set `VCC_TARGET_REMAINING_FRACTION=0.20` explicitly when regenerated; an older
candidate is reusable only if its authenticated spec matches every other input
descriptor and configuration field.

The deployment comparisons are `C` versus incumbent `B` and `D` versus `B`,
using the sequential primary gate on HepG2. Select deterministically:

1. retain `B` if neither `C` nor `D` passes;
2. select the sole passer if exactly one passes; or
3. if both pass, select `D` only if `D` also passes against `C`; otherwise
   select `C`.

The within-alpha `C-A` and `D-B` contrasts isolate target force, `D-C` isolates
the residual at the new target force, and `(D-C) - (B-A)` is the interaction
contrast. These mechanistic contrasts do not replace the deployment gates.
Within fraction 0.40, held-target rows in `C` and `D` must be count-identical
because the residual held rows are zero. Held rows are expected to change
across fractions 0.20 and 0.40, so cross-fraction count identity is not a valid
gate. Repeat the selected configuration on Jurkat as a non-harm check. Do not
tune an intermediate fraction after observing scores: the contract rejects
values outside the pre-registered pair.

The C/D structural comparator accepts only strict v2 specs, requires both arms
to use target remaining fraction `0.40`, and requires the alpha set to be
exactly `{0.00, 0.10}`. It authenticates and compares all v2 input descriptors,
including the generator script, perturbation map, STATE dimensions, and the
optional model config. Thus count identity cannot pass when the two arms were
produced by different code or model-side auxiliary files.

## Executed HepG2 sequence

HepG2 is the primary zero-shot context for this STATE checkpoint. The V5.1
power-zero residual at alpha `0.10` improved all six authenticated public
metrics relative to the STATE anchor and passed the paired-bootstrap evidence
screen. The planned `0.025` and `0.05` alpha jobs were therefore skipped; they
would not answer the next structural question and were never generated.

The first V6 experiment was P2: reliability-power `0.5` with alpha
`0.1248983248`, chosen to match the total residual energy of V5.1 alpha
`0.10`. Its immutable tag is `v6_relp05_ematch_a0124898`. P2 was evaluated
incrementally against the V5.1 incumbent, not against the weaker STATE-only
anchor. It failed the all-target and direct-target promotion gates, so the
power-zero V5.1 arm remained the incumbent and the sequence advanced to the
pre-registered P3 target-force factorial.

The isolated outputs are:

```text
artifacts/public_v6/candidates/hepg2/<tag>/
  spec.json
  prediction.h5ad
  generation.json
  generation_verified.json

artifacts/public_v6/scoring/hepg2/<tag>/
  prediction_view.h5ad
  truth_view.h5ad
  views.json
  cache_real/
  cell_eval2/results.csv
  scoring_verified.json
```

Each candidate receives a private real-side cache. The sequential screen
does not reuse a context cache because an unstamped or differently configured
cache would confound the comparison. Cache sharing may be introduced only
after the cache key, real-view SHA-256, scorer commit, scorer config digest,
and pdex version are bound in an immutable receipt.

The scorer is pinned to cell-eval2 commit
`5e64833518a6603a0301cbe28185d49c30f4a986`, declared version `0.16.0`, and
pdex `0.3.0`. Post-run validation reconstructs
`expr_mse_unbiased_capped_norm` from the complete per-target numerator
`expr_mse_unbiased_capped` and denominator `expr_distance_unbiased`, then
requires that ratio to match `agg_results.csv`. The derived metric is forbidden
from appearing as per-target rows. Direction fidelity and reach retain their
full 300-target axes and may contain genuine NaNs, but their finite subsets and
`metric_aggregation.csv` `n_used` counts must agree. NMAE may omit its
real-side-only subset; that omission is recorded and authenticated rather than
silently treated as 300 observations.

Do not run Jurkat candidates until one HepG2 alpha passes the primary
zero-shot gate. Jurkat was included in STATE training and is therefore only an
in-distribution non-harm check.

## Comparison

The HepG2 sequence is incremental: every new structural candidate is compared
with the current authenticated incumbent. Both sides require a completed
scoring receipt. The comparator re-hashes `results.csv`, `agg_results.csv`,
`metric_aggregation.csv`, and `run_meta.json`; binds a V6 result to its exact
candidate spec and panel manifest; checks shared scorer, truth-fingerprint,
and anchor identities; reconstructs official MSE as a ratio of sums; checks
the matched NMAE omission subset; reports direction-metric NaN masks; and runs
10,000 paired-target bootstrap draws with seed 20260901. For V6 arms it hashes
the immutable prediction/truth scoring-view H5ADs but never loads their
expression matrices:

```bash
.venv-state/bin/python scripts/compare_public_v6_sequential.py \
  --manifest dataset/public_v51/split_manifest.csv \
  --context HepG2 \
  --role primary \
  --baseline-label v51_alpha010 \
  --baseline-results artifacts/public_v51/scoring/hepg2/v51_alpha010_cell_eval2/results.csv \
  --baseline-scoring-receipt artifacts/public_v51/scoring/hepg2/scoring_receipt.json \
  --baseline-receipt-role v51_alpha010 \
  --candidate-label v6_relp05_ematch_a0124898 \
  --candidate-results artifacts/public_v6/scoring/hepg2/v6_relp05_ematch_a0124898/cell_eval2/results.csv \
  --candidate-scoring-receipt artifacts/public_v6/scoring/hepg2/v6_relp05_ematch_a0124898/scoring_verified.json \
  --candidate-candidate-spec artifacts/public_v6/candidates/hepg2/v6_relp05_ematch_a0124898/spec.json \
  --candidate-allow-legacy-v1 \
  --output-json artifacts/public_v6/comparisons/hepg2_v6_relp05_ematch_a0124898_vs_v51.json
```

The legacy flag is restricted to this already-sealed P2 artifact, whose spec
predates the strict v2 provenance contract. All newly generated P3 candidates
must use v2 and must not set a legacy flag.

The primary point-estimate gate requires both the all-target and direct-target
cohorts to preserve MSE, NMAE, and direction reach while strictly improving
PDS or Jaccard. Every evaluable held-target metric must also be non-regressing.
The report separately counts how many of all six metrics improve and how many
have a positive one-sided 90% bootstrap lower bound. Bootstrap uncertainty is
an advisory signal and cannot override a failed point-estimate safety gate.

Do not promote a candidate from an unauthenticated `results.csv` alone. After
selecting one HepG2 finalist, generate and score the identical configuration
for Jurkat and run the receipt-authenticated sequential comparator in
`non_harm` mode without copying or modifying any score directory:

```bash
.venv-state/bin/python scripts/compare_public_v6_sequential.py \
  --manifest dataset/public_v51/split_manifest.csv \
  --context Jurkat \
  --role non_harm \
  --baseline-label v51_alpha010 \
  --baseline-results artifacts/public_v51/scoring/jurkat/v51_alpha010_cell_eval2/results.csv \
  --baseline-scoring-receipt artifacts/public_v51/scoring/jurkat/scoring_receipt.json \
  --baseline-receipt-role v51_alpha010 \
  --candidate-label SELECTED_TAG \
  --candidate-results artifacts/public_v6/scoring/jurkat/SELECTED_TAG/cell_eval2/results.csv \
  --candidate-scoring-receipt artifacts/public_v6/scoring/jurkat/SELECTED_TAG/scoring_verified.json \
  --candidate-candidate-spec artifacts/public_v6/candidates/jurkat/SELECTED_TAG/spec.json \
  --output-json artifacts/public_v6/comparisons/jurkat_SELECTED_TAG_vs_v51.json
```

The top-level `experiment_gate` passes only when the all-target, direct-target,
and held-target non-regression gates all pass. The Jurkat non-harm role does
not require a strict PDS or Jaccard improvement.

Residual-only candidates must leave all 33 held-target groups count-identical
to the matched anchor. A candidate is not promoted if HepG2 expression MSE,
NMAE, or reach regresses beyond the pre-registered non-inferiority margin,
even when a discrimination metric improves.

## P4 STATE-anchor amplitude

After P3 retained the V5.1 incumbent, the next isolated factor is the raw
paired-STATE amplitude before tanh bounding. The strict candidate contract and
H100 wrapper accept only gamma `0.50`, `0.75`, or `1.00` and bind the selected
value in the immutable spec and generation receipt. P4 regenerates gamma
`1.00` as a strict-v2 control and requires a three-arm factor receipt proving
that all public residual, target-force, decoder, and generator inputs remain
fixed. Scorer identity is authenticated by the later pairwise comparison
receipts. The complete pre-registration, tags, commands, and deterministic
selection rule are in
[the P4 STATE-gamma plan](PUBLIC_V6_P4_STATE_GAMMA.md).
