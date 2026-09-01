# P3 alpha-pair count-identity audit

## Purpose

The P3 target-force experiment generates two new HepG2 arms at a forced
model-space target fold of 0.40:

- `p3_tf040_state_a000`: STATE only, residual alpha 0.00.
- `p3_tf040_p0_a010`: the same prediction path with residual alpha 0.10.

The public residual is pre-registered to be exactly zero for all 33
`held_target` rows. Therefore, the two generated count matrices must be
identical for every cell and gene in those groups. A held-target difference is
an implementation or provenance failure, not a model-selection tradeoff.

`scripts/compare_public_p3_count_identity.py` checks this invariant without
opening any treated truth. It authenticates both generation-verification
receipts, specifications, generation reports, H5AD files, and all recorded
generator inputs. It also requires:

- identical contexts, axes, source-control cell pairing, and annotations;
- identical generator inputs and checkpoint selection;
- identical generator configuration except for `residual_alpha`;
- the exact frozen manifest target order and routing labels;
- backed integer CSR count matrices.

Both arms must have strict `vcc-public-candidate-spec-v2` specifications and
passing `strict_generator_provenance_authenticated` checks. The authenticated
and matched inputs include the generator script, `pert_onehot_map.pt`,
the executable `generate_state_direct_counts.py` and
`infer_state_effect_prior.py` helpers, `var_dims.pkl`, and the optional STATE
`config.yaml`, in addition to the
controls, panel, residual, checkpoint, selection receipt, and support-gene
axis. A null model-config descriptor is allowed only when both generator
reports also declare no config. The comparator re-hashes every non-null input
and requires descriptor identity between arms. It additionally rejects any
pair other than target remaining fraction `0.40` with residual alphas exactly
`0.00` and `0.10`.

The matrix scan is bounded by `--chunk-rows`. Canonical logical CSR hashes are
updated row by row, so the hashes do not depend on HDF5 layout, sparse index
width, or chunk size. The tool never loads both complete prediction matrices
into memory.

## Run after both generation verifications pass

```bash
.venv-state/bin/python scripts/compare_public_p3_count_identity.py \
  --manifest dataset/public_v51/split_manifest.csv \
  --left-spec artifacts/public_v6/candidates/hepg2/p3_tf040_state_a000/spec.json \
  --left-verification artifacts/public_v6/candidates/hepg2/p3_tf040_state_a000/generation_verified.json \
  --left-prediction artifacts/public_v6/candidates/hepg2/p3_tf040_state_a000/prediction.h5ad \
  --right-spec artifacts/public_v6/candidates/hepg2/p3_tf040_p0_a010/spec.json \
  --right-verification artifacts/public_v6/candidates/hepg2/p3_tf040_p0_a010/generation_verified.json \
  --right-prediction artifacts/public_v6/candidates/hepg2/p3_tf040_p0_a010/prediction.h5ad \
  --chunk-rows 400 \
  --output-json artifacts/public_v6/comparisons/hepg2_p3_tf040_alpha_count_identity.json
```

The optional output is immutable: the command refuses to overwrite an
existing receipt. A passing receipt reports zero differing held rows and
entries, equal held logical hashes, and diagnostic direct-row difference
counts and hashes. Any held-target count difference exits nonzero and no
passing receipt is written.

## Interpretation boundary

This audit establishes only count-level implementation identity for the
held-target alpha contrast. It does not evaluate biological accuracy and does
not replace Cell-Eval scoring. Panel-wide scorer corrections can still make a
held-target metric move even when the submitted held-target counts are exactly
identical; this structural audit separates that scorer coupling from residual
leakage or generation drift.
