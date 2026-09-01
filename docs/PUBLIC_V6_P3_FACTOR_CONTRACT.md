# Immutable P3 four-arm factor contract

## Release gate

The factorial analyzer must not select a P3 arm until one immutable
`vcc-public-v6-p3-factor-contract-v1` receipt has passed. The receipt binds the
exact pre-registered identities:

| Arm | Artifact identity | Target remaining fraction | Residual alpha |
| --- | --- | ---: | ---: |
| A | `anchor` | 0.20 | 0.00 |
| B | `v51_alpha010` | 0.20 | 0.10 |
| C | `p3_tf040_state_a000` | 0.40 | 0.00 |
| D | `p3_tf040_p0_a010` | 0.40 | 0.10 |

The builder authenticates the completed V5.1 HepG2 scoring receipt and the
full A/B generation reports and prediction H5ADs. C and D must have strict-v2
specifications and passing strict generation-verification receipts. The tool
authenticates every shared generator input across the legacy generation-report
schema and the strict-v2 specification schema, including controls, frozen
panel, checkpoint selection, STATE auxiliary files, generator code, and the
power-zero residual.

For strict C/D, "generator code" includes file-level descriptors for the
top-level generator and both executable imported helpers:
`generate_state_direct_counts.py` and `infer_state_effect_prior.py`. The C/D
descriptors must be identical and are re-hashed during both generation
verification and factor-receipt authentication. The factor validator also
requires each helper descriptor in `generation_verified.json` to match the
corresponding strict spec, so a helper swap between planning and verification
or after verification fails closed. Historical A/B reports
predate strict-v2 and do not contain helper descriptors; the receipt records
this explicitly as read-only legacy provenance and never retroactively claims
that the current helper hashes were authenticated by A/B.

Strict C/D also share an authenticated STATE runtime-source contract: resolved
repository root, HEAD commit, Git tree, clean tracked worktree, and no
untracked or ignored files below `src/`; bytecode writes are disabled. The generation report commit and both
post-generation verification receipts must match it. The factor validator
rechecks the live repository state before selection.

The main repository `repository_commit` is retained per arm as noncausal
metadata and is not required to match. Historical A/B recorded HEAD
`1918764...` from a dirty tree, so HEAD alone cannot identify their runtime.
Exact generator/helper/input hashes and the STATE commit/tree remain mandatory.

All non-factor configuration fields, model metadata, stable source revisions,
var axes, obs annotations, and sampled source-control cells must be exact
across A/B/C/D. The C/D held-target count comparison is re-run in bounded
backed-CSR chunks. Any held-target count difference prevents receipt creation.
No expression-truth input is accepted by the factor-contract CLI, and the
builder does not consume treated expression values. Authentication of the
historical scoring receipt may hash its sealed scoring-view artifacts without
opening their matrices for analysis.

## Build

Run this only after both C and D generation-verification receipts pass:

```bash
.venv-state/bin/python scripts/build_public_v6_p3_factor_contract.py build \
  --manifest dataset/public_v51/split_manifest.csv \
  --v51-scoring-receipt artifacts/public_v51/scoring/hepg2/scoring_receipt.json \
  --arm-a-generation artifacts/public_v51/full/hepg2_anchor.json \
  --arm-a-prediction artifacts/public_v51/full/hepg2_anchor.h5ad \
  --arm-b-generation artifacts/public_v51/full/hepg2_v51_alpha010.json \
  --arm-b-prediction artifacts/public_v51/full/hepg2_v51_alpha010.h5ad \
  --arm-c-spec artifacts/public_v6/candidates/hepg2/p3_tf040_state_a000/spec.json \
  --arm-c-verification artifacts/public_v6/candidates/hepg2/p3_tf040_state_a000/generation_verified.json \
  --arm-c-generation artifacts/public_v6/candidates/hepg2/p3_tf040_state_a000/generation.json \
  --arm-c-prediction artifacts/public_v6/candidates/hepg2/p3_tf040_state_a000/prediction.h5ad \
  --arm-d-spec artifacts/public_v6/candidates/hepg2/p3_tf040_p0_a010/spec.json \
  --arm-d-verification artifacts/public_v6/candidates/hepg2/p3_tf040_p0_a010/generation_verified.json \
  --arm-d-generation artifacts/public_v6/candidates/hepg2/p3_tf040_p0_a010/generation.json \
  --arm-d-prediction artifacts/public_v6/candidates/hepg2/p3_tf040_p0_a010/prediction.h5ad \
  --chunk-rows 400 \
  --output-json artifacts/public_v6/comparisons/hepg2_p3_factor_contract.json
```

The output path is immutable; an existing receipt is never overwritten.

## Authenticate before selection

The factorial analyzer should call
`authenticate_factor_contract_receipt(...)` directly. A manual audit can use:

```bash
.venv-state/bin/python scripts/build_public_v6_p3_factor_contract.py validate \
  --receipt artifacts/public_v6/comparisons/hepg2_p3_factor_contract.json \
  --manifest dataset/public_v51/split_manifest.csv \
  --v51-scoring-receipt artifacts/public_v51/scoring/hepg2/scoring_receipt.json \
  --arm-c-spec artifacts/public_v6/candidates/hepg2/p3_tf040_state_a000/spec.json \
  --arm-d-spec artifacts/public_v6/candidates/hepg2/p3_tf040_p0_a010/spec.json
```

Validation re-hashes the receipt's primary artifacts, all shared generator
inputs, and the residual artifact. It then reloads the small generation,
specification, verification, and V5.1 scoring JSON files to re-derive arm tags,
factor levels, non-factor equality, output bindings, and strict-verification
links. Factorial selection must consume this authenticated result rather than
trusting receipt booleans alone.

The expensive four-matrix structural computation is performed when the
immutable receipt is built. Later analyzer validation re-hashes the four bound
prediction files and re-derives all small-artifact links; it does not repeat
the backed matrix scan. Thus the stored axes/source-pairing and held-count
results are cryptographically bound to the exact prediction bytes on which
they were computed.
