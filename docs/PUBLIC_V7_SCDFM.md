# Public V7 scDFM Adapter Experiment

## Decision and scope

V7 keeps the authenticated STATE `g100` arm fixed at
`state_effect_weight = 1.0`. The scDFM experiment is an isolated residual
modeling lane; it cannot modify the STATE anchor, the V6 P4 artifacts, or the
official challenge inputs. No artifact produced by this lane is eligible for
an official submission until it passes the complete public-data evaluation
contract.

The registered candidate is

\[
\widehat{\Delta}_{c,p}
= \widehat{\Delta}^{\mathrm{STATE},\gamma=1}_{c,p}
+ w\,R^{\mathrm{scDFM}}_{c,p},
\qquad w\in\{0,0.05,0.10,0.20\}.
\]

The first V7 experiment treats scDFM as a distributional residual model. It is
not allowed to replace the STATE mean response with an absolute flow output.
The `w=0` arm must reuse the authenticated `g100` artifact exactly.

## What the paper establishes

The ICLR 2026 paper introduces three useful components:

1. conditional flow matching in log-normalized gene-expression space;
2. a PAD-Transformer that repeatedly injects the perturbation condition and
   applies self- and control-cross differential attention;
3. an unbiased multi-kernel MMD endpoint loss with dynamic RBF bandwidths.

The paper trains on Norman K562 CRISPRa and ComboSciPlex A549 drug data. Its
Norman configuration uses 5,029 genes, evaluates 1,000 selected genes, uses a
512-dimensional four-layer backbone, a batch size of 96, 100,000 optimization
steps, MMD weight 0.5, a correlation-derived boolean kNN mask with `k=30`, and a
100-step Euler rollout. These facts support the architecture and objective,
not direct transfer of the released checkpoint to VCC 2026.

The upstream graph builder ranks both positive and negative correlations by
absolute magnitude, but the model receives only a boolean attention mask. It
does not receive signed edge weights and should not be described as a causal
or signed regulatory network.

Primary sources:

- paper: <https://arxiv.org/abs/2602.07103>;
- conference paper: <https://openreview.net/forum?id=QSGanMEcUV>;
- official source: <https://github.com/AI4Science-WestlakeU/scDFM>.

## Why the released checkpoint is not a VCC zero-shot model

The two tasks differ in every important generalization axis.

| Property | Published scDFM genetic benchmark | VCC 2026 |
|---|---|---|
| Perturbation mechanism | CRISPRa | CRISPRi |
| Context | K562 only | three unseen anonymous cell lines |
| Holdout | genes or combinations within K562 | unseen context and target response |
| Model output | continuous log-normalized subset | sparse raw integer counts |
| Gene output | 1,000 evaluation genes | fixed 18,533-gene axis |
| Cells per target | upstream code samples 128 | exactly 400 per context-target |
| Target encoding | learned selected-gene token | all 300 targets require coverage |

The pinned Norman vocabulary contains 5,033 tokens: 5,029 genes and four
special tokens. Its exact overlap with the released VCC files is only 4,882 of
18,533 output genes and 62 of 300 targets. The other 238 target symbols cannot
receive their own released-checkpoint gene embedding. Unknown-token fallback
would collapse biologically distinct targets and is forbidden.

The paper itself lists multi-context ARC/state and Virtual Cell Challenge
evaluation as future work. Its full-vocabulary imputation head is also stated
as an optional future addition and is not used in the experiments. Therefore,
neither the paper result nor the pretrained checkpoint demonstrates the
double-zero-shot capability required here.

## Pinned upstream boundary

The official checkout is an immutable, Git-ignored research dependency:

| Field | Pinned value |
|---|---|
| Commit | `2cf6bca1f044e74c4e1dc586892c0495880cf125` |
| Tree | `a04130b07020505a609158cbe31e9e65083c0d79` |
| Code license | MIT |
| License SHA-256 | `dedf7a6ce9cb9f1062d6f387a6a164320a442d61591627f81ce6aedc7c8e2fdb` |

`scripts/authenticate_scdfm_source.py` rejects a different commit or tree, a
dirty checkout, a modified license, and missing runtime files. Project code
must not patch the checkout in place. Any adapter belongs in this repository
and must be included in the model receipt.

The paper is CC BY 4.0 and the source is MIT. The checkpoint archive has no
separate model card, weight license, or author-provided checksum. Its use and
redistribution therefore require a separate review; this repository records a
local checksum without claiming that the source-code license governs weights.

The official Google Drive assets were downloaded and authenticated locally on
2026-09-01. These SHA-256 values bind the exact bytes used by this branch; they
are project-computed receipts, not checksums published by the authors.

| Archive | Bytes | Local SHA-256 | Authenticated contents |
|---|---:|---|---|
| `scDFM_ckpts.zip` | 5,931,951,086 | `373e69c80b88e1735cd65afe3039385cdc91fe0ef33e64b324228e029df4e937` | 71 safe members, 9 checkpoints, all CRCs valid |
| `norman.zip` | 614,006,395 | `794564cef2c0ee8f8d7fbe2ee69e87cd4eba3f14a995bd28639b6d330d76cf24` | Norman AnnData and split file |
| `combosciplex.zip` | 702,210,085 | `60095672660ca3d983fe85cafb26541df3139790d7f87e1deb0e62283f0235f4` | ComboSciPlex AnnData |

`scripts/authenticate_scdfm_assets.py` checks exact official byte lengths and
the registered project digests, decompresses every member to validate its CRC
without extracting it, and rejects absolute, parent-traversing, encrypted,
duplicate, or symbolic-link members. It also stream-hashes required members
and requires each extracted file to match the corresponding decompressed bytes,
not merely the member size. The local receipt is
`dataset/scdfm/receipts/assets_member_bound_v2.json` and remains outside Git;
its v2 contract rejects legacy size-only extracted-member bindings. Its sanitized
tracked summary is [`assets_authentication.json`](../results/scdfm_v7/assets_authentication.json).

The released Norman holdout fold-0 checkpoint was then loaded on CPU with
PyTorch `weights_only=True` and its expected SHA-256. It is a 697,879,461-byte
file at iteration 70,000 with 219 model tensors and 60,359,169 model-state
elements. Its perturbation embedding has shape `6000 x 512`. The checkpoint
contains optimizer and scheduler state but does not embed the model config,
gene vocabulary, or graph mask, so those inputs must be authenticated
separately. See the sanitized
[`holdout fold-0 audit`](../results/scdfm_v7/holdout_fold0_checkpoint_audit.json).
The separate sanitized
[`VCC compatibility audit`](../results/scdfm_v7/vcc_compatibility.json) records
why that checkpoint is not directly eligible for prediction or submission.
Its full local v2 receipt is
`dataset/scdfm/receipts/vcc_compatibility_config_bound_v2.json`; it binds the
exact adapter-config SHA-256 and rejects legacy configs without the registered
experiment schema.

The upstream runtime pins were installed in an isolated Python 3.11.13
environment rather than modifying the STATE environment. `pip check` reports
no broken requirements; the key versions are PyTorch 2.7.1+cu126, torchvision
0.22.1+cu126, timm 1.0.19, anndata 0.12.1, scanpy 1.11.4, JAX 0.6.2, pertpy
1.0.1, and scvi-tools 1.3.3. The official environment's scvi-tools 0.20.3 pin
cannot import with anndata 0.12.1 because it relies on a removed private sparse
API; the project pin is the smallest tested compatibility correction and is
recorded explicitly rather than silently reproducing a broken environment. The
sanitized environment receipt is
[`results/scdfm_v7/environment.json`](../results/scdfm_v7/environment.json).

Before official preprocessing or training, authenticate the pinned source and
preflight every statically discovered runtime import without constructing a
model or opening a dataset:

```bash
dataset/scdfm/envs/scdfm/bin/python scripts/preflight_scdfm_environment.py \
  --source dataset/external_repos/scDFM
dataset/scdfm/envs/scdfm/bin/python -m pip check
```

The preflight includes the network, optimal-transport, and compound-processing
dependencies used by the official paths (`requests`, `beautifulsoup4`, `POT`,
`pertpy`, and RDKit). It classifies `fast_transformers` and `flash_attn` as
optional scGPT attention backends because the registered origin/PAD model does
not select them. The sanitized environment record binds the current
`requirements/scdfm.txt` SHA-256, authenticated source commit and tree, and the
SHA-256 of the deterministic JSON emitted by this preflight. Any dependency or
source change requires regenerating those bindings before training.

## Reproducibility defects that must not enter V7

The pinned source has several observable differences from the paper and
several unresolved interfaces:

- `instantiate_model()` ignores the supplied token count and hidden size and
  instantiates the model defaults;
- `if split_method == 'additive' or 'combinations'` is always true, so the
  intended unseen branch is unreachable;
- the training script does not establish a deterministic global seed;
- the `devices` and `test_only` configuration fields are unused;
- loading a checkpoint does not restore the training iteration and there is
  no clean inference-only entry point;
- the paper specifies batch 96, 100,000 steps, and Euler-100, while `run.sh`
  specifies batch 48 and 200,000 steps and the code uses RK4-20 inference;
- the validation call in the training loop is constructed from the test split,
  as also reported in unresolved upstream issue 7;
- the public multi-GPU guidance references a script absent from the pinned
  tree.

Relevant upstream reports:

- preprocessing mismatch: <https://github.com/AI4Science-WestlakeU/scDFM/issues/5>;
- possible test-derived checkpoint selection: <https://github.com/AI4Science-WestlakeU/scDFM/issues/7>;
- unresolved multi-GPU interface: <https://github.com/AI4Science-WestlakeU/scDFM/issues/3>.

V7 therefore uses an independent data split, seed, checkpoint-selection, and
inference wrapper. Upstream code may be imported only after source
authentication and only through a reviewed adapter.

## Data roles and leakage firewall

All fitted transforms, highly variable gene selection, graph construction,
normalization statistics, and target representations must be fitted inside
each training fold.

| Dataset | Permitted role | Explicit exclusion |
|---|---|---|
| ARC/STATE support cells | Multi-context transition training | No challenge perturbed cells |
| K562 GWPS pseudobulk | Target mean and reliability supervision | Not a single-cell count distribution |
| K562-essential/RPE1 pseudobulk | Cross-context residual calibration | Not a count-emitter target |
| Jurkat raw single cells | Distribution training outside a sealed secondary non-harm partition | The authenticated non-harm manifest is excluded from every fit and preprocessing transform |
| HepG2 raw single cells | Sealed primary whole-context evaluation | Excluded from fitting and preprocessing |
| Feng multi-iPSC | Response representation pretraining | Normalized data; never trains raw-count emission |
| Norman | Upstream CRISPRa reproduction only | Not evidence for CRISPRi transfer |
| ComboSciPlex | Upstream drug reproduction only | Not genetic target supervision |
| VCC A/B/C controls | Inference conditioning and format validation | No fitted perturbation labels exist |

The primary split holds out the complete HepG2 context. Within every remaining
training context, complete ESM2 target clusters are held out. Jurkat has an
additional fail-closed secondary partition registered at
`artifacts/scdfm/v7/splits/jurkat_non_harm.json`; training cannot start until
that manifest is created, authenticated, and excluded from every fitted
transform. A cell line, donor, target cluster, perturbation group, or fitted
graph may not cross its registered boundary.

The generation command may accept only:

- a frozen model and its receipt;
- the frozen `g100` anchor artifact;
- released VCC controls and target identifiers;
- the official gene axis;
- a registered configuration and random seed.

It may not accept any measured treated-cell matrix.

## V7 model changes required for real zero-shot use

### Continuous target conditioning

The upstream learned gene-ID perturbation embedding is replaced by a projected
5,120-dimensional ESM2 vector. Every training and inference target must have a
unique authenticated embedding. This avoids mapping 238 missing VCC targets
to one padding token. A missing requested embedding is a hard error.
The registered configuration is currently fail-closed: the exact ESM2 model
identifier, immutable revision, weights digest, protein-sequence source,
gene-to-protein mapping and digest, pooling rule, and embedding receipt are all
explicitly `UNRESOLVED_REQUIRED`. Consequently Gate 1b is incomplete and
training is forbidden until every field is replaced by authenticated values
and `provenance_complete` and `gate_1b_allowed` are deliberately enabled.

### Context conditioning

The released control population is summarized without cell-line labels. The
context encoder is trained with whole-cell-line dropout and receives basal
expression, detection rate, depth, and optional frozen STATE embeddings. It
must not receive an inferred manual cell-line identity.

### Full-axis output

The flow operates on registered gene subsets for tractable attention. Subset
predictions are assembled only through deterministic panels with recorded
coverage. Genes not predicted by the flow retain the `g100` anchor. The first
phase does not train an unconstrained full-axis imputation head.

### Mean protection

The distribution-only arm centers the proposed flow residual within each
context-target group in registered continuous space. The compositor verifies
continuous-space centroid identity only; it explicitly makes no raw-count
claim. A later authenticated count renderer must separately preserve the
anchor integer group sum for every scored gene exactly. This keeps expression
MSE and PDS centroid changes detectable as contract violations instead of
hoping that an MMD loss will preserve them.

A later mean-residual arm may be considered only if the distribution-only arm
passes. Its residual must be reliability-gated using source target coverage,
cell count, source expression, cross-context agreement, and network support.

## Registered experiment sequence

### Phase 0: upstream reproduction

1. Authenticate the source revision.
2. Download the immutable Norman, ComboSciPlex, and checkpoint archives.
3. Record archive size, SHA-256, member paths, and extracted-file hashes.
4. Reproduce one Norman fold without using test metrics for checkpoint choice.
5. Compare paper, shell-script, and checkpoint configurations explicitly.

Failure to reproduce is recorded; it is not repaired by silently changing the
reference result.

### Phase 1: project-native smoke

Phase 1 has two distinct gates. Gate 1a checks only synthetic flow-matching and
MMD objective wiring. It deliberately does not instantiate ESM2 conditioning,
create context or target-cluster splits, load a STATE artifact, or call the
centered-residual compositor. Gate 1b, which is not yet complete, must run a
small single-H100 adapter with continuous ESM2 perturbation conditions, one
held context, one held target cluster, and an authenticated immutable
`state_effect_weight=1.0` artifact. Gate 1b must verify finite
forward/backward passes, deterministic restart, and no train/validation
identity overlap before compact training is allowed.

The synthetic objective-contract portion completed on one NVIDIA H100 80 GB as
Slurm job `861790` in four seconds. It passed all 17 checks, used 67,563,520
bytes of peak allocated GPU memory, and produced finite CFM, MMD, total-loss,
gradient, and post-step parameter values. This validates objective wiring and
the GPU environment only; it is not evidence of prediction quality. The
sanitized receipt is tracked at
`results/scdfm_v7/h100_contract_smoke.json`.

Diagnostic job `861750` first exposed an environment conflict: loading the
cluster CUDA 12.6 module placed system cuDNN 9.5 ahead of the cuDNN 9.20 bundled
with the pinned PyTorch wheel. The corrected launcher intentionally uses the
Slurm NVIDIA driver and the wheel-bundled CUDA/cuDNN runtime.

The original released-checkpoint compatibility test completed as Slurm job
`861784` in seven seconds on one NVIDIA H100 80 GB. It authenticated the exact
source, checkpoint, and 5,033-token Norman vocabulary; instantiated the
unmodified 60,359,169-element upstream model; and loaded all 219 model tensors
strictly with no missing or unexpected keys. The loaded file descriptor was
independently rehashed. Two evaluation forwards over a tiny synthetic tensor
were bitwise identical and finite. The test used a
temporary all-False interaction mask, restored no optimizer or scheduler, and
did not read the released split pickle. Its continuous outputs ranged from
`-2.4886429` to `2.0936260`, reinforcing that the native model output is not a
valid nonnegative raw-count submission. The sanitized result is
[`results/scdfm_v7/h100_released_checkpoint_smoke.json`](../results/scdfm_v7/h100_released_checkpoint_smoke.json).
That historical run verified only that the imported module path was below the
authenticated worktree; it predates the stronger immutable Git-object loader
and is therefore explicitly superseded as an executable-source provenance
attestation. The current launcher snapshots every file below `src` from the
pinned commit, verifies each Git blob ID, executes upstream modules only from
those in-memory bytes, and records SHA-256 for every executed module. A fresh
H100 run is required before the hardened released-checkpoint gate can pass.
Neither the historical result nor a future hardened rerun is evidence of
biological or leaderboard performance.

### Phase 2: compact training

Use one H100, hidden size 256, four layers, 1,000-gene subsets, MMD weight 0.5,
and 20 Euler steps. Train at least three registered seeds. Larger models or
multiple GPUs are justified only after this experiment clears the scientific
gate.

### Phase 3: matched residual sweep

Generate the same cells and random seeds for residual weights
`{0, 0.05, 0.10, 0.20}`. At the continuous-factor boundary, `w=0` is a
byte-identical copy of the authenticated STATE factor. After count rendering,
the zero arm must reproduce the authenticated `g100` raw-count candidate.
All arms use the same decoder, gene panels, control cells, and target order.

### Phase 4: selection

Score every arm with all six official public-validation metrics. Selection is
based on registered paired comparisons, not the public leaderboard.

## Promotion and stop rules

An scDFM arm is promoted only if all of the following hold:

1. the HepG2 whole-context gate passes;
2. Jurkat is non-inferior under the registered margin;
3. PDS, expression MSE, and NMAE do not regress;
4. the distribution-only arm preserves registered continuous centroids and
   post-render integer group sums under their separate contracts;
5. improvements reproduce across at least three seeds and held target
   clusters;
6. every source, dataset, split, checkpoint, configuration, and output hash is
   authenticated;
7. the official output shape and raw-count constraints pass locally.

Stop the lane if fidelity gains require a material MSE/PDS regression, if the
benefit disappears under whole-context holdout, or if unknown targets share an
embedding. In those cases, retain `g100` rather than ensemble a weaker model.

## Naming rule

Two unrelated parameters are commonly called gamma:

- `state_effect_weight = 1.0` is the fixed VCC STATE anchor selected by P4;
- `mmd_weight = 0.5` is the scDFM endpoint-distribution loss coefficient.

V7 code and receipts must use these explicit names and must never expose a
generic `gamma` field.

## Initial commands

```bash
.venv-state/bin/python -m venv dataset/scdfm/envs/scdfm
dataset/scdfm/envs/scdfm/bin/pip install -r requirements/scdfm.txt
dataset/scdfm/envs/scdfm/bin/pip check

dataset/scdfm/envs/scdfm/bin/python scripts/authenticate_scdfm_source.py \
  --source dataset/external_repos/scDFM \
  --output dataset/scdfm/receipts/source.json

dataset/scdfm/envs/scdfm/bin/python scripts/audit_scdfm_vcc_compatibility.py \
  --output dataset/scdfm/receipts/vcc_compatibility_config_bound_v2.json

dataset/scdfm/envs/scdfm/bin/python scripts/authenticate_scdfm_assets.py \
  --extracted-root dataset/scdfm/extracted \
  --output dataset/scdfm/receipts/assets_member_bound_v2.json

dataset/scdfm/envs/scdfm/bin/python scripts/audit_scdfm_checkpoint.py \
  --checkpoint dataset/scdfm/extracted/ckpts/holdout/fold0/checkpoint.pt \
  --expected-sha256 5730a05918e2e727f8a558f21db23ae793a8deb74af5c28b905d852af92354d7 \
  --output dataset/scdfm/receipts/checkpoint_holdout_fold0.json

sbatch --test-only slurm/h100_scdfm_contract_smoke.sbatch
sbatch slurm/h100_scdfm_contract_smoke.sbatch

sbatch --test-only slurm/h100_scdfm_released_checkpoint_smoke.sbatch
sbatch slurm/h100_scdfm_released_checkpoint_smoke.sbatch
```

Once a flow proposal exists, the distribution-only factor arms are composed
with a strict continuous-space centroid invariant. Both input receipts must
bind the exact array bytes, selected array key, axes, experiment seed,
representation, and permitted data role. Their expected receipt hashes are
provided independently so a substituted receipt is rejected. Version-2 input
receipts also bind the canonical context-major/target-minor group layout plus
the exact cell-axis and latent-axis identity hashes; all three identities must
match before elementwise composition:

```bash
.venv-state/bin/python scripts/scdfm_centered_residual.py \
  --anchor artifacts/scdfm/v7/factors/anchor.npy \
  --flow artifacts/scdfm/v7/factors/flow.npy \
  --anchor-receipt artifacts/scdfm/v7/factors/anchor_provenance.json \
  --anchor-receipt-sha256 "$ANCHOR_RECEIPT_SHA256" \
  --flow-receipt artifacts/scdfm/v7/factors/flow_provenance.json \
  --flow-receipt-sha256 "$FLOW_RECEIPT_SHA256" \
  --flow-semantics absolute_endpoint \
  --weight 0.05 \
  --output artifacts/scdfm/v7/factors/candidate_w005.npy \
  --receipt artifacts/scdfm/v7/factors/candidate_w005.json
```

The contract permits only registered weights and rejects any STATE effect
weight other than exactly `1.0`. `absolute_endpoint` first forms `flow-anchor`;
`residual_proposal` accepts an explicitly residual-valued producer output. Both
routes center that residual per context-target and verify that every candidate
gene centroid equals the anchor within the fixed registered `1e-10` tolerance.
The zero-weight arm is a byte-identical copy of the authenticated anchor.
The official control manifest binds the ordered context list and requires the
flattened group count to equal `context_count * target_count`; both producer
receipts must carry the same manifest hash.

All config, axis, manifest, array, and provenance inputs are opened through
non-symlink regular-file descriptors. Hashing and parsing/loading use the same
descriptor, and lexical output paths reject existing or dangling symlinks.

This output remains a continuous-expression factor. It makes no raw-count or
integer-sum claim and cannot be submitted. A separately authenticated count
renderer must enforce nonnegative integer output, exact axes and dimensions,
and any required count-space sum invariant.

The receipts are local data artifacts and remain outside Git. The configuration,
adapter, tests, and Slurm launchers are tracked on
`experiment/v7-scdfm-gamma1`.
