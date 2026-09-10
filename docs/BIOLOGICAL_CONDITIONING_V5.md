# Biological conditioning v5: ESM2, Lingshu, and fusion

## Scope

This new public-only pretraining screen tests whether authenticated biological
features improve target-response conditioning after the failed v3 flow and v4
symbol-only Lingshu experiments. It fits small response adapters, not a flow
generator. It neither generates challenge cells nor authorizes a submission.

The genuine frozen Lingshu features and public training cache are unchanged.
The second feature source is the Arc competition support artifact
`dataset/state_support/extracted/ESM2_pert_features.pt`, authenticated using
`scripts/authenticate_esm2_target_features.py`. Its registered SHA-256 is
`a210e1cc7901513999b2bca3836ba9e2f203cd008be4e9a9d6412a2267de9748`.
The source archive and member identity were also rechecked read-only.
This verifies the supplied artifact, **not** its upstream model ID, weights,
protein sequences, isoform mapping or pooling: those remain unknown. Feature
dimension alone is not model provenance. No feature redistribution or new
license grant is implied. See `results/scdfm_v7/esm2_target_features_authentication.json`
and the existing provenance boundary in `docs/DATA_PROVENANCE.md`.

Every arm uses the same 145 exact-symbol K562 training targets. `TAZ` is excluded
from all arms because it has no exact feature key and the historical symbol is
ambiguous; no substitution with TAFAZZIN or WWTR1 is made. All 512 existing pilot
genes remain in the reconstructed error. No challenge-treated data or cached
`val_*` expression arrays are loaded.

## Frozen comparison

The trainer writes `contract.json` before fitting any effects. It refuses an
existing output directory and authenticates code, cache and both feature inputs.

- Genuine ESM2, genuine Lingshu, and ESM2+Lingshu fusion.
- Three feature-association shuffles for each single-feature family.
- Three incremental fusion controls: **genuine ESM2 unchanged**, Lingshu shuffled.
- Independently tuned constant mean effect and zero effect.

The three fixed repeats each have five target-held-out outer folds and three
inner folds. Each vector is L2-normalized. Centering, row-energy normalization,
input/output PCA and regression are all fitted within the relevant training
fold. Fusion gives the two centered feature blocks equal total energy and
shares the same **total** input-rank budget as single-feature arms. It does not
double model capacity just because there are two blocks.

The v4 25-candidate grid and independently selected constant gain are reused.
The feature arms share outer/inner splits, tuning budget and per-fold mean gain.
Final research exports are refitted on all 145 training targets, with their
hyperparameters selected by a separate fixed inner CV, not outer losses.

Each family must beat the optimized constant by at least 1%, beat every required
negative control in every repeat, have exploratory paired target-bootstrap
intervals wholly below zero against those controls, and beat zero effect.
Fusion must additionally beat ESM2 alone. These are exploratory compute gates,
not independently confirmed scientific findings. Passing still does not permit
deployment: all exports have `submission_allowed=False`, `independent_test=False`
and require authenticated sidecars and explicit input normalization/block order.

## Split history and limits

This is reused **development** data. The gene axis was selected before this CV;
cached controls can recur; biological targets are correlated. The Lingshu
training lineage also contains 14 members of clusters globally excluded by the
separate V7 protocol. A v5-derived model therefore cannot be claimed compliant
with that protocol merely because none of its 300 directly scored symbols was
used. Do not open or score that reserved panel as part of v5.

Existing K562/RPE1 validation, historical HepG2 and Jurkat evaluation panels have
previous selection exposure. Do not relabel them untouched. Any future
multi-context experiment needs an explicit new lineage/split contract, globally
excluded related-target clusters where applicable, and control-derived recipient
context features. This centroid screen measures neither cell distributions nor
full-axis output, and supplies no evidence of an improved challenge score.

## Run with W&B

Use the authorized BioHPC compute host (four CPU threads), or a Slurm allocation;
never an Anvil/AIDA login node. The small matrix fits do not require an H100.
Offline W&B records real fold progress, family/control MSE and separate family
quality decisions. It uploads no data or credentials. Pretraining success does
not imply that post-training is scientifically justified.

```bash
.venv-state/bin/python scripts/run_training_with_wandb.py \
  --stage pretrain --mode offline --group public-biological-v5 \
  --output-dir artifacts/tracking/biological_conditioning_v5 \
  --summary-file artifacts/lingshu_scdfm/biological_conditioning_v5/result.json \
  -- .venv-state/bin/python scripts/train_biological_conditioning_v5.py \
  --cache artifacts/lingshu_scdfm/public_cache_expanded_train_v3.npz \
  --esm2-features dataset/state_support/extracted/ESM2_pert_features.pt \
  --lingshu-embeddings artifacts/lingshu_scdfm/v1/lingshu_embeddings.npz \
  --lingshu-receipt artifacts/lingshu_scdfm/v1/lingshu_embeddings.json \
  --output-dir artifacts/lingshu_scdfm/biological_conditioning_v5
```

Use fresh paths for any explicitly registered new run, never overwrite this
experiment. Read `result.json:decisions` for scientific status and the W&B
`tracking.json` for execution/logging status. Unit tests are synthetic and are
not evidence that real-data conditioning works.

## Completed run: 2026-09-09

The real run completed in 120.31 seconds on the authorized BioHPC compute host,
using 28,449 public K562 training cells, 145 targets and 512 genes. All 147
relevant synthetic tests passed before launch. All 15 outer folds and all
12 feature arms completed. No new Anvil job or challenge submission was made.
An independent read-only audit reproduced every OOF metric and gate, checked
all 180 fold records and selected specs, and verified all export/log receipts.

| Arm | Repeated held-target MSE | Difference from constant |
| --- | ---: | ---: |
| Optimized constant | 0.01307749978 | Reference |
| ESM2 | 0.01307795945 | 0.00352% worse |
| Lingshu | 0.01307972164 | 0.01699% worse |
| ESM2 + Lingshu | 0.01308230140 | 0.03672% worse |
| Zero effect | 0.01330588687 | 1.74641% worse |

All three conditioning screens **failed**. Fusion is also 0.03320% worse than
ESM2 alone and loses to one genuine-ESM2/shuffled-Lingshu control. The small
differences do not establish harm or useful feature signal; their paired
intervals against the constant span zero. Beating zero effect is insufficient
because the constant performs better. Results are not directly comparable to
v4's MSE because v5 uses a different shared panel and fixed fold seeds.

Final all-target research refits retained nonzero conditional branches: ESM2
input/output ranks 32/4, residual gain 1; Lingshu 32/16, residual gain 0.25;
fusion 32/16, residual gain 1. All selected penalty 0.1 and mean gain 0.5.
Their inner-CV selections do not override the outer-CV failure. Do not deploy
or post-train these exports as validated checkpoints.

Real offline W&B run `526a459abce5` recorded 17 events, exit code 0, and zero
tracking errors. All three family-quality flags are false. Logs are in
`artifacts/tracking/biological_conditioning_v5/`; this is not a cloud dashboard.
Expression, weights and target tables were not uploaded.

Recorded identities:

- Contract: `7eaa69be0ff6e0a69f36990bdce7d0c98794ecdf26eda89d2e6bf176977572b5`.
- OOF errors: `477eca2477376fb4ca95aa5c03650bdd613003e45d096ff22dfc19c409f2570c`.
- ESM2 export: `29fe5895adf4d5b8eac6eda6f1e20374029c5bf5b8fc7c843dc6a2ba3cd43c5b`.
- Lingshu export: `4051248796b899384ea6ba117d032ac8c074a5a9496571755a2ef6a1dc1d6c42`.
- Fusion export: `26e08bf1c38f94d7e46a9aa507fff23567322c816bc141f8b5bd3193fb2a2bb3`.

Next hypothesis: test a larger public target universe before scaling model
capacity. The available GWPS pseudobulk data provide thousands of targets,
but require a new exclusion-aware extraction and whole-cluster CV contract.
Only a passing result should lead to cross-context calibration, then a
distribution-aware flow pilot. See `PUBLIC_BIOLOGICAL_NEXT_STEPS.md` for the
concrete source/coverage audit and proposed sequence. This follow-on is a
proposal, not an executed or preregistered experiment. No score gain is claimed.
