# Public auxiliary v10: deduplicated data and actual flow pretraining pilot

2026-09-10. New development experiment, not a change to frozen v8.1 results or
v9's candidate-only authorization. The user now authorizes exploratory VCC
attempts once prediction/schema/provenance validation passes; the old 1% local
improvement gate is not a prerequisite for an honest exploratory submission.
This pilot itself emits 482-gene continuous predictions, not a VCC file.

## Data preparation

Input is the exact completed v9 candidate plan SHA256
`bde224820c144517dbd49f8740c9d3da750f4c688b2081607ebbca25032df060`.
Preserve 487 auxiliary targets, 5,075 groups, 57,207 treated original rows,
1,616 fit-reference and 3,232 context controls. No panel56, protected772,
tuning-anchor, sham, H1, RPE1, HepG2, challenge-treated or Feng rows are admitted.

Create a separate expression-materialization contract before X reads. Full-hash
BOTH already-local raw source handles against their exact acquisitions, check
unchanged handle identities, and reconstruct the v9 metadata/roles before any
expression decode. Stream only the authorized original-row union in chunks of
at most256, preserving deterministic row IDs. Before any materialization, bind
the official18533-gene metadata axis in `dataset/controls/gene_names.csv`
(SHA256 `25bfa66715e186bebabce7ac788bbcea47e2bf59ca70be1f8f3a06f2f0e47201`).
Filter the original7563 normalization/512 output symbols by exact membership
in that official axis, preserving their order. The new axes are7097 and482;
466 normalization and30 output genes are explicitly excluded, not zero-filled.
Normalize over those7097 shared measured symbols in float64 to log1p(CP10000),
then retain482 outputs as float32. Zero measured library fails closed; negative,
nonfinite, fractional raw counts, ambiguous axes or changed sources fail closed.

Deduplicate rows per source. Store source-level numeric NPZ arrays with exact
original row IDs, role/group/pool mappings, and mean/std context features derived
only from each pool's32 context donors. No repeated group-wise expression cache.
Save per-source hashes/size bounds and a completed two-source receipt. Cache
loading validates exact arrays, dimensions, row roles, pool derivation and
external completion hash. Materialization does not authorize model fitting.

Data preparation limits: approved BioHPC compute host, two numerical CPU
threads, no GPU,8GiB peak process RSS,1hour checkpointed elapsed,512MiB per
archive and1GiB total outputs. No network, credentials or new cloud download.

## Source-specific pretraining roles and features

Before fitting, create a new training contract pinning cache/receipt, this
protocol, complete code closure, source roles and GO artifacts. Each source's
targets rank by SHA256(canonical JSON [20260926, source, target]), ties by exact
symbol. The first ceil20% are held tuning targets; remaining targets fit:
K562299fit/75held, Jurkat90fit/23held. These are auxiliary development holdouts,
not the56-target outer panel and not an untouched final test. No response-based
target selection, checkpoint selection, early stopping or hyperparameter search.

Parse only the pinned local GO GAF/OBO acquisition. Build each source vocabulary
from its fitting targets only. Encode exact-symbol direct accepted GO terms as
binary multi-hot, L2-normalize each nonempty row, append explicit GO-availability
and GenePT-availability flags. No IEA/ND/NOT, ancestors or alias guesses.
GenePT is unavailable in this first pilot and its flag remains0; no fabricated
GenePT embeddings are supplied. Later acquired GenePT requires a new experiment.
The matched-control context is concatenated mean/std over the same482 genes,
computed from context32 only; no recipient/source one-hot or response-derived
context encoder is used.

Train separate models per source, never a shared K562/Jurkat treated checkpoint.
Compare three independently fitted arms per source: true GO, shuffled complete
target rows within fit/held partitions separately, and fit-target-mean constant
conditioning. Preserve identical initialization and sampling/time RNG across
arms. The unmatched source is not part of fitting, tuning or model selection.

## Actual flow objective and fixed optimization

Use an unpaired conditional population flow, not measured before/after pairs.
Fitting starts are sampled from fit-reference16, destinations from treated cells
of the same source/target/batch. Sample target uniformly, batch within target
uniformly, then source/destination cells. Never use context32 as a fitting
response/prior or add the exact starting cell as an extra conditioner.

Representation: positive log-expression x maps to z=sqrt(x), exact zeros map to
z=-0.25. Decode with ReLU(z)^2. This fixed zero-atom representation is used in
training and inference; no inference-only negative clipping or extra noise.
Zero coordinates may activate by crossing0, and nonzero values may deactivate.
A zero-initialized velocity head begins at the control baseline within numerical
tolerance. Nonnegativity follows from the decoder, not evidence of accuracy.

Learn velocity on linear interpolants between encoded control and treated
cells, with ordinary mean squared flow-matching loss. Residual MLP width128,
two blocks, separate state/target/control-context/time projections. AdamW,
600steps per arm, batch128, learning rate1e-4, weight decay1e-4, gradient norm
clip1, seed20260926. Six fits total,3,600 optimizer updates. No full training
recipe or convergence claim follows from this small fixed-budget pilot.

Evaluate held auxiliary targets from context32 donors, disjoint from fit16;
compare model and unchanged control on exactly those donors. Heun integration,
16steps. Report target-balanced expression centroid MSE, distribution/variance,
zero/negative fraction and target sensitivity with per-target results when
available. Control donors also provide context covariates, but not trained
response labels. Report dependencies from reused source/batch pools explicitly.
Fixed final checkpoint, no score-based checkpoint or arm selection in this run.

## Recording, replay and resource boundaries

W&B0.19.11 offline, stage `pretrain`, one run per source/arm. Log every actual
optimizer step/loss/gradient norm and final evaluation, durable scalar journal.
No expression/code/checkpoint/console/environment upload or automatic cloud sync.
Save safe numeric model and AdamW state plus typed JSON structure, exact
features, history and provenance. Never load an arbitrary pickle. Restore saved
state and replay held evaluation without refitting before marking completion.

Training: approved BioHPC compute host, two CPU threads, one actually verified
RTX2080Ti, at most6GiB allocated GPU memory,16GiB process RSS,2hours checkpointed
per arm,8GiB recorded outputs. Use the verified GPU UUID/device selection;
no takeover of existing processes, head-node computation or H100 claim.
The runner first loads/authenticates caches in a GPU-disabled CPU phase, then
restores the registered GPU visibility before first CUDA initialization.

## Progression

The official-axis alignment was discovered and registered before any v10 raw
materialization or optimizer fit. It changes both normalization and output axis;
absolute v10 losses must not be compared causally with v8.1. Use within-v10
matched controls and negative-condition comparisons instead.

This establishes real auxiliary flow pretraining only if optimization and saved
checkpoint replay succeed. Public-only post-training must restore the trained
checkpoint and conditioning/optimizer lineage; new random fitting is not
post-training. Feng needs validated barcode/guide/target/donor joins, target
exclusions, donor-separated roles and gene/count axes before expression reads.

Keep PromoterAI as a separate output-gene-feature ablation. Existing local
promoter composition descriptors are not PromoterAI weights or predictions.
Do not conflate the two. New GenePT/PromoterAI artifacts need verified provenance
and permitted use, not an unreported feature swap inside this pilot.

For an exploratory VCC attempt, first implement/test the new flow-to-full-count
adapter: explicit18533-gene alignment,7097 normalization mask, untouched
unmodeled genes anchored to raw controls, zero-residual count identity, finite
nonnegative integer counts and correct per-context/perturbation quotas. Use the
installed official prep/validator and current submission-limit check; save entry
ID and poll the same upload. Six attempts/day is the user's stated allowance,
not yet independently verified here. Do not upload a mislabeled old model or
claim the482-gene pilot is already submission-ready.
