# Bayesian Public-Prior Baseline v0

## Official result

The validation submission reached `published` on 2026-08-26 with no server error.

| Field | Value |
|---|---:|
| Entry ID | `4gqHnJ61sXnaMRRcLPTc` |
| Overall | `-0.020587639341217333` |
| Rank at publication | `173` |
| Partition | `val` |
| Panel | `vcc2026-val-1` |
| Anchor set | `vcc2026-valA-r4+vcc2026-valB-r4+vcc2026-valC-r4` |

### Normalized component scores

| Component | Score |
|---|---:|
| Perturbation discrimination | `-0.000022441888027582818` |
| Expression accuracy | `0.0` |
| DE log-fold-change accuracy | `-0.012492210789393074` |
| DE direction fidelity | `-0.11212822275427158` |
| DE direction reach | `0.049985586492981675` |
| DE significance overlap | `-0.04886854710859345` |

The overall value is the unweighted mean of these six scores. Zero is the context-mean baseline on the official normalized scale; one is the split-half experimental reference. Rank is volatile and should not be treated as a fixed artifact property.

## Model

The model uses public Perturb-seq effects on a harmonized log1p(CP10K) scale. It estimates context similarity from the released control profiles, constructs context-conditioned generic effects, and adds shrunk direct effects for 17 validation targets with public measurements. A raw-count generator starts from real control cells and applies sparse thinning or increments while retaining cell-level depth and most background heterogeneity.

The ESM2 transfer candidate was not deployed:

- best held-target relative SSE: `1.0182816528067589`;
- nearest-neighbor relative SSE: `1.828381935880335`;
- median held-target latent cosine: `-0.1864543855190277`;
- deployed target-specific ESM scale: `0.0`.

This is therefore a context-weighted generic public response plus 17 direct targets, not a successful ESM2 or STATE zero-shot predictor.

## Production validation

- H100: NVIDIA H100 80GB HBM3;
- H100 fit job: `793706`, exit `0:0`;
- generation and packaging job: `793708`, exit `0:0`;
- shape: `360000 × 18533`;
- context-target groups: `900`;
- cells per group: `400`;
- stored nonzeros: `1921377517`;
- official target verification: passed;
- raw non-negative integer counts: passed;
- control-cell rejection check: passed;
- package container validation: passed.

Key artifact hashes recorded outside Git:

| Artifact | SHA256 |
|---|---|
| Public-effect cache | `8cd33d85c946ff234680d253504b6cc0ced23663e9f205bf745fc4c4d66e4ac4` |
| H100 prior | `8df237226a2d37295724c90d646660ed188e8d3ae9ef366f715d9b60799118e4` |
| Submission container | `f7fe68961fa8b5315d33e903045936871f6c7d33284b54de2f409986f24614c3` |

## Interpretation

The near-zero perturbation-discrimination score confirms that the generic response does not identify the 283 unseen targets. Direction reach is the only positive normalized component, while direction fidelity and significance overlap are the largest deficits. The next model should prioritize target-specific biology, sign calibration, and DE-set cardinality before increasing generator complexity.

## STATE development candidate

This candidate has not been submitted and has no leaderboard score yet. It is
documented separately from the published Bayesian baseline to prevent local
proxy metrics from being mistaken for official results.

### Training and checkpoint selection

- architecture: `state_sm`, 162,626,484 parameters, 128-cell set size,
  672 hidden features, and four transformer layers;
- support: six permitted public matrices and 5,120-dimensional ESM2
  perturbation embeddings;
- training: one NVIDIA H100 80GB, Slurm job `859832`, 20,000 optimizer steps,
  successful completion;
- zero-shot validation context: HepG2;
- selected checkpoint: step 16,000, validation loss `1.666636586`;
- selection rule: minimum recorded validation loss, earliest step on ties;
- later losses: `1.735803962` at step 18,000 and `1.701111555` at step 20,000.

The explicit step-16,000 archive is required because the upstream checkpoint
callback can save weights before validation at the same step boundary. H100 job
`860433` used the selected checkpoint to generate a diagnostic sparse
effect-prior for all 900 context-target groups. That prior is not the preferred
final adapter and has not been submitted.

### Preferred direct count adapter

For each source control cell, the adapter runs STATE with the target embedding
and with the non-targeting embedding on identical raw-log1p input, then uses
their paired cellwise difference. Non-target effects use the smooth bound
`0.6 * tanh(delta / 0.6)`; the target is forced to a `0.20` remaining fold.
Only observed shared-gene counts are reweighted, so source-zero genes remain
zero. The 456 challenge-only genes are copied exactly, and deterministic
largest remainder restores the exact source library after composition changes.

This is intentionally conservative. It preserves genuine challenge-cell
sparsity, library depth, and unmodeled genes, but cannot generate de novo
expression and does not model perturbation-induced library-size changes. The
effect bound and forced target fold are calibrated safeguards rather than
learned context-specific biology.

The model-input choice is also explicit. The largest support matrix contains
raw-log1p values, while the other five contain normalized log1p profiles; the
training loader supplied both as stored. This candidate uses elementwise
`log1p(raw)` for the challenge controls. CP10K or CP14K may be worth comparing
later, but neither is justified as an unambiguous reconstruction of the mixed
training representation.

### Verified smoke evidence

The completed CPU plumbing smoke evaluated one target across contexts A, B, and
C with one cell per group. It produced a `3 x 18,533` `int32` CSR matrix with
12,768 stored nonzeros. Every internal schema check passed; each output library
matched its source library exactly, all 456 current-only genes matched the
source counts exactly, all cells changed on modeled shared genes, and no target
knockdown failure was recorded. This smoke is too small for the official
contract and is not performance evidence.

A second production-shape-per-group smoke used 400 cells for the same target in
each context. Its chunks were exactly `[128, 128, 128, 16]`, and the canonical
`1,200 x 18,533` `int32` CSR output contained 6,385,748 stored nonzeros. ABCD1
target totals changed from `140` to `10` in A, `938` to `162` in B, and `158`
to `12` in C. All structural, library, challenge-only-count, and target-
knockdown gates passed. This still covers only one of 300 targets and therefore
is not a full-contract or performance test.

### HepG2 log-normalized proxy

The selected checkpoint was also evaluated on the released HepG2 holdout:
9,386 cells, including 4,976 controls and 4,410 treated cells from 68
perturbations. Current `cell-eval2` mean metrics in log-normalized space were:

| Raw proxy metric | Mean |
|---|---:|
| Direction fidelity yield | `0.496288` |
| Direction reach | `0.250694` |
| LFC NMAE | `0.968758` |
| Significant-gene Jaccard | `0.002487` |
| Unbiased expression distance | `0.002238` |
| Unbiased expression MSE | `0.000466` |
| Capped unbiased expression MSE | `0.001821` |
| Real-mass ratio | `0.011470` |
| PDS cosine | `0.635865` |

These values are not official normalized VCC scores. The public HepG2 matrix is
already log normalized and does not expose recoverable integer counts, while
the competition evaluator scores raw-count submissions. The proxy also uses
upstream direct STATE output rather than the final paired count adapter.

### Current status

Full H100 direct generation, the 360,000-cell scientific gates, official
`vcc prep --require-counts --dry-run`, package creation, submission, and the
leaderboard result are all pending. No claim about improvement over the
published Bayesian baseline is justified until those stages complete and the
official entry is published.
