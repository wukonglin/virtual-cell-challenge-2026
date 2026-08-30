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
