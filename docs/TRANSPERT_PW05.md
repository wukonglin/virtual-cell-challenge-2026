# TransPert pw05 — similarity-weighted public transfer with magnitude compression

## Official result

The validation submission reached `published` on 2026-09-11 with no server error.

| Field | Value |
|---|---:|
| Entry ID | `asT16iuxJgjNkMiaeXU8` |
| Overall | `0.04342560673511914` |
| Rank at publication | `417` |
| Partition | `val` |
| Panel | `vcc2026-val-1` |
| Anchor set | `vcc2026-valA-r4+vcc2026-valB-r4+vcc2026-valC-r4` |

### Normalized component scores

| Component | Score |
|---|---:|
| Perturbation discrimination | `0.2609355767351552` |
| Expression accuracy | `0.0` |
| DE log-fold-change accuracy | `0.04465120497690356` |
| DE direction fidelity | `-0.0963960183208042` |
| DE direction reach | `0.06472426577470614` |
| DE significance overlap | `-0.013361388755245876` |

### Raw aggregates

| Aggregate | Value | No-skill reference |
|---|---:|---:|
| `pds_cosine` | `0.6178966926793015` | `0.5` |
| `expr_mse_unbiased_capped_norm` | `3.811041924780969` | clamped to `0` after scaling |
| `de_wilcoxon_lfc_nmae` | `0.9736299683004721` | `~1.001` |
| `de_wilcoxon_direction_fidelity_yield_raw` | `0.4838178783625024` | `0.5` |
| `de_wilcoxon_direction_reach_raw` | `0.13629355481176195` | — |
| `de_wilcoxon_sig_jaccard` | `0.02550205968784984` | `0.02`–`0.04` baseline, `0.38`–`0.42` replicate |

This is the first positive overall score recorded in this repository. The previous
best tracked entry is `STATE weights v1` at `-0.0059348511`.

## Model

There is no neural network, no training, and no GPU in this model. It is a
similarity-weighted lookup over public Perturb-seq data plus two scalar transforms.

1. **Transfer.** For each panel target `g`, take every public reference cell line that
   knocked down `g`, and average their measured log2 fold-change vectors. Each source
   is weighted by `softmax(r / 0.05)` on the control-profile similarity `r` between
   that reference line and the challenge context, and by a cell-count factor
   `n / (n + shrink_cells)`. Targets with no public measurement fall back to the mean
   response. 283 of the 300 panel targets have public support, mostly from the
   genome-scale Replogle 2022 K562 screen and Feng 2026 iPSC.
2. **Magnitude compression, `--lfc-power 0.5`.** Apply `sign(x) * |x| ** 0.5` per gene.
   This is rank-preserving, so no gene changes position in the response ordering, but
   it compresses the dynamic range a transferred profile inherits from its source line.
3. **Amplitude pinning, `--target-mean-lfc 0.0898`.** Rescale so mean `|log2FC|` equals
   `0.0898`, matching the amplitude of the previously scored configuration so that a
   shape change is evaluated as a shape change.
4. **Emission.** Resample real control cells with replacement and scale gene-wise by
   `2 ** lfc`, then round stochastically to integers. Cell-level depth and background
   heterogeneity are inherited from the real controls.

`--no-shrink` is required. Combining reference-table shrinkage with amplitude
re-inflation manufactures an extreme tail and was measured locally at `-0.150`.

## Why the magnitude transform is the active ingredient

The earlier same-family entry at identical amplitude returned `pds` `+0.211` with
`nmae` `-0.207`. Read together those say the *ordering* of the transferred response
carries real information while the *magnitudes* are worse than supplying none. Step 2
targets exactly that asymmetry, and it is where nearly all the gain appears:

| Component | Earlier same-family entry | This entry | Change |
|---|---:|---:|---:|
| DE log-fold-change accuracy | `-0.207253` | `0.044651` | `+0.251904` |

In raw terms the NMAE aggregate moved to `0.9736` against a no-skill reference of
about `1.001`, so predicted fold changes now beat supplying none.

## Pre-registered local prediction

The configuration was chosen on a local held-out benchmark (2025 H1 carved to the 2026
shape, holding out `arc-vcc`) before submission, and the expected score was recorded in
advance.

| Component | Local prediction | Official result | Error |
|---|---:|---:|---:|
| Perturbation discrimination | `0.378` | `0.261` | `-0.117` |
| Expression accuracy | `0` | `0` | `0` |
| DE log-fold-change accuracy | `-0.039` | `0.045` | `+0.084` |
| DE direction fidelity | `-0.184` | `-0.096` | `+0.088` |
| DE direction reach | `0.078` | `0.065` | `-0.014` |
| DE significance overlap | `-0.035` | `-0.013` | `+0.022` |
| **Overall** | **`0.033`** | **`0.043`** | **`+0.010`** |

The overall was predicted to within `0.010`, but mean absolute per-component error was
`0.054`. The local benchmark is usable for ranking whole candidates and is **not**
reliable for attributing a gain to a specific component: here a large `pds`
overestimate happened to cancel three underestimates. The local benchmark also
over-penalizes DE direction fidelity, which is a repeat of a previously observed bias.

## Production validation

- Build host: `cbsuvlaminck6.biohpc.cornell.edu`, CPU only, no accelerator;
- no Slurm job and no GPU hours were consumed by this model;
- build wall time: `195.5` s;
- shape: `360000 × 18533`;
- context-target groups: `900`;
- cells per group: `400`;
- contexts: `A` `120000`, `B` `120000`, `C` `120000`;
- stored nonzeros: `2052757743` against the `4750000000` cap;
- maximum counts per cell: `53052` against the `1000000` cap;
- packaged artifact SHA256:
  `dabf317725d431bb5fc39fa4a34beab5005023563cc61e25f7451c63b06957cb`;
- `vcc prep` reported `targets: verified against the official list` and
  `normalization: counts-preserved`;
- upload `3.1 GiB` in `4m31s`, then server-side scoring to `published`.

## Reproduction

    python scripts/build_transpert_pw05_submission.py transpert \
        -o out/tp_pw05 --no-shrink --lfc-power 0.5 --target-mean-lfc 0.0898

Inputs are the official `data/controls/` release and public Perturb-seq reference
tables under `data/processed/reftables/`. Neither is committed here. Source checksums
for the ported model are recorded in `results/transpert_pw05/submission.json`; the
code originates from commit `095e16d` of the contributor's VC2026 working tree.

## Limitations and failure modes

- **DE direction fidelity is below chance.** The raw yield-scaled fidelity is `0.4838`
  against a chance level of `0.5`. This is the only component actively costing points
  and is the first thing to attack.
- **DE significance overlap is at baseline.** Raw Jaccard `0.0255` sits inside the
  `0.02`–`0.04` baseline band while the split-half replicate reaches `0.38`–`0.42`.
  This component has by far the most headroom and the model currently extracts none of it.
- **No unseen-target capability.** 17 of 300 targets have no public measurement and
  receive the mean response. On the D/E/F panel that count may change, and the model
  has no mechanism to predict a target it has never seen measured anywhere.
- **Amplitude is a fitted constant.** `0.0898` was inherited from an earlier entry, not
  derived for this shape. It is the obvious next one-dimensional sweep.
- **Rollback.** This model adds only new files under `scripts/transpert/`,
  `scripts/build_transpert_pw05_submission.py`, `results/transpert_pw05/`, and this
  document. Reverting the merge commit fully removes it; no existing pipeline imports it.

## On the receipt rank

Rank `417` is a snapshot at publication, not a property of the artifact. This
repository already documents the same effect: `STATE weights v1` is a byte-identical
resubmission of the earlier STATE artifact and returned an identical score vector at
rank `298` rather than `278`. Ranks in this repository should continue to be read as
receipts, and the overall score compared instead.
