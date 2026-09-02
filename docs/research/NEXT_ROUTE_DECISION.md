# Next-Route Decision Memo

Date: 2026-09-02. Branch `experiment/v7-scdfm-gamma1` at `fbb12a06ae1755d799c8961c67792e81a2f14601`.
Evidence: [`NEXT_ROUTE_PRIMARY_SOURCE_AUDIT.md`](NEXT_ROUTE_PRIMARY_SOURCE_AUDIT.md) and the
tracked measurement `results/scdfm_v7/pds_reachability_hepg2_p4_g100.json`, produced by
`scripts/diagnose_pds_reachability.py` and tested by
`tests/test_diagnose_pds_reachability.py`.

Every measurement below was re-derived by four independent adversaries and an
adjudicator, including one from-scratch h5py-only reimplementation that never imported
the diagnostic. An earlier draft of this memo recommended abandoning the V7 lane for a
discrimination-first route; that recommendation did not survive the adversarial pass and
has been withdrawn. The corrections are recorded in section 7 rather than deleted.

## Summary

**Do not change route on this evidence.** Retain `public_hepg2_p4_g100_raw_candidate` as
the incumbent. The one question that could have voided the STATE-anchored route is now
closed: the team asked the organiser on 2026-09-02 and STATE may be used in the
competition. The V7 scDFM lane cannot be dismissed on the invariance argument that
looked decisive, and the discrimination-first alternative is arithmetically insufficient
on its own. What the audit *does* establish is a first-order defect nobody had named:
this team's raw expression error is **4.79x the no-skill baseline**, hidden behind a
score clamp that reads as `0.0000`. That is the cheapest real target available.

---

## 1. Where the score actually is

`PRIMARY-SOURCE FACT` — Live leaderboard, 50 rows, panel `vcc2026-val-1`, anchor set
`vcc2026-valA-r4+vcc2026-valB-r4+vcc2026-valC-r4`, read 2026-09-02 (latest row stamped
`2026-09-02T04:13:03Z`). The payload publishes **raw** metric values alongside the scaled
ones, so no back-solving is needed.

| Component | Top-50 median | Top-50 max | Top-50 min | This team, `STATE weights v1` |
|---|---:|---:|---:|---:|
| `pds` | +0.6921 | +0.8165 | +0.5439 | **-0.0055** |
| `reach` | +0.1765 | +0.2379 | +0.1074 | +0.0762 |
| `nmae` | +0.1236 | +0.1838 | -0.0036 | -0.0294 |
| `mse` | +0.0759 | +0.2605 | 0.0000 | **0.0000 (clamped)** |
| `jac` | -0.0028 | +0.0173 | -0.0303 | -0.0301 |
| `fid` | -0.0133 | +0.0227 | -0.0688 | -0.0467 |
| overall | +0.1683 | +0.2169 | +0.1425 | **-0.0059** |

Raw values, same source: top-50 `pds_cosine` spans **0.745853 to 0.868179** with median
0.812122; the implied effective `r` is 0.9514, inside the published 0.927-0.984 band.
This team's raw `pds_cosine` is **0.4973690078037904** and its raw
`expr_mse_unbiased_capped_norm` is **4.738905630752523**, both recorded locally in
`results/state_direct_v0/submission.json`.

### 1.1 The expression-scale defect

`LOCAL REPOSITORY FACT` + `PRIMARY-SOURCE FACT` — The published no-skill baseline `b` for
`expr_mse_unbiased_capped_norm` is 0.986-0.992 and the replicate anchor `r` is
0.028-0.045. This team's raw value is **4.7389**, i.e. **4.79x worse than emitting the
context mean for every target**. Scaled, that is
`(4.7389 - 0.989) / (0.0365 - 0.989) = -3.94`, clamped to `0.0000` because `mse` is the
one component bounded on `[0, 1]`.

The scoreboard entry `0.0000` therefore does **not** mean "equal to the baseline". It
means "floored". Reading it as parity was an error in this project's own prose and is
corrected here. Among the top 50, `sd(scoreMse) = 0.0799` exceeds `sd(scorePds) = 0.0555`,
so expression accuracy is the widest lever in that cohort, not the narrowest.

### 1.2 Leverage, and why no single component is a route

Moving each component from this team's vector to the top-50 median:

| Component | Delta overall | Share of the total available gain |
|---|---:|---:|
| `pds` | +0.11627 | 64.1% |
| `nmae` | +0.02551 | 14.1% |
| `reach` | +0.01672 | 9.2% |
| `mse` | +0.01265 | 7.0% |
| `fid` | +0.00557 | 3.1% |
| `jac` | +0.00456 | 2.5% |
| **total** | **+0.18127** | 100% |

`pds` carries 64.1% of the budget and 63-72% of every top-50 team's own score. **It is
still not a route on its own.** Freeze the other five components at this team's recorded
values and set `scorePds` to the live world-number-one value 0.8165044: overall becomes
**0.1310682**, still below the rank-50 threshold of **0.1424575**. A `pds`-only entry
would need raw `pds_cosine` 0.899379, above the current leader's 0.868179 and therefore
unreachable; even a perfect `scorePds = 1.0` reaches only about rank 34.

The minimal sufficient set is **`pds` + `nmae` + `reach`**, which at top-50 medians gives
overall **0.1525601**, roughly rank 42.

## 2. What can and cannot move `pds_cosine`

`PRIMARY-SOURCE FACT` — `pds_cosine` takes the per-group **count-summed** `bulk_lognorm`
pseudobulk, subtracts the real control, removes **all 300 panel target genes**, and for
each target ranks the cosine distance to its own real effect against the other 299,
scoring `1 - rank/(n-1)` with mid-rank ties. Chance is exactly 0.5.

**A pure amplitude rescale is exactly inert.** Scaling the predicted effect by 0.25, 0.5,
1, 2, 4, 8 or 16 returns `pds_cosine = 0.564526198439242` in all seven arms,
bit-identical. This holds here only because the prediction's control pseudobulk is
**bit-identical** to the real one (`max |pred_ctrl - real_ctrl| = 0.0`, corroborated by
`views.json: prediction_controls_match_sealed_truth_exactly: true`); a candidate that
perturbed its own control cells would break the equivalence.

**This must not be read as "stop calibrating amplitude".** `pds` is blind to global
expression scale by construction, and global expression scale is precisely where this
team is 4.79x off. Amplitude calibration is inert for one component and is the direct fix
for another.

**A transformation preserving each group's per-gene integer count sum leaves `pds_cosine`
exactly unchanged — but not `expr_mse_unbiased_capped_norm`.** The earlier claim that
both are invariant is **wrong**. `external/cell-eval2/src/cell_eval2/moments.py:137`
computes a delete-one jackknife
`v_ig = log1p(TS * (P_pg - y_ig) / (S_p - lib_i))` that reads individual cells and
libraries; `correction_for:465-486` routes `bulk_lognorm` to it and raises if it is
absent. Measured on the real artifact: an exact per-gene count permutation leaves
`bulk_lognorm` bit-identical while moving the jackknife term by **+18.75% and +20.75%**,
with the trace cap non-binding (`C_pred/C_real = 0.310` against `PRED_TRACE_CAP_K = 1.0`).
Per-cell library standard deviation within a group fell from 7340.2 to 596.4 — a large
distributional change that `pds` cannot see and `mse` can.

## 3. Why the incumbent is at chance on `pds`, measured

Measured on the sealed HepG2 panel in the scorer's own ranked space (6,807 genes after
removing the 300 panel targets), from
`results/scdfm_v7/pds_reachability_hepg2_p4_g100.json`:

| Quantity | Value |
|---|---:|
| `pds_cosine`, as scored | 0.564526198439242 |
| Same prediction with the panel exclusion **off** | 0.743188405797101 |
| **Cost of the panel exclusion alone** | **0.178662** |
| Top-1 retrieval, ranked space | **3 of 300** (chance 1) |
| Top-1 retrieval, before exclusion | 32 of 300 |
| Matched cosine, mean / median / min / max | +0.0248 / +0.0124 / -0.1985 / +0.2503 |
| **Targets with a matched cosine at or below zero** | **37.3%** |
| Mismatched cosine, mean / sd | +0.0134 / 0.0598 |
| Separation `z` | +0.1906 |
| Mean competitors nearer than the true match | 130.2 of 299 |

### 3.1 Attribution, corrected

Two simulated references carry the candidate's own directional accuracy with **perfectly
independent** error, averaged over three registered seeds:

| Reference | `pds_cosine` |
|---|---:|
| Homogeneous — every target at the mean accuracy 0.0248 | **0.9111** |
| Heterogeneous — each target at its own **signed** matched cosine | **0.6263** |
| `g100`, actual | **0.5645** |

The attribution follows directly:

- **per-target signal deficit: 0.2847** (homogeneous minus heterogeneous);
- **target-correlated error: 0.0618** (heterogeneous minus actual).

The signal deficit dominates by **4.6 to 1**. An earlier draft reported only the
homogeneous reference and concluded that the shortfall was structured error; that
conclusion is withdrawn. Most of the gap is that per-target directional accuracy is both
tiny and dispersed, with 37.3% of targets predicted anti-correlated with their own truth
and top-1 retrieval at 3 of 300 against a chance of 1. This is a small mean-rank drift,
not partial identification.

### 3.2 The structural finding that does survive

`g100` emits a genuinely cross-target-structured common response — mean pairwise cosine
among predicted effects **0.1157**, close to the truth's **0.1394** — but that common
response points the wrong way: `cos(predicted shared axis, real shared axis) = 0.1005`.
Removing it is therefore worth only **+0.01214**, reaching a true maximum of
**0.576667 at fraction 1.05**, on a surface flat across `[1.00, 1.10]`. The earlier
figure of +0.0119 at 1.1 was a coarse-grid artifact.

### 3.3 Information beats amplitude, quantified

- Information-free perturbation applied at 0.1x to 4.0x the delta norm moves `pds` from
  0.5645 to between 0.5395 and 0.5646 — that is, adding energy without information does
  nothing or hurts.
- Rotating the predicted delta toward the truth by only `t = 0.02 / 0.05 / 0.10` reaches
  `pds` **0.686 / 0.827 / 0.943**.

Two percent of aligned signal is worth more than four hundred percent of unaligned
amplitude. Any route that raises `pds` must add *aligned direction*, and the amount
required is small.

## 4. The six required answers

**1. Which route attacks the current failure modes?** Three distinct defects are now
named, and they call for different work: a global expression-scale error of 4.79x
(amplitude and library calibration); a per-target signal deficit with 37.3% of targets
anti-correlated (target representation and reliability gating); and a common response
pointing 84 degrees away from the truth (cross-context transfer). None of the three
candidate methods addresses all three, and the incumbent addresses none of them well.

**2. Which zero-shot axis is actually demonstrated?**

| Method | Demonstrated axis | Matches the organiser's stated axis? |
|---|---|---|
| STATE ST | Unseen cell **context**, seen perturbations, held-out context's controls in training | **Yes** |
| scDFM | Unseen CRISPRa target identity and dual combinations, **within one seen cell line** | No |
| PerturbDiff | Unseen (context x perturbation) **combination** under partial coverage, with 30% of held-out-donor perturbations moved back into training | No |
| PertMind | Unseen cell context for small molecules; zero-shot task transfer onto genetic reverse-ranking | Partly, but the output is ternary |
| STATE SE | None for perturbation response | No |
| Flow Matching / MFM | None biological | No |

**3. What information is genuinely unavailable for the 300 VCC targets?** Same-target
CRISPRi single-cell evidence for **283 of 300**. Same-target evidence on the official
18,533-gene axis for **all 300**. Raw integer counts for any perturbed cell on that axis.
The identity of contexts A/B/C. The upstream ESM2 lineage. And, after 2026-10-22, the
D/E/F target list itself.

**4. Why should any proposal beat the incumbent?** On the evidence gathered, the honest
answer is that no proposal in this audit has been shown to. The incumbent's three defects
are now measured, which is a precondition for improving them, not a demonstration that
any of the three candidate methods will.

**5. What would falsify a proposal quickly?** See section 5.

**6. What can be tested on CPU before requesting an H100?** All of section 5. No GPU is
justified until it returns numbers.

## 5. Recommended next experiments

Both are CPU-only, reversible, and produce a registered number. Run them in parallel.

### EXS-1 — Expression-scale repair

**Hypothesis.** The raw `expr_mse_unbiased_capped_norm` of 4.7389 is a global scale and
library-calibration failure, not a per-target modelling failure, and can be brought below
the no-skill baseline of ~0.989 by rescaling the emitted counts alone.

**Method.** On the sealed HepG2 panel, sweep a global effect-attenuation factor and a
per-cell library-matching correction over the existing `g100` counts, rescore raw
`expr_mse_unbiased_capped_norm` and raw `de_wilcoxon_lfc_nmae` with the pinned scorer,
and record `pds_cosine` at every point to confirm it does not move.

**Registered pass/fail.** PASS if some setting reaches raw
`expr_mse_unbiased_capped_norm <= 0.900` (that is, better than the no-skill baseline)
**while** raw `pds_cosine` stays within 1e-6 of 0.564526198439242 and raw
`de_wilcoxon_lfc_nmae` does not rise above 1.017691308794254. FAIL otherwise.

**Why it is worth doing first.** It is the only defect that is plainly a calibration
problem, it is worth up to +0.0434 on `mse` alone at the top-50 maximum, and it requires
no new biology. If it fails, the expression error is structural and that is itself a
major finding.

### PDT-1 — Public-transfer direction test

**Hypothesis.** A direct transfer of measured public CRISPRi *trans* response directions,
applied to the sealed HepG2 control pseudobulk, raises the per-target matched cosine
above `g100`'s +0.0248 and reduces the 37.3% anti-correlated fraction.

**Falsifier.** It does not. In that case the public effect atlases carry no transferable
*trans* direction for this context, and the correct next act is to acquire same-target
evidence rather than to build a better model.

**Arms**, all pseudobulk-only, scored through `scripts/diagnose_pds_reachability.py`:

| Arm | Prediction |
|---|---|
| `A0` | `g100` as scored — must reproduce `pds_cosine = 0.564526198439242` |
| `A1` | `g100` with the shared component removed at fraction 1.05 — expected 0.576667 |
| `A2` | Direct K562 GWPS same-target *trans* log-fold direction, gene-mapped onto the HepG2 axis; `g100` elsewhere |
| `A3` | `A2` with the shared component removed |
| `A4` | Shuffled-target control: `A2` with the target-to-effect assignment permuted under seed 20260901 |
| `A5` | Oracle — must reproduce exactly 1.000000 |

**Registered pass/fail.** PASS if `A2` or `A3` reaches `pds_cosine >= 0.6263` — the
heterogeneous independent-error reference, i.e. the score `g100`'s own per-target accuracy
would earn with unstructured error — **and** exceeds the shuffled control `A4` by at
least 0.0400, **and** reduces the anti-correlated target fraction below 0.300. FAIL
otherwise.

**Threshold justification.** 0.6263 is not arbitrary: it is the measured point at which a
candidate stops losing anything to error structure. `A1` is known to reach only 0.576667,
so shared-response removal alone cannot clear the bar. The 0.0400 shuffled margin is 3.3x
the entire shared-response channel and 7.4x the 0.0054 full width of the `g050`/`g075`/
`g100` spread that P4 selected within.

## 6. What is rejected, and what is not

**Rejected outright.**

| Rejected | Reason |
|---|---|
| The released scDFM Norman checkpoint as a VCC predictor | 62/300 targets and 4,882/18,533 genes in vocabulary; `Vocab.encode` raises `KeyError` on the other 238; CRISPRa not CRISPRi; output is a clamped continuous vector over 1,000 genes with var names `'0'..'999'` |
| Released STATE ST checkpoints used directly | `pert_onehot_map.pt` is literally `np.eye(2024)` with **0/300** VCC targets; the cell-type input is a closed four-way one-hot with no A/B/C row |
| PerturbDiff as the route | Output is `normalize_total` -> `log1p` -> `/10` with a `relu` head over 2,000 HVG, not raw counts on 18,533 genes; released `cov_encoding` is a closed one-hot; its Replogle preprocessing inherits an on-target knockdown-efficacy filter that reads perturbed outcomes; no unseen-target-identity result is reported |
| PertMind as a predictor | Withdrawn manuscript, no source code released, retrieval KG unreleased so no reported number is reproducible, ternary output with no count semantics, altered Apache text on the weights |
| SE-600M in front of the transition model | Arc's own shipped evaluations: ST-HVG beats ST-SE on DE ranking in Jurkat 23/26, RPE1 24/26, K562 about 22/26; HepG2 reverses 13-12-1 |
| Byte-reproducing `prediction.h5ad` as an acceptance test | Structurally impossible: no registered renderer consumes a continuous anchor, the axes are disjoint, and the pinned producer hashes have drifted |
| Opening Gate 1b by editing status fields | Forbidden by the project's own rules; the upstream facts were never released |

**Explicitly NOT rejected: the V7 scDFM residual lane.** The invariance argument against
it does not hold. The compositor pins the per-cell mean of `log1p` — the legacy `lognorm`
functional — while the competition scores under `bulk_lognorm` count sums, and the
compositor's own receipt records `raw_count_invariance_claimed: False`. The leak is
first-order in the residual weight: at the smallest registered weight `w = 0.05` a
compositor-legal residual already rotates every predicted delta by **6.83 degrees**, and
`pds` moves by `+0.0003 / -0.0080 / -0.0446` at `w = 0.05 / 0.10 / 0.20`. Since section
3.3 shows that a rotation of 1.8-3.6 degrees *toward the truth* would reach the
leaderboard band, the constraint does not bind.

The V7 lane must therefore be judged on **information**, not on inertness: uninformed
residuals measurably destroy score, and an oracle-steered residual honouring the
non-negativity constraint reaches +0.230 raw `pds` at a small displacement and +0.435 at
the shipped displacement — an upper bound, explicitly truth-steered and not attainable.
Its five execution blockers (no producer for the anchor, axis-disjoint panels, the
non-negativity gate, the ~320 GB compositor ceiling, and two undefined hash fields) all
still stand and are unchanged by this memo.

## 7. Ordered next actions

**Tier 0 — governance and hygiene. Hours, no GPU.**

| # | Action | Status / unblocks |
|---|---|---|
| 0.1 | Prize eligibility of a STATE-derived entry | **Done 2026-09-02.** The team asked the organiser: STATE may be used in the competition. Archive the written reply beside this memo. The remaining Arc licensing questions — `MODEL_LICENSE` §1.2 "use as a prior" as a Derivative Work, and the §1.4 Commercial Entity determination — are Arc matters, not organiser matters, and stay open |
| 0.2 | Trace and sever the HepG2-in-fitted-stages lineage through `artifacts/bayes_public_effects_v1.npz` (68 HepG2-derived rows) | Open. Gates the HepG2 zero-shot claim |
| 0.3 | Quarantine the three unregistered large files, especially `competition_val_template.h5ad` (7.16 GB) | Open. Possible firewall exposure |
| 0.4 | `jwsup` write access on the mirror | **Owner decision 2026-09-02: retained.** The exposure is accepted rather than remediated, so the control becomes detection: install an `origin`-versus-mirror `git ls-remote \| diff` tripwire, and keep publishing authoritative history to `origin` only |
| 0.5 | Register numeric promotion margins per component, replacing "material" and "the registered margin" | Open. Makes rule 2 and the stop rule enforceable |
| 0.6 | Regenerate the receipts bound to the superseded config sha256 `257b28a8...` | Open. Gates every V7 boundary citation |

**Tier 1 — decisive CPU measurements. Run EXS-1 and PDT-1 in parallel; both are
specified in section 5.**

| # | Action | Decision it produces |
|---|---|---|
| 1.1 | **EXS-1**, expression-scale repair | PASS -> the 4.79x expression error is calibration and is fixable without new biology. FAIL -> it is structural, which is itself a major finding |
| 1.2 | **PDT-1**, public-transfer direction test | PASS -> public CRISPRi atlases carry transferable *trans* direction. FAIL -> acquire same-target evidence rather than build models |
| 1.3 | Repeat `scripts/diagnose_pds_reachability.py` on the Jurkat sealed panel | Whether the diagnosis is context-specific or general |
| 1.4 | ESM2 information test under whole-cluster holdout against gene-identity, GO/STRING, basal-only and shuffled controls | Whether continuous target conditioning carries information at all |
| 1.5 | Resolve the Feng modality; add machine-readable receipts for the six unreceipted files | Whether the same-modality union is 282 or 100 |
| 1.6 | Gate-integrity fixes: freeze `AuthorizedH5Source`, absolute forbidden-import check, CSR read path, per-stage authorisation | Whether any current receipt attests to what it claims |

**Tier 2 — only after Tier 1 returns numbers. Still no GPU.**
Act on whichever of EXS-1 and PDT-1 passed; register the holdout design (whole-context
and whole-target-cluster, three seeds); and measure the count renderer's ceiling before
building on it — it reweights only genes already observed in each source cell, and 57.24%
of entries are exactly zero.

**Tier 3 — H100.** Nothing in the current evidence justifies a request. The V7 lane's five
execution blockers are unchanged, and section 6 replaces the false inertness argument with
an information test the lane must pass first.

**Tier 4 — D/E/F calendar, independent of the above.**
Before 2026-10-22: remove every dependency on the literal labels A/B/C and on the current
300-target list; confirm the STATE disclosure obligation now that use is permitted. From
2026-10-22 to 2026-11-05 23:59 UTC the final entry runs blind, two scored submissions per
day, only the last one counting.

## 8. Corrections to the earlier draft

Recorded rather than deleted, because the errors are instructive.

| Claim as first written | Status | Corrected |
|---|---|---|
| A count-sum-preserving transformation leaves `pds` **and** `mse` exactly unchanged | **REFUTED** | True for `pds` only; the `mse` jackknife is per-cell and moves by +18.75%/+20.75% |
| The V7 residual cannot move the scored surface because the compositor preserves the group mean | **REFUTED** | The compositor preserves the mean of `log1p`, not the count sum the scorer uses; the leak is first-order in `w` |
| At the same directional accuracy an independent-error prediction scores 0.904, so the shortfall is structured error | **OVERSTATED** | The homogeneous reference discards per-target dispersion. With signed heterogeneity preserved it is 0.6263, so the shortfall is 4.6:1 a signal deficit |
| Shared-response removal peaks at +0.0119 at fraction 1.1 | **OVERSTATED** | +0.01214 at fraction 1.05; the 8-point grid missed the optimum, and the surface is flat over [1.00, 1.10] |
| Top-50 raw `pds_cosine` is 0.75-0.87, back-solved from `b` and `r` | **OVERSTATED** | Raw values are published directly: 0.745853-0.868179, median 0.812122, effective `r` 0.9514. Our own raw value 0.4973690078 is recorded locally |
| `pds` is the dominant lever, therefore a `pds`-first route | **REFUTED as a route** | `pds` carries 64.1% of the budget, but the world-number-one `pds` with our other five components still lands at 0.1311, below the rank-50 threshold 0.1425. Minimal sufficient set is `pds` + `nmae` + `reach` |
| Our `mse` score of 0.0000 means parity with the baseline | **REFUTED** | It is a clamp. Raw 4.7389 against a baseline of ~0.989 is 4.79x worse than emitting the context mean |
| Stop the V7 lane and switch to a discrimination-first route | **WITHDRAWN** | Do not change route on this evidence |

## 9. What this memo does not establish

It does not establish that EXS-1 or PDT-1 will pass. It does not establish how the
leaderboard's top teams reach raw `pds_cosine` 0.746-0.868. It does not resolve the ESM2
lineage, the STATE prize-eligibility question, or any of the eight open items in the
audit. It does not measure anything on the challenge panel itself — the HepG2 proxy panel
is target-disjoint from the VCC 300, and the one same-panel number available
(raw `pds_cosine` 0.4974 against the top-50's 0.746-0.868) is a leaderboard receipt, not
a local measurement. And it does not retire the V7 lane; it removes the false argument
against it and replaces it with an information test the lane must pass.
