# Public flow diagnosis and conservative residual experiment

## Registered follow-up scope

Written after observing the failed development metrics of Anvil job 20539599
and before fitting follow-up residual baselines. This is post-hoc engineering
on an exposed H1 development set, not untouched validation, a leaderboard score,
or evidence of unseen-target generalization.

Reuse only the authenticated `h1_dev_pilot_v1` cache: K562 and Jurkat for fitting,
H1 for development reporting. Preserve its 772 excluded targets, source roles,
matched-control groups, 7,006-gene normalization axis and fixed 512 outputs.
No new raw expression reads, challenge-treated data, checkpoint promotion,
Feng post-training, challenge submission, paid API or GCP transfer is included.
Execution is CPU-bounded on the BioHPC compute host, not an Anvil login node.

### Frozen-checkpoint diagnosis

Authenticate the downloaded suite, summaries, weights and tracking receipts from
job 20539599; bind hashes in a diagnostic contract before numeric evaluation.
Reconstruct all three arms without fitting; check saved conditioning transforms
and replay original development samples and 16-step Heun integration. Check
agreement with original metrics before interpreting results.

Report predicted versus observed treated-minus-control displacement magnitude,
direction, shared cross-target shifts, and variance relative to both controls
and treated cells. Check the analytic zero-velocity/control identity. An optional
all-zero-target NTC-to-NTC stress test is explicitly out of distribution: the
model has no trained null condition, so this is not a ground-truth NTC test.
Keep negative continuous log-expression visible; never silently clip.

### Conservative mean-effect baselines

Freeze six arms: unchanged controls; source-balanced same-target mean-delta
transfer; additive GO plus matched-control-context ridge with true, fixed
shuffled and train-constant GO; and context-only ridge without target masks.
No hyperparameter selection using H1.

Ridge uses fixed lambda=10 with a penalized intercept (zero-effect prior),
separately training-standardized blocks divided by square-root block width,
fixed context standard-deviation floor 0.1, and source-then-target-then-batch
balanced weights. Seeds for target shuffling and evaluation are 20260910. Fit
transforms independently in each leave-one-training-source-out fold; report
target overlap explicitly. Same-target transfer is a separate unshrunk diagnostic
arm, not an unseen-target predictor. Evaluate every arm, even when worse than
controls, on the same fixed development samples. Report centroid MSE,
distributional diagnostics and negative fractions. Reused controls do not
justify independent-cell confidence intervals. Record offline W&B with accurate
stage/provenance labels; a fresh residual fit is not resumed post-training.

### Optional output-promoter branch

Prepare independently implemented reference-sequence features for output genes,
keeping GO target function and matched-control state as separate inputs. Initial
GC/CpG/window descriptors are not PromoterAI embeddings. Resolve exact symbols
to versioned reference IDs; preserve ambiguity/missingness; freeze strand, TSS
and window choices without expression feedback; record source hashes. Acquire
only a bounded public reference after source/terms checks. No gated Illumina
weights, license acceptance, derived PromoterAI features or performance claim
is included. Promoter augmentation needs a registered ablation before fitting.

## Results

### Completed frozen-checkpoint replay

Contract: `artifacts/public_flow/h1_dev_pilot_v1/diagnostic_contract_v1.json`,
SHA-256 `cf8ff2b61e9ae7bb87422947a62ad47d308ca1e570d798f2ec5fcc8ee2e429d9`.
Result: `diagnostic_results_v1.json` in the same directory,
SHA-256 `f8c1093b14eb23df3788768a3b61177e38f613db019e66bee8f954ac0ea957f1`.
The exact saved GO means/scales matched reconstruction; all three CPU replays
matched the original H100 metrics within 8.5e-8. The zero-velocity identity check
preserved controls exactly. No model fitting was performed by this replay.

For true GO, training shift cosine was 0.62688 and training centroid MSE fell
from 0.0546355 (controls) to 0.0318972. On H1, cosine fell to 0.025926 and MSE
rose from 0.0113449 to 0.0251152. Thus fitting source responses did not establish
cross-context transfer. Development predicted shift RMS was 0.118974 versus
observed 0.099510. The shift-energy contribution 0.0146520 minus twice the
alignment contribution 0.00088173 exactly explains excess MSE 0.0137703.

This is not demonstrated variance collapse: true-GO development variance was
1.00475 times controls and 0.97893 times treated, with zero exact duplicates.
The common cross-target component accounts for 22.3% of predicted H1 shift
energy versus 8.24% observed: elevated, but most wrong-direction energy remains
target-specific. Constant GO is different: 99.94% common-shift energy. Removing
one common offset is therefore not sufficient evidence of a fix for true GO.

The zero-target sham is explicitly OOD, not a trained NTC condition. It must not
be interpreted as validation of a genuine NTC classifier or intervention.

### Completed conservative baselines

Contract: `mean_effect_contract_v1.json`,
SHA-256 `b586749790772a0b1af8dc8179e64a167cd1d8619ac41904e3c262023b68b6c8`.
All six fixed arms completed in `mean_effect_v1/`; no arm was selected/promoted.

| Arm | H1 centroid MSE | H1 MMD squared | MSE reduction vs controls |
| --- | ---: | ---: | ---: |
| Unchanged controls | 0.011344863 | 0.010863827 | 0% |
| Same-target public mean transfer | 0.030817760 | 0.029010769 | -171.65% |
| GO + control-context ridge | 0.011326819 | 0.010847126 | 0.1590% |
| Shuffled GO + control-context ridge | 0.011328122 | 0.010848366 | 0.1476% |
| Constant GO + control-context ridge | 0.011333857 | 0.010853707 | 0.0970% |
| Control-context-only ridge | 0.011333867 | 0.010853715 | 0.0969% |

The tiny GO-versus-shuffled difference (1.30e-6 MSE) is not convincing semantic
conditioning evidence. True GO wins against controls in 14/27 groups; shuffled
GO in 15/27. Each inner source holdout has only four admitted groups and three
overlapping targets (HIRA, NISCH, RNF20); true GO slightly loses to shuffled in
both. Unshrunk same-target transfer worsens error in both source holdouts and
all 27 H1 groups. This supports conservative control anchoring, not a claim of
reliable biological transfer or improved challenge scores.

All mean-shift models preserve control variance by construction. Their variance
ratio is not evidence of learned distributional quality. Ridge output still
contains about 8.6% negative continuous log-expression entries; their magnitude
was not included in this baseline summary. These are not valid count emitters.
No clipping or count reconstruction was silently applied.

### Independent output-promoter features completed

Expression-free reference features are ready at
`dataset/reference/promoter_grch38_gencode48_v1_validated_recovery/`.
They use pinned [GENCODE 48 annotation](https://www.gencodegenes.org/human/release_48.html)
and explicit GRCh38 sequence coordinates through the
[Ensembl batched region endpoint](https://rest.ensembl.org/documentation/info/sequence_region_post).
Responses were checked for exact query, assembly, orientation and length;
GRCh38.p14 chromosome metadata supplied bounds.

There are **501/512 genes present**: 500 unique MANE TSS choices and one consensus
TSS. Eleven missing exact symbols remain masked, without alias guesses. Windows
are strand-correct [-1000, +100) around the reference TSS, which is not claimed
to be the recipient cell's active promoter. Four independent descriptors are
GC fraction, CpG dinucleotide fraction, CpG observed/expected and ambiguous-base
fraction, with validity/missingness masks. These are **not PromoterAI embeddings**,
TF-binding predictions or learned regulatory effects.

Initial parsing exposed valid unquoted numeric and repeated metadata attributes.
Narrow corrections were tested and the full authenticated annotation was
successfully audited before DNA fetching. Failed/unstarted campaign records are
retained. The 33,712,235-byte annotation was reused, never redownloaded. The final
campaign made 12 API requests totaling 618,693 response bytes. Across annotation
plus API bodies: **34,330,928 bytes**, plus a small checksum manifest. No GCP,
paid embedding service, gated model asset or expression fitting was used.

| Artifact | SHA-256 |
| --- | --- |
| Final reference plan | `720aef191b03d2d75e3e1b85c907d8f98928259c90b277cc7d8dfbe256d14e80` |
| `features.npz` | `224d8deb2896b05d7da775d23215750bf092b50c61f25be62e2dd254dbbcb760` |
| `complete.json` | `f87f9d6f9d45ddc057681c7b45bb4283e851416d0cbf1fadc055e4b703cec452` |

No model yet consumes these promoter descriptors. Test true descriptors versus
shuffled-output-gene descriptors versus mask-only in a separately registered
public-source ablation. Coverage alone does not establish predictive value.

### Tracking, verification and next gate

A separate expression-free GO audit confirmed a numerical defect in the legacy
constant-feature transform inside the small source-holdout folds. With ten
training targets, 182/275 constant GO columns acquired nonzero standardized
roundoff; with nine targets, 138/275 did. Identical inputs were centered via
floating-point summation and divided by tiny positive standard deviations.
Those columns are duplicate intercepts, not target-identity leakage, but they
change effective ridge shrinkage. Consequently fine inner-fold comparisons of
constant versus context-only arms are not valid ablations. The full 16-target
H1 fit had zero affected columns; the true-versus-shuffled conclusion is
unchanged. Every v1 file and checkpoint is preserved. The separate opt-in
`scripts/public_flow_stable_conditioning.py` now sets exactly constant training
columns to their first observed value and unit scale. It preserves fold-local
admission/missingness, rejects PCA explicitly, and identifies its v2 method in
provenance. Its 36 synthetic tests cover fractional and extreme constants,
variable columns, held-out rows and multiple missingness patterns. No v1 model
has been rerun or retrospectively replaced; future experiment registration must
explicitly choose this implementation.

Frozen replay W&B run `5fb05aa84f58`: stage validation, two recorded events,
zero tracking errors. Six separately named residual/reference runs each have
three events and zero tracking errors. One-step residual fits are labeled
pretrain, not resumed Feng post-training. All runs are offline with durable
local journals and SDK buffers; no cloud sync is claimed.

Keep the recommended separation of target function, recipient control state,
and optional output-gene promoter properties. Before any larger flow run:

1. Improve public-source target overlap and matched-control replicate precision;
   freeze source/donor holdouts and preserve the 772 exclusions.
2. Add an explicitly trained null/control anchor and a nonnegative output
   representation; validate its unchanged-control identity and mean effects.
3. Calibrate effect strength only inside public training-source folds. H1 has
   already been inspected repeatedly and cannot select this calibration.
4. Test independently acquired promoter descriptors with a registered feature
   ablation; do not substitute the PromoterAI leaderboard name for evidence.
5. Connect Feng only after a defensible parent, compatible measured-gene/count
   contract and authenticated resume path exist. No current checkpoint qualifies
   for automatic full-scale post-training or challenge submission.

Final combined synthetic regression: **689 passed in 12.99 seconds**, covering
flow/conditioning, row admission, GO construction, Feng parsing, replay,
baselines, promoter references and W&B. The original pilot trainer, conditioning
module, baseline v1 and three V7 exclusion/admission seals retain their recorded
hashes. Dataset and model artifacts remain Git-ignored; unrelated user edits
were preserved. No Git push was performed in this continuation.
