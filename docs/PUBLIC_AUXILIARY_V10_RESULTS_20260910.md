# Public GO/context flow pretraining v10

Actual six-arm GPU pretraining completed on the BioHPC compute node on
2026-09-10 at18:53:12UTC. This is a trained flow model, unlike the earlier
two-head kernel fits. It is not Feng post-training or a VCC score improvement.

## Data and execution

The separately registered cache contains62,055 unique cells:57,207 treated
cells plus disjoint fit-reference/context controls, from public K562 and Jurkat.
All56 earlier panel targets and772 protected targets remain excluded.
Before v10 materialization/training, the original axes were projected by exact
symbol onto the official VCC gene list:7,097 measured normalization genes and
482 modeled genes. No aliases, zero filling, or new response rows were added.
This changes the loss scale from v8.1; cross-version absolute losses are not
an apples-to-apples model comparison.

Each source has a separate fit-only GO vocabulary and three arms: true GO,
shuffled whole feature rows, and fit-mean constant features. GO dimensions are
1,788(K562) and609(Jurkat), followed by GO-availability and GenePT-absent flags.
GenePT was not available: a bounded official archive attempt returned HTTP504
with zero payload bytes. No GenePT or PromoterAI embeddings were used.

Source target roles:K562299fit/75held; Jurkat90fit/23held. Context consists of
the float64 mean and population standard deviation of32 disjoint controls,
then float32 conversion. Every model used600 actual AdamW steps, batch128,
two128-wide blocks, a nonnegative zero-atom flow decoder and16-step Heun rollout.
Final checkpoints were fixed in advance, not selected by held-out outcomes.

Device: verified RTX2080Ti GPU2, not a head/login node. Training took119.879s,
peak process RSS1,489,354,752bytes, peak PyTorch allocation39,408,128bytes.
The latter is allocator usage, not total GPU driver memory. Saved outputs
were43,607,898bytes. There was no H100 allocation or cloud transfer in this run.

## Results: source-local public target holdouts

| Metric | K562,75targets/756groups | Jurkat,23targets/212groups |
|---|---:|---:|
| Control centroid MSE |0.03528138|0.03828359|
| True-GO centroid MSE |0.03698655|0.04174875|
| True-GO MSE change vs control |4.8331% worse|9.0513% worse|
| True-GO MMD change vs control |5.2360% worse|7.8917% worse|
| True-GO MSE improvement vs shuffled |1.3369%|2.0558%|
| True-GO MSE improvement vs constant |1.1065%|0.7811%|
| Variance / control variance |1.0172|1.0740|
| Negative prediction fraction |0|0|

True GO provides modest target discrimination, but the overall learned shift
is harmful relative to unchanged controls. Jurkat true-GO MMD is also0.7294%
worse than the constant-feature arm. Variance does not show strong collapse;
nonnegativity follows from the decoder and is not evidence of biological
accuracy. Shuffles include availability flags, so this is not a pure semantic
GO test. Repeated control donors are dependent observations. These are public
development holdouts, not independent cross-source or challenge evaluations.

## Artifact and logging audit

Six model+AdamW checkpoints were saved as finite numeric NPZ plus JSON metadata,
without pickle. Every checkpoint was restored and its evaluation replayed.
An independent CPU audit restored all six and reproduced every group and
aggregate metric within rtol1e-5/atol1e-7 (maximum aggregate difference1.956e-7).
The GPU runner had already required exact same-device replay.

W&B recorded every optimizer step and each final evaluation:3,606 events,
zero logging errors, six genuine offline SDK runs. Their scalar journals were
independently reconstructed from saved histories and evaluations. These logs
are local; they have **not** been synced to a W&B cloud dashboard.

| Binding | SHA256 |
|---|---|
| Protocol |`c2809ce815810912859db57fb3c7d4d9c37e65db01130c8a2c2faeef9346acd9`|
| Materialization contract |`8aa59a6228aff58065aa290e6b639057759c797be375845582a79cd8f8d12218`|
| Cache completion |`721692af02a822e56e6af8b2a0c311dbe28507d831ba56a7380e1ea6a811c80a`|
| Training contract |`590a5916c6a9c76e263f2c8e4e45ed9ba41def7682a6694ce82d87b1b7074f2b`|
| Training completion |`78e0e7d8e0fd443f6ba6ca3b4d3d02aeef42e166b50908aa28a3e1100a2b9a67`|

Artifacts: `artifacts/public_flow/auxiliary_v10_20260910/`.
Combined cache/flow/runner/conditioning/W&B regression:479passed before launch.
New native-count adapter:71 tests,153 related tests passed separately.

## Next: exploratory counts, then measured post-training

The user explicitly authorizes exploratory VCC attempts; the old1% scientific
promotion gate is not a mandatory submission gate. Technical checks remain
mandatory. A separately registered candidate will use fixed equal weights on
the two true-GO models, released non-targeting controls only, and unchanged
fit-only GO vocabularies. Count conversion uses the original7,097-gene control
depth and unbiased rounding for the482 modeled outputs; all18,051 unmodeled
native gene counts remain exactly anchored. Total libraries need not be
conserved. No clipping, thinning or favorable-seed retries are allowed.

At this report's initial writing, the inference generator/count writer are
under review: no full candidate, official CLI validation, package, or VCC
submission has completed. A read-only authenticated server check returned
`limit_reached:false`; that response does not independently establish a6/day
policy or reserve an attempt. Keep server limit checks enabled at upload.

Feng post-training must restore a real model/AdamW parent and first authenticate
barcode/guide/target/donor joins, protected exclusions and donor-disjoint roles.
Never align Feng rows positionally. Promoter composition descriptors already
on disk are not PromoterAI embeddings; actual PromoterAI acquisition/use has
separate model/license requirements. Both remain explicit research branches,
not features claimed by this candidate.

## Completed generation and submitted exploratory test

The following continuation supersedes the earlier pending-generation snapshot.
Real inference completed at19:23:37UTC. Its contract SHA is
`2d8c218e99a42b65481424c88d431ed093c60575c171f2b9e539dc1a12f9cd81`;
generation completion SHA is
`94cf6080ce71ff22ff7910fa11918e20cdc07f8b3812878feb4c63105659cfd4`.

The genuine candidate has360,000 cells,18,533 genes,900 exact400-cell groups,
and2,089,876,228 nonzeros. Every value passed streaming integer/nonnegative
checks; native cell libraries range833–51,359. No clipping/thinning was used.
The16,746,907,728-byte H5AD SHA is
`5b534d8290341ef25560761a0bf4396cf86b22ed711852ea0989a8ced305d0ce`.
Generation took857.66s and peaked at3.61GB process RAM. Intermediate files
are preserved; resource accounting counts hard-linked files more than once.

An independent2,000-cell audit across five actual groups in A/B/C verified
exact preservation of every18,051 unmodeled count. There were300,098 changed
modeled entries, with no negative/fractional values. Individual library drift
ranged−3.508% to+2.730%. Audit receipt SHA:
`3c70983690acc95456b2ea5c464c22353da48f6a6057381711b79e60f3fbf47f`.
These are anchoring/safety results, not predictive accuracy.

Official VCC CLI0.2.0 validated and packaged the real file, preserving counts,
all genes and quotas. Peak packaging RAM36.06GB; duration250.45s.
Package completion SHA:
`d459fee2e7b52fb83f2e707d268112125edc74f8b672358574a47f17deab6052`.
The3,317,166,080-byte package SHA is
`e44fad6228758e16048c84dbba7800146c9fa05d4b7e0358ed2a8270842081f7`.

**Submitted model:** `GO_context_flow_v10_exploratory`.
**VCC entry:** `0ivHnDQV8wi5WQH14o2b`.
The full upload succeeded with MD5 verification and scoring was launched.
At19:48:37UTC the server status became `published`, and the bounded monitor
finished successfully. Tokens were transient, hidden, and not stored. Official daily
and server in-flight checks stayed enabled; no duplicate entry was created.

New tests:124 combined writer/generator/packager tests and12 mocked uploader
tests passed. An additional installed-CLI full-shape synthetic smoke also
passed, but its files are explicitly `SYNTHETIC_NOT_FOR_SUBMISSION` and were
never sent to the production uploader. The actual package above is separate.

### Published VCC result

Entry `0ivHnDQV8wi5WQH14o2b`, model `GO_context_flow_v10_exploratory`:

| Server scaled metric | Value |
| --- | ---: |
| Overall | -0.06402252994606115 |
| PDS | 0.0014949020105450145 |
| MSE | 0 |
| Jaccard | -0.06447729870746306 |
| NMAE | -0.02754696883178635 |
| FID | -0.28615245372524145 |
| REACH | -0.007453360422421081 |

Rank was589 at retrieval, not a permanent rank. Panel:`vcc2026-val-1`,
partition:`val`, anchor:`vcc2026-valA-r4+vcc2026-valB-r4+vcc2026-valC-r4`.
The durable sanitized server receipt is `upload/status_00028.json`, also tracked
as [submission.json](../results/go_context_flow_v10/submission.json), SHA256
`9f689d73f752920a121d5d6c6ac22257d15fa4ba1c87c896752ae6f312f45af8`.

The real upload/evaluation path succeeded; predictive improvement did not.
FID is the weakest scaled component. In VCC this means **DE direction fidelity**
(`de_wilcoxon_direction_fidelity_yield_raw`), not Fréchet Inception Distance.
It warrants diagnosis of predicted DE directions and significance/yield; the
aggregate score alone does not identify a cause. Public MMD and variance checks
are auxiliary diagnostics, not the official FID. See the
[primary-source metric audit](research/NEXT_ROUTE_PRIMARY_SOURCE_AUDIT.md).
MSE0 is a scaled competition score,
not zero raw prediction error. The older screenshot's -0.036 result has no
verified matching anchor here, so it is not used as a controlled improvement
comparison. No checkpoint, blend, seed or preprocessing change was chosen from
this result, and the same file was not reuploaded under another entry.

Continue the separately versioned public-only GenePT/GO conditioning experiment:
availability-only, correct GenePT and shuffled GenePT, crossed with original
and source-fit-only calibrated branch scaling. Preserve all98 held targets;
source-transfer evaluation is development evidence, not an untouched final test.
Promoter/Feng components remain separate pending their admission requirements.

## Next conditioning and data work

The official GenePT API retry succeeded:574,395,233bytes, published MD5
`3f6ce4317e3a0091978ae5cb8fbf05a3`, SHA
`6193575dcbd7bbf214c8ca3eb518cb3d13272443f98ecd6c26402411a7ca745e`.
Location:`dataset/reference/genept/10833191_api_retry_20260910/`.
No GCP billing project, paid API or credentials were used. Previous failed
attempts were preserved. This acquisition did not change the submitted model.

A bounded, non-executing symbolic audit of the entire Ada-text pickle found
93,800 name keys referencing33,230 stored vector objects, all1,536-dimensional.
This is not93,800 distinct biological genes or necessarily33,230 distinct
numeric vectors. All300 official target symbols are covered. Public fit
coverage is296/299 K562 and88/90 Jurkat; all75/23 held symbols are covered.
Missing fit names:ALG1L,C1orf109,POLR2B,C7orf26,CCDC144NL. Only whitelisted
data opcodes and bounded references were interpreted; no pickle callable was
executed and no numeric embedding vectors were retained/admitted to training.
Safe numeric conversion and source-bound conditioning subsequently completed;
see[the GenePT feature report](GENEPT_NUMERIC_V11_RESULTS_20260910.md). That
continuation found four exact duplicate-vector pairs and quarantined all eight
names for the first test, leaving297/300official vectors usable plus GO. No new
predictive-model training has yet used those features.

The current combined GO representations distinguish284 of300 official targets;
seven exact-collision groups contain23 targets. GO is missing forHMGXB3 and
TMEM104. These limitations motivate complementary embeddings, not a change to
the already-frozen submission.

A read-only diagnostic on128 public fit examples found target/context projection
RMS ratios0.03250(K562) and0.04874(Jurkat) at control-start states. Centered
target RMS was0.02316/0.03204, versus centered context RMS0.06747/0.09789.
This is not a causal importance estimate; branch balancing is a hypothesis
for the next registered public-only ablation, not an established fix.

The [Feng v11 draft](FENG_POSTTRAINING_V11_DRAFT_20260910.md) proposes a573-cell
donor-separated pilot and reports exact stable-ID metadata coverage for all
482/7,097 genes. A raw-CSV gene-axis audit is still needed before count admission.
Its explicit8+8 control design and genuinely resumed AdamW updates must be
registered separately. No Feng or PromoterAI post-training has run.
