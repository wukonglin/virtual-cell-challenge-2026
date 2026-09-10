# Sparse control-anchored public decoder v4: frozen protocol

2026-09-10. Freeze this protocol, implementation and input hashes before
decoding the existing cache for the new experiment. This is exposed-public
development, not an untouched test. Preserve all v7/v1/v2/reliability/v3 seals.

## Question, cohort and comparisons

Does replacing additive positive-part output with an explicitly sparse,
two-part control-anchored decoder preserve zeros and improve response prediction?
Reuse exactly v3's authenticated 56-target K562/Jurkat cohort, 398 groups,
512 genes, fixed four 42-fit/14-held target folds, both source directions,
four train-vocabulary GO tables, and 772 protected exclusions. No new raw
expression, cloud download, H1/RPE1/HepG2/challenge/Feng expression or GenePT
acquisition is admitted. The matched context control pool is already in the
cache and may now also supply positive-expression donors at inference.

Six fixed arms: unchanged control, context-only, true GO, and v3's three
role-contained packet shuffles (20260921, 20260922, 20260923). Keep the v3
conditioning design, stable train-only transforms, independent control-context
pool, source/target/batch weighting, penalized intercept and lambda10 unchanged.
No hyperparameter selection, new interaction block or source of annotations.
Target-by-context interactions are a separate next experiment, not confounded
with this output comparison. The two response heads change their units, so the
same lambda does not imply equal effective shrinkage across v3 and v4.

## Training-only two-part response

For each training group and gene, let q be the empirical fraction of values
strictly greater than zero, and xbar the mean of all log-normalized values.
Both reference and treated occupancies use the SAME fixed smoothing:

`s(q) = (32*q + 0.5) / 33`.

The occupancy target is `logit(s(q_treated)) - logit(s(q_reference))`. It is
not sample-specific Jeffreys inference. Different group sample sizes must not
produce a response for all-zero/all-positive or identical empirical occupancy
populations. Nonlinear finite-sample bias is not claimed to be eliminated.

For each fold, derive a positive-expression prior per gene using only unique
positive values in its fitting groups' reference and treated cells. Deduplicate
original source/row identities; repeated reference copies are not extra cells.
No held-source values or held-target treated values enter the prior. Record
positive support counts and unsupported genes. Use the fixed fraction
alpha=1/32 to define `m = (xbar + alpha*prior)/(q + alpha)` for both populations.
The amplitude target is `log(m_treated) - log(m_reference)`. A gene with no
training positive support receives zero amplitude response; do not invent a
positive prior or drop the gene. Both heads are fitted in one concatenated
ridge response with the same fixed group weights, not per-gene fitted weights.

## Sparse inference and donor provenance

Use each group's independent reference cells as the anchor. Predicted occupancy
shift is clipped to [-4,4]; log-amplitude shift to [-log4,log4]. These fixed
bounds and every clipping event are recorded. No outcome-based gain tuning.

Compute continuous requested occupancy:
`q_req = clip(q_anchor + sigmoid(logit(s(q_anchor)) + delta_occ) - s(q_anchor), 0, 1)`.
For exactly zero occupancy effect, use q_anchor directly to avoid roundoff.
Requested positive count is `floor(n_anchor*q_req + 0.5)`. This is a population
occupancy choice, not rounding expression to UMI counts. With n=32, shifts
smaller than about 1/64 commonly leave the support count unchanged; report it.

Preserve the existing support if its count is unchanged. To reduce support,
retain a fixed ranked subset of existing positives; to increase it, activate
a fixed ranked subset of existing zeros. Never reshuffle all support just
because a prediction was requested. Rankings are fixed by seed20260924,
group/gene/original-row identity and purpose, never by arm or treated outcomes.

Retained positive values are multiplied by exp(clipped delta_amp). New positive
values borrow deterministic donors in this order: original positive anchors,
positive matched-context controls, then the fitting-only positive prior. All
donor identities/types are saved; no held-treated donor is permitted. If no
positive source exists, keep zero and report an unresolved activation. Donor
ordering is identical across arms; it does not learn joint cell-state dependence.
For exact zero heads or explicit NTC, preserve exact anchor values and support
(after lossless float64 conversion), before smoothing, rank changes or effect
arithmetic. This is not preservation of the original dtype/byte layout.
Missing GO is not an NTC instruction.

These remain continuous log-normalized expressions, not raw counts, a full-gene
emitter, flow-pretraining weights or a VCC-ready submission.

## Metrics and fixed progression screen

Primary target-pooled centroid MSE and secondary batch-macro MSE/MMD retain
the exact v3 definitions. Add target-pooled occupancy MSE: equally average
predicted and observed per-gene nonzero fractions across matched batches within
each target before squaring, then macro-average targets and sources. Record
control occupancy MSE, zero fractions, continuous versus realized occupancy,
unchanged-support counts, activations/deactivations, clipping, and unresolved
activations. Never pool different sources' error vectors together.

In BOTH held sources, the true-GO candidate must satisfy:

- Conditioning: primary MSE improves at least1% relative to controls and each
  predeclared shuffle, and is strictly better than context-only.
- Distribution: batch-macro MMD is no worse than every comparator; occupancy
  MSE is no worse than control and context-only. Absolute aggregate zero-rate
  error versus treated is no worse than control. No post-score tolerance change.
- Safety: finite nonnegative outputs, exact null/zero identities, full saved
  artifact replay, zero unresolved-activation fraction, and occupancy/amplitude
  clipping fractions no greater than1% each.

Every condition is predeclared and reported even if another already fails.
The gated occupancy clipping fraction refers to head clipping at +/-4;
clipping requested probabilities into [0,1] is a separate local diagnostic,
not an additional unregistered gate or a silently combined clipping rate.
The screen is an engineering gate, not a statistical significance test. A pass
can make a public promoter ablation eligible, but is not automatic evidence of
a compatible Feng post-training parent or a valid competition package.

## Execution, monitoring and submission boundaries

Run 48 fold/arm evaluations (40 two-head ridge fits and eight references) on
an approved BioHPC compute host, two numerical threads, GPUs disabled. Save
safe NPZ coefficients, transforms, priors/support, heads, donor provenance,
predicted values and original cache mappings. Replay effects, donor generation,
metrics and roles after saving; require fresh artifact directories only.
Record54 offline W&B validation events (eight folds plus one aggregate per arm),
zero tracking errors, and do not claim cloud synchronization.

Readiness monitoring is read-only and fail-closed. It authenticates suite,
completion, contract/protocol, per-arm summaries and tracking receipts; optimizer
completion alone never passes a scientific gate. A bounded local watcher may
record changes for up to24hours without creating jobs, downloading or uploading.
It provides local files/stdout, not automatic chat notifications. A new candidate
requires new authenticated pins and a restarted watcher; immutable failed
results do not become successful with time.

Promoter ablations require the conditioning/distribution/safety gates. Feng
post-training additionally requires an authenticated compatible trainable
parent and donor/batch eligibility. This ridge diagnostic is not such a flow
parent by default. Submission additionally requires full official raw-count
output, exact genes/contexts/targets/cells, eligibility/provenance, format and
scientific QC, an official dry-run receipt, current credential/quota checks and
a one-candidate submission record. The user's conditional submission request
does not convert these 512-gene log-normalized outputs into valid VCC inputs.
