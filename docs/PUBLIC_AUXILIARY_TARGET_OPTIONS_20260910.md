# Future public auxiliary training: metadata-only audit

Update: the proposed metadata candidate plan has now been separately
[registered and replayed as v9](PUBLIC_AUXILIARY_V9_RESULTS_20260910.md).
Expression materialization and training remain unregistered. The original
audit below is historical and its counts were independently reproduced.

2026-09-10. This is a capacity/design audit, not a new row-admission contract,
materialized dataset, trained model or score claim. It does not alter the running
v8.1 experiment. No new expression values, downloads or model fitting were used.

## Available replication beyond the fixed panel

The same eligibility rules were applied independently to each public source:
at least eight treated cells per target/batch, 72 exact non-targeting controls
per batch, at least two qualified batches per target, at most 32 treated cells
per group, no four-batch truncation, and all 772 protected exclusions.

| Quantity | K562 | Jurkat | Combined |
|---|---:|---:|---:|
| Eligible targets | 430 | 169 | 543 unique |
| Targets outside the common 56 | 374 | 113 | 487 unique |
| Eligible groups, all targets | 4,642 | 2,200 | 6,842 |
| Treated-cell capacity, all targets | 49,262 | 30,158 | 79,420 |
| Additional groups outside the 56 | 3,689 | 1,386 | 5,075 |
| Additional treated-cell capacity | 38,928 | 18,279 | 57,207 |

These are metadata-derived capacities, not newly admitted/decoded cells.
The existing 48 K562 and 53 Jurkat control pools suffice; this expansion would
not add new source/batch identities or independent cell contexts.

Observed non-control target counts are 2,057 / 2,393. After protected exclusions,
1,426 / 1,644 remain; 607 / 295 have at least one qualifying batch, and only
430 / 169 have the required two. Raw target overlap is 2,055, but qualified
overlap before exclusions is 76; exclusions remove 20, leaving the current 56.
Thus replication eligibility and the protected-target intersection limit the
panel, not the 64-target ceiling or GO availability.

Most additional targets occur in the other source's raw metadata but fail its
eligibility rules: 373 of the 374 extra K562 targets occur in Jurkat, and 86 of
the 113 extra Jurkat targets occur in K562. They are source-eligibility-specific,
not evidence of biologically exclusive gene expression.

## GO availability is not a fitted vocabulary guarantee

Accepted direct GO annotations cover 53/56 panel targets, 409/430 eligible K562
targets, 159/169 eligible Jurkat targets, and 515/543 unique eligible targets.
Additional-target coverage is 356/374 and 106/113. These counts use the same
exact-symbol, no IEA/ND/NOT, no ancestor expansion and no alias-guessing policy.

This is annotation-source coverage, not guaranteed coverage in a frozen
fold-specific vocabulary. Existing feature tables emit only the old 56/42 target
sets, so auxiliary training requires new authenticated feature artifacts and
receipts. No new GenePT acquisition or compatible feature receipt was verified.

## A clean future experiment

Use two separate source-specific auxiliary warm starts, if later registered:

- K562's 374 extra eligible targets only for K562-fit to Jurkat-held evaluation.
- Jurkat's 113 extra eligible targets only for Jurkat-fit to K562-held evaluation.

Exclude all 56 validation-panel genes and all 772 protected targets from auxiliary
training globally. Do not combine K562 and Jurkat treated labels into one shared
pretrained checkpoint and then claim that the opposite source is unseen.

Keep the original source/batch-global 16 fitting-reference / 16 tuning-anchor
split during auxiliary response and prior fitting. Training on all 32 anchors
would contaminate later inner tuning. Keep context32 as conditioning donors only
and exclude sham8 from fitting. Clone the allowed warm start for each outer fold;
preserve inner 28/14 selection and outer 42/14 refitting/evaluation on the panel.

Every learned encoder, transform, prior, optimizer/checkpoint, GO vocabulary and
early-stopping choice must respect these roles, not just the final head. Fix the
feature-vocabulary policy, architecture, comparisons and compute budget before
admitting additional expression. Relaxing minimum replication or training on
held-source treated cells would be a different evaluation regime, requiring its
own protocol rather than an unreported exception.

This option was identified without choosing on partial v8.1 results. It is not
authorized by the current cache contract and has not been implemented or launched.

## Audit evidence and limits

The audit loaded the frozen v7 row contract and sources manifest, with SHA-256
`c2fe975be373f3c7eb6be20488a0307ebd4f7f355d87b5672977f814b328f644` and
`df6fc7a7a7da6d4ba2a241c0a73849a7aeb15a0f42c71de6cd69d72f288d3516`.
It used `prepare_public_replicated_v7.read_metadata_sources` and the existing
`eligible_batches` logic. Source sizes and unchanged open-handle signatures
were checked; recomputed metadata SHA-256 matched:

- K562: `8b7b42af6f3a1b41497850ea65c01e81d0524b235c16cfc7695914adb425948f`
- Jurkat: `8d48a963e217f0fd7d42e599f8b2ad3c4e51bb5238aa0cb2842959a56e844fd0`

The audit explicitly blocked HDF5 reads from `/X` and `/X/*` while reading obs/var
metadata and X shape/encoding attributes. It did not freshly hash both whole raw
files; their previous acquisition/materialization checks remain in the v7 record.
Any new materializer must authenticate both complete raw sources before X reads.

GO came only from the already-local authenticated acquisition:
`ce799783141d4156e1db15e6f7ceedbd044762accae7124de3e3841e62e7ae29`.
The audit created no standalone machine-readable admission artifact; this document
records the reviewed tool-output findings and their limits.
