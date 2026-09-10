# Feng v11 public-only post-training: unregistered admission draft

2026-09-10. **Draft only: no new Feng expression admission, count decoding,
training, download, upload or automatic promotion is authorized by this file.**
This audit read existing receipts, authenticated targeted assignment metadata,
GO metadata, checkpoint descriptors, and HDF5 `var` gene-identity metadata only.
No Feng `X`, normalized expression, RNA-CSV data row or guide-UMI value was read.

## Outcome

There is now a genuine resumable v10 model plus AdamW state. The next gate is
Feng data/axis admission, not another randomly initialized “post-training” run.
The targeted screen is the useful first source. Two issues must be resolved:

1. Exact line/batch strata have at most **23 explicit controls**. The v10
   fit16/context32 rule cannot yield even one Feng group. A separately declared
   smaller-pool pilot or justified technical-inlet pooling is necessary.
2. Raw CSV gene-row identities are not yet audited. Curated Feng symbol names
   miss 6 modeled and 134 normalization names, but **all 482 modeled and all
   7,097 normalization genes have one-to-one exact Ensembl-ID matches in its
   `var` metadata**. This is promising metadata evidence, not raw-CSV authority.
   Do not zero-fill missing names or silently change the normalization axis.

## Existing acquisition and exact joins

The already-local publisher RNA counts are Figshare article **27989294 v2**, MIT.
The acquisition receipt verifies 7,675,835,277 bytes, publisher MD5 and local
SHA-256; no new transfer is needed. This audit did not rehash the three large
RNA archives. Their frozen acquisition identities are:

| Screen | Compressed bytes | SHA-256 |
| --- | ---: | --- |
| Fitness | 1,445,924,890 | `bae144af382f81132720d10300daffc6e105f122eb5b5bdb65d5d796b64c1a19` |
| Nonfitness | 1,748,940,996 | `20532e0527cb35ccc8052ec8801fd05c765b0d057adc0e36ae0d5dd313b48070` |
| Targeted | 4,480,969,391 | `cbda45f0645f1ce06488ba733fb95da04145a6ff2bd8d6d197633d2534386471` |

Files are under `dataset/raw/feng_rna_counts/`. Companion assignment metadata
is under `metadata/`; its columns are exactly `Cell_ID`, `Batch`, `Guide_Call`,
`Cell_Line`. Targeted metadata was freshly SHA-256 checked against
`d6057946570edbbcd3406f4e55d8f338f602ef1b7cc71cffca101133e35b098d`
before this metadata audit. It has 1,161,864 unique cell IDs, 19 cell lines,
10 donor prefixes, 142 batch labels, 8,241 explicit NTCs and 526,842 unassigned
cells. Fitness/nonfitness have only 36/12 explicit NTCs in their complete
metadata; do not relabel unassigned cells as controls.

RNA columns and metadata rows are reordered; positional joins are forbidden.
The frozen screen-specific mappings in [feng_authorized_csv.py](../scripts/feng_authorized_csv.py) are:

- Fitness/nonfitness: metadata ID = literal `MP-` + original RNA column ID.
- Targeted: `P<pool>_I<inlet>_<barcode>` maps exactly to
  `PC-P<pool>-D3_I<inlet>_<barcode>` for this all-day-3 source only.
- Targeted RNA column `P2_I48_CAACCTCAGCCCAGCT-1` has no annotation; explicitly
  exclude it. Targeted ordered RNA-axis SHA is
  `aaef994eeceb132a72cb9ef876205ce11b02bb4476f4d7e37deedfbbf8c64460`.
- NTC requires the full match `NonTarget_[ACGT]{20}`. A treated target candidate
  removes only the final underscore plus 20-base sequence. Preserve the full
  original guide string; reject ambiguous/unassigned calls, and separately
  validate target identities and exclusion equivalences before admission.
- Donor is the `Cell_Line` prefix before its first underscore, as documented in
  the existing author-code audit. Preserve the full line and exact batch too.
  Two lines from one donor are not independent held donors.

The raw source and the normalized `dataset/raw/feng2026/feng24_preprocessed.h5ad`
compilation overlap. Do not combine them as independent training examples or
use normalized values as raw RNA counts. The compilation was not freshly
full-file authenticated here and is only a diagnostic gene-identity reference.

## Small exact-batch candidate, without merging inlets

Counters use exact `(target, Cell_Line, Batch)` treated groups, exact
`(Cell_Line, Batch)` controls, at least 8 treated cells, and the v10 global
828 exclusions (772 protected + 56 panel targets):

| Minimum explicit controls | Groups | Targets | Treated cells, cap 32/group |
| --- | ---: | ---: | ---: |
| 8 | 183 | 84 | 1,553 |
| 16 | 42 | 36 | 357 |
| 24, 32, 48 or 64 | 0 | 0 | 0 |

Also retain the union of **98 v10 held-target identities** as response-excluded
through post-training. Only `MRPL39` intersects the 36-target candidate; removing
it yields **41 groups, 35 targets, 349 treated cells and 14 control pools**.
The resulting response denylist has 926 distinct identities before any further
versioned stable-ID equivalence audit.

A small coherent **proposed**, not registered, split is:

| Role | Entire donors | Groups | Target identities | Treated | Control pools |
| --- | --- | ---: | ---: | ---: | ---: |
| Fit | eipl, iudw, jejf, kolf | 28 | 26 | 240 | 9 |
| Tune | oikd | 7 | 7 | 56 | 3 |
| Held development test | zapk | 6 | 6 | 53 | 2 |

The split is chosen from annotation coverage only. These are public development
donors, not an untouched challenge test. Tune/test target sets partly overlap
fit identities and partly do not: tune shares `GATA6`; test shares `INS`,
`INTS6`, `UQCR11`. Report those two evaluation cases separately, rather than
calling every held response unseen-target evidence. Many targets have only one
eligible donor, so target/donor effects are partly confounded.

For this small pilot only, a NEW protocol could register disjoint **context8
and control-base8** per exact line/batch, hash-selected from the explicit NTCs.
Fit uses control-base8 as source population; held evaluation also uses its own
control-base8. Context8 is conditioning only, never a response target. Do not
repeat 8 cells and call them 32 independent controls. The population mean/std
still has 964 coordinates, but its smaller sample size creates a stated
conditioning-distribution change relative to v10's context32.

There would be **573 unique selected cells** (349 treated + 224 controls) and
4,066,581 count elements over 7,097 genes, below the existing authorized CSV
consumer's 8,000,000-element cap. This bounds the output matrix, not CSV scan
time: a single gzip pass still traverses the full archive's row text. Resource
limits and a recoverable single-pass materialization plan must be registered.

Do not silently pool technical inlets. As a metadata-only counterfactual,
collapsing `_InletN` within the same line/screen/pool/day would create 17,058
groups across 330 targets and 56 qualifying control pools at NTC>=48, with
391,234 treated cells capped at 32/group, before the additional 98 exclusions.
This much larger option needs independent experimental/batch justification and
a new protocol; the counts alone do not authorize that merge.

## Gene axis and fixed GO conditioning

The curated compilation has 36,518 unique symbol entries. Exact current-symbol
intersection is 476/482 outputs and 6,963/7,097 normalization genes. All missing
outputs resolve by exact stable ID in K562/Feng `var` metadata:

| Frozen output symbol | Exact Ensembl ID | Curated Feng symbol |
| --- | --- | --- |
| C1orf109 | ENSG00000116922 | AIRIM |
| C4orf48 | ENSG00000243449 | NICOL1 |
| FOPNL | ENSG00000133393 | CEP20 |
| HIST1H1D | ENSG00000124575 | H1-3 |
| HSPB11 | ENSG00000081870 | IFT25 |
| RARS | ENSG00000113643 | RARS1 |

An exact-ID diagnostic finds one unique counterpart for all 482/7,097 genes.
Before counts, authenticate a raw-CSV gene-axis-only scan and its gene identity
crosswalk; require the same one-to-one coverage and preserve frozen model order.
Do not silently strip versions, choose one duplicate, infer aliases, or change
normalization to a convenient smaller intersection. If exact coverage cannot be
proved, this direct-resume design is blocked and needs a separately registered
representation-adaptation experiment.

Reuse each parent's fixed GO vocabulary: **1,788 terms + 2 flags** for K562,
**609 terms + 2 flags** for Jurkat. Project Feng annotations into those exact
ordered terms, binary L2-normalize, and append GO-present/GenePT-absent flags.
Do not refit/extend vocabulary or change target projection width during resume.
The authenticated GO audit covers 35/36 preliminary targets in both vocabularies;
`FAM103A1` has no accepted in-vocabulary terms. After withholding MRPL39, coverage
is 34/35. Preserve an explicit missing-modality flag or preregister exclusion;
do not fabricate GenePT features. Target-identity/exclusion validation is still
required even for a target with missing GO annotations.

## Genuine parent restoration and the next gate

Completed v10 pretraining receipt SHA:
`78e0e7d8e0fd443f6ba6ca3b4d3d02aeef42e166b50908aa28a3e1100a2b9a67`.
Training contract SHA:
`590a5916c6a9c76e263f2c8e4e45ed9ba41def7682a6694ce82d87b1b7074f2b`.
Parents are separate `replogle_k562__true` and `nadig_jurkat__true` directories
under `artifacts/public_flow/auxiliary_v10_20260910/pretraining/`:

| Parent | Checkpoint JSON SHA | Numeric NPZ SHA |
| --- | --- | --- |
| K562 true GO | `1122e3f392995ba47d464721906c4874e73d072469902c23f41656fdac12cafb` | `c56b64bc4da7cd1cc13b76b419932810b9530d7297c5e5693fa34b96f4fa5e04` |
| Jurkat true GO | `833fcc47351455828be130c9ef2642febe6360f9f881759fe16367dfa837dd9a` | `6ea0ddf5ce57ec41e5641bbda1e07a6acf32142a9821ba0cb631e6c1cd5262dc` |

Use the authenticated `restore_checkpoint` pathway in
[run_public_auxiliary_flow_v10.py](../scripts/run_public_auxiliary_flow_v10.py)
to restore the genuine model and every AdamW first/second moment at step 600.
Keep the parents separate; their differing target projection widths prohibit
weight averaging. A later child can share a declared Feng fit cohort, but must
then be described as K562+Feng or Jurkat+Feng, not single-source trained.

Do NOT call frozen `train_source` as post-training: it initializes new weights,
only accepts K562/Jurkat source labels, and enforces fit16/context32. Implement
a new Feng data consumer and resumed loop; never relabel Feng as K562. A
conservative draft budget is 100 extra updates per parent, batch 64 and learning
rate 3e-5 after restoring the original optimizer; record the deliberate LR
change, retained AdamW moments, and global steps 601-700. No early stopping or
checkpoint/strength selection on challenge scores. Keep the existing zero-atom
flow loss/decoder, and compare frozen parent versus resumed child versus exact
matched controls on the same held public donors. Also replay the preserved v10
held-target diagnostics to assess forgetting.

A new checkpoint schema must distinguish parent step 600 from post-training
steps/global step 700; frozen v10 checkpoint validation deliberately requires
its original fixed 600 steps. Reuse safe numeric packing, not an arbitrary
pickle. W&B stage must be `posttrain` with every actual optimizer update logged
and offline journals/replay, preserving parent IDs and hashes.

Before any of that: seal gene-axis audit, target/control/donor/guide joins,
exact cell/gene allowlists, source and metadata hashes, source-specific GO
projection receipts, resource bounds and a new training contract. The existing
`feng_authorized_csv.py` validates selected IDs/counts but explicitly does NOT
establish biological eligibility by itself. The current draft is preparation,
not evidence of higher scores or a launched post-training run.
