# Public replication-expanded cache v7: completed

2026-09-10. **Data preparation and verification completed successfully.**
Materialization finished at 2026-09-10T15:24:49.624495+00:00 on
cbsuvlaminck3.biohpc.cornell.edu. No model fitting, post-training, new scientific
score or VCC submission occurred.

See the [frozen protocol](PUBLIC_REPLICATED_V7_PROTOCOL_20260910.md).
This implements the data-expansion step proposed after
[v6 nested validation](PUBLIC_NESTED_V6_RESULTS_20260910.md), whose predictive
quality gates remain failed. This cache is not a new model-quality result.

## What changed

Only the maximum of four batches per target/source was removed. Minimum eight
treated cells per batch, minimum two eligible batches per target/source, 72 NTCs
per batch, the 32-treated-cell cap, seed/ranking rules, target exclusions and
gene axes are unchanged. All eligible batches for the same 56 targets were
registered before newly admitted expression rows were decoded.

| Quantity | Old v2 cache | Expanded v7 cache |
|---|---:|---:|
| Targets | 56 | 56 |
| Matched source/target/batch groups | 398 | 1,767 |
| Distinct source/batches | 97 | 101 |
| Retained treated cells | 4,083 | 22,213 |
| Unique physical source/cells, all pools | 11,067 | 29,485 |
| Shared normalization genes | 7,563 | 7,563 |
| Output genes | 512 | 512 |

K562 contributes 953 groups and 10,334 treated cells; Jurkat contributes 814
groups and 11,879 treated cells. There are 18,418 newly admitted physical rows.
The additional groups reuse many controls: 1,767 groups are NOT 1,767 independent
biological replicates. No new cell type was added.

Context/anchor/sham pools remain globally disjoint within source/batch, with
32/32/8 cells respectively. All 772 protected target exclusions remain in force.
RPE1, H1, HepG2, challenge-2026 and Feng expression were not opened.

## Extraction and verification

Both existing raw H5AD files passed full same-open SHA-256/size checks. Complete
source metadata and every selected row role were revalidated before any X value
was read. Extraction normalized 29,485 unique rows in 116 chunks of at most 256:
54 K562 chunks and 62 Jurkat chunks.

Normalization remains float64 log1p(CP10000) over the full ordered 7,563-gene
shared axis, followed by float32 conversion and selection of the same 512 genes.
Only registered rows were decoded; no whole X matrix was loaded and filtered.

All 398 retained parent groups preserve their original cell lists, normalized
values for every pool and context summaries. Expanded flat group IDs are not an
old prefix; an explicit semantic parent-to-expanded mapping is stored. A separate
reviewer compared uint32 views of the real float32 arrays and confirmed literal
bit equality, including zero signs, for the complete old subset.

The completed cache was reloaded through the new verifier. It checked schema,
axes, dtypes, nonnegative finite values, repeated-row consistency, independent
context reconstruction, bounded archive headers, receipt fields and parent
conservation. Neither verification run reread raw X values or selected models.

There are 2,568 passing related synthetic tests across 47 modules: 2,567 in the
combined regression and one subsequently added all-cell mapping test run
separately. The new registration/cache subset has 114 tests. Heterogeneous
synthetic cells allow an independent formula check of every expanded value,
including new rows; a value permutation that preserves structural and old-parent
checks is detected by this test. Real newly admitted rows were extracted by the
tested decoder, not independently reread for a second raw-expression audit.

## Resources

| Measurement | Actual | Registered limit |
|---|---:|---:|
| Compressed NPZ | 57,659,410 bytes (54.99 MiB) | 512 MiB |
| Expanded NPY members including headers | 315,859,340 bytes (301.23 MiB) | 512 MiB |
| Array payload | 315,856,524 bytes | Included above |
| Materialization peak RAM | 1,113,001,984 bytes (1.04 GiB) | 8 GiB |
| Numeric row chunk | At most 256 cells | 256 cells |

Execution used two numerical threads and no GPU, on the BioHPC compute node.
No login/head-node computation, new download, GCP transfer, paid embedding call
or H100 allocation occurred. This was data preparation, so no pretrain/posttrain
W&B events were invented. Future actual model fitting must retain W&B logging.

## Next: separately registered expanded-cohort validation

The data prerequisite is complete. The frozen v6 runner is not compatible with
this cache: it pins 398 groups and writes all candidate/fold arrays into a single
archive per arm. Do not point it at v7 or alter its completed code in place.

Next implementation should bind the new cache, preserve source/target holdouts,
register reference/tuning roles for all 101 pools, and store model/prediction
artifacts separately per fold/candidate to respect archive limits. Fix the
experimental comparisons and resource budget before fitting, and retain
control/context/additive/true/shuffled comparators, nonnegative control-anchored
outputs and W&B tracking. No expanded-cohort training is registered or running.

More replication is a testable data improvement, not an established score gain.
Promoter/Feng progression still needs successful public validation; Feng also
needs a compatible trainable flow parent, and submission needs a validated
full-gene raw-count emitter. No automatic promotion or new background watcher
was started.

## Reproducibility

Root: `artifacts/public_flow/replicated_v7_20260910/`.

- `registration/contract.json`: `c2fe975be373f3c7eb6be20488a0307ebd4f7f355d87b5672977f814b328f644`
- `registration/sources.json`: `df6fc7a7a7da6d4ba2a241c0a73849a7aeb15a0f42c71de6cd69d72f288d3516`
- `registration/metadata_audit.json`: `efe14f290a285f97d5c4201f2129fe0f82dd647fc5f1ed2d2e47127c7ccd25d9`
- `registration/complete.json`: `67788667880a03437c17606d36709c9651c4dc46194fcb287c363f85c56af2f7`
- `cache.npz`: `292e52e7e2818e754f8c9176ff17d9b1542c07d7f1de2a43ed40a0912132f783`
- `cache.npz.receipt.json`: `a3cd83308ab7fa8fd45bdc5a4de5acce4768e7c7d1134919724e24071261ad0e`

Frozen new source SHA-256:

- `prepare_public_replicated_v7.py`: `b4517402e869cad02ab370d2892fdf44c41746519478b8649c5b84033ac6ca26`
- `public_replicated_cache_v7.py`: `b8f18f734d5ac959df75a26c35ff70a3c92d2c7917e9d4f8cb492e068360fd9c`
- Protocol: `6434ffb3aa29bf5774113cb896fdf9d6cb2ad9ae00e873d643ca8baa755fd989`
