# GenePT numeric conversion v11

This is feature preparation, not training or model promotion. The submitted
GO/context v10 checkpoint and VCC entry remain unchanged.

The input is the already-local official Zenodo 10833191 Ada-text archive,
SHA256 `6193575dcbd7bbf214c8ca3eb518cb3d13272443f98ecd6c26402411a7ca745e`,
574,395,233 bytes, CC BY 4.0. No download, paid API, GCP or GPU is needed.
Only `GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle` is decoded.

The audited format is a primitive protocol-4 dictionary mapping 93,800 name
keys to 33,230 stored finite 1,536-dimensional numeric vector objects. These
are name keys sharing stored vector objects, not 93,800 distinct biological
genes or independently proven biological synonyms. A custom data-only parser
accepts only PROTO4, FRAME, EMPTY_DICT, MEMOIZE, MARK, SHORT_BINUNICODE,
EMPTY_LIST, BINFLOAT, BINGET, LONG_BINGET, APPENDS, SETITEMS and STOP. Every
other opcode is rejected before reading its argument. No callable, import,
general unpickling, extension, persistent ID, arbitrary object or cycle can
be instantiated. Frames, dimensions, references, finite values and EOF are
checked. Full archive authentication precedes parsing on the same open file.

Before numeric decoding, registration freezes the converter/protocol hashes
and exact sorted union of the v10 K562/Jurkat fit/held target names and the
300 official target names. These are identities only, not expression. Roles
remain inherited unchanged from training contract
`590a5916c6a9c76e263f2c8e4e45ed9ba41def7682a6694ce82d87b1b7074f2b`.
No held or challenge response is read. Selecting known pretrained rows for
held names is not fitting a transform on them.

Export float32 raw embeddings, Unicode target names, uint8 availability and
int32 source-vector reference IDs and canonical binary64-vector SHA256 values
into NPZ readable with allow_pickle=False.
Keep original values unnormalized; report maximum float32 rounding error.
Missing symbols get zero vectors and availability=0. No alias guessing,
identity borrowing or fitted mean imputation. Shared vector IDs report actual
archive object aliases, not independently proven biological equivalence.
Report exact duplicate object/content groups across fit/held and fit/official
names, excluding missing vectors; they flag future admission review, not
automatic permission to fit or evidence of biological identity.

Bounds: two CPU cores/threads, 600 seconds, 3 GiB address space, 2 GiB peak RSS,
128 MiB cumulative output. Only approved BioHPC compute hosts, never login/head;
GPUs hidden. The full-conversion wall guard includes provenance reconstruction.
Bounded regular-file readers reject leaf symlinks/FIFOs. Fresh, non-overwriting
directories; failed attempts preserved. Archive identity is checked immediately
after hashing and after parsing, and code/protocol/target provenance is replayed
before completion. Final RSS/time/output checks include numeric NPZ replay.
Full-shape, finite numeric replay and role-specific coverage enter the receipt.

Later training must register source-fit-only transforms and separate GO scale
balancing from GenePT addition. Do not infer a score gain from target coverage.
