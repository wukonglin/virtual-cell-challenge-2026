# GenePT numeric features: actual conversion and admission findings

Completed on the BioHPC compute node at2026-09-10T19:53:33UTC. This is real
numeric feature preparation, not new flow pretraining, Feng post-training or
a submission. No network, GCP, paid API, GPU or pickle execution was used.

The verified public Ada archive yielded a4,083,232-byte numeric NPZ containing
787 requested names and1,536 embedding coordinates. Coverage is782/787:
K562 fit296/299, Jurkat fit88/90, all75/23 public held targets, and all300 VCC
target names. Missing:ALG1L,C1orf109,C7orf26,CCDC144NL,POLR2B. Missing rows are
zero with an explicit availability flag; no aliases were guessed.

The full stream had93,800 name keys and33,230 stored vector objects, matching
the independent non-executing audit. Selected rows refer to778 stored objects.
Every selected original binary64 value is exactly representable in float32:
observed conversion error0.0. All NPZ arrays replayed with allow_pickle=False.
The run took56.74s, peakRSS1,159,491,584bytes, within its2CPU/2GiB/10minute bounds.

## Ambiguous duplicate representations

Four pairs share the same stored object and identical binary64/float32 values:

| Names | Current roles |
| --- | --- |
| BRD1 / BRPF1 | Both official VCC targets |
| LTB4R2 / NOP9 | K562 fit / Jurkat fit |
| MBD4 / MED1 | Official VCC target / K562 fit |
| SKP2 / UXT | Both K562 fit |

These are observations about the released representation, **not proof of
biological synonymy or true target equivalence**. No duplicate group crosses
the current fit/held response boundary, but MBD4/MED1 crosses fit/official.

For the first downstream pilot, conservatively quarantine GenePT features for
all eight names:zero vector, availability0, no silent selection of one member.
Keep exact target labels, GO features and original response roles unchanged.
The raw converted artifact remains immutable. The expected post-quarantine
coverage is774/787:K562 fit292/299, Jurkat fit87/90, held75/75 and23/23,
official297/300. The separate source-bound materialization below has now
completed and verified these exact counts.

## Provenance and verification

Artifacts:`artifacts/public_flow/genept_numeric_v11_20260910/`.

| Binding | SHA256 |
| --- | --- |
| Converter | `029968dbc72d92e2ecfaaab16b33b4d66781e31db8e1a4075d975fac3e2faebd` |
| Protocol | `b2970a671be4c475c00f48207f37e05e6a786b7d0616e3aff2fa7c8df9a476e4` |
| Registration | `965efbbc88ee7e6f2309af3d487260b247d9b2618331607c33d2eb3e59b36987` |
| Completion | `f565eb23256722ed3b85d5960a71086138670aff55d58d2c2660dc8fe23cdc57` |
| Numeric NPZ | `f1c29e064be43a6a2ebfc03371fb23537edf401582e8d5e2187967ebd3b83536` |
| Decoded Ada member | `fd297510ddd3040744033fde0b0f2cf15a40ac8b2fd2fb02f10667295e55c862` |

388 synthetic parser/numeric/registration tests passed before launch; an
independent review tested malicious opcode rejection and numeric/alias replay.
Full archive hash and same-handle identity checks preceded decoding; code,
protocol and source target provenance were reauthenticated before completion.

Next:source-fit-only feature transformations and a factorial public validation
experiment, as described in[the v11 draft](GENEPT_GO_FLOW_V11_DRAFT_20260910.md).
Coverage is not evidence of predictive quality. The scored v10 model used no
GenePT and remains unchanged.

## Source-bound feature preparation completed

At20:06:02UTC, the reviewed materializer completed the actual quarantine,
source-fit-only statistics, all-target transforms, and correct/masked/shuffled
arms for seeds20260926/20260927. All six numeric archives and four arm-report
JSON files are bound by the completion receipt; both transform records and
quarantine/sharing audits are bound too. Every numeric archive was replayed.

K562 statistics use292 available fitting targets, scalar RMS
0.01041000315195989. Jurkat uses87, RMS0.010521640144601675. Both retain1,536
coordinates; available fit means are near0 and scalar mean-square near1 after
transformation. Official/held features do not fit these statistics. No exact
cross-role duplicate representation remains after quarantine. All98 held
response targets remain outside fitting, and no response values were read.

Actual runtime1.59s; peakRSS326,197,248bytes;19,828,841artifact bytes before the
small completion receipt. No GPU, network, predictive-model training, W&B
upload or submission was performed. These are ready feature artifacts, not a
new trained checkpoint. The factorial model/runner still needs implementation
and a separate execution contract.

Artifacts:`artifacts/public_flow/genept_conditioning_v11_20260910/`.

| Binding | SHA256 |
| --- | --- |
| Conditioning helper | `a80f7f64921203b6897c31af20ceb6e306cddf3e7ffdbb739d9d063425efec13` |
| Materializer | `c78c63845e234b60cfbf5939faddf66fc7dc342a46facc6858b6771a26396d48` |
| Protocol | `63f50295655cecc21e93362817012c5fb6181df55beb7ea877aa4cca618784e4` |
| Registration | `ad310a0bfc23e919d7dc001401782523d00fa58b9395b20f6b29b90d71909c04` |
| Completion | `46e602682284ce76ef06ecb3c22bf675f7a891fabc8208155eaab6b4ea9a0db9` |

Before launch,555 combined conversion, conditioning, materializer and frozen
flow/runner regression tests passed, including a synthetic full787x1536
materialization. The helper and materializer received independent review.

A fresh independent read-only audit authenticated23 files and recomputed the
real statistics/transforms without calling the feature helper's fit/transform
functions. Both source means and scalar RMS, all787x1536transformed values, and
both seeds' three arms reproduced exactly:maximum numerical replay error0.
Full arm reports and RNG donor maps matched, availability was preserved, and
no donor crossed fitting/held roles. All98held and300official identities stayed
outside statistic fitting. Audit runtime0.511s, peakRSS108,363,776bytes,2CPU
cores with GPUs hidden. No files, responses or predictive models were changed.
This verifies feature preparation, not prediction quality or automatic training
admission. No new GPU training job was launched in this continuation.
