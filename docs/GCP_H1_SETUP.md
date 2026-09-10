# H1 acquisition: free-allowance gate and isolated Google Cloud setup

Date: 2026-09-09. Billing project: `vcc-dataset`.

## Scope and current boundary

The requested first acquisition is only
`gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/`, into
`/fs/cbsuvlaminck3/workdir/yz3482/virtual-cell-challenge-2026/vcc_2025_h1/`.
The whole destination is Git-ignored, including CSV/JSON metadata.
The CLI is installed and its local project is set to `vcc-dataset`. The user
completed OAuth; an active account and `billingEnabled: true` were verified on
2026-09-09. The user subsequently confirmed active Arc Marketplace linkage and
zero prior usage of the 2 TB monthly allowance. That confirmation is recorded
as user attestation, not an independently queried Google credit balance.
The selected seven-file processed-data transfer **completed and passed checksum
verification at 2026-09-09 21:44:05 UTC**, using explicit
`--billing-project vcc-dataset` throughout. The destination remains mode 0700.
No billing, IAM, subscription or training-job changes have been made.

### Inventory and selected transfer

The live 2025 prefix contained **49 objects / 5,844,589,074,216 bytes**.
Its **42 FASTQ objects total 5,810,226,748,327 bytes** and are explicitly excluded:
raw sequencing is unnecessary for the current count-matrix modelling route and
the whole prefix would exceed the free allowance. Do not use recursive `cp`.

The approved seven processed files total **34,362,325,889 bytes** (34.36 GB,
1.7182% of a decimal 2 TB). They comprise the train/validation/test H5ADs, three
perturbation-count CSVs and gene_names.csv. Exact generations, sizes and upstream
CRC32C/available MD5 values are bound by `vcc_2025_h1/acquisition_plan.json`:
SHA-256 `93c8f811bee27aa9c61c7b5c9b36f313bfb1118f6254a2324e35687b729a9b8e`.

`scripts/acquire_vcc_h1.py` permits only those seven names. It uses the official
CLI sequentially, pins every source generation and billing project, reserves
each full object's size before transfer, and independently verifies CRC32C,
available upstream MD5, and local SHA-256. Existing files require checksum
verification; incomplete reservations or partials halt for review. The campaign
cap is **50 GB**, with an additional **100 GB monthly safety reserve**. The
private project/month ledger lives outside Git under `.vcc-gcp/transfer-ledger/`.

The download-only environment `.venv-vcc-download` contains google-crc32c 1.8.0
with its compiled verifier; training environments are unchanged. Completion
is established by the fully verified `vcc_2025_h1/acquisition_complete.json`,
not by the presence of filenames or a queued process. No expression matrix has
been inspected or model trained by this acquisition.

Final verified and reserved bytes both equal **34,362,325,889**. All seven
official-client manifest entries are `OK`, with one full reservation per file
and no failed or repeated transfers. Completion receipt SHA-256:
`6e01022c48a682e381d907ebb4fb8ef7f781ef99f3a11550528bbec2efd90d7b`.
The acquisition/CLI safety regression suite passed **92 offline tests**.
This records transfer bytes and integrity, not independently posted billing
credits; Marketplace credits may appear asynchronously.

Configured zero retries disables media/resumable retry loops; the installed SDK
can still retry metadata requests or refresh authentication after HTTP 401.
Object-byte reservations are conservative local accounting, not a live Google
billing meter, and do not account for independent activity by other processes.

**The 2 TB benefit is a conditional monthly credit, not an automatic stop.**
[Arc's access instructions](https://github.com/ArcInstitute/arc-virtual-cell-atlas#accessing-the-data)
require the exact billing project to subscribe through Google Cloud Marketplace.
The `--billing-project` flag alone does not activate the offer. Credits may be
delayed 24–48 hours or until billing-cycle end. Project billing linkage does not
prove subscription or remaining allowance.

## Private CLI runtime

The task-specific runtime is outside Git:
`/fs/cbsuvlaminck3/workdir/yz3482/.vcc-gcp/` (mode 0700).
Its `config/` directory is the only CLI configuration directory used by
`bash scripts/vcc_gcloud.sh`. No shell startup files, default Google login,
system packages or Application Default Credentials need to be changed.

Checksum-pinned official package: Google Cloud CLI **584.0.0**, Linux x86_64.

- [Package used](https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz)
- SHA-256: `309a8fd47df8d4d5694c798b9d0d8968ae736a58f5dceba60eb2cf520b541460`
- [Official installation instructions/checksum source](https://docs.cloud.google.com/sdk/docs/install-sdk)

Verify the archive checksum before unpacking/executing it. This upstream URL is
rolling: future contents must not be accepted under the old checksum. The
version-named 584.0.0 archive has a different checksum and was **not** unpacked
or executed; the installed archive is the exact link/checksum pair from Google's
installation page. The installed CLI reports 584.0.0 and bundled Python 3.14.7.
The wrapper disables usage reporting, update checks and
HTTP debug logging; it applies sequential storage transfers, zero configured
retries, no sliced downloads and mandatory checksum checking. These settings
are **not** a billing guarantee. `VCC_GCP_RUNTIME` can select another absolute
private runtime path for another compute host or for tests. Configuration
isolation is not a security sandbox: inherited explicit authentication overrides
are not cleared. None were configured on this host during setup.

Verification: shell syntax check, real CLI version/config checks, private
directory permissions, and **19 offline wrapper tests passed**. The installed
584.0.0 GCS download implementation passes `storage/max_retries` to its transfer
client, and documents threshold zero as disabling sliced downloads. This does
not establish that there can never be retransmission at another network layer.

## Authentication (user's own terminal)

From an appropriate BioHPC compute session:

```bash
cd /fs/cbsuvlaminck3/workdir/yz3482/virtual-cell-challenge-2026
bash scripts/vcc_gcloud.sh auth login --no-launch-browser
bash scripts/vcc_gcloud.sh config set project vcc-dataset
bash scripts/vcc_gcloud.sh billing projects describe vcc-dataset
```

Open the Google URL in your own trusted browser and complete the exchange in
that terminal. **Do not send passwords, OAuth codes, refresh tokens or service
account keys through chat or commit them.** The last command is the GA equivalent
of `gcloud beta billing projects describe`; it is read-only and reports billing
association, not Atlas enrollment or remaining free usage.
[Authentication reference](https://docs.cloud.google.com/sdk/gcloud/reference/auth/login),
[billing command](https://docs.cloud.google.com/sdk/gcloud/reference/billing/projects/describe).

## Required checks before any Atlas request

1. Confirm an authorized Google identity can use `vcc-dataset`; do not create a
   new project, link billing, change IAM or enable paid services as a workaround.
2. In [Arc's Marketplace listing](https://console.cloud.google.com/marketplace/product/bigquery-public-data/arc-institute),
   select **vcc-dataset** and verify the active offer and its free-usage terms.
   Confirm what operations/egress are covered, the allowance period and units.
3. Establish current-period usage for that project, including other people,
   failed downloads, pending charges and concurrent transfers. A zero-dollar
   billing report does not establish zero bytes consumed. If usage or enrollment
   cannot be verified, keep the transfer blocked and ask the provider/project
   administrator for confirmation. No documented real-time Arc remaining-byte
   API was found in this audit.
4. Once eligibility and metadata-operation coverage are established, inventory
   only the H1 prefix. Record explicit object names, generations, sizes, CRC32C,
   MD5 where available, encoding and modification times. Freeze the approved
   manifest and its hash before any media transfer. The current object count,
   total size and remaining allowance are still **unverified**.
5. Budget conservatively against at most **2,000,000,000,000 bytes** per month
   until offer units are confirmed, minus verified prior usage, outstanding
   reservations and a safety margin. Reserve each full object size before an
   attempt; do not refund interrupted/uncertain transfers. Account for all
   request retries and concurrent usage, not only final local file sizes.
6. Check destination free space, including temporary files and later conversion
   expansion. Use an exact-generation, explicit-file transfer plan and a shared
   per-project/per-period lock. Verify existing files by size and checksum, not
   existence alone; never overwrite unverified local data. Stop on errors.
7. Publish a completion receipt only after checksum validation. Record attempted
   byte reservations separately from verified downloaded bytes and existing
   verified bytes. A local manifest cannot prove account-wide free usage.

Do **not** run the user's recursive `cp` command before these gates pass. It has
no maximum-spend/maximum-byte flag. `--manifest-path` is an audit aid, not a cap;
`--no-clobber` is not checksum verification.
[Storage copy reference](https://docs.cloud.google.com/sdk/gcloud/reference/storage/cp),
[Requester Pays](https://docs.cloud.google.com/storage/docs/requester-pays),
[retry controls](https://docs.cloud.google.com/storage/docs/retry-strategy).

Alerts-only billing budgets do not cap spend. The current spend-cap preview does
not list Cloud Storage among supported services and allows latency/in-flight
overages; it cannot enforce this download's free-only condition.
[Google budget documentation](https://docs.cloud.google.com/billing/docs/how-to/budgets),
[spend-cap coverage](https://docs.cloud.google.com/billing/docs/how-to/budgets-spend-caps).

## Approved modelling route after acquisition

GenePT/GO target conditioning + **control-derived** context conditioning →
public CRISPRi flow pretraining → public-only post-training.

- H1 2025 is an additional public CRISPRi context, not evidence that 2026 treated
  challenge data may be used. Register the split and any target-family exclusions
  before inspecting expression; keep any existing V7 seal intact and record new
  lineage instead of reusing contaminated checkpoints.
- Build context representations from matched controls only. Compare inexpensive
  control-PCA with any frozen cell encoder; keep ESM2 and negative target-feature
  controls in the embedding ablation. Do not assume GenePT/GO improves scores.
- Evaluate context transfer on held-out public contexts, explicitly labeling
  previously explored RPE1/Jurkat/HepG2 as development rather than untouched test.
  Use actual single-cell counts, aligned genes and explicit missing-gene masks;
  do not treat population-mean Replogle artifacts as individual cells.
- Promote a flow model only after distributional and perturbation-effect checks
  (including MMD, variance, covariance, library sizes and target-feature
  shuffling) beat matched controls and simpler baselines. No new H100 job or
  submission is authorized by a failed gate.
- Record pretraining and post-training through the existing W&B wrapper with
  source manifests, split hashes, model/feature versions, negative controls and
  quality metrics. Offline logging remains available without a W&B cloud login.
  Never upload Google credentials or raw data through tracking.

See [the evidence and dataset shortlist](research/EMBEDDINGS_AND_PUBLIC_FLOW_DATA_20260909.md)
and [the new prototype status](PUBLIC_CRISPRI_FLOW_V1.md), plus
[the tracking runbook](WANDB_TRAINING.md). This is the approved direction,
not a claim that a new hybrid architecture has already been trained or validated.
