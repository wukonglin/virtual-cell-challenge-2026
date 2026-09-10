#!/usr/bin/env bash
# Isolated Google Cloud CLI for the H1 acquisition workflow.
# This wrapper is NOT a spending cap or evidence of free-tier eligibility.
set -euo pipefail
umask 077

vcc_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
vcc_workdir_parent=$(cd -- "$vcc_script_dir/../.." && pwd -P)
vcc_gcp_runtime=${VCC_GCP_RUNTIME:-"$vcc_workdir_parent/.vcc-gcp"}
if [[ "$vcc_gcp_runtime" != /* ]]; then
  printf 'VCC_GCP_RUNTIME must be an absolute path.\n' >&2
  exit 2
fi
if [[ ! -x "$vcc_gcp_runtime/google-cloud-sdk/bin/gcloud" || ! -d "$vcc_gcp_runtime/config" ]]; then
  printf 'Isolated Google Cloud CLI/config missing under %s; see docs/GCP_H1_SETUP.md.\n' "$vcc_gcp_runtime" >&2
  exit 2
fi

export CLOUDSDK_CONFIG="$vcc_gcp_runtime/config"
export CLOUDSDK_CORE_PROJECT=vcc-dataset
export CLOUDSDK_CORE_DISABLE_USAGE_REPORTING=true
export CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK=true
export CLOUDSDK_CORE_LOG_HTTP=false
export PYTHONNOUSERSITE=1

# Make later transfers simpler to audit. These do not impose a byte/spend cap.
export CLOUDSDK_STORAGE_PROCESS_COUNT=1
export CLOUDSDK_STORAGE_THREAD_COUNT=1
export CLOUDSDK_STORAGE_MAX_RETRIES=0
export CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD=0
export CLOUDSDK_STORAGE_CHECK_HASHES=always

exec "$vcc_gcp_runtime/google-cloud-sdk/bin/gcloud" "$@"
