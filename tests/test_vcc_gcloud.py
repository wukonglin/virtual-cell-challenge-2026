"""Offline Google Cloud wrapper checks using only a synthetic executable."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "scripts" / "vcc_gcloud.sh"
EXPECTED_ENV = {
    "CLOUDSDK_CORE_PROJECT": "vcc-dataset",
    "CLOUDSDK_CORE_DISABLE_USAGE_REPORTING": "true",
    "CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK": "true",
    "CLOUDSDK_CORE_LOG_HTTP": "false",
    "PYTHONNOUSERSITE": "1",
    "CLOUDSDK_STORAGE_PROCESS_COUNT": "1",
    "CLOUDSDK_STORAGE_THREAD_COUNT": "1",
    "CLOUDSDK_STORAGE_MAX_RETRIES": "0",
    "CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD": "0",
    "CLOUDSDK_STORAGE_CHECK_HASHES": "always",
}


@pytest.fixture
def layout(tmp_path):
    workdir = tmp_path / "workdir with spaces"
    scripts = workdir / "synthetic-repo" / "scripts"
    scripts.mkdir(parents=True)
    wrapper = scripts / "vcc_gcloud.sh"
    shutil.copyfile(WRAPPER, wrapper)
    return workdir, wrapper


def fake_sdk(runtime):
    """The only executable installed in each test is this local recorder."""
    config = runtime / "config"
    config.mkdir(parents=True)
    executable = runtime / "google-cloud-sdk" / "bin" / "gcloud"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, stat, sys\n"
        "config = pathlib.Path(os.environ['CLOUDSDK_CONFIG'])\n"
        "probe = config / 'synthetic-permission-probe'\n"
        "with probe.open('x') as stream: stream.write('synthetic')\n"
        "directory = config / 'synthetic-directory-probe'\n"
        "directory.mkdir()\n"
        "mask = os.umask(0o077)\n"
        "payload = {'argv': sys.argv[1:], 'umask': mask, "
        "'file_mode': stat.S_IMODE(probe.stat().st_mode), "
        "'directory_mode': stat.S_IMODE(directory.stat().st_mode), "
        "'cwd': os.getcwd(), "
        "'environment': {key: value for key, value in os.environ.items() "
        "if key.startswith('CLOUDSDK_') or key == 'PYTHONNOUSERSITE'}}\n"
        "print(json.dumps(payload))\n"
        "sys.exit(int(os.environ.get('SYNTHETIC_GCLOUD_EXIT_CODE', '0')))\n"
    )
    executable.chmod(0o700)
    return executable


def clean_environment():
    # No actual cloud/auth environment is passed even to the fake executable.
    return {key: value for key, value in os.environ.items()
            if not key.startswith(("CLOUDSDK_", "GOOGLE_", "GCLOUD_", "VCC_GCP_"))}


def run_fake(wrapper, *args, env=None, cwd=None):
    environment = clean_environment()
    environment.update(env or {})
    return subprocess.run(["bash", str(wrapper), *args], env=environment, cwd=cwd,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          timeout=10, check=False)


def test_default_runtime_is_under_project_parent_not_current_directory(layout, tmp_path):
    workdir, wrapper = layout
    runtime = workdir / ".vcc-gcp"
    fake_sdk(runtime)
    elsewhere = tmp_path / "unrelated-cwd"
    elsewhere.mkdir()
    result = run_fake(wrapper, "version", cwd=elsewhere)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["environment"]["CLOUDSDK_CONFIG"] == str(runtime / "config")
    assert payload["cwd"] == str(elsewhere)
    assert payload["argv"] == ["version"]
    assert not (elsewhere / ".vcc-gcp").exists()


def test_absolute_override_and_argv_are_forwarded_without_interpolation(layout, tmp_path):
    _, wrapper = layout
    runtime = tmp_path / "alternate isolated runtime"
    fake_sdk(runtime)
    args = ["storage", "cp", "gs://synthetic-bucket/file with spaces", "literal'quote",
            'literal"quote', "$(should-not-run)", "$HOME", "--", ""]
    result = run_fake(wrapper, *args, env={"VCC_GCP_RUNTIME": str(runtime)})
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["argv"] == args
    assert payload["environment"]["CLOUDSDK_CONFIG"] == str(runtime / "config")


def test_inherited_project_telemetry_and_storage_settings_are_overridden(layout):
    workdir, wrapper = layout
    runtime = workdir / ".vcc-gcp"
    fake_sdk(runtime)
    hostile = {key: "SYNTHETIC_UNSAFE_INHERITED_VALUE" for key in EXPECTED_ENV}
    hostile["CLOUDSDK_CONFIG"] = "/synthetic/unrelated/config"
    result = run_fake(wrapper, "config", "list", env=hostile)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)["environment"]
    for key, value in EXPECTED_ENV.items():
        assert observed[key] == value
    assert observed["CLOUDSDK_CONFIG"] == str(runtime / "config")
    assert "SYNTHETIC_UNSAFE_INHERITED_VALUE" not in result.stdout


def test_private_umask_applies_to_files_and_directories_created_by_cli(layout):
    workdir, wrapper = layout
    fake_sdk(workdir / ".vcc-gcp")
    result = run_fake(wrapper, "version")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["umask"] == 0o077
    assert payload["file_mode"] == 0o600
    assert payload["directory_mode"] == 0o700


@pytest.mark.parametrize("code", [0, 7, 42])
def test_cli_exit_status_is_preserved(layout, code):
    workdir, wrapper = layout
    fake_sdk(workdir / ".vcc-gcp")
    result = run_fake(wrapper, "version", env={"SYNTHETIC_GCLOUD_EXIT_CODE": str(code)})
    assert result.returncode == code
    assert json.loads(result.stdout)["argv"] == ["version"]


@pytest.mark.parametrize("missing", ["everything", "config", "executable", "execute-bit"])
def test_missing_or_nonexecutable_runtime_fails_before_cli(layout, missing):
    workdir, wrapper = layout
    runtime = workdir / ".vcc-gcp"
    if missing != "everything":
        executable = fake_sdk(runtime)
        if missing == "config":
            (runtime / "config").rmdir()
        elif missing == "executable":
            executable.unlink()
        else:
            executable.chmod(0o600)
    result = run_fake(wrapper, "version")
    assert result.returncode == 2
    assert "Isolated Google Cloud CLI/config missing" in result.stderr
    assert result.stdout == ""
    assert not (runtime / "config" / "synthetic-permission-probe").exists()


@pytest.mark.parametrize("override", ["relative", ".vcc-gcp", "../sibling", "~/runtime"])
def test_relative_runtime_override_fails_even_when_default_is_available(layout, override):
    workdir, wrapper = layout
    runtime = workdir / ".vcc-gcp"
    fake_sdk(runtime)
    result = run_fake(wrapper, "version", env={"VCC_GCP_RUNTIME": override})
    assert result.returncode == 2
    assert "VCC_GCP_RUNTIME must be an absolute path" in result.stderr
    assert result.stdout == ""
    assert not (runtime / "config" / "synthetic-permission-probe").exists()


def test_empty_override_uses_documented_default_runtime(layout):
    workdir, wrapper = layout
    runtime = workdir / ".vcc-gcp"
    fake_sdk(runtime)
    result = run_fake(wrapper, "version", env={"VCC_GCP_RUNTIME": ""})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["environment"]["CLOUDSDK_CONFIG"] == str(runtime / "config")


@pytest.mark.parametrize("path", ["vcc_2025_h1/metadata.json", "vcc_2025_h1/manifest.tsv",
                                  "vcc_2025_h1/private/acquisition_receipt.json"])
def test_h1_acquisition_metadata_is_ignored_by_git(path):
    result = subprocess.run(["git", "check-ignore", "--no-index", "--", path], cwd=REPO,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == path

