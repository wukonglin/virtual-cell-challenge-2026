"""Synthetic tracking tests: no cloud access, real datasets, or GPU training."""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import run_training_with_wandb as wrapper  # noqa: E402
import wandb_training as tracking  # noqa: E402


class FakeConfig(dict):
    def update(self, values, *, allow_val_change=False):
        super().update(values)


class FakeRun:
    def __init__(self, *, log_error=False, finish_error=False):
        self.logged = []
        self.defined = []
        self.summary = {}
        self.config = FakeConfig()
        self.finished = []
        self.log_error = log_error
        self.finish_error = finish_error

    def define_metric(self, *args, **kwargs):
        self.defined.append((args, kwargs))

    def log(self, data, *, step):
        self.logged.append((dict(data), step))
        if self.log_error:
            raise RuntimeError("synthetic SDK log failure")

    def finish(self, *, exit_code):
        self.finished.append(exit_code)
        if self.finish_error:
            raise RuntimeError("synthetic SDK finish failure")


class FakeSDK:
    def __init__(self, **kwargs):
        self.run = None
        self.created_run = FakeRun(**kwargs)
        self.initialized = []

    @staticmethod
    def Settings(**kwargs):
        return kwargs

    def init(self, **kwargs):
        self.initialized.append(kwargs)
        self.run = self.created_run
        return self.run


@pytest.fixture(autouse=True)
def isolated_rank_and_tracking_environment(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("WANDB_") or key in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fake_sdk(monkeypatch):
    sdk = FakeSDK()
    original = tracking.TrainingTracker
    monkeypatch.setattr(wrapper, "TrainingTracker", lambda *args, **kwargs: original(*args, sdk=sdk, **kwargs))
    return sdk


def events(path):
    return [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]


def receipt(path):
    return json.loads((path / "tracking.json").read_text())


def command_args(output, source, *options):
    return ["--stage", "pretrain", "--output-dir", str(output), *options,
            "--", sys.executable, "-c", source]


def test_scalar_allowlist_drops_paths_secrets_arrays_and_arbitrary_quality():
    payload = {
        "step": 4, "loss": 1.25, "mmd": -0.05,
        "secret": "SYNTHETIC_PRIVATE_VALUE", "path": "/private/data.h5ad",
        "expression": [[1, 2]], "status": "pass", "passed": True,
        "mse": {"true": 0.1, "constant": 0.2, "zero": 0.3,
                "shuffled_20260909": 0.4, "secret": 5.0},
        "decision": {"conditioning_screen_passed": False, "private": "private"},
    }
    result = tracking.scalar_metrics(payload)
    assert result == {
        "train/global_step": 4, "train/loss": 1.25, "train/mmd": -0.05,
        "cv/mse/true": 0.1, "cv/mse/constant": 0.2, "cv/mse/zero": 0.3,
        "cv/mse/shuffled_20260909": 0.4, "quality/conditioning_screen_passed": 0,
    }
    assert tracking.scalar_metrics({"schema": "vcc-lingshu-scdfm-public-quality-preflight-v1", "status": "fail"}) == {
        "quality/public_gate_passed": 0}
    assert tracking.scalar_metrics({"schema": "vcc-lingshu-scdfm-public-quality-preflight-v1", "passed": 1}) == {}


@pytest.mark.parametrize("value", [True, False, None, "1", float("nan"), float("inf"), -float("inf")])
def test_nonfinite_and_nonnumeric_values_are_dropped(value):
    assert tracking.scalar_metrics({"loss": value, "mmd": value}) == {}


@pytest.mark.parametrize("value", [-1, 1.5, 1.0, True, "3"])
def test_steps_require_nonnegative_integers(value):
    assert tracking.scalar_metrics({key: value for key in (
        "step", "best_step", "last_step", "completed_repeat", "outer_fold")}) == {}
    assert tracking.scalar_metrics({"step": 0}) == {"train/global_step": 0}


def test_json_lines_reject_invalid_and_oversized_payloads():
    for line in ("not JSON", "[1,2]", "{broken", "{" + " " * 65536 + "}"):
        assert tracking.line_metrics(line) == {}
    assert tracking.line_metrics(' {"step": 2, "mmd": -0.01, "private": "omit"}') == {
        "train/global_step": 2, "train/mmd": -0.01}


def test_biological_feature_metrics_and_quality_remain_allowlisted(tmp_path):
    arms = ("esm2", "lingshu", "fusion", "esm2_shuffled_20260960",
            "lingshu_shuffled_20260960", "fusion_shuffled_lingshu_20260960")
    payload = {"mse": {**dict.fromkeys(arms, 0.1), "fusion_private_label": 9.0},
               "decisions": {"esm2": {"conditioning_screen_passed": True},
                             "lingshu": {"conditioning_screen_passed": False},
                             "fusion": {"conditioning_screen_passed": False},
                             "private_label": {"conditioning_screen_passed": True}}}
    expected = {**{f"cv/mse/{arm}": 0.1 for arm in arms},
                "quality/esm2_conditioning_screen_passed": 1,
                "quality/lingshu_conditioning_screen_passed": 0,
                "quality/fusion_conditioning_screen_passed": 0}
    assert tracking.scalar_metrics(payload) == expected
    sdk = FakeSDK()
    tracker = tracking.TrainingTracker(tmp_path / "tracking", stage="pretrain", sdk=sdk)
    tracker.log(expected)
    tracker.finish(0)
    logged = sdk.run.logged[0][0]
    assert all(logged[key] == value for key, value in expected.items())
    assert not any("private_label" in key for key in logged)


def test_config_only_known_typed_flags_and_verified_hashes():
    command = ["python", "/private/train.py", "--steps", "20", "--learning-rate=1e-4",
               "--feature-mode", "true", "--token", "SECRET", "--data", "/private/data",
               "--batch-size", "nan", "--dropout", "inf", "--seed", "3"]
    assert tracking.config_from_command(command) == {
        "steps": 20, "learning_rate": 1e-4, "feature_mode": "true", "seed": 3}
    assert tracking.safe_config({"entrypoint_sha256": "a" * 64,
        "parent_checkpoint_sha256": "/private/model.pt", "steps": True,
        "feature_mode": "SECRET", "raw_argv": command, "api_key": "SECRET"}) == {
        "entrypoint_sha256": "a" * 64}


@pytest.mark.parametrize("label", ["/private/path", "name with spaces", "https://private", "", "x" * 129])
def test_labels_reject_paths_and_unbounded_strings(label):
    with pytest.raises(ValueError):
        tracking.safe_label(label, "name")


def test_repeated_optimizer_steps_are_preserved_and_sdk_capture_disabled(tmp_path):
    sdk = FakeSDK()
    tracker = tracking.TrainingTracker(tmp_path / "tracking", stage="posttrain", sdk=sdk)
    tracker.log(tracking.scalar_metrics({"step": 10, "loss": 1.0}))
    tracker.log(tracking.scalar_metrics({"step": 10, "validation_velocity_mse": 0.4}))
    tracker.finish(0)
    assert [item[1] for item in sdk.run.logged] == [1, 2]
    assert [item[0]["train/global_step"] for item in sdk.run.logged] == [10, 10]
    assert len(events(tmp_path / "tracking")) == 2
    settings = sdk.initialized[0]["settings"]
    assert settings["console"] == "off"
    for key in ("disable_git", "disable_code", "disable_job_creation", "x_disable_meta",
                "x_disable_stats", "x_disable_machine_info"):
        assert settings[key] is True
    assert settings["save_code"] is False
    assert settings["x_save_requirements"] is False
    result = receipt(tmp_path / "tracking")
    assert result["cloud_synced"] is False
    assert result["execution_status"] == "completed"
    assert result["scientific_quality_inferred_from_exit"] is False


def test_log_direct_call_still_removes_unknown_metrics(tmp_path):
    sdk = FakeSDK()
    tracker = tracking.TrainingTracker(tmp_path / "tracking", stage="validation", sdk=sdk)
    tracker.log({"train/loss": 1.0, "private_path": "/private", "secret": 5,
                 "cv/mse/SECRET": 0.4, "train/cfm": math.nan,
                 "train/global_step": -1, "validation/best_step": 1.5})
    tracker.finish(0)
    assert sdk.run.logged == [({"train/loss": 1.0}, 1)]


def test_huge_integers_are_safely_rejected():
    assert tracking.finite_scalar(10 ** 1000) is False
    assert tracking.scalar_metrics({"loss": 10 ** 1000}) == {}
    assert tracking.safe_config({"steps": 10 ** 1000}) == {}


def test_context_target_diagnostics_preserve_signed_mmd_only_known_fields():
    result = tracking.scalar_metrics({"deployment_rollout_diagnostics": {"kind_metrics": {
        "context": {"model_mmd2": -0.0001, "model_centroid_mse": 0.1, "secret": 0.5},
        "target": {"control_mmd2": -0.0002, "clamped_high_fraction": 0.01},
        "SECRET": {"model_mmd2": 3.0}}}})
    assert result == {"validation/context/model_mmd2": -0.0001,
                      "validation/context/model_centroid_mse": 0.1,
                      "validation/target/control_mmd2": -0.0002,
                      "validation/target/clamped_high_fraction": 0.01}


def test_fresh_summary_merges_only_allowlisted_provenance(tmp_path):
    sdk = FakeSDK()
    tracker = tracking.TrainingTracker(tmp_path / "tracking", stage="posttrain", sdk=sdk)
    tracker.summary({"cache_sha256": "a" * 64, "data_path": "/private",
        "training_config": {"steps": 20, "learning_rate": 0.001, "api_key": "SECRET"},
        "model_config": {"hidden_size": 128, "layers": 2, "feature_mode": "true",
                         "checkpoint": "/private/checkpoint.pt"}})
    tracker.finish(0)
    expected = {"cache_sha256": "a" * 64, "steps": 20, "learning_rate": 0.001,
                "hidden_size": 128, "layers": 2, "feature_mode": "true"}
    assert sdk.run.config == expected
    assert receipt(tmp_path / "tracking")["config"] == expected
    assert receipt(tmp_path / "tracking")["tracking_errors"] == 0


def test_sdk_failures_preserve_local_journal_and_training_exit(tmp_path):
    sdk = FakeSDK(log_error=True, finish_error=True)
    output = tmp_path / "tracking"
    tracker = tracking.TrainingTracker(output, stage="posttrain", sdk=sdk)
    tracker.log(tracking.scalar_metrics({"step": 1, "loss": 2.0}))
    tracker.finish(17)
    assert events(output) == [{"event_index": 1, "train/global_step": 1, "train/loss": 2.0}]
    result = receipt(output)
    assert result["training_exit_code"] == 17
    assert result["execution_status"] == "failed"
    assert result["tracking_errors"] == 2
    assert result["cloud_synced"] is False


def test_online_without_auth_does_not_start_child(tmp_path, fake_sdk):
    marker = tmp_path / "child-started"
    code = wrapper.main(command_args(tmp_path / "tracking", f"open({str(marker)!r},'w').close()",
                                     "--mode", "online", "--entity", "test-entity"))
    assert code == 78
    assert not marker.exists()
    assert fake_sdk.initialized == []


def test_child_wandb_is_disabled_and_api_settings_are_removed(tmp_path, fake_sdk, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "SYNTHETIC_DO_NOT_PASS_TO_CHILD")
    monkeypatch.setenv("WANDB_ENTITY", "unexpected-entity")
    monkeypatch.setenv("WANDB_CONFIG_PATHS", "/private/settings.yaml")
    marker = tmp_path / "child-env.json"
    source = ("import json,os; "
              f"open({str(marker)!r},'w').write(json.dumps({{k:v for k,v in os.environ.items() if k.startswith('WANDB_')}}))")
    assert wrapper.main(command_args(tmp_path / "tracking", source)) == 0
    assert json.loads(marker.read_text()) == {"WANDB_MODE": "disabled", "WANDB_DISABLED": "true"}
    assert os.environ["WANDB_API_KEY"] == "SYNTHETIC_DO_NOT_PASS_TO_CHILD"


@pytest.mark.parametrize("rank", ["0", "1"])
def test_all_ranks_execute_but_only_global_rank_zero_tracks(tmp_path, fake_sdk, monkeypatch, rank):
    monkeypatch.setenv("RANK", rank)
    marker = tmp_path / f"rank-{rank}"
    output = tmp_path / "tracking"
    source = f"open({str(marker)!r},'w').close(); print('{{\"step\": 1, \"loss\": 0.1}}')"
    assert wrapper.main(command_args(output, source)) == 0
    assert marker.exists()
    assert len(fake_sdk.initialized) == (1 if rank == "0" else 0)
    assert output.exists() is (rank == "0")


def test_rank_resolution_uses_global_rank_not_local_rank():
    assert tracking.global_rank({"LOCAL_RANK": "1"}) == 0
    assert tracking.global_rank({"SLURM_PROCID": "3", "LOCAL_RANK": "0"}) == 3
    assert tracking.global_rank({"RANK": "2", "SLURM_PROCID": "0"}) == 2
    with pytest.raises(ValueError):
        tracking.global_rank({"RANK": "-1"})


def test_stale_summary_is_rejected_before_child_or_sdk(tmp_path, fake_sdk):
    stale = tmp_path / "old-summary.json"
    stale.write_text('{"loss": 0.01}')
    marker = tmp_path / "child-started"
    with pytest.raises(SystemExit) as error:
        wrapper.main(command_args(tmp_path / "tracking", f"open({str(marker)!r},'w').close()",
                                  "--summary-file", str(stale)))
    assert error.value.code == 2
    assert not marker.exists()
    assert not fake_sdk.initialized


def test_child_failure_exit_and_fresh_summary_are_preserved(tmp_path, fake_sdk):
    summary = tmp_path / "summary.json"
    output = tmp_path / "tracking"
    payload = {"schema": "vcc-lingshu-scdfm-public-quality-preflight-v1", "status": "fail",
               "private": "DO_NOT_TRACK", "best_step": 4}
    source = (f"import json,sys; open({str(summary)!r},'w').write({json.dumps(payload)!r}); "
              "print('{\"step\": 4, \"loss\": 0.5}'); sys.exit(19)")
    assert wrapper.main(command_args(output, source, "--summary-file", str(summary))) == 19
    result = receipt(output)
    assert result["training_exit_code"] == 19
    assert result["execution_status"] == "failed"
    assert result["event_count"] == 2
    assert events(output)[1]["quality/public_gate_passed"] == 0
    assert "DO_NOT_TRACK" not in (output / "metrics.jsonl").read_text()


def test_missing_requested_summary_counts_tracking_error_not_training_failure(tmp_path, fake_sdk):
    output = tmp_path / "tracking"
    assert wrapper.main(command_args(output, "print('completed')", "--summary-file",
                                     str(tmp_path / "missing.json"))) == 0
    assert receipt(output)["training_exit_code"] == 0
    assert receipt(output)["tracking_errors"] == 1


def test_wrapper_finish_exception_preserves_child_exit(tmp_path, fake_sdk, monkeypatch):
    def broken_finish(self, *args, **kwargs):
        self.journal.close()
        raise RuntimeError("synthetic local finalization failure")
    monkeypatch.setattr(tracking.TrainingTracker, "finish", broken_finish)
    assert wrapper.main(command_args(tmp_path / "tracking", "import sys; sys.exit(23)")) == 23


def test_oversized_stdout_line_cannot_smuggle_json_tail(tmp_path, fake_sdk):
    output = tmp_path / "tracking"
    source = ("print('x' * 65537 + '{\"step\": 99, \"loss\": 9}'); "
              "print('{\"step\": 1, \"loss\": 0.1}')")
    assert wrapper.main(command_args(output, source)) == 0
    assert events(output) == [{"event_index": 1, "train/global_step": 1, "train/loss": 0.1}]


def test_real_pinned_offline_sdk_does_not_capture_private_inputs(tmp_path):
    """Inspect actual serialized SDK files, not only arguments to a mock."""
    assert importlib.metadata.version("wandb") == tracking.SDK_VERSION
    sentinels = ["SYNTHETIC_PRIVATE_CONFIG_68f7", "SYNTHETIC_PRIVATE_ENV_994c",
                 "SYNTHETIC_PRIVATE_ARGV_7ea2", "SYNTHETIC_PRIVATE_PAYLOAD_522a"]
    (tmp_path / "config-defaults.yaml").write_text(f"private_secret:\n  value: {sentinels[0]}\n")
    output = tmp_path / "tracking"
    env = dict(os.environ, WANDB_API_KEY=sentinels[1], WANDB_NOTES=sentinels[1],
               WANDB_CONFIG_PATHS=str(tmp_path / "config-defaults.yaml"))
    source = ("import json,sys; "
              f"print(json.dumps({{'step': 2, 'loss': 0.5, 'private': {sentinels[3]!r}}})); "
              "print(json.dumps({'step': 2, 'validation_velocity_mse': 0.4})); sys.exit(7)")
    completed = subprocess.run([sys.executable, str(REPO / "scripts/run_training_with_wandb.py"),
        *command_args(output, source), "--token", sentinels[2]], cwd=tmp_path, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=45)
    assert completed.returncode == 7, completed.stderr
    assert len(events(output)) == 2
    result = receipt(output)
    assert result["mode"] == "offline"
    assert result["cloud_synced"] is False
    assert result["tracking_errors"] == 0
    assert result["training_exit_code"] == 7
    saved = [path for path in output.rglob("*") if path.is_file()]
    assert any(path.suffix == ".wandb" for path in saved), "Real SDK offline run missing"
    for path in saved:
        content = path.read_bytes()
        for sentinel in sentinels:
            assert sentinel.encode() not in content, f"Private sentinel leaked into {path.name}"
    assert all(path.name != "wandb-metadata.json" for path in saved)
    assert all(path.name != "requirements.txt" for path in saved)


@pytest.mark.skipif(os.name != "posix", reason="Process-group signal behavior is POSIX-specific")
def test_signal_forwards_to_owned_child_group_and_preserves_interruption(tmp_path):
    """Only synthetic processes started here are ever signaled."""
    ready = tmp_path / "ready"
    child_pid_file = tmp_path / "child.pid"
    child_signal = tmp_path / "child.signal"
    grandchild_signal = tmp_path / "grandchild.signal"
    grandchild_code = (
        "import signal,time,sys; from pathlib import Path; "
        f"signal.signal(signal.SIGTERM, lambda s,f: (Path({str(grandchild_signal)!r}).write_text(str(s)), sys.exit(0))); "
        f"Path({str(ready)!r}).touch(); time.sleep(20)")
    child_code = (
        "import os,signal,subprocess,sys,time; from pathlib import Path; "
        f"signal.signal(signal.SIGTERM, lambda s,f: (Path({str(child_signal)!r}).write_text(str(s)), sys.exit(143))); "
        f"Path({str(child_pid_file)!r}).write_text(str(os.getpid())); "
        f"subprocess.Popen([sys.executable,'-c',{grandchild_code!r}]); time.sleep(20)")
    output = tmp_path / "tracking"
    process = subprocess.Popen([sys.executable, str(REPO / "scripts/run_training_with_wandb.py"),
        *command_args(output, child_code)], cwd=tmp_path, env=dict(os.environ),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    child_pid = None
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.05)
        assert ready.exists(), "Synthetic child group failed to become ready"
        child_pid = int(child_pid_file.read_text())
        assert os.getpgid(process.pid) == process.pid
        assert os.getpgid(child_pid) == child_pid
        assert child_pid != process.pid
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 143, (stdout, stderr)
        assert child_signal.read_text() == str(signal.SIGTERM)
        assert grandchild_signal.read_text() == str(signal.SIGTERM)
        assert receipt(output)["execution_status"] == "interrupted"
        assert receipt(output)["training_exit_code"] == 143
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
