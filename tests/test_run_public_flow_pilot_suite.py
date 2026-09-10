"""Mocked launcher checks: no real datasets, GPUs, Slurm jobs, or W&B calls."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import run_public_flow_pilot_suite as suite


ARTIFACTS = {"cache": "cache.npz", "contract": "contract.json", "source-manifest": "sources.json",
             "target-features": "go/features.npz", "feature-receipt": "go/receipt.json"}


def argument(command, key):
    return command[command.index("--" + key) + 1]


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    root = tmp_path / "pilot"
    (root / "go").mkdir(parents=True)
    for key, relative in ARTIFACTS.items():
        (root / relative).write_bytes(("synthetic-" + key).encode())
    for key, value in {"SLURM_JOB_ID": "314159", "SLURM_JOB_NODELIST": "a001",
                       "SLURMD_NODENAME": "a001", "SLURM_STEP_ID": "0"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(socket, "gethostname", lambda: "a001")
    monkeypatch.setattr(socket, "getfqdn", lambda: "a001.example.test")
    def allocation_nodes(command, *, text):
        assert command == ["scontrol", "show", "hostnames", "a001"]
        assert text is True
        return "a001\n"
    monkeypatch.setattr(suite.subprocess, "check_output", allocation_nodes)
    tracking_python = tmp_path / "tracking-python"
    tracking_python.write_text("#!/bin/sh\nexit 99\n")
    tracking_python.chmod(0o700)
    return root, ["--pilot-dir", str(root), "--tracking-python", str(tracking_python)]


def write_mock_outputs(command, *, bad=None):
    marker = command.index("--")
    wrapper, training = command[:marker], command[marker + 1:]
    mode = argument(training, "feature-mode")
    output = Path(argument(training, "output-dir"))
    tracking = Path(argument(wrapper, "output-dir"))
    assert not output.exists() and not tracking.exists()
    output.mkdir()
    tracking.mkdir()
    weights = output / "weights.npz"
    weights.write_bytes(("synthetic weights " + mode).encode())
    pins = {name.replace("-", "_") + "_sha256": argument(training, name + "-sha256") for name in ARTIFACTS}
    configuration = {"steps": 200, "batch_size": 64, "hidden_size": 128, "layers": 2,
                     "learning_rate": 0.0003, "weight_decay": 0.0001, "seed": 20260909,
                     "integration_steps": 16, "shuffle_seed": 20260910, "feature_mode": mode}
    summary = {"schema": "vcc-public-flow-pilot-summary-v1", "status": "completed_unpromoted_pilot",
               "stage": "pretrain", "last_step": 200, "configuration": configuration,
               "weights_sha256": suite.sha(weights), "contract_sha256": pins["contract_sha256"],
               "source_manifest_sha256": pins["source_manifest_sha256"], "provenance": pins,
               "runtime": {"device": "cuda", "hostname": "a001", "slurm_job_id": "314159", "gpu_name": "NVIDIA H100 80GB HBM3"},
               "dev_used_for_fitting_or_selection": False, "checkpoint_selection": "fixed_final_step_only",
               "promotion_decision": "not_performed", "full_axis_submission_ready": False}
    record = {"schema": "vcc-training-tracking-v1", "run_id": hashlib.sha256(mode.encode()).hexdigest()[:12],
              "stage": "pretrain", "mode": "offline", "execution_status": "completed", "training_exit_code": 0,
              "tracking_errors": 0, "event_count": 200, "cloud_synced": False,
              "config": {**pins, **configuration}, "scientific_quality_inferred_from_exit": False}
    if bad == "wrong_mode":
        summary["configuration"]["feature_mode"] = "other"
    elif bad == "wrong_hash":
        summary["contract_sha256"] = "0" * 64
    elif bad == "short_training":
        summary["last_step"] = 2
    elif bad == "tracking_errors":
        record["tracking_errors"] = 1
    elif bad == "online_tracking":
        record["mode"] = "online"
        record["cloud_synced"] = True
    elif bad == "duplicate_run_id":
        record["run_id"] = "a" * 12
    elif bad == "wrong_weights":
        summary["weights_sha256"] = "0" * 64
    if bad != "missing_summary":
        (output / "summary.json").write_text(json.dumps(summary))
    if bad != "missing_tracking":
        (tracking / "tracking.json").write_text(json.dumps(record))
    (output / "steps.jsonl").write_text("".join(json.dumps({"step": i + 1, "train_loss": 1.0}) + "\n" for i in range(200)))
    (tracking / "metrics.jsonl").write_text("".join(json.dumps({"event_index": i + 1, "train/global_step": i + 1}) + "\n" for i in range(200)))
    return mode


def test_three_arms_sequential_fresh_authenticated_and_separately_tracked(pilot, monkeypatch):
    root, args = pilot
    calls, completed = [], []
    def fake_call(command):
        assert len(calls) == len(completed), "An arm started before the prior arm finished"
        calls.append(command)
        completed.append(write_mock_outputs(command))
        return 0
    monkeypatch.setattr(suite.subprocess, "call", fake_call)
    assert suite.main(args) == 0
    assert completed == ["true", "constant", "shuffled"]
    assert len({argument(c[:c.index("--")], "name") for c in calls}) == 3
    for call, mode in zip(calls, completed):
        marker = call.index("--")
        wrapper, training = call[:marker], call[marker + 1:]
        assert wrapper[0] == args[-1]
        assert training[0] == sys.executable
        assert argument(wrapper, "mode") == "offline"
        assert argument(wrapper, "stage") == argument(training, "stage") == "pretrain"
        assert argument(wrapper, "output-dir") == str(root / "tracking" / mode)
        assert argument(wrapper, "summary-file") == str(root / "runs" / mode / "summary.json")
        assert argument(training, "output-dir") == str(root / "runs" / mode)
        assert argument(training, "device") == "cuda"
        for name, relative in ARTIFACTS.items():
            assert argument(training, name) == str(root / relative)
            assert argument(training, name + "-sha256") == suite.sha(root / relative)
        for name, expected in {"steps": "200", "batch-size": "64", "hidden-size": "128", "layers": "2",
                               "seed": "20260909", "shuffle-seed": "20260910", "integration-steps": "16"}.items():
            assert argument(training, name) == expected
        assert not any("resume" in item or "checkpoint" in item for item in training)
    status = json.loads((root / "suite.json").read_text())
    assert status["completed"] is True
    assert status["cloud_synced"] is False
    assert status["promotion_performed"] is False and status["submission_performed"] is False


@pytest.mark.parametrize("existing", ["runs", "tracking", "suite.json", "dangling-runs"])
def test_existing_output_never_overwritten_or_reused(pilot, monkeypatch, existing):
    root, args = pilot
    if existing == "dangling-runs":
        (root / "runs").symlink_to(root / "absent")
    elif existing == "suite.json":
        (root / existing).write_text("preserve me")
    else:
        (root / existing).mkdir()
    monkeypatch.setattr(suite.subprocess, "call", lambda _: pytest.fail("Training started with stale output"))
    with pytest.raises(FileExistsError):
        suite.main(args)


@pytest.mark.parametrize("relative", list(ARTIFACTS.values()))
@pytest.mark.parametrize("kind", ["missing", "symlink"])
def test_all_inputs_checked_before_any_outputs_or_training(pilot, monkeypatch, relative, kind):
    root, args = pilot
    target = root / relative
    target.unlink()
    if kind == "symlink":
        backing = root / "backing"
        backing.write_bytes(b"untrusted")
        target.symlink_to(backing)
    monkeypatch.setattr(suite.subprocess, "call", lambda _: pytest.fail("Training started without valid inputs"))
    with pytest.raises((ValueError, FileNotFoundError)):
        suite.main(args)
    assert not (root / "runs").exists() and not (root / "tracking").exists()


@pytest.mark.parametrize("missing", ["SLURM_JOB_ID", "SLURM_JOB_NODELIST"])
def test_missing_allocation_refuses_before_hashing(pilot, monkeypatch, missing):
    root, args = pilot
    monkeypatch.delenv(missing)
    monkeypatch.setattr(suite, "sha", lambda _: pytest.fail("Artifact work happened on head node"))
    monkeypatch.setattr(suite.subprocess, "call", lambda _: pytest.fail("Training started on head node"))
    with pytest.raises(RuntimeError):
        suite.main(args)
    assert not (root / "runs").exists()


@pytest.mark.parametrize("failed_arm", [0, 1, 2])
def test_nonzero_arm_stops_suite_without_retry(pilot, monkeypatch, failed_arm):
    root, args = pilot
    modes = []
    def fake_call(command):
        mode = argument(command[command.index("--") + 1:], "feature-mode")
        index = len(modes)
        modes.append(mode)
        if index == failed_arm:
            return 23
        write_mock_outputs(command)
        return 0
    monkeypatch.setattr(suite.subprocess, "call", fake_call)
    assert suite.main(args) != 0
    assert modes == ["true", "constant", "shuffled"][:failed_arm + 1]
    status = json.loads((root / "suite.json").read_text())
    assert status["completed"] is False
    assert status["arms"][-1]["exit_code"] == 23


@pytest.mark.parametrize("bad", ["missing_summary", "missing_tracking", "wrong_mode", "wrong_hash",
                                "short_training", "tracking_errors", "online_tracking", "wrong_weights"])
def test_exit_zero_without_authentic_completion_evidence_fails(pilot, monkeypatch, bad):
    root, args = pilot
    calls = []
    def fake_call(command):
        calls.append(command)
        write_mock_outputs(command, bad=bad)
        return 0
    monkeypatch.setattr(suite.subprocess, "call", fake_call)
    try:
        result = suite.main(args)
    except (RuntimeError, ValueError, FileNotFoundError):
        result = 1
    assert result != 0, f"False success for {bad}"
    assert len(calls) == 1, "Invalid arm evidence must stop before another arm"
    if (root / "suite.json").exists():
        assert json.loads((root / "suite.json").read_text())["completed"] is False


def test_duplicate_tracking_run_id_is_not_three_independent_runs(pilot, monkeypatch):
    root, args = pilot
    calls = []
    def fake_call(command):
        calls.append(command)
        write_mock_outputs(command, bad="duplicate_run_id")
        return 0
    monkeypatch.setattr(suite.subprocess, "call", fake_call)
    try:
        result = suite.main(args)
    except (RuntimeError, ValueError):
        result = 1
    assert result != 0
    assert len(calls) == 2


def test_slurm_script_is_bounded_one_h100_and_syntax_valid():
    script = REPO / "slurm/anvil_public_flow_pilot.sbatch"
    text = script.read_text()
    for directive in ("--partition=ai", "--constraint=H100", "--nodes=1", "--ntasks=1",
                      "--gpus-per-node=1", "--time=00:20:00", "--no-requeue"):
        assert "#SBATCH " + directive in text
    assert "set -euo pipefail" in text
    assert "'H100' in torch.cuda.get_device_name(0)" in text
    assert "torch.cuda.device_count() == 1" in text
    assert "exec srun" in text
    assert "PIP_RETRIES=0" in text
    subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)


def test_slurm_shell_refuses_no_allocation_before_any_work(tmp_path):
    script = REPO / "slurm/anvil_public_flow_pilot.sbatch"
    env = {"PATH": os.environ["PATH"], "VCC_PROJECT_DIR": str(tmp_path), "VCC_TRAIN_PYTHON": "/nonexistent/model-python"}
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Submit through Slurm" in result.stderr
    assert not list(tmp_path.iterdir())


def test_inherited_slurm_environment_on_wrong_host_refused_before_hashing(pilot, monkeypatch):
    root, args = pilot
    monkeypatch.setattr(socket, "gethostname", lambda: "login01.example.test")
    monkeypatch.setattr(suite, "sha", lambda _: pytest.fail("Artifact work happened on login host"))
    monkeypatch.setattr(suite.subprocess, "call", lambda _: pytest.fail("Training started on login host"))
    with pytest.raises(RuntimeError, match="allocated compute node"):
        suite.main(args)
    assert not (root / "runs").exists()


def fake_allocation_tools(tmp_path, *, current_host):
    bin_dir = tmp_path / "mock-bin"
    bin_dir.mkdir()
    for name, body in {"hostname": f"printf '%s\\n' '{current_host}'",
                       "scontrol": "printf '%s\\n' 'a001'",
                       "srun": "exit 91"}.items():
        tool = bin_dir / name
        tool.write_text("#!/bin/sh\n" + body + "\n")
        tool.chmod(0o700)
    project = tmp_path / "project"
    project.mkdir()
    env = {"PATH": str(bin_dir) + ":" + os.environ["PATH"], "VCC_PROJECT_DIR": str(project),
           "VCC_TRAIN_PYTHON": "/nonexistent/model-python", "SLURM_JOB_ID": "314159", "SLURM_JOB_NODELIST": "a001"}
    return project, env


def test_slurm_shell_refuses_stale_allocation_on_login_host(tmp_path):
    project, env = fake_allocation_tools(tmp_path, current_host="login01")
    result = subprocess.run(["bash", str(REPO / "slurm/anvil_public_flow_pilot.sbatch")],
                            cwd=project, env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "outside the allocated compute node" in result.stderr
    assert not list(project.iterdir())


@pytest.mark.parametrize("kind", ["directory", "dangling_symlink"])
def test_tracking_environment_freshness_checked_before_srun_or_install(tmp_path, kind):
    project, env = fake_allocation_tools(tmp_path, current_host="a001")
    target = project / ".venv-tracking"
    if kind == "directory":
        target.mkdir()
    else:
        target.symlink_to(project / "missing")
    result = subprocess.run(["bash", str(REPO / "slurm/anvil_public_flow_pilot.sbatch")],
                            cwd=project, env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Tracking environment already exists" in result.stderr
    assert list(project.iterdir()) == [target]
