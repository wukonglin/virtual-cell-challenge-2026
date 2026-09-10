"""Synthetic admission/execution checks; no real cells, fitting or W&B."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_public_effect_reliability as runner
from public_reliability_tracking import ReliabilityTracker
from test_wandb_training import FakeSDK


@pytest.fixture
def registration(tmp_path, monkeypatch):
    pilot = tmp_path / "pilot"
    for relative in runner.FILES.values():
        path = pilot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"opaque bytes; not a NumPy archive")
    expected = {name + "_sha256": runner.sha256_file(pilot / relative) for name, relative in runner.FILES.items()}
    monkeypatch.setattr(runner, "EXPECTED", expected)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Synthetic fixed diagnostic protocol")
    output = tmp_path / "registration"
    digest = runner.register(pilot, protocol, output)
    return pilot, output, digest


def test_registration_does_not_decode_opaque_expression(registration):
    _, output, digest = registration
    record = runner.validate_registration(output / "contract.json", digest)
    assert record["registration_expression_decoded"] is False
    assert record["model_fitting_allowed"] is False
    assert record["new_raw_expression_allowed"] is False
    assert record["automatic_promotion"] is False


def test_changed_input_rejected(registration):
    pilot, output, digest = registration
    (pilot / "cache.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Input bytes"):
        runner.validate_registration(output / "contract.json", digest)


@pytest.mark.parametrize("field,value", [("model_fitting_allowed", True), ("new_raw_expression_allowed", True),
    ("automatic_promotion", True), ("post_hoc_development_audit", False), ("stage", "pretrain")])
def test_contract_role_flags_cannot_be_relaxed(registration, field, value):
    _, output, _ = registration
    path = output / "contract.json"
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="policy"):
        runner.validate_registration(path, runner.sha256_file(path))


def test_protocol_changed_after_registration_is_rejected(registration):
    _, output, digest = registration
    record = json.loads((output / "contract.json").read_text())
    Path(record["protocol"]["path"]).write_text("changed selection rules")
    with pytest.raises(ValueError, match="SHA"):
        runner.validate_registration(output / "contract.json", digest)


def test_code_pin_changed_is_rejected(registration):
    _, output, _ = registration
    path = output / "contract.json"
    record = json.loads(path.read_text())
    record["code_sha256"][runner.MODULES[0]] = "a" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Implementation"):
        runner.validate_registration(path, runner.sha256_file(path))


def synthetic_report():
    aggregate = {key: 0.1 for key in runner.METRICS}
    def sources():
        return [{"source": source, "aggregate": dict(aggregate)} for source in sorted(runner.SOURCES)]
    return {"aggregate": aggregate, "sources": sources(), "seed_summaries": [
        {"seed": seed, "aggregate": dict(aggregate), "sources": sources()} for seed in range(20260920, 20260952)]}


@pytest.mark.parametrize("mutation", ["seed", "missing_source", "nan", "negative_energy", "bad_agreement"])
def test_invalid_statistics_rejected(mutation):
    report = synthetic_report()
    if mutation == "seed":
        report["seed_summaries"][0]["seed"] += 100
    elif mutation == "missing_source":
        report["sources"].pop()
    elif mutation == "nan":
        report["aggregate"]["real_half_crossdot"] = float("nan")
    elif mutation == "negative_energy":
        report["aggregate"]["real_group_energy"] = -0.1
    else:
        report["aggregate"]["real_half_agreement"] = 1.1
    with pytest.raises(ValueError):
        runner.verify_report(report)


def test_zero_energy_agreement_is_undefined_not_zero():
    report = synthetic_report()
    report["aggregate"]["real_half_agreement"] = None
    report["aggregate"]["real_half_crossdot"] = -0.2
    runner.verify_report(report)


def test_fake_slurm_does_not_bypass_head_node_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "login03.anvil")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setattr(runner, "load_cache", lambda *a, **k: pytest.fail("Must not decode"))
    with pytest.raises(RuntimeError, match="compute host"):
        runner.run(tmp_path / "missing", "a" * 64, tmp_path / "output")


@pytest.mark.parametrize("tracking_failure", [False, True])
def test_synthetic_run_has_35_offline_events_no_model(registration, tmp_path, monkeypatch, tracking_failure):
    _, registration_dir, digest = registration
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "cbsuvlaminck3.biohpc.cornell.edu")
    calls = []
    def load(*args):
        calls.append(args)
        return {"synthetic_only": True}, {"excluded_targets": list(range(772))}
    monkeypatch.setattr(runner, "load_cache", load)
    monkeypatch.setattr(runner.statistics, "compute_reliability", lambda values: copy.deepcopy(synthetic_report()))
    sdk = FakeSDK(log_error=tracking_failure)
    monkeypatch.setattr(runner, "ReliabilityTracker", lambda *args, **kwargs: ReliabilityTracker(*args, sdk=sdk, **kwargs))
    output = tmp_path / "run"
    if tracking_failure:
        with pytest.raises(ValueError, match="W&B"):
            runner.run(registration_dir / "contract.json", digest, output)
        assert not (output / "complete.json").exists()
    else:
        assert runner.run(registration_dir / "contract.json", digest, output) == 0
        complete = json.loads((output / "complete.json").read_text())
        assert complete["tracking_events"] == 35 and complete["tracking_errors"] == 0
        assert complete["model_fitting_performed"] is False
        assert complete["prior_v2_screen_changed"] is False
    assert len(calls) == 1 and len(sdk.run.logged) == 35
    assert sdk.initialized[0]["mode"] == "offline"
    assert [values["analysis/event_index"] for values, _ in sdk.run.logged] == list(range(1, 36))
