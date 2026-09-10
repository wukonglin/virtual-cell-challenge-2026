"""Synthetic-only diagnostic registration, reconstruction and metric tests."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_public_flow_pilot as diag
import train_public_flow_pilot as pilot
from public_crispri_flow import ConditionalPublicFlow, FlowConfig
from test_train_public_flow_pilot import registered_cache, go_artifacts


@pytest.fixture(autouse=True)
def single_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.fixture
def completed(tmp_path):
    cache, contract, sources, _, cache_path, cache_sha = registered_cache(tmp_path, "h1_dev")
    feature_path, feature_sha, receipt_path, receipt_sha = go_artifacts(tmp_path)
    table = pilot.load_go_table(feature_path, feature_sha, receipt_path, receipt_sha, cache.train.targets)
    source_inputs = {"cache": cache_sha, "contract": cache.contract_sha256, "source-manifest": cache.source_manifest_sha256,
                     "target-features": feature_sha, "feature-receipt": receipt_sha}
    module = lambda name: pilot.sha256_file(Path(diag.__file__).with_name(name))
    provenance = {"cache_sha256": cache_sha, "target_features_sha256": feature_sha, "feature_receipt_sha256": receipt_sha,
                  "entrypoint_sha256": module("train_public_flow_pilot.py"), "flow_module_sha256": module("public_crispri_flow.py"),
                  "conditioning_module_sha256": module("public_flow_conditioning.py"), "go_builder_sha256": module("build_go_target_features.py")}
    complete = tmp_path / "completed"
    outcomes = []
    for index, mode in enumerate(diag.ARMS):
        run = complete / "runs" / mode
        track = complete / "tracking" / mode
        run.mkdir(parents=True)
        track.mkdir(parents=True)
        config = pilot.PilotConfig(feature_mode=mode)
        conditions, transform, coverage = pilot.target_conditions(cache, table, config)
        model_cfg = FlowConfig(len(cache.genes), conditions["train"].shape[1], len(cache.genes)*2, config.hidden_size, config.layers)
        model = ConditionalPublicFlow(model_cfg)
        with torch.no_grad():
            model.output.bias.fill_(.01)
        # Fake synthetic completion, not an actual model-training claim.
        weights_sha = pilot.save_weights(run / "weights.npz", model, transform)
        groups, aggregate = pilot.evaluate(model, cache.dev, conditions["dev"], device="cpu", seed=20260910, steps=16)
        summary = {"schema": "vcc-public-flow-pilot-summary-v1", "status": "completed_unpromoted_pilot", "stage": "pretrain",
                   "last_step": 200, "configuration": asdict(config), "model_config": asdict(model_cfg),
                   "weights_sha256": weights_sha, "contract_sha256": cache.contract_sha256,
                   "source_manifest_sha256": cache.source_manifest_sha256, "representation": pilot.REPRESENTATION,
                   "runtime": {"slurm_job_id": "20539599"}, "dev_used_for_fitting_or_selection": False,
                   "checkpoint_selection": "fixed_final_step_only", "provenance": provenance, "output_gene_count": len(cache.genes),
                   "target_transform": transform.provenance(), "go_targets_present": coverage, "dev_groups": groups}
        (run / "summary.json").write_text(json.dumps(summary))
        tracking = {"execution_status": "completed", "training_exit_code": 0, "tracking_errors": 0, "mode": "offline",
                    "cloud_synced": False, "event_count": 202, "run_id": f"{index:012x}"}
        (track / "tracking.json").write_text(json.dumps(tracking))
        outcomes.append({"arm": mode, "exit_code": 0, "summary_sha256": pilot.sha256_file(run / "summary.json"),
                         "tracking_receipt_sha256": pilot.sha256_file(track / "tracking.json"), "run_id": tracking["run_id"]})
    suite = {"schema": "public-flow-pilot-suite-v1", "completed": True, "stage": "pretrain", "slurm_job_id": "20539599",
             "promotion_performed": False, "submission_performed": False, "tracking_mode": "offline", "cloud_synced": False,
             "arms": outcomes, "input_sha256": source_inputs, "contract_sha256": cache.contract_sha256, "cache_sha256": cache_sha}
    (complete / "suite.json").write_text(json.dumps(suite))
    return tmp_path, complete, cache, table


def registration(completed):
    root, complete, _, _ = completed
    path = root / "diagnostic_contract.json"
    digest = diag.register(root, complete, path)
    return path, digest


def test_registration_never_decodes_npz(completed, monkeypatch):
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("Registration decoded numeric arrays"))
    path, digest = registration(completed)
    receipt = pilot.authenticated_json(path, digest)
    assert receipt["registration_npz_decoded"] is False
    assert receipt["method"]["sham"].endswith("not a trained NTC condition")
    with pytest.raises(FileExistsError):
        diag.register(completed[0], completed[1], path)


def test_complete_frozen_replay(completed):
    path, digest = registration(completed)
    output = completed[0] / "diagnostics.json"
    result = diag.run(path, digest, output)
    assert result["new_fitting_performed"] is False
    assert result["zero_velocity_identity"]["maximum_absolute_coordinate_difference"] == 0
    assert result["tracking_rollout_arm"] == "true"
    assert len(result["arms"]) == 3
    for arm in result["arms"].values():
        assert arm["original_dev_replay"]["passed"]
        assert arm["ood_zero_target_sham"]["not_a_trained_ntc_condition"]
        assert arm["ood_zero_target_sham"]["all_target_features_including_masks_zeroed"]
    with pytest.raises(FileExistsError):
        diag.run(path, digest, output)


@pytest.mark.parametrize("change", ["cache", "weights", "summary", "suite", "tracking"])
def test_tampered_inputs_rejected_before_cache_decode(completed, monkeypatch, change):
    path, digest = registration(completed)
    entry = json.loads(path.read_text())["inputs"][change if change in {"cache", "suite"} else "true/" + change]
    with Path(entry["path"]).open("ab") as stream:
        stream.write(b"tampered")
    monkeypatch.setattr(pilot, "load_cache", lambda *a, **kw: pytest.fail("Arrays decoded before identities verified"))
    with pytest.raises(ValueError, match="SHA-256"):
        diag.run(path, digest, completed[0] / "result.json")


def test_changed_methods_and_wrong_contract_sha_rejected(completed, monkeypatch):
    path, digest = registration(completed)
    monkeypatch.setattr(pilot, "load_cache", lambda *a, **kw: pytest.fail("Unexpected cache load"))
    with pytest.raises(ValueError, match="SHA-256"):
        diag.run(path, "a"*64, completed[0] / "result.json")
    content = json.loads(path.read_text())
    content["method"]["heun_steps"] = 32
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="methods"):
        diag.run(path, pilot.sha256_file(path), completed[0] / "result.json")


@pytest.mark.parametrize("field", ["target_mean", "target_scale", "model.output.bias"])
def test_weight_transform_and_finiteness_validation(completed, field):
    _, complete, cache, table = completed
    summary = json.loads((complete / "runs/true/summary.json").read_text())
    weight_path = complete / "runs/true/weights.npz"
    with np.load(weight_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays[field].flat[0] = np.nan if field.startswith("model.") else 123
    np.savez(weight_path, **arrays)
    with pytest.raises(ValueError, match="mean/scale|finiteness"):
        diag.restore_model(cache, table, summary, weight_path, pilot.sha256_file(weight_path))


@pytest.mark.parametrize("field", ["model_config", "target_transform", "output_gene_count"])
def test_invalid_architecture_and_transform_identity(completed, field):
    _, complete, cache, table = completed
    summary = json.loads((complete / "runs/true/summary.json").read_text())
    summary[field] = None
    weight_path = complete / "runs/true/weights.npz"
    with pytest.raises(ValueError, match="architecture|provenance"):
        diag.restore_model(cache, table, summary, weight_path, pilot.sha256_file(weight_path))


def test_displacement_exact_alignment_and_error_decomposition():
    controls = np.arange(24).reshape(8,3) / 10
    treated = controls + np.array([.1, .2, -.3])
    metrics, predicted, observed = diag.displacement_metrics(treated, controls, treated)
    assert metrics["displacement_cosine"] == pytest.approx(1)
    assert metrics["displacement_norm_ratio"] == pytest.approx(1)
    assert metrics["model_centroid_mse"] == 0
    assert metrics["variance_ratio_to_control"] == pytest.approx(1)
    assert metrics["mse_excess_decomposition"] == pytest.approx(-metrics["control_centroid_mse"])
    identity, _, _ = diag.displacement_metrics(controls, controls, treated)
    assert identity["displacement_cosine"] is None
    assert identity["predicted_shift_rms"] == 0


def test_variance_denominators_negative_outputs_and_zero_variance():
    control = np.array([[0., 1.], [1., 2.], [2., 3.]])
    predicted = 2 * control - 4
    metrics, _, _ = diag.displacement_metrics(predicted, control, np.ones_like(control))
    assert metrics["variance_ratio_to_control"] == pytest.approx(4)
    assert metrics["variance_ratio_to_treated"] is None
    assert metrics["negative_fraction"] == pytest.approx(3/6)
    assert metrics["minimum_prediction"] == -4


def test_common_shift_uses_equal_target_weights_not_batch_counts():
    rows = [{"source": "A", "target": "one"}, {"source": "A", "target": "one"}, {"source": "A", "target": "two"}]
    vectors = [np.array([1.,0.]), np.array([1.,0.]), np.array([-1.,0.])]
    result = diag.shared_shift(rows, vectors, vectors)["A"]["predicted"]
    assert result["common_shift_energy_fraction"] == 0
    assert result["mean_pairwise_target_cosine"] == -1


def test_sham_uses_disjoint_ntcs_and_zeros_masks(completed, monkeypatch):
    _, _, cache, table = completed
    features, _, _ = pilot.target_conditions(cache, table, pilot.PilotConfig())
    seen = []
    def rollout(model, controls, targets, context):
        assert np.all(targets == 0)
        seen.append(controls.copy())
        return controls.copy()
    monkeypatch.setattr(diag, "_rollout", rollout)
    # Poisoning treated values must not affect a controls-only stress test.
    changed = replace(cache.dev, treated=np.full_like(cache.dev.treated, np.nan))
    result = diag.diagnose_split(None, changed, features["dev"], sham=True)
    assert len(seen) == len(cache.dev.targets)
    assert result["aggregate"]["predicted_shift_rms"] == 0


def test_mismatched_replay_is_not_silently_accepted():
    with pytest.raises(ValueError, match="reproduce"):
        diag.replay_check([{"model_centroid_mse": .2}], {"dev_groups": [{"model_centroid_mse": .1}]})


def test_cli_refuses_head_node_before_arrays(monkeypatch):
    monkeypatch.setattr(socket := diag.socket, "gethostname", lambda: "login02.anvil.rcac.purdue.edu")
    monkeypatch.setattr(diag, "run", lambda *a, **kw: pytest.fail("Ran on login node"))
    with pytest.raises(RuntimeError, match="compute hosts"):
        diag.main(["run", "--diagnostic-contract", "absent", "--diagnostic-contract-sha256", "a"*64, "--output", "absent"])
