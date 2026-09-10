"""Synthetic-only bounded training and file-contract tests; never real data."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_public_flow_pilot as pilot  # noqa: E402
import build_go_target_features as go  # noqa: E402
import wandb_training  # noqa: E402
from public_flow_conditioning import FeatureTable  # noqa: E402

CONTRACT, SOURCES = "a" * 64, "b" * 64


@pytest.fixture(autouse=True)
def single_threaded_cpu():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def cache_arrays():
    rng = np.random.default_rng(771)
    arrays = {"schema": np.array(pilot.SCHEMA), "representation": np.array(pilot.REPRESENTATION),
              "contract_sha256": np.array(CONTRACT), "source_manifest_sha256": np.array(SOURCES),
              "genes": np.array(["G1", "G2", "G3", "G4"])}
    for prefix, sources, targets in (("train", ("K562", "K562", "H1", "H1"), ("A", "B", "A", "B")),
                                     ("dev", ("RPE1", "RPE1"), ("A", "B"))):
        controls, treated, context = [], [], []
        for group, target in enumerate(targets):
            control = (rng.uniform(1, 3, (16, 4)) + group * .1).astype(np.float32)
            destination = (rng.uniform(1, 3, (8, 4)) + group * .1 + (.2 if target == "A" else .5)).astype(np.float32)
            controls.append(control)
            treated.append(destination)
            context.append(np.concatenate((control.astype(np.float64).mean(0), control.astype(np.float64).std(0))))
        arrays.update({prefix + "_control": np.concatenate(controls),
                       prefix + "_control_group": np.repeat(np.arange(len(targets)), 16),
                       prefix + "_treated": np.concatenate(treated),
                       prefix + "_group": np.repeat(np.arange(len(targets)), 8),
                       prefix + "_targets": np.array(targets), prefix + "_sources": np.array(sources),
                       prefix + "_context": np.array(context, dtype=np.float32)})
    return arrays


def save_cache(tmp_path, arrays=None, name="cache.npz"):
    path = tmp_path / name
    np.savez(path, **(cache_arrays() if arrays is None else arrays))
    return path, pilot.sha256_file(path)


def get_cache(tmp_path, arrays=None, name="cache.npz"):
    path, digest = save_cache(tmp_path, arrays, name)
    return pilot.load_cache(path, digest, CONTRACT, SOURCES)


def registered_cache(tmp_path, profile="rpe1_dev"):
    """Tiny fake contracts: named raw sources intentionally do not exist."""
    arrays = cache_arrays()
    second = "nadig_jurkat" if profile == "h1_dev" else "vcc2025_h1_train"
    recipient = "vcc2025_h1_train" if profile == "h1_dev" else "replogle_rpe1"
    arrays["train_sources"] = np.array(["replogle_k562"] * 2 + [second] * 2)
    arrays["dev_sources"] = np.array([recipient] * 2)
    source_specs, registered = [], []
    for source, role in (("replogle_k562", "train"), (second, "train"), (recipient, "dev")):
        digest = lambda suffix: hashlib.sha256((source + suffix).encode()).hexdigest()
        spec = {"id": source, "role": role,
                "path": str(tmp_path / (source + "-MUST-NOT-OPEN.h5ad")),
                "sha256": digest("bytes"), "size_bytes": 123,
                "receipt_path": str(tmp_path / (source + "-MUST-NOT-OPEN.json")),
                "receipt_sha256": digest("receipt"), "receipt_name": source + ".h5ad",
                "control_label": "non-targeting"}
        source_specs.append(spec)
        registered.append({**spec, "metadata_sha256": digest("metadata"), "cells_total": 1000,
                           "groups": [{"target": target, "batch": f"batch-{j}",
                                       "control_rows": list(range(j * 16, (j + 1) * 16)),
                                       "treated_rows": list(range(200 + j * 8, 208 + j * 8))}
                                      for j, target in enumerate(("A", "B"))]})
    source_path = tmp_path / "sources.json"
    source_path.write_text(json.dumps({"schema": "public-flow-source-manifest-v1",
                                      "source_profile": profile, "sources": source_specs}))
    source_sha = pilot.sha256_file(source_path)
    contract = {"schema": "public-flow-row-contract-v1", "source_profile": profile,
                "seed": 20260909, "representation": pilot.REPRESENTATION,
                "recipient_treated_allowed": False, "challenge_2026_treated_allowed": False,
                "source_manifest_sha256": source_sha, "sources": registered,
                "shared_genes": arrays["genes"].tolist(), "output_genes": arrays["genes"].tolist(),
                "excluded_targets": ["DENIED"],
                "policy": {"max_targets_per_source": 16, "max_batches_per_target": 2, "min_cells": 8,
                           "max_controls_per_group": 64, "max_treated_per_group": 32,
                           "matching": "exact source and batch; labeled non-targeting only",
                           "genes": "unique shared symbols; fixed SHA256-ranked subset; no effect selection",
                           "normalization": "CP10000 over full shared measured axis, then log1p, then output subset",
                           "go": "direct primary terms; exact symbols; no IEA/ND/NOT/obsolete/alternate; fit vocabulary train only",
                           "optimizer": "fresh per-arm AdamW; no prior weights; fixed steps; no dev checkpoint selection",
                           "training": {"steps": 200, "batch_size": 64, "hidden_size": 128, "layers": 2,
                                        "learning_rate": 0.0003, "weight_decay": 0.0001, "gradient_clip_norm": 1.0,
                                        "seed": 20260909, "shuffle_seed": 20260910,
                                        "arms": ["true", "constant", "shuffled"],
                                        "sampler": "uniform source then target then matched batch; unpaired cells"},
                           "evaluation": {"scope": "development only; no score claim",
                                          "when": "fixed final step only", "heun_steps": 16, "clipping": False,
                                          "centroid_mse": "mean over modeled genes of squared population-mean difference",
                                          "variance_ratio": "mean predicted population gene variance / mean treated population gene variance",
                                          "mmd2": "biased RBF MMD; kernel exp(-mean gene squared distance / 2); bandwidth fixed at 1 on log1p CP10k",
                                          "scale": pilot.REPRESENTATION}}}
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    arrays["contract_sha256"] = np.array(pilot.sha256_file(contract_path))
    arrays["source_manifest_sha256"] = np.array(source_sha)
    cache_path, cache_sha = save_cache(tmp_path, arrays)
    cache = pilot.load_cache(cache_path, cache_sha, str(arrays["contract_sha256"]), source_sha)
    return cache, contract_path, source_path, contract, cache_path, cache_sha


@pytest.mark.parametrize("profile", ["rpe1_dev", "h1_dev"])
def test_registered_source_profiles_are_metadata_only(tmp_path, profile):
    cache, contract_path, source_path, contract, _, _ = registered_cache(tmp_path, profile)
    result = pilot.validate_registration(cache, pilot.PilotConfig(), contract_path, source_path)
    assert result["source_profile"] == profile
    assert result["registration_authenticated"] is True
    assert result["raw_source_files_opened_by_trainer"] is False
    assert result["source_roles"]["vcc2025_h1_train"] == ("dev" if profile == "h1_dev" else "train")
    assert all(not Path(source["path"]).exists() for source in contract["sources"])


@pytest.mark.parametrize("change", ["gene-order", "normalization", "min-cells-bool", "role", "source-sha",
                                    "excluded-target", "group-count", "control-count", "row-duplicate", "row-overlap",
                                    "steps", "learning-rate", "gradient-clip", "evaluation", "recipient-allowed",
                                    "profile", "source-order", "policy-type", "training-type"])
def test_changed_registration_contract_rejected_before_training(tmp_path, change):
    cache, contract_path, source_path, contract, _, _ = registered_cache(tmp_path)
    if change == "gene-order":
        contract["output_genes"].reverse()
    elif change == "normalization":
        contract["policy"]["normalization"] = "normalize over modeled genes only"
    elif change == "min-cells-bool":
        contract["policy"]["min_cells"] = True
    elif change == "role":
        contract["sources"][2]["role"] = "train"
    elif change == "source-sha":
        contract["sources"][0]["sha256"] = "f" * 64
    elif change == "excluded-target":
        contract["excluded_targets"].append("A")
    elif change == "group-count":
        contract["sources"][0]["groups"].pop()
    elif change == "control-count":
        contract["sources"][0]["groups"][0]["control_rows"].pop()
    elif change == "row-duplicate":
        contract["sources"][0]["groups"][0]["control_rows"][1] = 0
    elif change == "row-overlap":
        contract["sources"][0]["groups"][0]["treated_rows"] = list(range(8))
    elif change in {"steps", "learning-rate", "gradient-clip"}:
        key = {"steps": "steps", "learning-rate": "learning_rate", "gradient-clip": "gradient_clip_norm"}[change]
        contract["policy"]["training"][key] *= 2
    elif change == "evaluation":
        contract["policy"]["evaluation"]["when"] = "select best checkpoint"
    elif change == "recipient-allowed":
        contract["recipient_treated_allowed"] = True
    elif change == "profile":
        contract["source_profile"] = "unknown"
    elif change == "source-order":
        contract["sources"].reverse()
    elif change == "policy-type":
        contract["policy"] = []
    elif change == "training-type":
        contract["policy"]["training"] = []
    contract_path.write_text(json.dumps(contract))
    cache = replace(cache, contract_sha256=pilot.sha256_file(contract_path))
    with pytest.raises(ValueError):
        pilot.validate_registration(cache, pilot.PilotConfig(), contract_path, source_path)


@pytest.mark.parametrize("changes", [{"steps": 1}, {"batch_size": 8}, {"integration_steps": 2},
                                    {"gradient_clip_norm": .5}, {"shuffle_seed": 1}])
def test_actual_cli_config_must_equal_registered_defaults(tmp_path, changes):
    cache, contract_path, source_path, _, _, _ = registered_cache(tmp_path)
    with pytest.raises(ValueError, match="hyperparameters|evaluation"):
        pilot.validate_registration(cache, pilot.PilotConfig(**changes), contract_path, source_path)


def test_contract_and_manifest_files_are_both_authenticated(tmp_path):
    cache, contract_path, source_path, _, _, _ = registered_cache(tmp_path)
    for path in (contract_path, source_path):
        original = path.read_text()
        path.write_text(original + "\n")
        with pytest.raises(ValueError, match="SHA-256"):
            pilot.validate_registration(cache, pilot.PilotConfig(), contract_path, source_path)
        path.write_text(original)


@pytest.mark.parametrize("text", ['{"key": 1, "key": 2}', '{"value": NaN}', '[]'])
def test_authenticated_json_is_strict(tmp_path, text):
    path = tmp_path / "input.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        pilot.authenticated_json(path, pilot.sha256_file(path))


def test_authenticated_payload_is_not_reopened_for_parsing(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text('{"value": 1}')
    source_sha = pilot.sha256_file(source)
    arrays = tmp_path / "arrays.npz"
    np.savez(arrays, example=np.arange(3))
    arrays_sha = pilot.sha256_file(arrays)
    original = pilot.authenticated_bytes
    def authenticate_then_replace(path, expected, maximum):
        payload = original(path, expected, maximum)
        path.write_bytes(b"replaced after authentication")
        return payload
    monkeypatch.setattr(pilot, "authenticated_bytes", authenticate_then_replace)
    assert pilot.authenticated_json(source, source_sha) == {"value": 1}
    np.testing.assert_array_equal(pilot.load_npz(arrays, arrays_sha, {"example"})["example"], np.arange(3))


def test_declared_npy_allocation_is_checked_before_numpy_load(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(buffer, {"descr": "<f4", "fortran_order": False, "shape": (2 ** 50,)})
    path = tmp_path / "forged.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("example.npy", buffer.getvalue())
    monkeypatch.setattr(pilot.np, "load", lambda *a, **kw: pytest.fail("Unsafe array reached NumPy allocation"))
    with pytest.raises(ValueError, match="declared allocation"):
        pilot.load_npz(path, pilot.sha256_file(path), {"example"})


def test_authenticated_bytes_rejects_oversize_input_and_symlink(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="bounded"):
        pilot.authenticated_bytes(path, pilot.sha256_file(path), 1)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="non-symlink"):
        pilot.authenticated_json(link, pilot.sha256_file(path))


def feature_table():
    return FeatureTable("go", ("A", "B"), ("GO:0000001", "GO:0000002"),
                        np.eye(2), "synthetic-GO", "c" * 64)


def go_artifacts(tmp_path):
    index = go.AnnotationIndex({"A": frozenset({"GO:0000001"}), "B": frozenset({"GO:0000002"})},
                               {"A": ("UniProtKB:P1",), "B": ("UniProtKB:P2",)}, {}, {},
                               {"ontology_data_version": "synthetic", "ancestor_policy": "direct_only", "include_iea": False})
    result = go.build_features(index, target_ids=("A", "B"), train_target_ids=("A", "B"),
                               fold_id="synthetic", sources={"gaf_sha256": "a" * 64,
                               "obo_sha256": "b" * 64, "acquisition_sha256": "c" * 64})
    out = tmp_path / "go"
    receipt = go.write_artifacts(result, out)
    return out / "features.npz", receipt["npz_sha256"], out / "receipt.json", pilot.sha256_file(out / "receipt.json")


def test_cache_contract_and_matched_control_context_verified(tmp_path):
    cache = get_cache(tmp_path)
    assert cache.genes == ("G1", "G2", "G3", "G4")
    assert set(cache.train.sources) == {"K562", "H1"} and set(cache.dev.sources) == {"RPE1"}
    assert cache.contract_sha256 == CONTRACT


@pytest.mark.parametrize("change", ["schema", "representation", "contract", "sources-hash", "gene-duplicates",
                                    "float64", "nan", "negative", "context-treated", "too-few-controls",
                                    "too-few-treated", "group-negative", "group-outside", "group-bool",
                                    "recipient-in-train", "unseen-dev-target", "object-array", "extra-field"])
def test_bad_or_leaking_cache_rejected(tmp_path, change):
    arrays = cache_arrays()
    if change in {"schema", "representation"}:
        arrays[change] = np.array("wrong")
    elif change == "contract":
        arrays["contract_sha256"] = np.array("c" * 64)
    elif change == "sources-hash":
        arrays["source_manifest_sha256"] = np.array("c" * 64)
    elif change == "gene-duplicates":
        arrays["genes"][1] = arrays["genes"][0]
    elif change == "float64":
        arrays["train_control"] = arrays["train_control"].astype(np.float64)
    elif change in {"nan", "negative"}:
        arrays["train_treated"][0, 0] = np.nan if change == "nan" else -1
    elif change == "context-treated":
        arrays["dev_context"][0, :4] = arrays["dev_treated"][:8].mean(0)
    elif change in {"too-few-controls", "too-few-treated"}:
        name, labels = ("control", "control_group") if change.endswith("controls") else ("treated", "group")
        keep = arrays["train_" + labels] != 0
        keep[:7] = True
        arrays["train_" + name], arrays["train_" + labels] = arrays["train_" + name][keep], arrays["train_" + labels][keep]
    elif change in {"group-negative", "group-outside"}:
        arrays["train_group"][0] = -1 if change.endswith("negative") else 99
    elif change == "group-bool":
        arrays["train_group"] = arrays["train_group"].astype(bool)
    elif change == "recipient-in-train":
        arrays["dev_sources"] = np.array(["K562", "K562"])
    elif change == "unseen-dev-target":
        arrays["dev_targets"] = np.array(["A", "UNKNOWN"])
    elif change == "object-array":
        arrays["genes"] = np.array(["G1", "G2", "G3", "G4"], dtype=object)
    elif change == "extra-field":
        arrays["surprise"] = np.array(1)
    path, digest = save_cache(tmp_path, arrays)
    with pytest.raises(ValueError):
        pilot.load_cache(path, digest, CONTRACT, SOURCES)


def test_wrong_cache_sha_and_input_symlink_rejected(tmp_path):
    path, digest = save_cache(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        pilot.load_cache(path, "0" * 64, CONTRACT, SOURCES)
    link = tmp_path / "link.npz"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="non-symlink"):
        pilot.load_cache(link, digest, CONTRACT, SOURCES)


def test_go_export_authenticates_and_preserves_training_vocabulary(tmp_path):
    paths = go_artifacts(tmp_path)
    table = pilot.load_go_table(*paths, ("A", "B", "A"))
    assert table.target_ids == ("A", "B")
    np.testing.assert_array_equal(table.values, np.eye(2))
    with pytest.raises(ValueError, match="training IDs"):
        pilot.load_go_table(*paths, ("A", "OTHER"))


@pytest.mark.parametrize("change", ["receipt-sha", "npz-sha", "fingerprint", "vocabulary-scope", "present", "binary"])
def test_go_provenance_or_masks_cannot_drift(tmp_path, change):
    path, digest, receipt_path, receipt_digest = go_artifacts(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    if change == "receipt-sha":
        receipt_digest = "0" * 64
    elif change == "npz-sha":
        digest = "0" * 64
    elif change in {"fingerprint", "vocabulary-scope"}:
        receipt["feature_table_fingerprint" if change == "fingerprint" else "vocabulary_fit_scope"] = "bad"
        receipt_path.write_text(json.dumps(receipt))
        receipt_digest = pilot.sha256_file(receipt_path)
    else:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if change == "present":
            arrays["present"][0] = False
        else:
            arrays["features"][0, 0] = 2
        np.savez(path, **arrays)
        digest = pilot.sha256_file(path)
        receipt["npz_sha256"] = digest
        receipt["npz_size_bytes"] = path.stat().st_size
        receipt_path.write_text(json.dumps(receipt))
        receipt_digest = pilot.sha256_file(receipt_path)
    with pytest.raises(ValueError):
        pilot.load_go_table(path, digest, receipt_path, receipt_digest, ("A", "B"))


def test_sampler_matches_populations_and_balances_sources(tmp_path):
    cache = get_cache(tmp_path)
    sampler = pilot.BalancedSampler(cache.train)
    groups, ci, ti = sampler.draw(np.random.default_rng(18), 4000)
    np.testing.assert_array_equal(cache.train.control_group[ci], groups)
    np.testing.assert_array_equal(cache.train.group[ti], groups)
    source = np.array(cache.train.sources)[groups]
    assert .47 < np.mean(source == "H1") < .53
    targets = np.array(cache.train.targets)[groups]
    assert .47 < np.mean(targets == "A") < .53
    # Control/treated cell indices are independently drawn, not paired by row.
    assert len(np.unique(ci[groups == 0])) == 16
    assert len(np.unique(ti[groups == 0])) == 8


def test_sampler_does_not_overweight_sources_or_targets_with_more_batches(tmp_path):
    split = get_cache(tmp_path).train
    extra_groups = np.arange(4, 8)
    split = replace(split, control=np.concatenate([split.control] + [split.control[-16:]] * 4),
                    control_group=np.concatenate([split.control_group, np.repeat(extra_groups, 16)]),
                    treated=np.concatenate([split.treated] + [split.treated[-8:]] * 4),
                    group=np.concatenate([split.group, np.repeat(extra_groups, 8)]),
                    targets=split.targets + ("B",) * 4, sources=split.sources + ("H1",) * 4,
                    context=np.vstack([split.context] + [split.context[-1:]] * 4))
    groups, _, _ = pilot.BalancedSampler(split).draw(np.random.default_rng(59), 4000)
    sources, targets = np.array(split.sources)[groups], np.array(split.targets)[groups]
    assert .47 < np.mean(sources == "H1") < .53
    assert .45 < np.mean(targets[sources == "H1"] == "A") < .55


@pytest.mark.parametrize("mode", ["true", "constant", "shuffled"])
def test_feature_arm_shapes_masks_and_no_dev_refitting(tmp_path, mode):
    cache = get_cache(tmp_path)
    condition, transform, covered = pilot.target_conditions(cache, feature_table(), pilot.PilotConfig(feature_mode=mode))
    assert condition["train"].shape == (4, 4) and condition["dev"].shape == (2, 4)
    assert covered == 2
    np.testing.assert_array_equal(condition["train"][:, -2:], [[1, 0]] * 4)
    np.testing.assert_array_equal(condition["dev"][0], condition["train"][0])
    if mode == "constant":
        np.testing.assert_array_equal(condition["train"][:, :2], 0)
    assert transform.provenance()["fold_id"] == CONTRACT


def test_variance_centroid_and_registered_biased_mmd_diagnostics():
    rng = np.random.default_rng(22)
    control = rng.normal(size=(16, 4)).astype(np.float32)
    treated = control + .4
    metrics = pilot.diagnostics(treated, control, treated)
    assert metrics["model_centroid_mse"] == 0
    assert metrics["control_centroid_mse"] == pytest.approx(.16)
    assert metrics["variance_ratio"] == pytest.approx(1)
    assert metrics["model_mmd2"] == pytest.approx(0, abs=1e-12)
    reference = lambda a, b: np.exp(-np.mean((a[:, None, :] - b[None, :, :]) ** 2, axis=2) / 2).mean()
    expected_mmd = reference(control, control) + reference(treated, treated) - 2 * reference(control, treated)
    assert metrics["control_mmd2"] == pytest.approx(expected_mmd, abs=1e-7)
    assert 1 <= metrics["covariance_effective_rank"] <= 4
    assert metrics["duplicate_fraction"] == 0
    degenerate = pilot.diagnostics(np.ones((8, 2)), np.ones((8, 2)), np.ones((8, 2)))
    assert degenerate["variance_ratio"] is None
    assert degenerate["covariance_effective_rank"] == 0
    assert degenerate["duplicate_fraction"] == .875


def test_tiny_training_emits_every_step_saves_safe_weights_and_no_promotion(tmp_path, capsys):
    cache = get_cache(tmp_path)
    config = pilot.PilotConfig(steps=3, batch_size=8, hidden_size=8, layers=1, integration_steps=2)
    out = tmp_path / "run"
    result = pilot.train_pilot(cache, feature_table(), config, out)
    journal = [json.loads(line) for line in (out / "steps.jsonl").read_text().splitlines()]
    assert [row["step"] for row in journal] == [1, 2, 3]
    assert all(np.isfinite(row["loss"]) and row["gradient_norm"] > 0 for row in journal)
    emitted = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["step"] for row in emitted] == [1, 2, 3, 3]
    assert all("train/loss" in wandb_training.scalar_metrics(row) for row in emitted[:3])
    assert "validation/context/model_mmd2" in wandb_training.scalar_metrics(emitted[-1])
    assert "validation/context/negative_fraction" in wandb_training.scalar_metrics(emitted[-1])
    with np.load(out / "weights.npz", allow_pickle=False) as weights:
        assert all(not weights[key].dtype.hasobject for key in weights.files)
        assert np.any(weights["model.output.weight"] != 0)
    assert pilot.sha256_file(out / "weights.npz") == result["weights_sha256"]
    assert result["dev_used_for_fitting_or_selection"] is False
    assert result["count_emitter"] is False and result["full_axis_submission_ready"] is False
    assert result["checkpoint_selection"] == "fixed_final_step_only"
    assert result["promotion_decision"] == "not_performed"
    with pytest.raises(FileExistsError):
        pilot.train_pilot(cache, feature_table(), config, out)


def test_changing_dev_treated_values_cannot_change_fitted_weights(tmp_path):
    cache = get_cache(tmp_path)
    changed = replace(cache, dev=replace(cache.dev, treated=cache.dev.treated + 50))
    config = pilot.PilotConfig(steps=2, batch_size=8, hidden_size=8, layers=1, integration_steps=1)
    first = pilot.train_pilot(cache, feature_table(), config, tmp_path / "one")
    second = pilot.train_pilot(changed, feature_table(), config, tmp_path / "two")
    assert first["weights_sha256"] == second["weights_sha256"]
    assert first["target_transform"] == second["target_transform"]
    assert first["mmd_bandwidth_squared"] == second["mmd_bandwidth_squared"]
    assert first["deployment_rollout_diagnostics"] != second["deployment_rollout_diagnostics"]


def test_all_three_arms_share_initialization_and_training_rng(tmp_path, monkeypatch):
    cache = get_cache(tmp_path)
    original_loss = pilot.flow_matching_loss
    runs = []
    def record(model, control, treated, target, context, mask, *, time):
        runs[-1].append({"control": control.clone(), "treated": treated.clone(), "time": time.clone(),
                         "initial": {key: value.clone() for key, value in model.state_dict().items()} if not runs[-1] else None})
        return original_loss(model, control, treated, target, context, mask, time=time)
    monkeypatch.setattr(pilot, "flow_matching_loss", record)
    for mode in ("true", "constant", "shuffled"):
        runs.append([])
        config = pilot.PilotConfig(steps=2, batch_size=8, hidden_size=8, layers=1, integration_steps=1, feature_mode=mode)
        pilot.train_pilot(cache, feature_table(), config, tmp_path / mode)
    for other in runs[1:]:
        for key in runs[0][0]["initial"]:
            assert torch.equal(runs[0][0]["initial"][key], other[0]["initial"][key])
        for step in range(2):
            for key in ("control", "treated", "time"):
                assert torch.equal(runs[0][step][key], other[step][key])


def test_registered_defaults_and_mean_gene_distance_kernel():
    config = pilot.PilotConfig()
    assert (config.steps, config.batch_size, config.hidden_size, config.layers) == (200, 64, 128, 2)
    assert (config.learning_rate, config.weight_decay) == (3e-4, 1e-4)
    assert config.gradient_clip_norm == 1.0
    assert (config.seed, config.shuffle_seed, config.integration_steps) == (20260909, 20260910, 16)
    x, y = np.arange(16).reshape(8, 2).astype(float), np.arange(16).reshape(8, 2).astype(float) + .4
    assert pilot.rbf_mmd2(x, y) == pytest.approx(pilot.rbf_mmd2(np.tile(x, (1, 3)), np.tile(y, (1, 3))))


@pytest.mark.parametrize("changes", [{"steps": 0}, {"steps": 1001}, {"batch_size": True},
                                       {"hidden_size": 257}, {"integration_steps": 0},
                                       {"learning_rate": float("nan")}, {"learning_rate": 1},
                                       {"weight_decay": -1}, {"weight_decay": float("nan")},
                                       {"gradient_clip_norm": -1}, {"gradient_clip_norm": float("nan")},
                                       {"feature_mode": "other"}, {"seed": -1}, {"seed": True}])
def test_invalid_or_unbounded_training_config_rejected(changes):
    with pytest.raises(ValueError):
        pilot.PilotConfig(**changes)


def test_synthetic_cli_hash_binding_and_head_node_guard(tmp_path, monkeypatch):
    cache, contract_path, source_path, _, cache_path, cache_sha = registered_cache(tmp_path, "h1_dev")
    features, features_sha, receipt, receipt_sha = go_artifacts(tmp_path)
    args = ["--cache", str(cache_path), "--cache-sha256", cache_sha,
            "--target-features", str(features), "--target-features-sha256", features_sha,
            "--feature-receipt", str(receipt), "--feature-receipt-sha256", receipt_sha,
            "--contract", str(contract_path), "--source-manifest", str(source_path),
            "--contract-sha256", cache.contract_sha256,
            "--source-manifest-sha256", cache.source_manifest_sha256,
            "--output-dir", str(tmp_path / "run")]
    calls = []
    monkeypatch.setattr(pilot, "train_pilot", lambda *positional, **keywords: calls.append((positional, keywords)))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(pilot.socket, "gethostname", lambda: "aida")
    with pytest.raises(RuntimeError, match="head node"):
        pilot.main(args)
    monkeypatch.setattr(pilot.socket, "gethostname", lambda: "cbsuvlaminck3.biohpc.cornell.edu")
    assert pilot.main(args) == 0
    assert len(calls) == 1 and calls[0][0][2] == pilot.PilotConfig()
    assert calls[0][1]["provenance"]["cache_sha256"] == cache_sha
    assert calls[0][1]["provenance"]["target_features_sha256"] == features_sha
    assert calls[0][1]["provenance"]["registration_authenticated"] is True
    assert calls[0][1]["provenance"]["source_roles"]["vcc2025_h1_train"] == "dev"
    assert calls[0][1]["device"] == "cpu"
    with pytest.raises(ValueError, match="hyperparameters"):
        pilot.main(args + ["--steps", "1"])
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="allocated H100"):
        pilot.main(args + ["--device", "cuda"])
    with pytest.raises(SystemExit):
        pilot.main(args + ["--stage", "posttrain"])
