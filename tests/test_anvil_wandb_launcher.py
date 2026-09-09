"""Static and no-allocation guard tests; no Slurm, downloads or GPUs."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm/anvil_h100_tracked_training.sbatch"


def test_bash_syntax():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True, capture_output=True)


def test_refuses_head_node_before_any_work():
    env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
    result = subprocess.run(["bash", str(LAUNCHER), "pretrain", "true"], env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "SLURM_JOB_ID" in result.stderr


def test_stage_required_and_invalid_stage_precedes_setup():
    env = dict(os.environ, SLURM_JOB_ID="synthetic", SLURM_JOB_NODELIST="synthetic")
    for args in ([], ["wrong", "true"]):
        result = subprocess.run(["bash", str(LAUNCHER), *args], env=env,
                                capture_output=True, text=True)
        assert result.returncode == 2
        assert result.stdout == ""


def test_resources_isolation_and_no_automatic_experiment():
    source = LAUNCHER.read_text()
    for option in ("--partition=ai", "--qos=ai", "--account=bio250247-ai", "--ntasks=1",
                   "--gpus-per-node=1", "--mem=32G", "--cpus-per-task=8", "--time=01:00:00"):
        assert f"#SBATCH {option}" in source
    assert "torch.cuda.is_available()" in source
    assert "assert 'H100' in torch.cuda.get_device_name(0)" in source
    assert "torch.cuda.synchronize()" in source
    assert 'readonly TRACK_ENV="$PROJECT_DIR/.venv-tracking"' in source
    assert '"$TRACK_PYTHON" -m pip install' in source
    assert '"$TRAIN_PYTHON" -m pip install' not in source
    assert '${VCC_WANDB_MODE:-offline}' in source
    assert 'exec srun --ntasks=1 "$TRACK_PYTHON" scripts/run_training_with_wandb.py' in source
    assert '"${TRACK_ARGS[@]}" -- "$@"' in source
    assert '--summary-file "$VCC_TRAINING_PROVENANCE"' in source
    assert '--parent-checkpoint-sha256 "$VCC_PARENT_CHECKPOINT_SHA256"' in source
    assert 'eval ' not in source
    assert '--priority=' not in source
    assert 'scripts/train_lingshu_scdfm_effect.py' not in source
    assert 'vcc submit' not in source
