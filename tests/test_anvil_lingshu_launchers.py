"""Portable launcher contracts; no Slurm, external network, or GPU is needed."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = {kind: ROOT / 'slurm' / f'anvil_{prefix}_lingshu_scdfm_expanded_v3.sbatch'
             for kind, prefix in [('setup', 'setup'), ('test', 'test'),
                                  ('train', 'h100'), ('compare', 'compare')]}


def directive(text, name):
    match = re.search(rf'^#SBATCH --{re.escape(name)}=(.+)$', text, re.MULTILINE)
    return match.group(1) if match else None


class AnvilLingshuLauncherTests(unittest.TestCase):
    def test_bash_syntax(self):
        for path in LAUNCHERS.values():
            with self.subTest(path=path.name):
                subprocess.run(['bash', '-n', str(path)], check=True, capture_output=True)

    def test_no_head_node_execution(self):
        for path in LAUNCHERS.values():
            with self.subTest(path=path.name):
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith('SLURM_') and not key.startswith('VCC_')}
                result = subprocess.run(['bash', str(path)], env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('SLURM_JOB_ID', result.stderr)
                self.assertEqual(result.stdout, '')

    def test_scope_paths_and_isolation(self):
        for path in LAUNCHERS.values():
            with self.subTest(path=path.name):
                text = path.read_text()
                self.assertIn('${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}', text)
                self.assertIn('${SLURM_JOB_NODELIST:?', text)
                self.assertIn('unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT LD_PRELOAD VCC_TOKEN', text)
                self.assertIn('PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1', text)
                self.assertIn('HF_HOME="$PROJECT_DIR/.cache/huggingface"', text)
                self.assertIn('XDG_CACHE_HOME="$PROJECT_DIR/.cache"', text)
                self.assertNotIn('/home/fs01/', text)
                self.assertNotIn('/mnt/beegfs/', text)
                self.assertNotIn('export PYTHONPATH=', text)
                self.assertNotIn('.venv-state', text)
                self.assertNotRegex(text, r'(?:vcc_pat_|--priority|--nice|--qos=high|--qos=priority)')

    def test_ai_resources_and_real_h100_probe(self):
        text = LAUNCHERS['train'].read_text()
        for key, value in {'account': 'bio250247-ai', 'partition': 'ai', 'nodes': '1',
                           'qos': 'ai',
                           'ntasks': '1', 'gpus-per-node': '1', 'cpus-per-task': '8',
                           'mem': '32G', 'time': '01:00:00'}.items():
            self.assertEqual(directive(text, key), value)
        self.assertIn('torch.cuda.is_available()', text)
        self.assertIn('torch.cuda.device_count() == 1', text)
        self.assertIn("assert 'H100' in name", text)
        self.assertIn("torch.ones((32, 32), device='cuda')", text)
        self.assertIn('torch.cuda.synchronize()', text)
        self.assertIn("'aida_checkpoint_resumed': False", text)
        self.assertIn('#SBATCH --no-requeue', text)

    def test_cpu_phases_are_allocation_internal_not_separate_jobs(self):
        for kind in ('setup', 'test', 'compare'):
            with self.subTest(kind=kind):
                text = LAUNCHERS[kind].read_text()
                self.assertNotIn('#SBATCH', text)
                self.assertIn('test "${VCC_ANVIL_EXPANDED_PIPELINE:-}" = 1', text)
                self.assertIsNone(directive(text, 'account'))
                self.assertIsNone(directive(text, 'gpus-per-node'))
                self.assertIsNone(directive(text, 'gres'))

    def test_frozen_hashes_match_tracked_code(self):
        for kind in ('setup', 'test', 'train', 'compare'):
            text = LAUNCHERS[kind].read_text()
            checks = re.findall(r'^  ([a-f0-9]{64}) ((?:scripts|requirements)/\S+) \\$', text, re.MULTILINE)
            self.assertTrue(checks)
            for expected, relative in checks:
                with self.subTest(kind=kind, path=relative):
                    self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)
        for kind in ('test', 'train'):
            text = LAUNCHERS[kind].read_text()
            self.assertIn('3be669b7fc0966ec49870ecc5927b606b5f2b2fc352f3d8e46d1b5409e4095b7 artifacts/lingshu_scdfm/public_cache_v1.npz', text)
            self.assertIn('39d04694b579776e8c5fa9d73097287de2927629e1a01e19dbfcb2fc26341e77 artifacts/lingshu_scdfm/public_cache_expanded_train_v3.npz', text)
            self.assertIn('66d9f299212472102d292cb4e94c69a39f5ed700cd407b4f0fa5ac51d438f4cb artifacts/lingshu_scdfm/v1/lingshu_embeddings.npz', text)
            self.assertIn('scripts/prepare_lingshu_scdfm_expanded_train.py --verify-existing', text)
            self.assertLess(text.index('sha256sum --check -'), text.index('--verify-existing'))

    def test_setup_is_pinned_and_preserves_existing_source(self):
        text = LAUNCHERS['setup'].read_text()
        self.assertIn('module load conda/2026.03', text)
        self.assertIn('conda create -y --prefix "$ENV_DIR" python=3.11', text)
        self.assertIn('pip install -r requirements/lingshu-scdfm-route.txt', text)
        self.assertIn('2cf6bca1f044e74c4e1dc586892c0495880cf125', text)
        self.assertIn('a04130b07020505a609158cbe31e9e65083c0d79', text)
        self.assertIn('--untracked-files=all --ignored=matching', text)
        self.assertNotRegex(text, r'git\s+.*(?:reset|clean)\s')
        self.assertIn('if [[ ! -e "$SOURCE_DIR" ]]; then', text)
        self.assertIn('git clone --no-checkout https://github.com/AI4Science-WestlakeU/scDFM.git', text)
        self.assertLess(text.index('git -C "$SOURCE_DIR" checkout'), text.index('test "$(git -C "$SOURCE_DIR" rev-parse HEAD)"'))
        self.assertIn('CONDA_PKGS_DIRS="$PROJECT_DIR/.cache/conda_pkgs"', text)
        self.assertIn('PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"', text)
        self.assertIn('test "${SLURM_GPUS_ON_NODE:-}" = 1', text)
        self.assertIn('test "${SLURM_CPUS_PER_TASK:-}" = 8', text)
        self.assertIn("versions['torch'].split('+')[0] == '2.7.1'", text)
        self.assertIn("assert 'H100' in torch.cuda.get_device_name(0)", text)
        self.assertGreater(text.count('unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT LD_PRELOAD VCC_TOKEN'), 1)

    def test_integration_cannot_pass_with_skipped_pad(self):
        text = LAUNCHERS['test'].read_text()
        self.assertIn('--junitxml="$TEST_REPORT"', text)
        self.assertIn("assert not root.findall('.//skipped')", text)
        self.assertIn('test_authentic_pad_gradient_and_translation_equivariance', text)
        for test in ('test_lingshu_scdfm_effect.py', 'test_check_lingshu_scdfm_public_quality.py',
                     'test_compare_lingshu_scdfm_effect_v2.py', 'test_prepare_lingshu_scdfm_expanded_train.py',
                     'test_anvil_lingshu_launchers.py'):
            self.assertIn(f'tests/{test}', text)

    def test_registered_recipe_separate_artifacts_and_strict_gate(self):
        text = LAUNCHERS['train'].read_text()
        self.assertIn('for VCC_FEATURE_MODE in true shuffled constant; do', text)
        self.assertIn('--steps 4000 --validate-every 400 --batch-size 32 --seed 20260909', text)
        self.assertIn('artifacts/lingshu_scdfm/anvil_effect_v3_expanded', text)
        self.assertIn('mkdir "$RUN_ROOT"', text)
        self.assertNotIn('mkdir -p "$RUN_ROOT"', text)
        self.assertIn('if [[ "$VCC_CHECK_STATUS" -ne 1 ]]; then exit "$VCC_CHECK_STATUS"; fi', text)
        self.assertTrue(text.rstrip().endswith('--output "$RUN_ROOT/true/public_quality_preflight.json"'))
        self.assertIn('trap compare_on_exit EXIT', text)
        self.assertIn('if [[ "$VCC_PIPELINE_STATUS" -ne 0 ]]; then exit "$VCC_PIPELINE_STATUS"; fi', text)
        self.assertIn('bash "$PROJECT_DIR/slurm/anvil_compare_lingshu_scdfm_expanded_v3.sbatch"', text)
        self.assertIn('--root artifacts/lingshu_scdfm/anvil_effect_v3_expanded', LAUNCHERS['compare'].read_text())
        for path in LAUNCHERS.values():
            text = path.read_text()
            self.assertNotRegex(text, r'(?:scripts/generate_|vcc submit|--skip-limit-check|--debug-test-features|--cpu\b)')

    def test_setup_and_test_complete_in_same_job_before_fitting(self):
        text = LAUNCHERS['train'].read_text()
        setup = text.index('bash "$PROJECT_DIR/slurm/anvil_setup_lingshu_scdfm_expanded_v3.sbatch"')
        test = text.index('bash "$PROJECT_DIR/slurm/anvil_test_lingshu_scdfm_expanded_v3.sbatch"')
        train = text.index('scripts/train_lingshu_scdfm_effect.py \\\n')
        self.assertLess(setup, test)
        self.assertLess(test, train)
        self.assertIn('export VCC_ANVIL_EXPANDED_PIPELINE=1', text)
        self.assertNotIn('sbatch ', text)


if __name__ == '__main__':
    unittest.main()
