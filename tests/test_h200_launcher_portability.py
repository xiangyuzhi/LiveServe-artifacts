"""Exercise launcher path checks without Python model imports or GPU commands."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = Path(
    "benchmarks/interactive/H200/run_original_baseline_vs_eviction_gpu45.sh"
)


class H200LauncherPortabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="liveserve launcher ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "repo with spaces"
        for relative in (LAUNCHER, Path("scripts/env_defaults.sh")):
            dest = self.root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / relative, dest)
        for relative in (
            "benchmarks/interactive/H200/trace_replay_driver.py",
            "benchmarks/interactive/H200/client_bargein.py",
            "benchmarks/interactive/H200/aggregate_capacity.py",
            "benchmarks/interactive/H200/aggregate_audio_rtf.py",
            "benchmarks/interactive/H200/trace/total_traces_part1.txt",
            "profiling/scheduler_analysis/scripts/scheduler_policy_common.sh",
            "vllm-omni/vllm_omni/benchmarks/audio_continuity.py",
            "deploy/H200/qwen3_omni_moe_multi_replicas_2gpu.yaml",
            "deploy/H200/qwen3_omni_moe_multi_replicas_2gpu_prefix_cache_only.yaml",
        ):
            dest = self.root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.touch()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.env = {
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "TS": "cpu_test",
            "MODEL": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        }
        self.marker = self.root / "executed-command"
        for name in ("python3", "vllm-omni", "nvidia-smi", "setsid", "curl"):
            self.executable(self.bin / name)

    def executable(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'#!/bin/sh\nprintf touched > "{self.marker}"\nexit 97\n')
        path.chmod(0o755)
        return path

    def run_check(self, overrides=None):
        env = self.env | (overrides or {})
        result = subprocess.run(
            ["/bin/bash", str(self.root / LAUNCHER), "--check-env"],
            env=env,
            cwd=self.tmp.name,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertFalse(self.marker.exists(), result.stdout + result.stderr)
        self.assertFalse((self.root / "results").exists())
        return result

    def test_path_commands_and_repo_relative_outputs_from_other_cwd(self):
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        values = dict(
            line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
        )
        self.assertEqual(values["PYBIN"], str(self.bin / "python3"))
        self.assertEqual(values["VLLM_BIN"], str(self.bin / "vllm-omni"))
        self.assertEqual(
            values["RESULT_ROOT"],
            str(
                self.root
                / "results/interactive_original/interactive_original_gpu45_eviction_cpu_test"
            ),
        )
        self.assertEqual(values["TEXT_GPUS"], "4,5")
        self.assertEqual(values["VARIANTS"], "baseline lru_control eviction_only")
        self.assertEqual(values["CONCURRENCY_LIST"], "8 16 24 32 48 64 96 128 160")
        self.assertEqual((values["TIME_START"], values["TIME_LIMIT"]), ("6780", "6840"))

    def test_repo_venv_and_sibling_cli_precede_path(self):
        python = self.executable(self.root / ".venv-liveserve/bin/python")
        cli = self.executable(python.parent / "vllm-omni")
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"PYBIN={python}\n", result.stdout)
        self.assertIn(f"VLLM_BIN={cli}\n", result.stdout)

    def test_explicit_overrides_and_spaces(self):
        python = self.executable(self.root / "custom env/bin/python")
        cli = self.executable(self.root / "custom env/bin/omni")
        model = self.root / "model weights"
        model.mkdir()
        result_root = self.root / "custom results"
        result = self.run_check(
            {
                "LIVESERVE_PYBIN": str(python),
                "PYBIN": "missing-python",
                "LIVESERVE_VLLM_BIN": str(cli),
                "VLLM_BIN": "missing-cli",
                "LIVESERVE_MODEL": str(model),
                "MODEL": "/missing-model",
                "EXP_ROOT": "/unused",
                "RESULT_ROOT": str(result_root),
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for key, value in (
            ("PYBIN", python),
            ("VLLM_BIN", cli),
            ("MODEL", model),
            ("RESULT_ROOT", result_root),
        ):
            self.assertIn(f"{key}={value}\n", result.stdout)
        self.assertFalse(result_root.exists())

    def test_plain_overrides_and_external_trace_directory(self):
        trace_dir = self.root / "external trace"
        trace_dir.mkdir()
        (trace_dir / "total_traces_part2.txt").touch()
        result = self.run_check(
            {
                "PYBIN": str(self.bin / "python3"),
                "VLLM_BIN": "vllm-omni",
                "TRACE_PARTS": "2",
                "LIVESERVE_TRACE_DIR": str(trace_dir),
                "EXP_ROOT": str(self.root / "external results"),
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"RESULT_ROOT={self.root}/external results/", result.stdout)

    def test_early_model_and_command_errors(self):
        for override, message in (
            ({"MODEL": ""}, "set LIVESERVE_MODEL or MODEL"),
            ({"MODEL": "/missing-model"}, "model directory not found"),
            ({"PYBIN": "missing-python"}, "python not found or not executable"),
            (
                {"VLLM_BIN": "missing-cli"},
                "vllm-omni (set LIVESERVE_VLLM_BIN or VLLM_BIN) not found",
            ),
        ):
            with self.subTest(override=override):
                result = self.run_check(override)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)

    def test_missing_inputs_fail_before_server_start(self):
        for override, message in (
            ({"TRACE_PARTS": "99"}, "trace file not found"),
            ({"TRACE_PARTS": "1,2"}, "space-separated part numbers"),
            ({"VARIANTS": "unknown"}, "unknown variant"),
            ({"BASELINE_DEPLOY_YAML": "/missing.yaml"}, "deploy yaml not found"),
        ):
            with self.subTest(override=override):
                result = self.run_check(override)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)

    def test_gpu_scope_validation_unchanged(self):
        result = self.run_check({"TEXT_GPUS": "0,1"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("TEXT_GPUS must be exactly 4,5 or 5,4", result.stderr)
        result = self.run_check({"CUDA_VISIBLE_DEVICES": "5,4"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("expected '4,5'", result.stderr)


if __name__ == "__main__":
    unittest.main()
