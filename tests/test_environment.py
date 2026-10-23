"""Exercise interpreter resolution independently of the developer's machine."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class InterpreterResolutionTests(unittest.TestCase):
    def resolve(self, root, **overrides):
        env = {k: v for k, v in os.environ.items() if k not in ("LIVESERVE_PYBIN", "PYBIN_FALLBACK")}
        env.update(overrides)
        return subprocess.check_output(
            ["bash", "-c", 'source "$1"; liveserve_resolve_pybin "$2"', "env-test",
             str(ROOT / "scripts/env_defaults.sh"), str(root)], env=env, text=True).strip()

    @staticmethod
    def executable(path):
        path.parent.mkdir(parents=True)
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)

    def test_explicit_interpreter_wins(self):
        with tempfile.TemporaryDirectory(prefix="liveserve path ") as directory:
            root = Path(directory)
            self.executable(root / ".venv-liveserve/bin/python")
            self.assertEqual(self.resolve(root, LIVESERVE_PYBIN="/chosen path/python"), "/chosen path/python")

    def test_local_serving_then_cpu_environment(self):
        with tempfile.TemporaryDirectory(prefix="liveserve path ") as directory:
            root = Path(directory)
            self.executable(root / ".venv-liveserve-dev/bin/python")
            self.assertEqual(self.resolve(root), str(root / ".venv-liveserve-dev/bin/python"))
            self.executable(root / ".venv-liveserve/bin/python")
            self.assertEqual(self.resolve(root), str(root / ".venv-liveserve/bin/python"))

    def test_portable_fallback_when_environment_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.resolve(directory), "python3")
            self.assertEqual(self.resolve(directory, PYBIN_FALLBACK="python3.12"), "python3.12")


if __name__ == "__main__":
    unittest.main()
