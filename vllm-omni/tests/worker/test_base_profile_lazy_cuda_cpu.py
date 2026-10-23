"""Exercise the production profile method against a CPU-only lazy vLLM parent.

Extracting this method avoids importing torch/vLLM GPU runtime during CPU CI;
the code under test is compiled directly from worker/base.py, not reimplemented.
The parent contract matches vLLM 0.20 Worker.profile(is_start, profile_prefix).
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class Recorder:
    def __init__(self):
        self.events = []

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")


class OmniTorchRecorder(Recorder):
    def set_trace_filename(self, filename):
        self.events.append(("filename", filename))


class LazyParent:
    def profile(self, is_start=True, profile_prefix=None):
        self.parent_calls.append((is_start, profile_prefix))
        if self.profiler_config is None or self.profiler_config.profiler is None:
            raise RuntimeError("Profiling is not enabled")
        if is_start:
            if self.profiler is None:
                self.profiler = Recorder()
                self.constructed += 1
            self.profiler.start()
        elif self.profiler is not None:
            self.profiler.stop()


def production_worker():
    source = Path(__file__).resolve().parents[2] / "vllm_omni/worker/base.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OmniGPUWorkerBase")
    method = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "profile")
    # Compiling a class preserves the real zero-argument super() closure.
    cls = ast.ClassDef(name="OmniGPUWorkerBase", bases=[ast.Name(id="LazyParent", ctx=ast.Load())],
                       keywords=[], body=[method], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = dict(LazyParent=LazyParent, time=SimpleNamespace(time=lambda: 1234567.0))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["OmniGPUWorkerBase"]


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.worker_class = production_worker()
        stub = ModuleType("vllm_omni.profiler")
        stub.OmniTorchProfilerWrapper = OmniTorchRecorder
        self.modules = patch.dict(sys.modules, {"vllm_omni.profiler": stub})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def worker(self, kind):
        worker = self.worker_class()
        config = SimpleNamespace(profiler=kind) if kind is not None else None
        worker.vllm_config = SimpleNamespace(profiler_config=config, model_config=SimpleNamespace(stage_id=1))
        worker.profiler_config = config
        worker.profiler = None
        worker.rank = 3
        worker.parent_calls = []
        worker.constructed = 0
        return worker

    def test_cuda_lazily_constructs_once_and_forwards_start_stop(self):
        worker = self.worker("cuda")
        worker.profile(True, "first")
        recorder = worker.profiler
        worker.profile(False, "end")
        worker.profile(True, "second")
        self.assertIs(worker.profiler, recorder)
        self.assertEqual(worker.constructed, 1)
        self.assertEqual(recorder.events, ["start", "stop", "start"])
        self.assertEqual(worker.parent_calls, [(True, "first"), (False, "end"), (True, "second")])

    def test_cuda_stop_before_start_is_parent_noop(self):
        worker = self.worker("cuda")
        worker.profile(False)
        self.assertIsNone(worker.profiler)
        self.assertEqual(worker.constructed, 0)
        self.assertEqual(worker.parent_calls, [(False, None)])

    def test_disabled_profile_still_raises_omni_configuration_error(self):
        worker = self.worker(None)
        for start in (True, False):
            with self.subTest(start=start), self.assertRaisesRegex(RuntimeError, "For omni models"):
                worker.profile(start)
        worker.vllm_config.profiler_config = SimpleNamespace(profiler=None)
        with self.assertRaisesRegex(RuntimeError, "For omni models"):
            worker.profile()
        self.assertEqual(worker.parent_calls, [])

    def test_torch_preserves_custom_wrapper_filename_and_stop(self):
        worker = self.worker("torch")
        worker.profiler = OmniTorchRecorder()
        worker.profile(True)
        worker.profile(False)
        worker.profile(True, "custom")
        self.assertEqual(worker.profiler.events,
            [("filename", "stage1_rank3_1234567"), "start", "stop", ("filename", "custom"), "start"])
        self.assertEqual(worker.parent_calls, [])


if __name__ == "__main__":
    unittest.main()
