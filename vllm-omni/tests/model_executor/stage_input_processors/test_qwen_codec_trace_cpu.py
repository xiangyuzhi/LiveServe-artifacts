# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise real codec trace writers without importing the GPU runtime."""

from __future__ import annotations

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


def production_writers(logger):
    source = Path(__file__).resolve().parents[3] / "vllm_omni/model_executor/stage_input_processors/qwen3_omni.py"
    names = {"_append_code2wav_trace", "_maybe_trace_code2wav_chunk", "_maybe_trace_code2wav_process"}
    tree = ast.parse(source.read_text(), filename=str(source))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != len(names):
        raise AssertionError("Production codec trace functions are missing")
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = {"os": os, "json": json, "logger": logger}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


class CodecTraceTests(unittest.TestCase):
    def setUp(self):
        self.logger = Mock()
        self.writers = production_writers(self.logger)
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_disabled_trace_does_not_touch_filesystem(self):
        for kind in ("chunk", "process"):
            with self.subTest(kind=kind), patch("builtins.open") as opened, patch.object(os, "makedirs") as mkdir:
                os.environ[f"OMNI_CODEC_{kind.upper()}_TRACE_PATH"] = " \t "
                self.writers[f"_maybe_trace_code2wav_{kind}"]({"z": 1})
                opened.assert_not_called()
                mkdir.assert_not_called()
        self.logger.debug.assert_not_called()

    def test_separate_paths_append_sorted_utf8_json_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {kind: Path(directory) / kind / "trace.jsonl" for kind in ("chunk", "process")}
            for kind, path in paths.items():
                os.environ[f"OMNI_CODEC_{kind.upper()}_TRACE_PATH"] = f" {path} "
            self.writers["_maybe_trace_code2wav_chunk"]({"z": 1, "a": "音频"})
            self.writers["_maybe_trace_code2wav_chunk"]({"z": 2})
            self.writers["_maybe_trace_code2wav_process"]({"event": "process"})
            self.assertEqual(paths["chunk"].read_text(), '{"a": "音频", "z": 1}\n{"z": 2}\n')
            self.assertEqual(paths["process"].read_text(), '{"event": "process"}\n')
        self.logger.debug.assert_not_called()

    def test_io_failure_is_swallowed_and_preserves_each_log_message(self):
        with tempfile.TemporaryDirectory() as directory:
            for kind in ("chunk", "process"):
                with self.subTest(kind=kind):
                    os.environ[f"OMNI_CODEC_{kind.upper()}_TRACE_PATH"] = directory
                    self.writers[f"_maybe_trace_code2wav_{kind}"]({"z": 1})
                    self.logger.debug.assert_called_with(f"Failed to write Code2Wav {kind} trace", exc_info=True)

    def test_serialization_failure_is_swallowed_without_partial_record(self):
        with tempfile.TemporaryDirectory() as directory:
            for kind in ("chunk", "process"):
                with self.subTest(kind=kind):
                    path = Path(directory) / f"{kind}.jsonl"
                    os.environ[f"OMNI_CODEC_{kind.upper()}_TRACE_PATH"] = str(path)
                    self.writers[f"_maybe_trace_code2wav_{kind}"]({"unsupported": object()})
                    self.assertEqual(path.read_bytes(), b"")
                    self.logger.debug.assert_called_with(f"Failed to write Code2Wav {kind} trace", exc_info=True)


if __name__ == "__main__":
    unittest.main()
