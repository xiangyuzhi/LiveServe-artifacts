# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only: python tests/profile/test_interaction_trace.py (no vLLM/torch)."""

import ast
import importlib.util
import json
import os
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT / "vllm_omni/profiler/interaction_trace.py"
ADAPTER = ROOT / "vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py"
SCHEDULER = ROOT / "vllm_omni/core/sched/omni_ar_scheduler.py"


def load_trace(directory):
    with patch.dict(os.environ, {"OMNI_INTERACTION_TRACE_DIR": directory}):
        spec = importlib.util.spec_from_file_location("isolated_interaction_trace", TRACE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def events(directory):
    return [json.loads(line) for path in Path(directory).glob("*.jsonl") for line in path.read_text().splitlines()]


class InteractionTraceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.modules = []

    def tearDown(self):
        for module in self.modules:
            module.shutdown()
        self.tmp.cleanup()

    def trace(self, directory=None):
        module = load_trace(self.tmp.name if directory is None else directory)
        self.modules.append(module)
        return module

    def test_disabled_does_not_start_thread_or_inspect_payload(self):
        module = self.trace("")
        with patch.object(module, "_TraceWriter", side_effect=AssertionError("writer started")):
            module.emit("disabled", payload=object())
        self.assertFalse(module.enabled())
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_threads_have_unique_sequence_and_reserved_fields_are_protected(self):
        module = self.trace()
        workers = [threading.Thread(target=lambda: [module.emit("sample", pid=-1, seq=-1) for _ in range(100)])
                   for _ in range(4)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        self.assertTrue(module.flush())
        module.shutdown()
        rows = [row for row in events(self.tmp.name) if row["event"] == "sample"]
        self.assertEqual([row["seq"] for row in rows], list(range(1, 401)))
        self.assertTrue(all(row["pid"] == os.getpid() and row["monotonic_ns"] > 0 for row in rows))
        self.assertEqual(module.status()["dropped_total"], 0)
        self.assertEqual(events(self.tmp.name)[-1]["written"], 400)

    def test_never_converts_tensor_like_objects_and_bounds_fields(self):
        module = self.trace()

        class TensorLike:
            def __repr__(self):
                raise AssertionError("must not stringify tensors")

            def __float__(self):
                raise AssertionError("must not copy tensor scalar")

        module.emit("safe", payload=TensorLike(), long_string="x" * 5000,
                    values=list(range(1000)), invalid=float("nan"))
        self.assertTrue(module.flush())
        row = next(row for row in events(self.tmp.name) if row["event"] == "safe")
        self.assertEqual(row["payload"], "<omitted>")
        self.assertEqual(len(row["long_string"]), 512)
        self.assertEqual(len(row["values"]), 32)
        self.assertIsNone(row["invalid"])

    def test_queue_overflow_is_nonblocking_and_counted(self):
        module = self.trace()
        with patch.object(module.threading.Thread, "start"):
            writer = module._TraceWriter(self.tmp.name, 2)
        for index in range(7):
            writer.emit("overflow", {"index": index})
        self.assertEqual(writer.queue.qsize(), 2)
        self.assertEqual(writer.dropped, 5)
        writer.stop.set()
        writer._run()
        summary = events(self.tmp.name)[-1]
        self.assertEqual((summary["attempted"], summary["written"], summary["dropped_total"]), (7, 2, 5))

    def test_unwritable_destination_does_not_escape_to_inference(self):
        bad_directory = Path(self.tmp.name) / "is-a-file"
        bad_directory.write_text("x")
        module = self.trace(str(bad_directory))
        module.emit("first")
        module._writer.thread.join(2)
        module.emit("second")
        self.assertIsNotNone(module.status()["error"])
        self.assertEqual(module.status()["dropped_total"], 2)

    @unittest.skipUnless(hasattr(os, "fork"), "fork is a Unix engine start mode")
    def test_fork_creates_independent_pid_file_without_inherited_lock(self):
        module = self.trace()
        module.emit("parent")
        self.assertTrue(module.flush())
        child_pid = os.fork()
        if child_pid == 0:
            try:
                module.emit("child")
                success = module.flush()
                module.shutdown()
                os._exit(0 if success else 1)
            except BaseException:
                os._exit(2)
        _, status = os.waitpid(child_pid, 0)
        self.assertEqual(status, 0)
        self.assertEqual(len(list(Path(self.tmp.name).glob("*.jsonl"))), 2)
        child = next(row for row in events(self.tmp.name) if row["event"] == "child")
        self.assertEqual(child["pid"], child_pid)
        self.assertEqual(child["seq"], 1)

    def test_actual_adapter_enqueue_and_send_share_job_and_connector_key(self):
        # Execute the real method bodies with small CPU collaborators; importing
        # the entire adapter would unnecessarily require torch/vLLM here.
        tree = ast.parse(ADAPTER.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OmniChunkTransferAdapter")
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"save_async", "_send_single_request", "_observe_first_chunk_published"}]
        rows = []
        sent = []
        namespace = {
            "trace_emit": lambda event, **fields: rows.append({"event": event, **fields}),
            "trace_enabled": lambda: True, "time": time,
            "unflatten_payload": lambda value: value,
            "torch": SimpleNamespace(Tensor=type("FakeTensor", (), {})),
            "Request": object, "logger": SimpleNamespace(debug=lambda *args: None, error=lambda *args: None),
        }
        # Postponed annotations preserve the original union annotations.
        module_ast = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
                               + methods, type_ignores=[])
        exec(compile(ast.fix_missing_locations(module_ast), str(ADAPTER), "exec"), namespace)
        request = SimpleNamespace(request_id="engine-id", external_req_id="client-id", is_finished=lambda: False)
        payload = {"meta": {"finished": False}, "codes": {"audio": [1, 2]}}
        owner = SimpleNamespace(
            custom_process_next_stage_input_func=lambda **kwargs: payload,
            _interaction_job_seq=0, _interaction_trace_context={"stage_id": 1, "replica_id": None},
            _pending_save_reqs=[], _save_cond=threading.Condition(), put_req_chunk={"client-id": 0},
            connector=SimpleNamespace(stage_id=1, put=lambda **kwargs: (sent.append(kwargs) or True, 128, {})),
        )
        owner._observe_first_chunk_published = lambda *args: namespace["_observe_first_chunk_published"](owner, *args)
        namespace["save_async"](owner, pooling_output={}, request=request)
        task = owner._pending_save_reqs.pop()
        namespace["_send_single_request"](owner, task)
        self.assertEqual([row["event"] for row in rows], ["connector_enqueue", "connector_send_begin", "connector_send_end"])
        self.assertEqual({row["job_id"] for row in rows}, {"connector-1"})
        self.assertEqual(rows[-1]["connector_key"], "client-id_1_0")
        self.assertEqual(rows[-1]["size_bytes"], 128)
        self.assertIs(sent[0]["data"], payload)
        self.assertEqual(owner.put_req_chunk["client-id"], 1)
        # Disabling tracing leaves the original task and connector payloads intact.
        namespace["trace_enabled"] = lambda: False
        namespace["save_async"](owner, pooling_output={}, request=request)
        self.assertEqual(set(owner._pending_save_reqs[0]), {"pooling_output", "request", "is_finished"})

    def test_actual_scheduler_tracing_preserves_selection_and_deferred_restore(self):
        tree = ast.parse(SCHEDULER.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OmniARScheduler")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "schedule")
        protection = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_bp_first_chunk_protection")

        @dataclass
        class Output:
            num_scheduled_tokens: dict
            scheduled_new_reqs: list = field(default_factory=list)

        @dataclass
        class OmniOutput(Output):
            finished_requests_needing_kv_transfer: dict = field(default_factory=dict)

        class Base:
            def schedule(self):
                return Output({request.request_id: 1 for request in self.running})

        rows = []
        namespace = {
            "__name__": "isolated_scheduler", "__package__": "isolated_scheduler",
            "Base": Base, "SchedulerOutput": Output,
            "OmniSchedulerOutput": OmniOutput, "monotonic_ns": time.monotonic_ns,
            "trace_enabled": lambda: True, "trace_emit": lambda event, **fields: rows.append({"event": event, **fields}),
            "init_logger": lambda name: SimpleNamespace(exception=lambda *args: None),
        }
        # A class declaration supplies the __class__ closure needed by super().
        extracted = ast.ClassDef(name="ExtractedScheduler", bases=[ast.Name(id="Base", ctx=ast.Load())],
                                 keywords=[], body=[method, protection], decorator_list=[])
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, extracted], type_ignores=[])), str(SCHEDULER), "exec"), namespace)

        def owner():
            result = namespace["ExtractedScheduler"]()
            # Non-selected policy branches keep their production false defaults.
            for name in {
                "_bp_role_prefill_priority", "_bp_role_include_chunk_waiting_in_pending",
                "_bp_role_pressure_fast_warmup_arrival_window_s", "_bp_role_pressure_fast_warmup_include_running_arrivals",
                "_bp_role_pressure_fast_warmup_arrival_history", "_bp_kv_aware",
                "_bp_role_waiting_short_max_output_tokens", "_bp_role_waiting_short_min_count",
                "_bp_role_waiting_short_entry_only", "_bp_role_skip_global_suppressed_defer",
            }:
                setattr(result, name, False)
            result._interaction_job_seq = 0
            result._interaction_trace_context = {"stage_id": 1, "replica_id": None}
            result._omni_cancel_prefetch_under_live_pressure = lambda: None
            result._omni_release_expired_gpu_pins = lambda: None
            result._bp_pending_age_s = lambda: 0.25
            result._bp_recent_waiting_arrivals = lambda *args, **kwargs: 1
            result._bp_governor_role = "talker"
            result._bp_video_mix_active = lambda: False
            result._talker_lead_governor = SimpleNamespace(enabled=True, filter_running=lambda running, **kwargs: running[:1])
            result.chunk_transfer_adapter = None
            result.running = [SimpleNamespace(request_id=name, external_req_id="external-" + name, num_output_tokens=50)
                              for name in ("deferred", "selected")]
            result.waiting = [object()]
            result.requests = {request.request_id: request for request in result.running}
            return result

        traced = owner()
        traced_output = traced.schedule()
        namespace["trace_enabled"] = lambda: False
        baseline = owner()
        baseline_output = baseline.schedule()
        self.assertEqual(traced_output, baseline_output)
        self.assertEqual(traced_output.num_scheduled_tokens, {"selected": 1})
        self.assertEqual([request.request_id for request in traced.running], ["selected", "deferred"])
        self.assertEqual([request.request_id for request in traced.running], [request.request_id for request in baseline.running])
        defer = next(row for row in rows if row["event"] == "scheduler_defer")
        self.assertEqual((defer["request_id"], defer["reason"]), ("deferred", "lead_governor"))
        self.assertEqual(traced_output._interaction_job_id, "scheduler-1")


if __name__ == "__main__":
    unittest.main()
