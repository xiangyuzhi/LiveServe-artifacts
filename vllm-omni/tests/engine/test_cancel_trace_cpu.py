"""Exercise actual cancellation methods with CPU-only control-plane doubles."""

from __future__ import annotations

import ast
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

ROOT = Path(__file__).resolve().parents[2] / "vllm_omni"


def load_methods(path, class_name, names, events, *, tracing=True, base=object):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    cls.body = [n for n in cls.body if getattr(n, "name", None) in names]
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    cls.decorator_list = []
    selected = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[]
    )
    namespace = {
        "Base": base,
        "trace_enabled": lambda: tracing,
        "emit": lambda event, **fields: events.append({"event": event, **fields}),
        "logger": logging.getLogger(__name__),
    }
    exec(compile(ast.fix_missing_locations(selected), str(ROOT / path), "exec"), namespace)
    return namespace[class_name]()


class CancelTraceTests(unittest.IsolatedAsyncioTestCase):
    def frontend(self, events, tracing=True):
        obj = load_methods("entrypoints/async_omni.py", "AsyncOmni", {"abort"}, events, tracing=tracing)
        obj.engine = SimpleNamespace(abort_async=AsyncMock())
        obj.request_states = {"a": object(), "b": object()}
        obj.log_stats = False
        return obj

    async def test_frontend_one_shot_ids_and_explicit_unknown_drain(self):
        events = []
        obj = self.frontend(events)
        await obj.abort(iter(["a", "b"]))
        obj.engine.abort_async.assert_awaited_once_with(["a", "b"])
        self.assertEqual(obj.request_states, {})
        self.assertEqual(
            [e["event"] for e in events], ["frontend_abort_requested"] * 2 + ["frontend_abort_returned"] * 2
        )
        self.assertTrue(all(e["gpu_drain_confirmed"] is None for e in events[2:]))

    async def test_frontend_failure_preserves_state_without_false_return(self):
        events = []
        obj = self.frontend(events)
        obj.engine.abort_async.side_effect = RuntimeError("transport failed")
        with self.assertRaisesRegex(RuntimeError, "transport failed"):
            await obj.abort("a")
        self.assertIn("a", obj.request_states)
        self.assertEqual([e["event"] for e in events], ["frontend_abort_requested"])

    async def test_frontend_disabled(self):
        events = []
        obj = self.frontend(events, tracing=False)
        await obj.abort("a")
        self.assertEqual(events, [])
        self.assertNotIn("a", obj.request_states)

    def orchestrator(self, events, tracing=True):
        obj = load_methods(
            "engine/orchestrator.py",
            "Orchestrator",
            {
                "_handle_abort",
                "_abort_request_ids",
                "_release_request_bindings",
                "_cleanup_request_ids",
            },
            events,
            tracing=tracing,
        )
        obj.request_states = {"a": object(), "cfg": object()}
        obj._pd_kv_params = dict(obj.request_states)
        obj._stage_cum_tokens = dict(obj.request_states)
        obj._cfg_tracker = SimpleNamespace(abort_parents=Mock(return_value=["a", "cfg"]))
        obj.stage_pools = []
        for stage in range(3):
            bindings = {"a": stage % 2, "cfg": stage % 2}

            async def abort(ids, bindings=bindings):
                for rid in ids:
                    bindings.pop(rid, None)

            pool = SimpleNamespace(
                stage_id=stage,
                get_bound_replica_id=Mock(side_effect=bindings.get),
                abort_requests=AsyncMock(side_effect=abort),
                release_bindings=Mock(),
            )
            obj.stage_pools.append(pool)
        return obj

    async def test_abort_order_binding_snapshot_cfg_cleanup(self):
        events = []
        obj = self.orchestrator(events)
        await obj._handle_abort({"request_ids": ["a"]})
        self.assertEqual(events[0], {"event": "orchestrator_abort_received", "request_id": "a", "request_known": True})
        for stage in range(3):
            rows = events[1 + stage * 4 : 5 + stage * 4]
            self.assertEqual(
                [r["event"] for r in rows], ["stage_abort_dispatch_begin"] * 2 + ["stage_abort_dispatch_end"] * 2
            )
            self.assertTrue(all(r["stage_id"] == stage and r["replica_id"] == stage % 2 for r in rows))
            self.assertTrue(all(r["gpu_drain_confirmed"] is None for r in rows[2:]))
        self.assertEqual([e["event"] for e in events[-2:]], ["request_cleanup"] * 2)
        self.assertEqual(obj.request_states, {})
        self.assertEqual(obj._pd_kv_params, {})
        self.assertEqual(obj._stage_cum_tokens, {})

    async def test_dispatch_failure_not_mislabeled_as_cleanup(self):
        events = []
        obj = self.orchestrator(events)
        obj.stage_pools[0].abort_requests.side_effect = RuntimeError("pool failed")
        with self.assertRaisesRegex(RuntimeError, "pool failed"):
            await obj._handle_abort({"request_ids": ["a"]})
        self.assertFalse(any(e["event"] in {"stage_abort_dispatch_end", "request_cleanup"} for e in events))
        self.assertIn("a", obj.request_states)
        obj.stage_pools[1].abort_requests.assert_not_awaited()

    async def test_normal_cleanup_distinct_from_external_abort(self):
        events = []
        obj = self.orchestrator(events)
        await obj._cleanup_request_ids(["a"], abort=True)
        self.assertFalse(any(e["event"] == "orchestrator_abort_received" for e in events))
        self.assertEqual(events[-1]["event"], "request_cleanup")

    async def test_disabled_does_not_read_trace_metadata(self):
        events = []
        obj = self.orchestrator(events, tracing=False)
        for pool in obj.stage_pools:
            pool.get_bound_replica_id.side_effect = AssertionError("trace metadata read")
            del pool.stage_id
        await obj._handle_abort({"request_ids": ["a"]})
        self.assertEqual(events, [])
        self.assertEqual(obj.request_states, {})


class GenerationFinishTests(unittest.TestCase):
    def scheduler(self, events, tracing=True):
        class Base:
            def finish_requests(self, ids, status):
                self.parent_ids = ids
                ids = list(self.requests) if ids is None else ids
                result = []
                for rid in ids:
                    if self.requests.pop(rid, None) is not None:
                        result.append((rid, 7))
                return result

        obj = load_methods(
            "core/sched/omni_generation_scheduler.py",
            "OmniGenerationScheduler",
            {"finish_requests"},
            events,
            tracing=tracing,
            base=Base,
        )
        obj.requests = {"a": SimpleNamespace(external_req_id="http-a"), "b": SimpleNamespace(external_req_id="http-b")}
        obj.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=2))
        obj.adapter_ids = []
        obj.chunk_transfer_adapter = SimpleNamespace(finish_requests=lambda ids, *_: obj.adapter_ids.extend(ids))
        return obj

    def test_generator_shared_with_adapter_unknown_id_and_return_preserved(self):
        events = []
        obj = self.scheduler(events)
        result = obj.finish_requests(iter(["a", "missing"]), SimpleNamespace(name="FINISHED_ABORTED"))
        self.assertEqual(result, [("a", 7)])
        self.assertEqual(obj.adapter_ids, ["a", "missing"])
        self.assertEqual(obj.parent_ids, ["a", "missing"])
        self.assertEqual(events[0]["external_req_id"], "http-a")
        self.assertFalse(events[1]["was_known"])
        self.assertTrue(all(not e["remains_known"] for e in events[2:]))
        self.assertTrue(all(e["stage_id"] == 2 for e in events))

    def test_none_materializes_all(self):
        events = []
        obj = self.scheduler(events)
        self.assertEqual(obj.finish_requests(None, "abort"), [("a", 7), ("b", 7)])
        self.assertEqual(obj.adapter_ids, ["a", "b"])

    def test_disabled_passes_original_object_and_needs_no_config(self):
        events = []
        obj = self.scheduler(events, tracing=False)
        obj.chunk_transfer_adapter = None
        del obj.vllm_config
        ids = ["a"]
        self.assertEqual(obj.finish_requests(ids, "abort"), [("a", 7)])
        self.assertIs(obj.parent_ids, ids)
        self.assertEqual(events, [])

    def test_adapter_failure_has_no_false_return(self):
        events = []
        obj = self.scheduler(events)
        obj.chunk_transfer_adapter.finish_requests = Mock(side_effect=RuntimeError("adapter failed"))
        with self.assertRaisesRegex(RuntimeError, "adapter failed"):
            obj.finish_requests(["a"], "abort")
        self.assertIn("a", obj.requests)
        self.assertEqual([e["event"] for e in events], ["scheduler_finish_requested"])


if __name__ == "__main__":
    unittest.main()
