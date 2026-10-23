"""CPU contract checks without loading model/torch/CUDA dependencies.

Loads the entire real StagePool module with dependency stubs. The API module is
too heavyweight for this standalone test, so its registry and snapshot function
are compiled directly from their original AST, with only external helpers stubbed.
Run: python tests/engine/test_prefetch_route_cpu.py
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import logging
import os
import sys
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


def load_stage_pool():
    stubs = {}
    for name, attrs in {
        "vllm.logger": {"init_logger": logging.getLogger},
        "vllm.v1.engine": {"EngineCoreOutputs": object},
        "vllm_omni.metrics.stats": {"StageRequestStats": object, "StageStats": object},
        "vllm_omni.metrics.utils": {"count_tokens_from_outputs": lambda _: 0},
        "vllm_omni.profiler.interaction_trace": {"emit": lambda *a, **k: None, "enabled": lambda: False},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        stubs[name] = module
    spec = importlib.util.spec_from_file_location("stage_pool_cpu_test", ROOT / "vllm_omni/engine/stage_pool.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**stubs, spec.name: module}):
        spec.loader.exec_module(module)
    return module.StagePool


def load_api_contracts():
    path = ROOT / "vllm_omni/entrypoints/openai/api_server.py"
    tree = ast.parse(path.read_text())
    names = {"OmniPrefetchRegistry", "_build_omni_load_snapshot"}
    selected = ast.Module(body=[n for n in tree.body if getattr(n, "name", None) in names], type_ignores=[])
    namespace = {
        "Any": Any, "State": SimpleNamespace, "time": time,
        "_env_bool": lambda name, default=False: os.getenv(name, str(int(default))).lower() in {"1", "true", "yes", "on"},
        "_env_float": lambda name, default: float(os.getenv(name, str(default))),
        "_scheduler_policy_name_from_config": lambda _: None,
        "_scheduler_policy_name": lambda _: None,
        "_safe_qsize": lambda _: None,
    }
    exec(compile(selected, str(path), "exec"), namespace)
    return namespace["OmniPrefetchRegistry"], namespace["_build_omni_load_snapshot"]


StagePool = load_stage_pool()
Registry, load_snapshot = load_api_contracts()


class Client:
    stage_type = "llm"

    def __init__(self):
        self.requests = []
        self.aborts = []
        self.fail = False

    async def add_request_async(self, request, **kwargs):
        if self.fail:
            raise RuntimeError("submit rejected")
        self.requests.append(request.request_id)

    async def abort_requests_async(self, request_ids):
        self.aborts.extend(request_ids)

    def shutdown(self):
        pass


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {k: "0" for k in (
            "OMNI_PREFETCH_REPLICA_AFFINITY", "OMNI_PREFETCH_STAGE_ACTIVE_LOAD",
            "OMNI_PREFETCH_FAST_BYPASS", "OMNI_PREFETCH_SKIP_OFFLOAD_ON_MISS",
            "OMNI_PREFETCH_HISTORY_HIT_ENABLE",
        )})
        self.env.start()
        self.addCleanup(self.env.stop)

    def pool(self, enabled=True):
        os.environ["OMNI_PREFETCH_REPLICA_AFFINITY"] = str(int(enabled))
        processor = SimpleNamespace(add_request=Mock(), remove_request=Mock(), abort_requests=Mock())
        return StagePool(0, [Client(), Client()], output_processor=processor)

    def submit(self, pool, request_id, *, prefetch=False, session="session", prefix="prefix"):
        kv = ({"omni_scheduler_prefetch": True, "omni_prefetch_session_id": session,
               "omni_prefetch_committed_prefix_hash": prefix} if prefetch else
              {"omni_session_id": session, "omni_matched_prefix_hash": prefix})
        state = SimpleNamespace(sampling_params_list=[SimpleNamespace(extra_args={"kv_transfer_params": kv})])
        request = SimpleNamespace(request_id=request_id, mm_features=[])
        return asyncio.run(pool.submit_initial(request_id, state, request))

    def warmed(self, pool):
        self.assertEqual(self.submit(pool, "warm", prefetch=True), 0)
        pool.mark_stage_finished("warm", 0, completion_output=self.completion("warm", "stop"))
        pool.release_binding("warm")

    @staticmethod
    def completion(request_id, finish_reason):
        return SimpleNamespace(
            request_id=request_id, finished=True,
            outputs=[SimpleNamespace(finish_reason=finish_reason, token_ids=[])],
        )

    def finish_through_orchestrator(self, pool, output):
        """Run the actual orchestrator method with the actual StagePool."""
        path = ROOT / "vllm_omni/engine/orchestrator.py"
        tree = ast.parse(path.read_text())
        orchestrator = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Orchestrator")
        method = next(n for n in orchestrator.body if getattr(n, "name", None) == "_handle_processed_outputs")
        namespace = {"Any": Any, "_time": time, "trace_enabled": lambda: False,
                     "logger": logging.getLogger("orchestrator_cpu_test")}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        routed = []

        async def route(*args):
            routed.append(args)

        pool.build_stage_metrics = lambda *args, **kwargs: SimpleNamespace(pipeline_timings={})
        owner = SimpleNamespace(
            stage_pools=[pool],
            request_states={output.request_id: SimpleNamespace(stage_submit_ts={0: time.time()}, pipeline_timings={})},
            _stage_cum_tokens={}, _route_output=route,
        )
        asyncio.run(namespace["_handle_processed_outputs"](owner, 0, 0, [output]))
        self.assertEqual(len(routed), 1)

    def test_orchestrator_failed_or_aborted_prefetch_cannot_publish_route_hint(self):
        for reason in ("error", "abort", None):
            with self.subTest(finish_reason=reason):
                pool = self.pool()
                self.submit(pool, "warm", prefetch=True)
                # vLLM RequestOutput has no top-level error for FINISHED_ERROR;
                # its nested CompletionOutput carries finish_reason="error".
                self.finish_through_orchestrator(pool, self.completion("warm", reason))
                self.assertEqual(pool.active_request_snapshot(), {})
                self.assertEqual(pool._prefetch_route_requests, {})
                self.assertEqual(pool._prefix_routes, {})
                pool.release_binding("warm")
                self.assertEqual(self.submit(pool, "live"), 1)

    def test_orchestrator_successful_prefetch_publishes_route_hint(self):
        for reason in ("stop", "length"):
            with self.subTest(finish_reason=reason):
                pool = self.pool()
                self.submit(pool, "warm", prefetch=True)
                self.finish_through_orchestrator(pool, self.completion("warm", reason))
                self.assertEqual(pool.active_request_snapshot(), {})
                self.assertEqual(pool._prefetch_route_requests, {})
                self.assertIn(("session", "prefix"), pool._prefix_routes)
                pool.release_binding("warm")
                self.assertEqual(self.submit(pool, "live"), 0)

    def test_diffusion_completion_still_retires_activity_and_routes_output(self):
        pool = self.pool()
        for client in pool.clients:
            client.stage_type = "diffusion"
        pool.select_replica_id("image")
        # Diffusion completion has no text CompletionOutput finish reason.
        self.finish_through_orchestrator(pool, SimpleNamespace(request_id="image", finished=True))
        self.assertEqual(pool.active_request_snapshot(), {})
        self.assertEqual(pool.get_bound_replica_id("image"), 0)
        self.assertEqual(pool._prefix_routes, {})

    def test_default_off_preserves_round_robin_and_no_route_state(self):
        pool = self.pool(False)
        self.warmed(pool)
        self.assertEqual(self.submit(pool, "live"), 1)
        self.assertEqual(pool._prefix_routes, {})
        self.assertEqual(pool.active_request_snapshot(), {})

    def test_completed_prefetch_survives_binding_release(self):
        pool = self.pool()
        self.warmed(pool)
        self.assertIsNone(pool.get_bound_replica_id("warm"))
        self.assertEqual(self.submit(pool, "live"), 0)

    def test_inflight_prefetch_is_not_completed_route_hint(self):
        pool = self.pool()
        self.submit(pool, "warm", prefetch=True)
        self.assertEqual(self.submit(pool, "live"), 1)

    def test_wrong_prefix_and_session_do_not_reuse_hint(self):
        for kwargs in ({"prefix": "different"}, {"session": "different"}):
            pool = self.pool()
            self.warmed(pool)
            self.assertEqual(self.submit(pool, "live", **kwargs), 1)

    def test_expired_route_falls_back(self):
        pool = self.pool()
        self.warmed(pool)
        pool._prefix_routes[("session", "prefix")] = replace(
            pool._prefix_routes[("session", "prefix")], expires=time.monotonic() - 1)
        self.assertEqual(self.submit(pool, "live"), 1)

    def test_client_replacement_invalidates_hint(self):
        pool = self.pool()
        self.warmed(pool)
        pool.clients[0] = Client()
        self.assertEqual(self.submit(pool, "live"), 1)

    def test_shutdown_prevents_late_completion_hint(self):
        pool = self.pool()
        self.submit(pool, "warm", prefetch=True)
        pool.shutdown_replica(0)
        pool.mark_stage_finished("warm", 0, completion_output=self.completion("warm", "stop"))
        self.assertEqual(pool._prefix_routes, {})

    def test_client_replacement_during_prefetch_cannot_create_new_hint(self):
        pool = self.pool()
        self.submit(pool, "warm", prefetch=True)
        pool.clients[0] = Client()
        pool.mark_stage_finished("warm", 0, completion_output=self.completion("warm", "stop"))
        self.assertEqual(pool._prefix_routes, {})

    def test_older_completion_cannot_replace_newer_candidate(self):
        pool = self.pool()
        self.submit(pool, "older", prefetch=True)
        self.submit(pool, "newer", prefetch=True)
        pool.mark_stage_finished("newer", 1, completion_output=self.completion("newer", "length"))
        candidate = pool._prefix_routes[("session", "prefix")]
        pool.mark_stage_finished("older", 0, completion_output=self.completion("older", "length"))
        self.assertIs(pool._prefix_routes[("session", "prefix")], candidate)
        self.assertEqual(candidate.source_request_id, "newer")
        self.assertEqual(candidate.version, 2)

    def test_new_failed_prefetch_invalidates_completed_candidate(self):
        pool = self.pool()
        self.warmed(pool)
        pool.clients[0].fail = True
        with self.assertRaises(RuntimeError):
            self.submit(pool, "failed_new", prefetch=True)
        self.assertEqual(pool._prefix_routes, {})
        self.assertEqual(pool._prefetch_route_requests, {})

    def test_new_client_after_shutdown_can_publish_new_generation(self):
        pool = self.pool()
        self.warmed(pool)
        pool.shutdown_replica(0)
        pool.clients[0] = Client()
        pool._next_replica_id = 0
        self.submit(pool, "new_generation", prefetch=True)
        pool.mark_stage_finished("new_generation", 0, completion_output=self.completion("new_generation", "stop"))
        candidate = pool._prefix_routes[("session", "prefix")]
        self.assertEqual(candidate.client_generation, 2)
        self.assertEqual(candidate.source_request_id, "new_generation")

    def test_wrong_replica_finish_does_not_retire_real_activity(self):
        pool = self.pool()
        self.submit(pool, "live")
        pool.mark_stage_finished("live", 1)
        self.assertEqual(pool.active_request_snapshot(), {"live": 0})

    def test_update_failure_restores_prior_activity_and_binding(self):
        for was_active in (False, True):
            for failure_at in ("processor", "client", "cancel"):
                with self.subTest(was_active=was_active, failure_at=failure_at):
                    pool = self.pool()
                    self.submit(pool, "live")
                    if not was_active:
                        pool.mark_stage_finished("live", 0)
                    if failure_at == "processor":
                        pool.output_processor.add_request.side_effect = RuntimeError("processor failed")
                    elif failure_at == "client":
                        pool.clients[0].fail = True
                    else:
                        async def cancel(*args):
                            raise asyncio.CancelledError
                        pool.clients[0].add_request_async = cancel
                    state = SimpleNamespace(sampling_params_list=[SimpleNamespace(extra_args={})])
                    with self.assertRaises((RuntimeError, asyncio.CancelledError)):
                        asyncio.run(pool.submit_update("live", state, SimpleNamespace(request_id="live")))
                    self.assertEqual(pool.active_request_snapshot(), {"live": 0} if was_active else {})
                    self.assertEqual(pool.get_bound_replica_id("live"), 0)

    def test_failed_update_does_not_resurrect_finished_or_replace_newer_work(self):
        for during_await in ("finish", "release", "newer"):
            pool = self.pool()
            self.submit(pool, "live")
            async def fail_after_interleaving(*args):
                pool.mark_stage_finished("live", 0)
                if during_await != "finish":
                    pool.release_binding("live")
                if during_await == "newer":
                    pool.select_replica_id("live")
                raise RuntimeError("late rejection")
            pool.clients[0].add_request_async = fail_after_interleaving
            state = SimpleNamespace(sampling_params_list=[SimpleNamespace(extra_args={})])
            with self.assertRaises(RuntimeError):
                asyncio.run(pool.submit_update("live", state, SimpleNamespace(request_id="live")))
            self.assertEqual(pool.active_request_snapshot(), {"live": 1} if during_await == "newer" else {})
            self.assertEqual(pool.get_bound_replica_id("live"), 1 if during_await == "newer" else 0 if during_await == "finish" else None)

    def test_failed_update_with_new_binding_cleans_activity_and_binding(self):
        pool = self.pool()
        pool.clients[0].fail = True
        state = SimpleNamespace(sampling_params_list=[SimpleNamespace(extra_args={})])
        with self.assertRaises(RuntimeError):
            asyncio.run(pool.submit_update("new", state, SimpleNamespace(request_id="new")))
        self.assertEqual(pool.active_request_snapshot(), {})
        self.assertIsNone(pool.get_bound_replica_id("new"))

    def test_trace_alone_does_not_change_failed_update_binding_behavior(self):
        namespace = StagePool.__init__.__globals__
        with patch.dict(namespace, trace_enabled=lambda: True, emit=lambda *args, **kwargs: None):
            pool = self.pool(enabled=False)
            pool.clients[0].fail = True
            state = SimpleNamespace(sampling_params_list=[SimpleNamespace(extra_args={})])
            with self.assertRaises(RuntimeError):
                asyncio.run(pool.submit_update("new", state, SimpleNamespace(request_id="new")))
            self.assertEqual(pool.get_bound_replica_id("new"), 0)
            self.assertEqual(pool.active_request_snapshot(), {})

    def test_prefill_accounting_uses_engine_fields_without_asserting_residency(self):
        pool = self.pool()
        pool.output_processor.process_outputs = Mock(return_value=SimpleNamespace(reqs_to_abort=[], request_outputs=[]))
        records = []
        stats = SimpleNamespace(num_prompt_tokens=143, num_computed_tokens=15, num_cached_tokens=128,
                                num_local_cached_tokens=128, num_external_cached_tokens=0)
        raw = SimpleNamespace(outputs=[SimpleNamespace(request_id="live", prefill_stats=stats)],
                              timestamp=0, scheduler_stats=None)
        namespace = pool.process_llm_raw_outputs.__func__.__globals__
        with patch.dict(namespace, trace_enabled=lambda: True, emit=lambda event, **fields: records.append((event, fields))):
            asyncio.run(pool.process_llm_raw_outputs(1, raw))
        self.assertEqual(records[0][0], "cache_accounting_observed")
        self.assertEqual(records[0][1]["num_local_cached_tokens"], 128)
        self.assertEqual(records[0][1]["num_external_cached_tokens"], 0)
        self.assertIsNone(records[0][1]["cache_residency_confirmed"])
        self.assertIsNone(records[0][1]["prefetch_attribution_confirmed"])

    def test_missing_prefill_fields_are_unknown_and_absent_stats_emit_nothing(self):
        pool = self.pool()
        pool.output_processor.process_outputs = Mock(return_value=SimpleNamespace(reqs_to_abort=[], request_outputs=[]))
        records = []
        raw = SimpleNamespace(outputs=[SimpleNamespace(request_id="partial", prefill_stats=SimpleNamespace(num_cached_tokens=4)),
                                       SimpleNamespace(request_id="absent")], timestamp=0, scheduler_stats=None)
        namespace = pool.process_llm_raw_outputs.__func__.__globals__
        with patch.dict(namespace, trace_enabled=lambda: True, emit=lambda event, **fields: records.append(fields)):
            asyncio.run(pool.process_llm_raw_outputs(0, raw))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["num_cached_tokens"], 4)
        self.assertIsNone(records[0]["num_local_cached_tokens"])
        self.assertIsNone(records[0]["num_external_cached_tokens"])

    def test_busy_hint_replica_falls_back_to_least_loaded(self):
        pool = self.pool()
        self.warmed(pool)
        self.assertEqual(self.submit(pool, "busy"), 0)
        self.assertEqual(self.submit(pool, "next"), 1)

    def test_finished_stage_retires_activity_but_keeps_abort_route(self):
        pool = self.pool()
        self.submit(pool, "live")
        pool.mark_stage_finished("live", 0)
        self.assertEqual(pool.active_request_snapshot(), {})
        self.assertEqual(pool.get_bound_replica_id("live"), 0)
        asyncio.run(pool.abort_requests(["live"]))
        self.assertEqual(pool.clients[0].aborts, ["live"])

    def test_failed_submission_has_no_activity_or_future_hint(self):
        pool = self.pool()
        pool.clients[0].fail = True
        with self.assertRaisesRegex(RuntimeError, "submit rejected"):
            self.submit(pool, "warm", prefetch=True)
        pool.mark_stage_finished("warm", 0)
        self.assertEqual(pool.active_request_snapshot(), {})
        self.assertEqual(pool._prefix_routes, {})

    def test_initial_cancel_cleans_enabled_state_and_keeps_default_off_semantics(self):
        for stage_type in ("llm", "diffusion"):
            for enabled in (False, True):
                with self.subTest(stage_type=stage_type, enabled=enabled):
                    pool = self.pool(enabled=enabled)
                    async def cancel(*args, **kwargs):
                        raise asyncio.CancelledError
                    for client in pool.clients:
                        client.stage_type = stage_type
                        client.add_request_async = cancel
                    with self.assertRaises(asyncio.CancelledError):
                        self.submit(pool, "warm", prefetch=True)
                    self.assertEqual(pool.get_bound_replica_id("warm"), None if enabled else 0)
                    self.assertEqual(pool.active_request_snapshot(), {})
                    self.assertEqual(pool._prefetch_route_requests, {})
                    self.assertEqual(pool._prefix_routes, {})
                    if enabled and stage_type == "llm":
                        pool.output_processor.remove_request.assert_called_once_with("warm")

    def test_stream_update_reactivates_without_rerouting(self):
        pool = self.pool()
        self.submit(pool, "live")
        pool.mark_stage_finished("live", 0)
        state = SimpleNamespace(sampling_params_list=[SimpleNamespace(extra_args={})])
        replica = asyncio.run(pool.submit_update("live", state, SimpleNamespace(request_id="live")))
        self.assertEqual(replica, 0)
        self.assertEqual(pool.active_request_snapshot(), {"live": 0})

    def test_live_completion_does_not_create_prefetch_hint(self):
        pool = self.pool()
        self.submit(pool, "live")
        pool.mark_stage_finished("live", 0)
        self.assertEqual(pool._prefix_routes, {})

    def test_strict_registry_marks_candidate_without_claiming_hit(self):
        os.environ["OMNI_PREFETCH_REPLICA_AFFINITY"] = "1"
        os.environ["OMNI_PREFETCH_SKIP_OFFLOAD_ON_MISS"] = "1"
        registry = Registry()
        registry.mark_warm(session_id="s", prefix_hash="p", request_id="w", metadata={"prefetch_scope": "history_only"})
        policy = registry.resolve_live(session_id="s", prefix_hash="p")
        self.assertEqual(policy["lookup_state"], "warm_route_candidate")
        self.assertFalse(policy["prefetch_hit_used"])
        self.assertFalse(policy["skip_offload"])
        self.assertEqual(registry.stats()["live_hits"], 0)
        self.assertEqual(registry.stats()["live_route_candidates"], 1)
        self.assertFalse(registry.resolve_live(session_id="s", prefix_hash="missing")["skip_offload"])

    def test_legacy_registry_behavior_preserved(self):
        os.environ["OMNI_PREFETCH_FAST_BYPASS"] = "1"
        registry = Registry()
        registry.mark_warm(session_id="s", prefix_hash="p", request_id="w", metadata={"prefetch_scope": "full_prompt"})
        self.assertTrue(registry.resolve_live(session_id="s", prefix_hash="p")["prefetch_hit_used"])
        self.assertEqual(registry.stats()["live_hits"], 1)

    def test_pressure_switch_keeps_binding_count_visible(self):
        pool = self.pool()
        self.submit(pool, "live")
        pool.mark_stage_finished("live", 0)
        state = SimpleNamespace(engine_client=SimpleNamespace(engine=SimpleNamespace(stage_pools=[pool])))
        old = load_snapshot(state)
        self.assertEqual(old["max_stage0_active_per_replica"], 1)
        pool._stage_active_load_enabled = True
        new = load_snapshot(state)
        self.assertEqual(new["max_stage0_active_per_replica"], 0)
        self.assertEqual(new["total_active_bindings"], 1)
        self.assertEqual(new["stages"][0]["active_bindings"], 1)
        self.assertEqual(new["stages"][0]["pressure_source"], "stage_outstanding")

    def test_stage_activity_retires_thinker_without_hiding_active_audio(self):
        thinker, audio = self.pool(), self.pool()
        audio.stage_id = 1
        thinker.select_replica_id("live")
        audio.select_replica_id("live")
        thinker.mark_stage_finished("live", 0)
        state = SimpleNamespace(engine_client=SimpleNamespace(engine=SimpleNamespace(stage_pools=[thinker, audio])))
        old = load_snapshot(state)
        self.assertEqual((old["max_stage0_active_per_replica"], old["max_audio_active_per_replica"]), (1, 1))
        thinker._stage_active_load_enabled = audio._stage_active_load_enabled = True
        new = load_snapshot(state)
        self.assertEqual((new["max_stage0_active_per_replica"], new["max_audio_active_per_replica"]), (0, 1))
        self.assertEqual(new["total_active_bindings"], 2)
        self.assertFalse(new["stages"][0]["target_replica_admission_confirmed"])

    def test_native_prefetch_max_live_zero_counts_only_current_scheduler(self):
        path = ROOT / "vllm_omni/core/sched/omni_scheduler_mixin.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OmniSchedulerMixin")
        names = {"_omni_kv_transfer_params", "_omni_is_scheduler_prefetch", "_omni_live_pressure_count",
                 "_omni_cancel_prefetch_under_live_pressure"}
        cls.body = [n for n in cls.body if getattr(n, "name", None) in names]
        namespace = {"Request": object, "RequestStatus": SimpleNamespace(FINISHED_ABORTED="abort"),
                     "logger": logging.getLogger("native_prefetch_test")}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
        scheduler = namespace["OmniSchedulerMixin"]()
        scheduler._omni_env_bool = lambda *args: True
        scheduler._omni_env_int = lambda *args: 0
        scheduler.finish_requests = Mock()
        scheduler.waiting = scheduler.skipped_waiting = []
        pref = SimpleNamespace(request_id="pref", kv_transfer_params={"omni_scheduler_prefetch": True})
        scheduler.running = [pref]
        self.assertEqual(scheduler._omni_cancel_prefetch_under_live_pressure(), 0)
        scheduler.finish_requests.assert_not_called()
        scheduler.running.append(SimpleNamespace(request_id="live", kv_transfer_params={}))
        self.assertEqual(scheduler._omni_cancel_prefetch_under_live_pressure(), 1)
        scheduler.finish_requests.assert_called_once_with(["pref"], "abort")


if __name__ == "__main__":
    unittest.main(verbosity=2)
