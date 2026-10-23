"""Exercise actual prefetch API bodies with dependency boundaries stubbed."""
from __future__ import annotations

import ast
import asyncio
import json
import os
import time
import unittest
from collections.abc import AsyncIterator
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock


class NonStreamResponse:
    def __init__(self, reasons):
        self.choices = [SimpleNamespace(finish_reason=reason) for reason in reasons]
        self.usage = None


class AsyncioWithoutCancellationCounter:
    """Use real async operations, with Python 3.10's public Task interface."""

    def __getattr__(self, name):
        return getattr(asyncio, name)

    @staticmethod
    def current_task():
        return SimpleNamespace()


class PrefetchCompletionTests(unittest.TestCase):
    def fixture(self, *, strict, chunks=None, response=None):
        async def stream():
            for chunk in chunks or []:
                yield chunk

        controller = SimpleNamespace(
            decide=AsyncMock(return_value=(True, {"reason": "admitted"})),
            resolve_priority=Mock(return_value=(100, "test")), register_active=AsyncMock(),
            is_abort_requested=AsyncMock(return_value=False), finish=AsyncMock(return_value={}),
        )
        monitor = SimpleNamespace(record_prefetch_decision=AsyncMock(), record_prefetch_start=AsyncMock(),
                                  record_prefetch_finish=AsyncMock())
        handler = SimpleNamespace(create_chat_completion=AsyncMock(return_value=response if response is not None else stream()))
        path = Path(__file__).resolve().parents[2] / "vllm_omni/entrypoints/openai/api_server.py"
        tree = ast.parse(path.read_text())
        nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        for node in tree.body:
            if getattr(node, "name", None) in ("OmniPrefetchRegistry", "OmniPrefetchAdmissionController", "_consume_prefetch_stream",
                                                  "_abort_prefetch_engine_request", "_PrefetchRequestLifecycle",
                                                  "_execute_prefetch_session", "omni_prefetch_session"):
                node.decorator_list = []
                nodes.append(node)
        namespace = {
            "asyncio": asyncio, "json": json, "time": time, "cast": cast, "AsyncIterator": AsyncIterator,
            "_env_bool": lambda name, default=False: strict if name == "OMNI_PREFETCH_REPLICA_AFFINITY" else default,
            "_env_float": lambda name, default: default,
            "trace_enabled": lambda: False, "Omnichat": lambda raw: handler, "logger": Mock(),
            "HTTPStatus": HTTPStatus,
            "HTTPException": lambda **kwargs: RuntimeError(kwargs.get("detail", "HTTP error")),
            "_get_prefetch_admission_controller": lambda state: controller,
            "_get_runtime_monitor": lambda state: monitor, "_OMNI_EVICTION_METADATA_FIELDS": (),
            "ChatCompletionRequest": SimpleNamespace(model_validate=lambda payload: SimpleNamespace(**payload)),
            "ChatCompletionResponse": NonStreamResponse, "ErrorResponse": type("ErrorResponse", (), {}),
            "EngineGenerateError": type("EngineGenerateError", (Exception,), {}),
            "EngineDeadError": type("EngineDeadError", (Exception,), {}),
            "JSONResponse": lambda content, **kwargs: content,
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
        registry = namespace["OmniPrefetchRegistry"]()
        namespace["_get_prefetch_registry"] = lambda state: registry
        request = SimpleNamespace(model="model", request_id="prefetch", session_id="session",
                                  committed_prefix_hash="prefix", alias_prefix_hashes=[], prefix_token_estimate=None,
                                  lead_ms=5000, metadata={}, messages=[], stream_options=None, max_tokens=1,
                                  temperature=0, top_p=1, top_k=1)
        raw = SimpleNamespace(state=SimpleNamespace(), app=SimpleNamespace(state=SimpleNamespace(
            engine_client=SimpleNamespace(abort=AsyncMock()))))
        return namespace, request, raw, registry, controller, handler, monitor

    def endpoint(self, **kwargs):
        namespace, request, raw, registry, controller, _, _ = self.fixture(**kwargs)
        result = asyncio.run(namespace["omni_prefetch_session"](request, raw))
        return result, registry, controller

    @staticmethod
    def choice(reason):
        return "data: " + json.dumps({"choices": [{"delta": {"content": "x"}, "finish_reason": reason}]}) + "\n\n"

    def test_success_requires_valid_terminal_reason_and_DONE(self):
        for reason in ("stop", "length"):
            result, registry, _ = self.endpoint(strict=True, chunks=[self.choice(reason), "data: [DONE]\n\n"])
            self.assertEqual(result["status"], "SUCCESS")
            self.assertTrue(result["completion_confirmed"])
            decision = registry.resolve_live(session_id="session", prefix_hash="prefix")
            self.assertEqual(decision["candidate_state"], "api_completed_candidate")
            self.assertFalse(decision["prefetch_hit_used"])
            self.assertIsNone(decision["cache_reuse_confirmed"])

    def test_abort_error_missing_DONE_or_error_payload_never_marks_candidate(self):
        cases = [[], [self.choice("stop")], [self.choice("abort"), "data: [DONE]\n\n"],
                 [self.choice("error"), "data: [DONE]\n\n"], [self.choice(None), "data: [DONE]\n\n"],
                 [self.choice("stop"), 'data: {"error":{"message":"failed"}}\n\n', "data: [DONE]\n\n"],
                 [self.choice("stop"), "data: [DONE]\n\n", self.choice("stop")],
                 ["data: not-json\n\n", self.choice("stop"), "data: [DONE]\n\n"]]
        for chunks in cases:
            with self.subTest(chunks=chunks):
                result, registry, controller = self.endpoint(strict=True, chunks=chunks)
                self.assertEqual(result["status"], "FAILED")
                self.assertEqual(result["skip_reason"], "prefetch_completion_unconfirmed")
                self.assertEqual(registry._warm, {})
                self.assertEqual(controller.finish.call_args.kwargs["terminal_status"], "completion_unconfirmed")

    def test_default_off_keeps_old_API_success_behavior(self):
        result, registry, _ = self.endpoint(strict=False, chunks=[self.choice("abort")])
        self.assertEqual(result["status"], "SUCCESS")
        self.assertFalse(result["completion_confirmed"])
        self.assertEqual(len(registry._warm), 1)

    def test_non_stream_response_obeys_same_finish_reason_guard(self):
        for reasons, expected in ((["stop"], "SUCCESS"), (["length"], "SUCCESS"),
                                  (["abort"], "FAILED"), ([], "FAILED"), (["stop", "error"], "FAILED")):
            result, registry, _ = self.endpoint(strict=True, response=NonStreamResponse(reasons))
            self.assertEqual(result["status"], expected)
            self.assertEqual(bool(registry._warm), expected == "SUCCESS")


class PrefetchLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setup_endpoint(self, *, strict=True, response=None):
        values = PrefetchCompletionTests().fixture(strict=strict, response=response)
        namespace, request, raw, registry, _, handler, monitor = values
        cls = namespace["OmniPrefetchAdmissionController"]
        controller = cls.__new__(cls)
        controller._lock = asyncio.Lock()
        controller._inflight = 1  # Admission already owns one slot.
        controller._active = {}
        controller._prefetch_wall_samples = 0
        controller._prefetch_wall_ewma_ms = 0
        controller._deadline_bucket_stats = {}
        controller.deadline_alpha = .25
        controller._controller_stats = lambda: {"inflight": controller._inflight, "active": len(controller._active)}
        controller.decide = AsyncMock(return_value=(True, {"reason": "admitted", "inflight_reserved": True}))
        controller.resolve_priority = Mock(return_value=(100, "test"))
        controller.finish = AsyncMock(wraps=controller.finish)
        namespace["_get_prefetch_admission_controller"] = lambda state: controller
        return namespace, request, raw, registry, controller, handler, monitor

    async def test_normal_prefetch_without_Task_cancelling_stream_and_nonstream(self):
        for strict in (False, True):
            for nonstream in (False, True):
                with self.subTest(strict=strict, nonstream=nonstream):
                    ns, request, raw, registry, controller, handler, _ = self.setup_endpoint(strict=strict)
                    ns["asyncio"] = AsyncioWithoutCancellationCounter()
                    async def stream():
                        yield PrefetchCompletionTests.choice("stop")
                        yield "data: [DONE]\n\n"
                    handler.create_chat_completion.return_value = NonStreamResponse(["stop"]) if nonstream else stream()
                    result = await ns["omni_prefetch_session"](request, raw)
                    self.assertEqual(result["status"], "SUCCESS")
                    self.assertTrue(registry._warm)
                    self.assertEqual(controller._inflight, 0)
                    self.assertEqual(controller._active, {})
                    controller.finish.assert_awaited_once()

    async def test_propagated_cancel_without_Task_cancelling_still_cleans_up(self):
        ns, request, raw, registry, controller, handler, _ = self.setup_endpoint()
        ns["asyncio"] = AsyncioWithoutCancellationCounter()
        entered = asyncio.Event()
        async def create(*args):
            raw.state.request_metadata = SimpleNamespace(request_id="chatcmpl-owned")
            entered.set()
            await asyncio.Event().wait()
        handler.create_chat_completion.side_effect = create
        task = asyncio.create_task(ns["omni_prefetch_session"](request, raw))
        await entered.wait()
        task.cancel("legacy task cancellation")
        with self.assertRaisesRegex(asyncio.CancelledError, "legacy task cancellation"):
            await task
        self.assertFalse(registry._warm)
        self.assertEqual(controller._inflight, 0)
        self.assertEqual(controller._active, {})
        controller.finish.assert_awaited_once()
        raw.app.state.engine_client.abort.assert_awaited_once_with(["chatcmpl-owned"])

    async def test_cancel_during_generation_before_stream_is_returned(self):
        ns, request, raw, registry, controller, handler, _ = self.setup_endpoint()
        entered = asyncio.Event()
        async def create(*args):
            raw.state.request_metadata = SimpleNamespace(request_id="chatcmpl-header-override")
            entered.set()
            await asyncio.Event().wait()
        handler.create_chat_completion.side_effect = create
        task = asyncio.create_task(ns["omni_prefetch_session"](request, raw))
        await entered.wait()
        task.cancel("original cancellation")
        with self.assertRaisesRegex(asyncio.CancelledError, "original cancellation"):
            await task
        self.assertEqual(controller._inflight, 0)
        self.assertEqual(controller._active, {})
        controller.finish.assert_awaited_once()
        raw.app.state.engine_client.abort.assert_awaited_once_with(["chatcmpl-header-override"])
        self.assertEqual(registry._warm, {})

    async def test_cancelled_stream_closes_and_preserves_cancellation_with_cleanup_errors(self):
        ns, request, raw, registry, controller, handler, _ = self.setup_endpoint()
        entered = asyncio.Event()
        class Stream:
            def __aiter__(self):
                return self
            async def __anext__(self):
                entered.set()
                await asyncio.Event().wait()
            aclose = AsyncMock(side_effect=ValueError("close failure"))
        stream = Stream()
        async def create(*args):
            raw.state.request_metadata = SimpleNamespace(request_id="chatcmpl-owned")
            return stream
        handler.create_chat_completion.side_effect = create
        raw.app.state.engine_client.abort.side_effect = RuntimeError("abort failure")
        task = asyncio.create_task(ns["omni_prefetch_session"](request, raw))
        await entered.wait()
        task.cancel("keep me")
        with self.assertRaisesRegex(asyncio.CancelledError, "keep me"):
            await task
        self.assertEqual(controller._inflight, 0)
        self.assertEqual(controller._active, {})
        stream.aclose.assert_awaited_once()
        controller.finish.assert_awaited_once()
        self.assertFalse(registry._warm)

    async def test_serving_generator_swallowed_cancel_is_not_legacy_SUCCESS(self):
        for strict in (False, True):
            with self.subTest(strict=strict):
                ns, request, raw, registry, controller, handler, _ = self.setup_endpoint(strict=strict)
                entered = asyncio.Event()
                async def stream():
                    try:
                        entered.set()
                        await asyncio.Event().wait()
                        yield "never"
                    except asyncio.CancelledError:
                        return  # Real serving_chat aborts upstream and returns.
                handler.create_chat_completion.return_value = stream()
                task = asyncio.create_task(ns["omni_prefetch_session"](request, raw))
                await entered.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertFalse(registry._warm)
                self.assertEqual(controller._inflight, 0)
                self.assertEqual(controller._active, {})
                controller.finish.assert_awaited_once()

    async def test_nonstream_swallowed_cancel_is_not_SUCCESS(self):
        ns, request, raw, registry, controller, handler, _ = self.setup_endpoint()
        entered = asyncio.Event()
        async def create(*args):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return NonStreamResponse(["stop"])
        handler.create_chat_completion.side_effect = create
        task = asyncio.create_task(ns["omni_prefetch_session"](request, raw))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(registry._warm)
        self.assertEqual(controller._inflight, 0)
        controller.finish.assert_awaited_once()

    async def test_repeated_cancel_awaits_finish_lock_and_does_not_double_decrement(self):
        ns, request, raw, registry, controller, _, monitor = self.setup_endpoint(response=NonStreamResponse(["stop"]))
        entered = asyncio.Event()
        async def start(**kwargs):
            # Also reserve another request's slot, which cleanup must not consume.
            controller._inflight += 1
            await controller._lock.acquire()
            entered.set()
        monitor.record_prefetch_start.side_effect = start
        task = asyncio.create_task(ns["omni_prefetch_session"](request, raw))
        await entered.wait()
        # Normal nonstream success has entered shielded finish waiting for the lock.
        await asyncio.sleep(0)
        task.cancel("first")
        await asyncio.sleep(0)
        task.cancel("second")
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        controller._lock.release()
        with self.assertRaisesRegex(asyncio.CancelledError, "first"):
            await task
        self.assertEqual(controller._inflight, 1)
        self.assertEqual(controller._active, {})
        controller.finish.assert_awaited_once()
        # No owned cleanup/finish task is left running after the HTTP task exits.
        self.assertEqual({t for t in asyncio.all_tasks() if not t.done()}, {asyncio.current_task()})

    async def test_generation_failure_finishes_once_and_preserves_error(self):
        ns, request, raw, registry, controller, handler, _ = self.setup_endpoint()
        handler.create_chat_completion.side_effect = ValueError("generation failed")
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            await ns["omni_prefetch_session"](request, raw)
        self.assertEqual(controller._inflight, 0)
        self.assertEqual(controller._active, {})
        controller.finish.assert_awaited_once()
        raw.app.state.engine_client.abort.assert_not_awaited()
        self.assertFalse(registry._warm)

    async def test_timeout_uses_owned_engine_ID_and_finishes_once(self):
        ns, request, raw, registry, controller, handler, _ = self.setup_endpoint()
        ns["_env_float"] = lambda name, default: 5 if name == "OMNI_PREFETCH_STREAM_TIMEOUT_MS" else default
        closed = asyncio.Event()
        async def stream():
            try:
                await asyncio.Event().wait()
                yield "never"
            finally:
                closed.set()
        async def create(*args):
            raw.state.request_metadata = SimpleNamespace(request_id="chatcmpl-header-override")
            return stream()
        handler.create_chat_completion.side_effect = create
        result = await ns["omni_prefetch_session"](request, raw)
        self.assertEqual(result["status"], "ABORTED")
        self.assertEqual(result["skip_reason"], "prefetch_stream_timeout")
        self.assertTrue(closed.is_set())
        raw.app.state.engine_client.abort.assert_awaited_once_with(["chatcmpl-header-override"])
        controller.finish.assert_awaited_once()
        self.assertEqual(controller._inflight, 0)
        self.assertEqual(controller._active, {})
        self.assertFalse(registry._warm)

    async def test_cancel_before_generation_finishes_without_aborting_unowned_metadata(self):
        ns, request, raw, registry, controller, handler, monitor = self.setup_endpoint()
        raw.state.request_metadata = SimpleNamespace(request_id="old-unowned")
        entered = asyncio.Event()
        async def decision(**kwargs):
            entered.set()
            await asyncio.Event().wait()
        monitor.record_prefetch_decision.side_effect = decision
        task = asyncio.create_task(ns["omni_prefetch_session"](request, raw))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        handler.create_chat_completion.assert_not_awaited()
        raw.app.state.engine_client.abort.assert_not_awaited()
        controller.finish.assert_awaited_once()
        self.assertEqual(controller._inflight, 0)

    async def test_cleanup_error_does_not_replace_original_cancel(self):
        ns, request, raw, _, controller, handler, _ = self.setup_endpoint()
        entered = asyncio.Event()
        async def create(*args):
            entered.set()
            await asyncio.Event().wait()
        handler.create_chat_completion.side_effect = create
        real_finish = controller.finish._mock_wraps
        async def finish(*args, **kwargs):
            await real_finish(*args, **kwargs)
            raise ValueError("post-finish diagnostic failed")
        controller.finish.side_effect = finish
        task = asyncio.create_task(ns["omni_prefetch_session"](request, raw))
        await entered.wait()
        task.cancel("original")
        with self.assertRaisesRegex(asyncio.CancelledError, "original"):
            await task
        self.assertEqual(controller._inflight, 0)
        controller.finish.assert_awaited_once()


class PrefetchAdmissionOwnershipTests(unittest.IsolatedAsyncioTestCase):
    def setup_endpoint(self):
        ns, request, raw, registry, _, handler, monitor = PrefetchCompletionTests().fixture(strict=True)
        ns.update({
            "os": os, "_env_int": lambda name, default: default,
            "_env_int_list": lambda name, default: default,
            "_build_omni_load_snapshot": lambda state: {
                "orchestrator_request_qsize": 0, "max_stage0_active_per_replica": 0,
                "max_audio_active_per_replica": 0,
            },
        })
        monitor.policy_context = AsyncMock(return_value={})
        controller = ns["OmniPrefetchAdmissionController"]()
        controller.enabled = True
        controller.abort_on_live = True
        controller.abort_scope = "all"
        controller.resolve_priority = Mock(return_value=(100, "test"))
        controller.finish = AsyncMock(wraps=controller.finish)
        ns["_get_prefetch_admission_controller"] = lambda state: controller
        request.admission_control = True
        return ns, request, raw, registry, controller, handler, monitor

    async def test_bypass_endpoint_cannot_release_another_requests_reserved_slot(self):
        ns, request, raw, registry, controller, handler, _ = self.setup_endpoint()
        admitted, reservation = await controller.decide(request, raw.app.state)
        self.assertTrue(admitted)
        self.assertTrue(reservation["inflight_reserved"])
        lifecycle = ns["_PrefetchRequestLifecycle"](
            controller, "metered", raw, reserved_slot=reservation["inflight_reserved"])
        await controller.register_active(request_id="metered", engine_request_id="chatcmpl-metered",
                                         session_id="session", prefix_hash="prefix", admission=reservation)
        async def stream():
            yield PrefetchCompletionTests.choice("stop")
            yield "data: [DONE]\n\n"
        handler.create_chat_completion.return_value = stream()
        request.admission_control = False
        result = await ns["omni_prefetch_session"](request, raw)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertFalse(result["admission"]["inflight_reserved"])
        self.assertEqual(controller._inflight, 1)
        self.assertEqual(set(controller._active), {"metered"})
        request.admission_control = True
        admitted, decision = await controller.decide(request, raw.app.state)
        self.assertFalse(admitted)
        self.assertFalse(decision["inflight_reserved"])
        self.assertEqual(decision["reason"], "prefetch_inflight_limit")
        await lifecycle.finish("success")
        await lifecycle.finish("success")
        self.assertEqual(controller._inflight, 0)
        self.assertEqual(controller._active, {})
        # One release for the bypass (false), one for the real reservation.
        self.assertEqual([call.kwargs["release_slot"] for call in controller.finish.await_args_list], [False, True])

    async def test_live_abort_before_metadata_prevents_later_submission(self):
        ns, request, raw, registry, controller, handler, _ = self.setup_endpoint()
        raw.state.request_metadata = SimpleNamespace(request_id="unowned-old")
        entered = asyncio.Event()
        submitted = []
        async def create(*args):
            entered.set()
            await asyncio.Event().wait()
            submitted.append("must not submit after cancellation")
        handler.create_chat_completion.side_effect = create
        task = asyncio.create_task(ns["omni_prefetch_session"](request, raw))
        await entered.wait()
        decision = await controller.abort_for_live(raw.app.state, session_id="session", prefix_hash="prefix")
        self.assertEqual(decision["aborted"], 1)
        self.assertEqual(decision["engine_request_ids"], [])
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(submitted, [])
        raw.app.state.engine_client.abort.assert_not_awaited()
        controller.finish.assert_awaited_once()
        self.assertEqual(controller._inflight, 0)
        self.assertEqual(controller._active, {})
        self.assertFalse(registry._warm)

    async def test_live_abort_after_metadata_uses_actual_ID_in_create_and_stream(self):
        for during_create in (True, False):
            with self.subTest(during_create=during_create):
                ns, request, raw, registry, controller, handler, _ = self.setup_endpoint()
                entered = asyncio.Event()
                stream_closed = asyncio.Event()
                submitted = []
                actual_id = "chatcmpl-header-override"
                async def stream():
                    submitted.append(actual_id)
                    try:
                        entered.set()
                        await asyncio.Event().wait()
                        yield "never"
                    finally:
                        stream_closed.set()
                async def create(*args):
                    raw.state.request_metadata = SimpleNamespace(request_id=actual_id)
                    if during_create:
                        entered.set()
                        await asyncio.Event().wait()
                        submitted.append("must not submit after cancellation")
                    return stream()
                handler.create_chat_completion.side_effect = create
                task = asyncio.create_task(ns["omni_prefetch_session"](request, raw))
                await entered.wait()
                first = await controller.abort_for_live(raw.app.state, session_id="session", prefix_hash="prefix")
                second = await controller.abort_for_live(raw.app.state, session_id="session", prefix_hash="prefix")
                self.assertEqual(first["engine_request_ids"], [actual_id])
                self.assertEqual(second["aborted"], 0)
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(submitted, [] if during_create else [actual_id])
                self.assertEqual(stream_closed.is_set(), not during_create)
                self.assertGreaterEqual(raw.app.state.engine_client.abort.await_count, 1)
                self.assertTrue(all(call.args == ([actual_id],) for call in raw.app.state.engine_client.abort.await_args_list))
                self.assertEqual(controller._live_abort_requests, 1)
                controller.finish.assert_awaited_once()
                self.assertEqual(controller._inflight, 0)
                self.assertEqual(controller._active, {})
                self.assertFalse(registry._warm)

    async def test_legacy_active_entry_without_lifecycle_uses_supplied_engine_ID(self):
        _, _, raw, _, controller, _, _ = self.setup_endpoint()
        await controller.register_active(request_id="legacy", engine_request_id="chatcmpl-legacy",
                                         session_id="session", prefix_hash="prefix")
        result = await controller.abort_for_live(raw.app.state, session_id="session", prefix_hash="prefix")
        self.assertEqual(result["engine_request_ids"], ["chatcmpl-legacy"])
        raw.app.state.engine_client.abort.assert_awaited_once_with(["chatcmpl-legacy"])
        await controller.finish("legacy")
        self.assertEqual(controller._active, {})
        self.assertEqual(controller._inflight, 0)

    async def test_disabled_live_abort_preserves_normal_stream(self):
        ns, request, raw, registry, controller, handler, _ = self.setup_endpoint()
        controller.abort_on_live = False
        entered = asyncio.Event()
        release = asyncio.Event()
        async def stream():
            entered.set()
            await release.wait()
            yield PrefetchCompletionTests.choice("stop")
            yield "data: [DONE]\n\n"
        handler.create_chat_completion.return_value = stream()
        task = asyncio.create_task(ns["omni_prefetch_session"](request, raw))
        await entered.wait()
        result = await controller.abort_for_live(raw.app.state, session_id="session", prefix_hash="prefix")
        self.assertFalse(result["enabled"])
        self.assertFalse(task.done())
        self.assertFalse(await controller.is_abort_requested("prefetch"))
        self.assertEqual(controller._inflight, 1)
        release.set()
        self.assertEqual((await task)["status"], "SUCCESS")
        self.assertEqual(controller._inflight, 0)
        self.assertEqual(controller._active, {})
        self.assertTrue(registry._warm)
        raw.app.state.engine_client.abort.assert_not_awaited()


class PrefetchAdmissionDecisionTests(unittest.IsolatedAsyncioTestCase):
    """Exercise public admission decisions, including contention and boundaries."""

    async def test_rejection_order_and_accounting(self):
        # Later gates also fail in the early cases: the first reason must win.
        cases = (
            (None, 9, 5, 3, "vad_lead_too_short"),
            (599, 9, 5, 3, "vad_lead_too_short"),
            (600, 9, 5, 3, "orchestrator_queue_busy"),
            (600, 8, 5, 3, "thinker_pressure"),
            (600, 8, 4, 3, "audio_backlog"),
            (600, 8, 4, 2, "prefetch_deadline_too_tight"),
        )
        for lead_ms, queue, thinker, audio, reason in cases:
            with self.subTest(reason=reason, lead_ms=lead_ms):
                ns, request, raw, _, controller, _, _ = PrefetchAdmissionOwnershipTests().setup_endpoint()
                snapshot = {
                    "orchestrator_request_qsize": queue,
                    "max_stage0_active_per_replica": thinker,
                    "max_audio_active_per_replica": audio,
                }
                ns["_build_omni_load_snapshot"] = lambda state: snapshot
                request.lead_ms = lead_ms
                controller.deadline_aware = True
                admitted, decision = await controller.decide(request, raw.app.state)
                self.assertFalse(admitted)
                self.assertEqual(decision["reason"], reason)
                self.assertFalse(decision["inflight_reserved"])
                self.assertEqual(decision["snapshot"], snapshot)
                self.assertEqual(decision["controller"]["skipped"], 1)
                self.assertEqual(controller._accepted, 0)
                self.assertEqual(controller._inflight, 0)
                if reason == "prefetch_deadline_too_tight":
                    self.assertFalse(decision["deadline"]["admitted"])

    async def test_equal_pressure_limits_admit_but_full_inflight_rejects(self):
        ns, request, raw, _, controller, _, _ = PrefetchAdmissionOwnershipTests().setup_endpoint()
        request.lead_ms = controller.min_lead_ms
        ns["_build_omni_load_snapshot"] = lambda state: {
            "orchestrator_request_qsize": controller.max_request_qsize,
            "max_stage0_active_per_replica": controller.max_stage0_active_per_replica,
            "max_audio_active_per_replica": controller.max_audio_active_per_replica,
        }
        admitted, first = await controller.decide(request, raw.app.state)
        self.assertTrue(admitted)
        self.assertTrue(first["inflight_reserved"])
        admitted, second = await controller.decide(request, raw.app.state)
        self.assertFalse(admitted)
        self.assertEqual(second["reason"], "prefetch_inflight_limit")
        self.assertFalse(second["inflight_reserved"])
        self.assertEqual(controller._inflight, 1)
        self.assertEqual(controller._accepted, 1)
        self.assertEqual(controller._skipped, 1)

    async def test_concurrent_rejections_never_release_an_existing_slot(self):
        _, request, raw, _, controller, _, _ = PrefetchAdmissionOwnershipTests().setup_endpoint()
        controller._inflight = 1
        request.lead_ms = None
        results = await asyncio.gather(*(controller.decide(request, raw.app.state) for _ in range(16)))
        self.assertTrue(all(not admitted for admitted, _ in results))
        self.assertEqual({decision["reason"] for _, decision in results}, {"vad_lead_too_short"})
        self.assertEqual(controller._skipped, 16)
        self.assertEqual(controller._inflight, 1)
        self.assertEqual(controller._accepted, 0)

    async def test_cancel_while_rejection_waits_for_lock_preserves_counters(self):
        _, request, raw, _, controller, _, _ = PrefetchAdmissionOwnershipTests().setup_endpoint()
        request.lead_ms = None
        await controller._lock.acquire()
        task = asyncio.create_task(controller.decide(request, raw.app.state))
        try:
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            task.cancel("cancel admission wait")
            with self.assertRaisesRegex(asyncio.CancelledError, "cancel admission wait"):
                await task
            self.assertEqual(controller._skipped, 0)
            self.assertEqual(controller._accepted, 0)
            self.assertEqual(controller._inflight, 0)
        finally:
            controller._lock.release()


if __name__ == "__main__":
    unittest.main(verbosity=2)
