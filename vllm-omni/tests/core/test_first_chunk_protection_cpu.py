"""Real scheduler/adapter methods and governor, with CPU-only dependency stubs.

The Qwen chunk processor's frame normalization is stubbed; its actual accumulation,
chunk-size gate and payload construction run. No torch/CUDA import is required.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import threading
import unittest
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]


class Status(IntEnum):
    WAITING = 0
    RUNNING = 1
    WAITING_FOR_CHUNK = 2
    FINISHED_ABORTED = 3

    @classmethod
    def is_finished(cls, status):
        return status == cls.FINISHED_ABORTED


class Clock:
    value = 0.0

    def monotonic_ns(self):
        return int(self.value * 1e9)

    def time(self):
        return self.value


def real_class(path, name, methods, namespace, base=object):
    tree = ast.parse(path.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    selected = [n for n in original.body if getattr(n, "name", None) in methods]
    cls = ast.ClassDef(name=name, bases=[ast.Name(id="TestBase", ctx=ast.Load())], keywords=[],
                       body=selected, decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    namespace["TestBase"] = base
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def governor_class():
    path = ROOT / "vllm_omni/core/sched/talker_lead_governor.py"
    spec = importlib.util.spec_from_file_location("first_chunk_cpu_governor", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TalkerLeadGovernor


Governor = governor_class()


@dataclass
class Output:
    num_scheduled_tokens: dict
    scheduled_new_reqs: list
    scheduled_cached_reqs: object = None


class BaseScheduler:
    def schedule(self):
        self.running.extend(self.waiting)
        self.waiting.clear()
        return Output({r.request_id: 1 for r in self.running}, [])

    def finish_requests(self, ids, status):
        self.finish_seen = tuple(self.requests) if ids is None else (ids,) if isinstance(ids, str) else tuple(ids)
        aborted = []
        for req_id in self.finish_seen:
            request = self.requests.pop(req_id, None)
            if request is not None:
                aborted.append((req_id, 0))
        return aborted


class FirstChunkTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.events = []
        ns = {"__package__": "cpu_test_scheduler", "time": self.clock, "deque": deque,
              "monotonic_ns": self.clock.monotonic_ns, "RequestStatus": Status, "Iterator": Iterator,
              "trace_enabled": lambda: True, "trace_emit": lambda event, **kw: self.events.append((event, kw)),
              "unflatten_payload": lambda p: p, "logger": Mock(), "init_logger": lambda _: Mock(),
              "torch": SimpleNamespace(Tensor=type("Tensor", (), {}), tensor=lambda x, **_: x, bool=bool)}
        adapter_methods = {"_observe_first_chunk_ready", "_first_chunk_generation", "first_chunk_pending_ready",
                           "_forget_first_chunk", "_observe_first_chunk_published", "save_async", "_send_single_request",
                           "cleanup_receiver", "cleanup_sender", "cleanup", "finish_requests", "load_async",
                           "_process_chunk_queue", "_clear_chunk_ready", "restore_queues"}
        Adapter = real_class(ROOT / "vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py",
                             "OmniChunkTransferAdapter", adapter_methods, dict(ns))
        self.adapter = Adapter()
        self.adapter.first_chunk_observation_enabled = True
        self.adapter._first_chunk_lock = threading.Lock()
        self.adapter._first_chunk_generation_seq = 0
        self.adapter._first_chunk_state = {}
        self.adapter._interaction_job_seq = 0
        self.adapter._interaction_trace_context = {"stage_id": 1, "replica_id": 0}
        self.adapter._pending_save_reqs = deque()
        self.adapter._pending_load_reqs = deque()
        self.adapter._save_cond = threading.Condition()
        for key in ("put_req_chunk", "get_req_chunk"):
            setattr(self.adapter, key, defaultdict(int))
        self.adapter.code_prompt_token_ids = defaultdict(list)
        for key in ("request_payload", "request_ids_mapping", "requests_origin_status", "ready_chunk_info", "inflight_chunk_info"):
            setattr(self.adapter, key, {})
        for key in ("finished_requests", "requests_with_ready_chunks", "_cancelled_load_reqs", "_finished_load_reqs"):
            setattr(self.adapter, key, set())
        self.adapter.connector = SimpleNamespace(stage_id=1, config={"extra": {"codec_first_chunk_frames": 2, "codec_chunk_frames": 2}},
                                                 put=Mock(return_value=(True, 16, {})))
        self.adapter.custom_process_next_stage_input_func = lambda **_: self.payload()
        self.adapter.process_pending_chunks = Mock()
        self.adapter.restore_queues = Mock()
        self.adapter.postprocess_scheduler_output = Mock()
        ns.update({"SchedulerOutput": Output, "OmniSchedulerOutput": lambda **kw: SimpleNamespace(**kw)})
        self.Scheduler = real_class(ROOT / "vllm_omni/core/sched/omni_ar_scheduler.py", "OmniARScheduler",
                                    {"schedule", "_bp_first_chunk_protection", "finish_requests"}, ns, BaseScheduler)
        self.scheduler = self.Scheduler()
        self.scheduler.chunk_transfer_adapter = self.adapter
        self.scheduler._bp_first_chunk_protect_ms = 800
        self.scheduler._bp_resume_on_empty_backlog = True
        self.scheduler._bp_governor_role = "talker"
        self.scheduler.running = []
        self.scheduler.waiting = []
        self.scheduler.requests = {}
        self.scheduler._interaction_trace_context = {"stage_id": 1, "replica_id": 0}
        self.scheduler._interaction_job_seq = 0
        self.scheduler._talker_lead_governor = Governor(L_max=12, L_lo=6, hold_max=75, resume_on_empty_backlog=True)
        for key in ("_bp_role_prefill_priority", "_bp_role_include_chunk_waiting_in_pending", "_bp_kv_aware",
                    "_bp_role_pressure_fast_warmup_arrival_window_s", "_bp_role_pressure_fast_warmup_include_running_arrivals",
                    "_bp_role_pressure_fast_warmup_arrival_history", "_bp_role_waiting_short_max_output_tokens",
                    "_bp_role_waiting_short_min_count", "_bp_role_waiting_short_entry_only", "_bp_role_skip_global_suppressed_defer"):
            setattr(self.scheduler, key, False)
        self.scheduler._bp_pending_age_s = lambda: None
        self.scheduler._bp_recent_waiting_arrivals = lambda *_, **__: 0
        self.scheduler._bp_video_mix_active = lambda: False
        self.scheduler._omni_cancel_prefetch_under_live_pressure = lambda: None
        self.scheduler._omni_release_expired_gpu_pins = lambda: None

    @staticmethod
    def payload(empty=False, finished=False):
        return {"codes": {"audio": [] if empty else [1, 2, 3, 4]},
                "meta": {"codec_chunk_frames": 0 if empty else 2, "finished": finished}}

    def request(self, req_id, status=Status.RUNNING, tokens=1):
        request = SimpleNamespace(request_id=req_id, external_req_id="external-" + req_id,
                                  status=status, num_output_tokens=tokens, num_computed_tokens=1)
        request.is_finished = lambda: Status.is_finished(request.status)
        return request

    def register(self, request):
        self.adapter._observe_first_chunk_ready(request)

    def enqueue(self, request, frames=None):
        self.adapter.save_async({"codes": {"audio": frames or [[1] * 8]}}, request)
        return self.adapter._pending_save_reqs.popleft()

    def test_enqueue_empty_payload_and_failed_put_are_not_publication(self):
        r = self.request("new", tokens=999)
        self.register(r)
        task = self.enqueue(r)
        self.assertIsNone(self.adapter._first_chunk_state["new"]["published_ns"])
        self.adapter.custom_process_next_stage_input_func = lambda **_: None
        self.adapter._send_single_request(task)
        self.adapter.connector.put.assert_not_called()
        self.adapter.custom_process_next_stage_input_func = lambda **_: self.payload()
        self.adapter.connector.put.return_value = (False, 0, {})
        self.adapter._send_single_request(task)
        self.assertIsNone(self.adapter._first_chunk_state["new"]["published_ns"])
        self.adapter.connector.put.return_value = (True, 16, {})
        self.adapter._send_single_request(task)
        self.assertEqual(self.adapter._first_chunk_state["new"]["published_ns"], 0)
        self.assertEqual(sum(e == "connector_first_chunk_published" for e, _ in self.events), 1)

    def test_actual_qwen_frame_gate_ignores_large_output_token_count(self):
        path = ROOT / "vllm_omni/model_executor/stage_input_processors/qwen3_omni.py"
        fn = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == "talker2code2wav_async_chunk")
        ns = {"time": self.clock, "_qwen3_omni_audio_code_frames": lambda frames: frames,
              "_codec_process_trace_enabled": lambda: False, "_cfg_bool": lambda cfg, k, d: cfg.get(k, d),
              "_maybe_trace_code2wav_process": lambda _: None, "_maybe_trace_code2wav_chunk": lambda _: None,
              "torch": SimpleNamespace(tensor=lambda x, **_: x, bool=bool)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])), str(path), "exec"), ns)
        self.adapter.custom_process_next_stage_input_func = ns["talker2code2wav_async_chunk"]
        r = self.request("new", tokens=999)
        self.register(r)
        self.adapter._send_single_request(self.enqueue(r))
        self.adapter.connector.put.assert_not_called()
        self.adapter._send_single_request(self.enqueue(r))
        self.assertEqual(self.adapter.connector.put.call_count, 1)
        self.assertEqual(self.adapter._first_chunk_state["new"]["published_ns"], 0)

    def test_cancel_during_put_cannot_publish_or_remove_reused_generation(self):
        r = self.request("same")
        self.register(r)
        task = self.enqueue(r)
        self.adapter.custom_process_next_stage_input_func = lambda **_: self.payload(finished=True)
        entered, release = threading.Event(), threading.Event()

        def put(**_):
            entered.set()
            self.assertTrue(release.wait(2))
            return True, 16, {}

        self.adapter.connector.put = put
        worker = threading.Thread(target=self.adapter._send_single_request, args=(task,))
        worker.start()
        self.assertTrue(entered.wait(2))
        self.adapter.finish_requests([r.request_id], Status.FINISHED_ABORTED, {r.request_id: r})
        self.register(r)
        new_generation = self.adapter._first_chunk_state["same"]["generation"]
        release.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.adapter._first_chunk_state["same"]["generation"], new_generation)
        self.assertIsNone(self.adapter._first_chunk_state["same"]["published_ns"])

    def test_empty_terminal_is_not_publication(self):
        r = self.request("new")
        self.register(r)
        self.adapter.custom_process_next_stage_input_func = lambda **_: self.payload(empty=True, finished=True)
        self.adapter._send_single_request(self.enqueue(r))
        self.assertNotIn("new", self.adapter._first_chunk_state)
        self.assertFalse(any(e == "connector_first_chunk_published" for e, _ in self.events))

    def test_ready_deadline_does_not_refresh_and_waiting_for_chunk_is_excluded(self):
        r = self.request("new")
        self.register(r)
        self.scheduler.running = [r]
        self.clock.value = .79
        self.register(r)
        self.assertEqual(self.scheduler._bp_first_chunk_protection()[0], {"new"})
        r.status = Status.WAITING_FOR_CHUNK
        self.assertEqual(self.scheduler._bp_first_chunk_protection()[0], set())
        r.status = Status.RUNNING
        self.clock.value = .8
        self.assertEqual(self.scheduler._bp_first_chunk_protection()[0], set())

    def test_real_schedule_protects_newcomer_then_releases_after_publish(self):
        old, new = self.request("old"), self.request("new", Status.WAITING, tokens=999)
        s = self.scheduler
        s.running, s.waiting, s.requests = [old], [], {"old": old, "new": new}
        s._talker_lead_governor.filter_running([old], now=0, pending_count=0)
        old.num_output_tokens = 100
        self.clock.value = .1
        self.register(new)
        s.waiting = [new]
        # Use the same simulated monotonic clock in the actual governor.
        governor_module = sys.modules[Governor.__module__]
        original_clock = governor_module.time
        governor_module.time = SimpleNamespace(monotonic=lambda: self.clock.value)
        try:
            self.assertEqual(set(s.schedule().num_scheduled_tokens), {"new"})
            self.clock.value = .2
            self.assertEqual(set(s.schedule().num_scheduled_tokens), {"new"})
            self.adapter._send_single_request(self.enqueue(new))
            self.clock.value = .3
            self.assertEqual(set(s.schedule().num_scheduled_tokens), {"old", "new"})
        finally:
            governor_module.time = original_clock

    def test_episode_wall_time_cap_survives_relaying_newcomers(self):
        g = Governor(L_max=12, L_lo=6, hold_max=75, resume_on_empty_backlog=True)
        r = self.request("old")
        g.filter_running([r], now=0, pending_count=0)
        r.num_output_tokens = 100
        self.assertEqual(g.filter_running([r], now=.1, pending_count=1, defer_time_limit_s=.8), [r])
        self.assertEqual(g.filter_running([r], now=.8, pending_count=0, empty_backlog_protected=True, defer_time_limit_s=.8), [r])
        self.assertEqual(g.filter_running([r], now=.91, pending_count=0, empty_backlog_protected=True, defer_time_limit_s=.8), [])
        self.assertEqual(g.snapshot()["time_budget_force_resume"], 1)
        # A later positive backlog may start another episode; the budget is
        # deliberately not a lifetime cap for the entire request.
        self.assertEqual(g.filter_running([r], now=.92, pending_count=1, defer_time_limit_s=.8), [r])
        self.assertEqual(g.snapshot()["defer_episodes"], 2)

    def _reenter_after_real_chunk_wait(self):
        """Real load/WFC/ready transitions around the actual schedule method."""
        old, new = self.request("old", tokens=100), self.request("new", Status.WAITING, tokens=999)
        s, a = self.scheduler, self.adapter
        s.running, s.waiting, s.requests = [old], [new], {"old": old, "new": new}
        a.requests_with_ready_chunks.update({"old", "new"})
        a.waiting_for_chunk_waiting_requests = deque()
        a.waiting_for_chunk_running_requests = deque()
        a._recv_cond = threading.Condition()
        a._trace_stage2_event = Mock()
        a._chunk_priority_key = Mock(return_value=0)
        a.stage2_priority_policy, a.stage2_priority_first_chunks = "", 0
        a.restore_queues = type(a).restore_queues.__get__(a)

        def process(waiting, running):
            a._process_chunk_queue(waiting, a.waiting_for_chunk_waiting_requests, Status.WAITING, a._finished_load_reqs)
            a._process_chunk_queue(running, a.waiting_for_chunk_running_requests, Status.RUNNING, a._finished_load_reqs)

        a.process_pending_chunks = process
        governor_module = sys.modules[Governor.__module__]
        original_clock = governor_module.time
        governor_module.time = SimpleNamespace(monotonic=lambda: self.clock.value)
        self.addCleanup(setattr, governor_module, "time", original_clock)
        self.clock.value = .1
        self.register(new)
        self.assertEqual(set(s.schedule().num_scheduled_tokens), {"new"})
        # Consuming the real input chunk clears readiness; the next scheduler
        # call's real adapter method enqueues load and removes the WFC newcomer.
        a._clear_chunk_ready(SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(req_id="new")], scheduled_cached_reqs=None))
        self.clock.value = .2
        self.assertEqual(set(s.schedule().num_scheduled_tokens), {"old"})
        self.assertEqual(new.status, Status.WAITING_FOR_CHUNK)
        self.assertEqual(s._bp_first_chunk_protection()[0], set())
        a._finished_load_reqs.add("new")
        self.clock.value = .3
        self.assertEqual(set(s.schedule().num_scheduled_tokens), {"new"})
        self.assertEqual(new.status, Status.RUNNING)
        self.assertEqual(a._first_chunk_state["new"]["first_ready_ns"], 100_000_000)
        return old, new

    def test_real_chunk_wait_reentry_closes_after_successful_publish(self):
        _, new = self._reenter_after_real_chunk_wait()
        self.adapter._send_single_request(self.enqueue(new))
        self.clock.value = .4
        self.assertEqual(set(self.scheduler.schedule().num_scheduled_tokens), {"old", "new"})
        self.assertEqual(self.scheduler._bp_first_chunk_protection()[0], set())

    def test_real_chunk_wait_reentry_keeps_original_deadline(self):
        _, new = self._reenter_after_real_chunk_wait()
        self.clock.value = .9
        self.register(new)
        self.assertEqual(set(self.scheduler.schedule().num_scheduled_tokens), {"old", "new"})
        self.assertEqual(self.scheduler._bp_first_chunk_protection()[0], set())
        self.assertEqual(self.adapter._first_chunk_state["new"]["first_ready_ns"], 100_000_000)

    def test_first_chunk_entry_retains_min_pending_and_min_running(self):
        old, new = self.request("old", tokens=100), self.request("new")
        for option, threshold, running in (("min_pending_count", 2, [old, new]),
                                           ("min_running_count", 3, [old, new])):
            with self.subTest(option=option):
                g = Governor(L_max=12, L_lo=6, resume_on_empty_backlog=True, **{option: threshold})
                kwargs = dict(now=0, pending_count=0, empty_backlog_protected=True, defer_allowed=lambda r: r is old)
                self.assertEqual(g.filter_running(running, **kwargs), [])
                if option == "min_pending_count":
                    kwargs["pending_count"] = 2
                else:
                    running = [old, new, self.request("third")]
                self.assertEqual(g.filter_running(running, **kwargs), [old])

    def test_first_chunk_entry_retains_pressure_and_age_gates(self):
        old = self.request("old", tokens=100)
        kwargs = dict(pending_count=0, empty_backlog_protected=True)
        for option in ("min_pressure_active_calls", "min_backlog_active_steps"):
            with self.subTest(option=option):
                g = Governor(L_max=12, L_lo=6, **{option: 2})
                self.assertEqual(g.filter_running([old], now=0, **kwargs), [])
                self.assertEqual(g.filter_running([old], now=.01, **kwargs), [old])
        g = Governor(L_max=12, L_lo=6, min_pending_age_s=.2)
        self.assertEqual(g.filter_running([old], now=0, **kwargs), [])
        self.assertEqual(g.filter_running([old], now=.01, pending_age_s=.3, **kwargs), [old])

    def test_first_chunk_pressure_does_not_reset_episode_budget(self):
        g = Governor(L_max=12, L_lo=6, hold_max=1, resume_on_empty_backlog=True,
                     max_defer_episodes_per_window=1, defer_budget_empty_reset_steps=1)
        old = self.request("old", tokens=100)
        kwargs = dict(pending_count=0, empty_backlog_protected=True)
        self.assertEqual(g.filter_running([old], now=0, **kwargs), [old])
        self.assertEqual(g.filter_running([old], now=.01, **kwargs), [])  # hold forces resume
        self.assertEqual(g.filter_running([old], now=.02, **kwargs), [])  # budget prevents re-entry
        self.assertEqual(g.snapshot()["defer_budget_resets"], 0)
        self.assertEqual(g.snapshot()["defer_episodes"], 1)
        g.filter_running([old], now=.03, pending_count=0)  # one real empty observation
        self.assertEqual(g.filter_running([old], now=.04, **kwargs), [])
        self.assertEqual(g.snapshot()["defer_budget_resets"], 0)
        g.filter_running([old], now=.05, pending_count=0)
        g.filter_running([old], now=.06, pending_count=0)  # ordinary reset semantics remain
        self.assertEqual(g.snapshot()["defer_budget_resets"], 1)
        self.assertEqual(g.filter_running([old], now=.07, **kwargs), [old])

    def test_default_off_keeps_empty_pending_entry_closed(self):
        old, new = self.request("old", tokens=100), self.request("new")
        self.scheduler.running = [old, new]
        self.scheduler.requests = {"old": old, "new": new}
        self.register(new)
        self.scheduler._bp_first_chunk_protect_ms = 0
        self.assertEqual(set(self.scheduler.schedule().num_scheduled_tokens), {"old", "new"})
        self.assertEqual(self.scheduler._talker_lead_governor.snapshot()["defer_episodes"], 0)

    def test_governor_policy_order_and_entry_only_scope(self):
        old = self.request("old", tokens=100)
        g = Governor(L_max=12, L_lo=6)
        calls = []

        def policy(allowed):
            def check(request):
                self.assertIs(request, old)
                calls.append(allowed)
                return allowed
            return check

        self.assertEqual(g.filter_running([old], now=0, pending_count=1,
                                         defer_allowed=policy(False), defer_entry_allowed=policy(True)), [])
        self.assertEqual(calls, [False])  # A general veto skips the entry callback.
        calls.clear()
        self.assertEqual(g.filter_running([old], now=.01, pending_count=1,
                                         defer_allowed=policy(True), defer_entry_allowed=policy(False)), [])
        self.assertEqual(calls, [True, False])
        self.assertEqual(g.snapshot()["entry_policy_suppressed"], 1)
        self.assertEqual(g.filter_running([old], now=.02, pending_count=1), [old])
        entry_only = Mock(side_effect=AssertionError("entry callback cannot resume an existing episode"))
        self.assertEqual(g.filter_running([old], now=.03, pending_count=1,
                                         defer_entry_allowed=entry_only), [old])
        entry_only.assert_not_called()
        self.assertEqual(g.filter_running([old], now=.04, pending_count=1,
                                         defer_allowed=policy(False)), [])
        self.assertEqual(g.snapshot()["policy_force_resume"], 1)

    def test_governor_policy_exceptions_preserve_fail_open(self):
        old = self.request("old", tokens=100)
        for option in ("defer_allowed", "defer_entry_allowed"):
            with self.subTest(option=option):
                g = Governor(L_max=12, L_lo=6)
                failing = Mock(side_effect=ValueError("policy unavailable"))
                self.assertEqual(g.filter_running([old], now=0, pending_count=1, **{option: failing}), [old])
                failing.assert_called_once_with(old)
                self.assertEqual(g.snapshot()["policy_suppressed"], 0)

    def test_default_off_ignores_observer_and_preserves_governor(self):
        self.scheduler._bp_first_chunk_protect_ms = 0
        self.adapter.first_chunk_pending_ready = Mock(side_effect=AssertionError("must not observe"))
        self.assertEqual(self.scheduler._bp_first_chunk_protection(), (set(), None))
        self.scheduler._bp_first_chunk_protect_ms = 800
        self.scheduler._bp_resume_on_empty_backlog = False
        self.assertEqual(self.scheduler._bp_first_chunk_protection(), (set(), None))

    def test_finish_default_off_preserves_original_arguments(self):
        self.Scheduler.finish_requests.__globals__["trace_enabled"] = lambda: False
        self.adapter.first_chunk_observation_enabled = False
        self.adapter.finish_requests = Mock()
        old_method = BaseScheduler.finish_requests
        seen = []
        BaseScheduler.finish_requests = lambda owner, ids, status: seen.append(ids)
        try:
            for source in (None, "original-string", ["a", "b"]):
                self.scheduler.finish_requests(source, Status.FINISHED_ABORTED)
                self.assertIs(seen[-1], source)
                self.assertIs(self.adapter.finish_requests.call_args.args[0], source)
        finally:
            BaseScheduler.finish_requests = old_method

    def test_finish_default_off_iterator_with_adapter_cancels_all_targets(self):
        self.Scheduler.finish_requests.__globals__["trace_enabled"] = lambda: False
        self.adapter.first_chunk_observation_enabled = False
        for source in ((x for x in ["a", "b"]), iter(["a", "b"])):
            with self.subTest(iterator_type=type(source).__name__):
                requests = {k: self.request(k, Status.WAITING_FOR_CHUNK) for k in ("a", "b")}
                self.scheduler.requests = requests.copy()
                self.adapter.requests_origin_status = {k: Status.RUNNING for k in requests}
                result = self.scheduler.finish_requests(source, Status.FINISHED_ABORTED)
                self.assertEqual(result, [("a", 0), ("b", 0)])
                self.assertEqual(self.scheduler.requests, {})
                self.assertTrue(all(r.status == Status.RUNNING for r in requests.values()))
                self.assertEqual(list(source), [])
                self.assertEqual(self.events, [])

    def test_finish_default_off_iterator_without_adapter_is_consumed_once(self):
        self.Scheduler.finish_requests.__globals__["trace_enabled"] = lambda: False
        self.scheduler.chunk_transfer_adapter = None
        self.scheduler.requests = {k: self.request(k) for k in ("a", "b")}
        consumed = []

        def targets():
            for req_id in ("a", "b"):
                consumed.append(req_id)
                yield req_id

        source = targets()
        old_method = BaseScheduler.finish_requests
        seen = []

        def finish(owner, ids, status):
            seen.append(ids)
            return old_method(owner, ids, status)

        BaseScheduler.finish_requests = finish
        try:
            self.assertEqual(self.scheduler.finish_requests(source, Status.FINISHED_ABORTED), [("a", 0), ("b", 0)])
        finally:
            BaseScheduler.finish_requests = old_method
        self.assertIs(seen[0], source)
        self.assertEqual(consumed, ["a", "b"])
        self.assertEqual(self.scheduler.requests, {})

    def test_finish_default_off_empty_and_repeated_abort_are_idempotent(self):
        self.Scheduler.finish_requests.__globals__["trace_enabled"] = lambda: False
        self.adapter.first_chunk_observation_enabled = False
        self.scheduler.requests = {"a": self.request("a")}
        self.assertEqual(self.scheduler.finish_requests(iter(()), Status.FINISHED_ABORTED), [])
        self.assertIn("a", self.scheduler.requests)
        self.assertEqual(self.scheduler.finish_requests(iter(["a", "a", "unknown"]), Status.FINISHED_ABORTED), [("a", 0)])
        self.assertEqual(self.scheduler.finish_requests(iter(["a"]), Status.FINISHED_ABORTED), [])
        self.assertEqual(self.scheduler.requests, {})

    def test_finish_base_exception_is_propagated_without_returned_event(self):
        self.adapter.first_chunk_observation_enabled = False
        old_method = BaseScheduler.finish_requests
        failure = RuntimeError("base finish failed")

        def fail(owner, ids, status):
            self.assertEqual(tuple(ids), ("a",))
            raise failure

        BaseScheduler.finish_requests = fail
        try:
            for tracing in (False, True):
                with self.subTest(tracing=tracing):
                    self.Scheduler.finish_requests.__globals__["trace_enabled"] = lambda: tracing
                    self.events.clear()
                    self.scheduler.requests = {"a": self.request("a")}
                    with self.assertRaises(RuntimeError) as raised:
                        self.scheduler.finish_requests(iter(["a"]), Status.FINISHED_ABORTED)
                    self.assertIs(raised.exception, failure)
                    self.assertIn("a", self.scheduler.requests)
                    self.assertFalse(any(event == "scheduler_finish_returned" for event, _ in self.events))
        finally:
            BaseScheduler.finish_requests = old_method

    def test_finish_generator_and_abort_all_have_matching_trace_ids(self):
        for source in (None, (x for x in ["a", "b"])):
            self.events.clear()
            self.scheduler.requests = {k: self.request(k) for k in ("a", "b")}
            self.scheduler.finish_requests(source, Status.FINISHED_ABORTED)
            self.assertEqual(set(self.scheduler.finish_seen), {"a", "b"})
            returned = [r for e, r in self.events if e == "scheduler_finish_returned"]
            self.assertEqual(len(returned), 2)
            self.assertTrue(all(r["was_known"] and not r["remains_known"] for r in returned))


if __name__ == "__main__":
    unittest.main()
