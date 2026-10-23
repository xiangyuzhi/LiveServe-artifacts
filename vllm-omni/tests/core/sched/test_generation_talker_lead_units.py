from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.request import Request, RequestStatus

from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.core.sched.talker_lead_governor import TalkerLeadGovernor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_talker_lead_governor_prefers_omni_lead_units():
    governor = TalkerLeadGovernor(L_max=6, speaker_rate=25)
    request = SimpleNamespace(
        request_id="req-generation",
        num_output_tokens=0,
        omni_lead_units=0,
    )

    assert governor.filter_running([request], now=0.0, pending_count=1) == []

    request.omni_lead_units = 20

    assert governor.filter_running([request], now=0.001, pending_count=1) == [request]


def test_generation_scheduler_records_pooler_audio_as_lead_units():
    scheduler = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    scheduler._generation_lead_speaker_rate = 25
    request = SimpleNamespace(request_id="req-audio")

    scheduler._record_pooler_lead_units(
        request,
        {
            "audio": torch.zeros((1, 1, 44100)),
            "sr": torch.tensor(44100),
        },
    )

    assert request.omni_lead_units == 25


def test_generation_placeholder_extends_past_computed_tokens():
    request = SimpleNamespace(
        prompt_token_ids=[1, 2],
        _all_token_ids=[1, 2],
        num_computed_tokens=5,
    )

    OmniGenerationScheduler._append_generation_placeholder(request)

    assert len(request.prompt_token_ids) == 6
    assert len(request._all_token_ids) == 6
    assert request.prompt_token_ids[-4:] == [0, 0, 0, 0]


class _FakeQueue:
    def __init__(self, requests):
        self.requests = list(requests)

    def __bool__(self):
        return bool(self.requests)

    def __len__(self):
        return len(self.requests)

    def __iter__(self):
        return iter(self.requests)

    def peek_request(self):
        return self.requests[0]

    def pop_request(self):
        return self.requests.pop(0)

    def prepend_requests(self, requests):
        self.requests = list(requests) + self.requests

    def add_request(self, request):
        self.requests.append(request)


class _FakeBlocks:
    def __init__(self, block_ids):
        self.block_ids = block_ids

    def get_block_ids(self, allow_none=False):
        if allow_none and all(len(group) == 0 for group in self.block_ids):
            return None
        return self.block_ids


class _FakeKVCacheManager:
    def new_step_starts(self):
        pass

    def allocate_slots(self, request, num_new_tokens, num_lookahead_tokens=0):
        return _FakeBlocks(([],))

    def get_blocks(self, request_id):
        return _FakeBlocks(([42],))

    def get_num_common_prefix_blocks(self, request_id):
        return [0]

    def take_new_block_ids(self):
        return None


class _FakeEmptyKVCacheManager(_FakeKVCacheManager):
    def get_blocks(self, request_id):
        return _FakeBlocks(([],))


def _make_async_generation_scheduler(request, kv_cache_manager):
    scheduler = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    scheduler.max_num_scheduled_tokens = 1_000_000
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.kv_cache_manager = kv_cache_manager
    scheduler.running = []
    scheduler.waiting = _FakeQueue([request])
    scheduler.max_num_running_reqs = 1
    scheduler.requests = {request.request_id: request}
    scheduler.chunk_transfer_adapter = SimpleNamespace(
        finished_requests=set(),
        process_pending_chunks=lambda waiting, running: None,
        postprocess_scheduler_output=lambda output: None,
        restore_queues=lambda waiting, running: None,
    )
    scheduler._defer_lead_bounded_running = lambda: []
    scheduler._restore_deferred_running = lambda deferred: None
    scheduler.num_lookahead_tokens = 0
    scheduler.log_stats = False
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    scheduler.use_v2_model_runner = False
    scheduler.use_pp = False
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=False)
    scheduler.prev_step_scheduled_req_ids = set()
    scheduler.needs_kv_cache_zeroing = False
    scheduler.finished_req_ids = set()
    scheduler.encoder_cache_manager = SimpleNamespace(get_freed_mm_hashes=lambda: [])
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.policy = SchedulingPolicy.FCFS
    return scheduler


def test_waiting_async_generation_progress_resumes_as_cached_request():
    request = Request(
        request_id="req-resume",
        prompt_token_ids=[0, 0, 0, 0, 0, 0],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    request.num_computed_tokens = 5
    request.status = RequestStatus.PREEMPTED

    scheduler = _make_async_generation_scheduler(request, _FakeKVCacheManager())

    scheduler_output = OmniGenerationScheduler.schedule(scheduler)

    assert scheduler_output.scheduled_new_reqs == []
    assert scheduler_output.scheduled_cached_reqs.req_ids == ["req-resume"]
    assert scheduler_output.scheduled_cached_reqs.resumed_req_ids == {"req-resume"}
    assert scheduler_output.scheduled_cached_reqs.new_block_ids == [([42],)]
    assert scheduler_output.scheduled_cached_reqs.num_computed_tokens == [5]
    assert scheduler_output.num_scheduled_tokens == {"req-resume": 1}
    assert request.status == RequestStatus.RUNNING


def test_resumed_async_generation_keeps_empty_block_table_non_none():
    request = Request(
        request_id="req-resume-empty-blocks",
        prompt_token_ids=[0, 0],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    request.num_computed_tokens = 1
    request.status = RequestStatus.PREEMPTED

    scheduler = _make_async_generation_scheduler(request, _FakeEmptyKVCacheManager())

    scheduler_output = OmniGenerationScheduler.schedule(scheduler)

    assert scheduler_output.scheduled_cached_reqs.req_ids == ["req-resume-empty-blocks"]
    assert scheduler_output.scheduled_cached_reqs.resumed_req_ids == {"req-resume-empty-blocks"}
    assert scheduler_output.scheduled_cached_reqs.new_block_ids == [([],)]
