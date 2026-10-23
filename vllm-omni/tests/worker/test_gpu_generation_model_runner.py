from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyInputBatch:
    def __init__(self):
        self.req_ids = ["req-1"]
        self.req_id_to_index = {"req-1": 0}
        self.num_reqs = 1
        self.vocab_size = 10


class _PromptUpdateInputBatch(_DummyInputBatch):
    def __init__(self):
        super().__init__()
        self.num_prompt_tokens = np.zeros(1, dtype=np.int32)
        self.num_tokens_no_spec = np.zeros(1, dtype=np.int32)
        self.token_ids_cpu = np.zeros((1, 8), dtype=np.int32)
        self.is_token_ids = np.zeros((1, 8), dtype=bool)
        self.removed = []
        self.added = []

    def remove_request(self, req_id):
        self.removed.append(req_id)
        raise AssertionError("remove_request should be left to _update_states")

    def add_request(self, req_state):
        self.added.append(req_state.req_id)
        raise AssertionError("add_request should be left to _update_states")


class _ReqState:
    def __init__(self):
        self.req_id = "req-1"
        self.prompt_token_ids = [0]
        self.num_prompt_tokens = 1
        self.output_token_ids = []

    @property
    def num_tokens(self):
        return self.num_prompt_tokens + len(self.output_token_ids)


class _FinishedHookModel:
    def __init__(self):
        self.finished_req_ids = []

    def on_requests_finished(self, req_ids):
        self.finished_req_ids.extend(req_ids)


def _make_runner(multimodal_outputs):
    runner = object.__new__(GPUGenerationModelRunner)
    runner.execute_model_state = (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        multimodal_outputs,
        None,
    )
    runner.kv_connector_output = None
    runner.input_batch = _DummyInputBatch()
    runner.use_async_scheduling = False
    runner.device = torch.device("cpu")
    runner.supports_mm_inputs = False
    runner.speculative_config = None
    return runner


def test_sample_tokens_tensor_output():
    multimodal_outputs = torch.randn(1, 2, 3)
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.pooler_output) == 1
    assert output.pooler_output[0]["model_outputs"].shape == (2, 3)


def test_sample_tokens_list_output():
    multimodal_outputs = [torch.randn(2, 1)]
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.pooler_output) == 1
    assert output.pooler_output[0]["model_outputs"].shape == (2, 1)


def test_sample_tokens_list_allows_none_output():
    multimodal_outputs = [None]
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.pooler_output) == 1
    assert output.pooler_output[0]["model_outputs"] is None


def test_sample_tokens_dict_output():
    multimodal_outputs = {"audio": torch.randn(1, 4), "unused": None}
    runner = _make_runner(multimodal_outputs)

    output = GPUGenerationModelRunner.sample_tokens(runner)

    assert len(output.pooler_output) == 1
    assert "audio" in output.pooler_output[0]
    assert "unused" not in output.pooler_output[0]
    assert output.pooler_output[0]["audio"].shape == (1, 4)


def test_update_request_states_only_refreshes_cached_prompt_buffers():
    runner = object.__new__(GPUGenerationModelRunner)
    runner.model = object()
    runner.model_config = SimpleNamespace(model_stage="ming_tts")
    runner.input_batch = _PromptUpdateInputBatch()
    runner.requests = {"req-1": _ReqState()}
    runner.enable_prompt_embeds = True

    scheduler_output = SimpleNamespace(
        finished_req_ids=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-1"],
            prompt_token_ids={"req-1": [2, 3, 4]},
        ),
    )

    GPUGenerationModelRunner._update_request_states(runner, scheduler_output)

    req_state = runner.requests["req-1"]
    assert req_state.prompt_token_ids == [2, 3, 4]
    assert req_state.num_prompt_tokens == 3
    assert runner.input_batch.num_prompt_tokens[0] == 3
    assert runner.input_batch.num_tokens_no_spec[0] == 3
    assert runner.input_batch.token_ids_cpu[0, :3].tolist() == [2, 3, 4]
    assert runner.input_batch.is_token_ids[0, :3].tolist() == [True, True, True]
    assert runner.input_batch.removed == []
    assert runner.input_batch.added == []


def test_update_request_states_skips_prompt_refresh_for_unrelated_model():
    runner = object.__new__(GPUGenerationModelRunner)
    runner.model = _FinishedHookModel()
    runner.model_config = SimpleNamespace(
        model_stage="code2wav",
        architectures=["UnrelatedGenerationModel"],
    )
    runner.input_batch = _PromptUpdateInputBatch()
    runner.requests = {"req-1": _ReqState()}
    runner.enable_prompt_embeds = True

    scheduler_output = SimpleNamespace(
        finished_req_ids=["done-1"],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-1"],
            prompt_token_ids={"req-1": [2, 3, 4]},
        ),
    )

    GPUGenerationModelRunner._update_request_states(runner, scheduler_output)

    req_state = runner.requests["req-1"]
    assert runner.model.finished_req_ids == ["done-1"]
    assert req_state.prompt_token_ids == [0]
    assert req_state.num_prompt_tokens == 1
    assert runner.input_batch.num_prompt_tokens[0] == 0
    assert runner.input_batch.num_tokens_no_spec[0] == 0
    assert runner.input_batch.token_ids_cpu[0, :3].tolist() == [0, 0, 0]
    assert runner.input_batch.is_token_ids[0, :3].tolist() == [False, False, False]


class _CodecInputBatch(_PromptUpdateInputBatch):
    """CPU persistent-buffer fixture; lifecycle remains owned by _update_states."""

    def __init__(self, requests):
        super().__init__()
        self.req_ids = list(requests)
        self.req_id_to_index = {req_id: i for i, req_id in enumerate(requests)}
        self.num_reqs = len(requests)
        self.num_prompt_tokens = np.zeros(self.num_reqs, dtype=np.int32)
        self.num_tokens_no_spec = np.zeros(self.num_reqs, dtype=np.int32)
        self.num_computed_tokens_cpu = np.zeros(self.num_reqs, dtype=np.int32)
        self.token_ids_cpu = np.full((self.num_reqs, 128), -99, dtype=np.int32)
        self.is_token_ids = np.zeros((self.num_reqs, 128), dtype=bool)
        for req_id, req in requests.items():
            i = self.req_id_to_index[req_id]
            self.num_prompt_tokens[i] = req.num_prompt_tokens
            self.num_tokens_no_spec[i] = req.num_tokens
            self.token_ids_cpu[i, : req.num_prompt_tokens] = req.prompt_token_ids

    def update_req_spec_token_ids(self, *args):
        pass

    def condense(self):
        pass

    def refresh_metadata(self):
        pass


def _make_qwen_codec_runner(monkeypatch, prompts):
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    import vllm_omni.worker.gpu_model_runner as base_module
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration

    # Use the real Qwen method and real CachedRequestState, without constructing
    # weights, an engine, a GPU, or a tokenizer.
    model = Qwen3OmniMoeForConditionalGeneration.__new__(Qwen3OmniMoeForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = "code2wav"
    runner = object.__new__(GPUGenerationModelRunner)
    runner.model = model
    hf_config = Qwen3OmniMoeConfig()
    for name, value in {
        "audio_end_token_id": 151670,
        "audio_start_token_id": 151669,
        "audio_token_id": 151675,
        "image_token_id": 151655,
        "seconds_per_chunk": 2,
        "video_token_id": 151656,
        "vision_end_token_id": 151653,
        "vision_start_token_id": 151652,
    }.items():
        setattr(hf_config.thinker_config, name, value)
    # The pretrained Qwen3-Omni checkpoint opts into M-RoPE; the generic
    # Transformers config constructor does not supply these checkpoint values.
    hf_config.thinker_config.text_config.rope_parameters = {
        "rope_type": "default",
        "mrope_section": [24, 20, 20],
        "mrope_interleaved": True,
        "rope_theta": 1000000,
    }
    runner.model_config = SimpleNamespace(
        model_stage="code2wav",
        hf_config=hf_config,
        architectures=["Qwen3OmniMoeForConditionalGeneration"],
    )
    runner.get_model = lambda: model
    runner.requests = {
        req_id: CachedRequestState(
            req_id=req_id,
            prompt_token_ids=list(tokens),
            mm_features=[],
            sampling_params=None,
            generator=None,
            block_ids=([],),
            num_computed_tokens=0,
            output_token_ids=[],
        )
        for req_id, tokens in prompts.items()
    }
    runner.input_batch = _CodecInputBatch(runner.requests)
    runner.model_intermediate_buffer = {}
    runner.omni_prefix_cache = None
    runner.use_async_spec_decode = False
    runner.use_async_scheduling = False
    runner.enable_prompt_embeds = True
    runner.uses_mrope = True
    runner.uses_xdrope_dim = 0
    runner._may_reorder_batch = lambda _: None
    mrope_buffer = torch.full((3, 256), -99, dtype=torch.long)
    runner.mrope_positions = SimpleNamespace(cpu=mrope_buffer, np=mrope_buffer.numpy())
    for req in runner.requests.values():
        runner._init_mrope_positions(req)
    monkeypatch.setattr(base_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    return runner


def _codec_scheduler_output(prompts):
    req_ids = list(prompts)
    return SimpleNamespace(
        finished_req_ids=[],
        new_block_ids_to_zero=None,
        free_encoder_mm_hashes=[],
        num_scheduled_tokens={req_id: len(tokens) for req_id, tokens in prompts.items()},
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=req_ids,
            prompt_token_ids=prompts,
            resumed_req_ids=set(),
            num_computed_tokens=[0] * len(req_ids),
            new_block_ids=[None] * len(req_ids),
            num_output_tokens=[0] * len(req_ids),
            new_token_ids=[],
        ),
        scheduled_new_reqs=[],
        scheduled_spec_decode_tokens={},
    )


@pytest.mark.parametrize("old_length,new_length", [(32, 32), (32, 48), (48, 16)])
def test_qwen_codec_refresh_survives_base_update_and_mrope_prepare(monkeypatch, old_length, new_length):
    runner = _make_qwen_codec_runner(monkeypatch, {"req-1": [1] * old_length})
    new_codes = [2 + i % 17 for i in range(new_length)]
    output = _codec_scheduler_output({"req-1": new_codes})

    # Exactly the preprocessing sequence in execute_model; the base last-rank
    # branch does not copy prompt_token_ids and cannot repair the old bug.
    runner._update_request_states(output)
    runner._update_states(output)
    runner._calc_mrope_positions(output)

    req = runner.requests["req-1"]
    assert req.prompt_token_ids == new_codes
    assert req.num_prompt_tokens == new_length
    assert runner.input_batch.num_tokens_no_spec[0] == new_length
    assert runner.input_batch.num_prompt_tokens[0] == new_length
    assert runner.input_batch.token_ids_cpu[0, :new_length].tolist() == new_codes
    assert runner.input_batch.is_token_ids[0, :new_length].all()
    expected_positions = torch.arange(new_length).expand(3, -1)
    assert torch.equal(req.mrope_positions, expected_positions)
    assert req.mrope_position_delta == 0
    assert torch.equal(runner.mrope_positions.cpu[:, :new_length], expected_positions)
    assert runner.input_batch.removed == []
    assert runner.input_batch.added == []


def test_qwen_codec_refresh_uses_request_rows_not_scheduler_order(monkeypatch):
    runner = _make_qwen_codec_runner(monkeypatch, {"req-1": [1] * 32, "req-2": [3] * 48})
    # Scheduler order differs from the persistent batch; sizes change in both directions.
    prompts = {"req-2": [7] * 16, "req-1": [5] * 48}
    output = _codec_scheduler_output(prompts)
    runner._update_request_states(output)
    runner._update_states(output)
    runner._calc_mrope_positions(output)
    for req_id, codes in prompts.items():
        i = runner.input_batch.req_id_to_index[req_id]
        assert runner.input_batch.token_ids_cpu[i, : len(codes)].tolist() == codes
        assert runner.input_batch.num_prompt_tokens[i] == len(codes)
        assert runner.requests[req_id].mrope_positions.shape == (3, len(codes))
    expected_positions = torch.cat([torch.arange(48).expand(3, -1), torch.arange(16).expand(3, -1)], dim=1)
    assert torch.equal(runner.mrope_positions.cpu[:, :64], expected_positions)


def test_qwen_codec_refresh_updates_paused_request_before_readding(monkeypatch):
    runner = _make_qwen_codec_runner(monkeypatch, {"req-1": [1] * 32})
    runner.input_batch.req_id_to_index.clear()
    output = _codec_scheduler_output({"req-1": [9] * 48})
    runner._update_request_states(output)
    assert runner.requests["req-1"].prompt_token_ids == [9] * 48
    assert runner.requests["req-1"].mrope_positions.shape == (3, 48)
    assert runner.input_batch.token_ids_cpu[0, :32].tolist() == [1] * 32
    assert runner.input_batch.removed == []
    assert runner.input_batch.added == []


@pytest.mark.parametrize("missing", [None, {}])
def test_qwen_codec_missing_prompt_update_leaves_state_intact(monkeypatch, missing):
    runner = _make_qwen_codec_runner(monkeypatch, {"req-1": [1] * 32})
    output = _codec_scheduler_output({"req-1": [2] * 32})
    output.scheduled_cached_reqs.prompt_token_ids = missing
    runner._update_request_states(output)
    assert runner.requests["req-1"].prompt_token_ids == [1] * 32
    assert runner.input_batch.token_ids_cpu[0, :32].tolist() == [1] * 32


def test_qwen_codec_refresh_ignores_already_removed_request(monkeypatch):
    runner = _make_qwen_codec_runner(monkeypatch, {"req-1": [1] * 32})
    output = _codec_scheduler_output({"already-aborted": [2] * 32})
    runner._update_request_states(output)
    assert runner.requests["req-1"].prompt_token_ids == [1] * 32


@pytest.mark.parametrize("stage", ["thinker", "talker"])
def test_qwen_other_stages_do_not_enable_codec_refresh(monkeypatch, stage):
    runner = _make_qwen_codec_runner(monkeypatch, {"req-1": [1] * 32})
    runner.model.model_stage = stage
    runner.model_config.model_stage = stage
    output = _codec_scheduler_output({"req-1": [2] * 32})
    runner._update_request_states(output)
    assert runner.requests["req-1"].prompt_token_ids == [1] * 32


def test_qwen_codec_refresh_without_mrope(monkeypatch):
    runner = _make_qwen_codec_runner(monkeypatch, {"req-1": [1] * 32})
    runner.uses_mrope = False
    positions = runner.requests["req-1"].mrope_positions
    output = _codec_scheduler_output({"req-1": [2] * 32})
    runner._update_request_states(output)
    assert runner.input_batch.token_ids_cpu[0, :32].tolist() == [2] * 32
    assert runner.requests["req-1"].mrope_positions is positions


def test_qwen_codec_model_forward_receives_new_quantizer_major_chunks(monkeypatch):
    runner = _make_qwen_codec_runner(monkeypatch, {"req-1": [1] * 32, "req-2": [3] * 48})
    prompts = {"req-2": list(range(16)), "req-1": list(range(48, 96))}
    output = _codec_scheduler_output(prompts)
    runner._update_request_states(output)
    runner._update_states(output)
    runner._calc_mrope_positions(output)
    counts = [len(prompts[req_id]) for req_id in runner.input_batch.req_ids]
    packed = torch.cat(
        [torch.from_numpy(runner.input_batch.token_ids_cpu[i, :count].copy()) for i, count in enumerate(counts)]
    )
    captured = {}

    def capture_decoder_input(codes, left_context, seq_counts):
        captured.update(codes=codes.clone(), left_context=left_context, seq_counts=seq_counts)
        return []

    # Exercise the real model's packing and per-request metadata boundary;
    # the decoder itself is deliberately a CPU spy, not a waveform proof.
    runner.model.generate_audio = capture_decoder_input
    runner.model.forward(
        input_ids=packed,
        positions=runner.mrope_positions.cpu[:, : sum(counts)],
        seq_token_counts=counts,
        runtime_additional_information=[{"meta": {"left_context_size": 1}}, {"meta": {"left_context_size": 0}}],
    )
    assert captured["codes"].shape == (2, 16, 3)
    assert captured["codes"][0].flatten().tolist() == prompts["req-1"]
    assert captured["codes"][1, :, :1].flatten().tolist() == prompts["req-2"]
    assert captured["codes"][1, :, 1:].count_nonzero() == 0
    assert captured["seq_counts"] == counts
    assert captured["left_context"] == [1, 0]
