# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
    talker2code2wav_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _req(rid: str, *, finished: bool = False):
    return SimpleNamespace(
        external_req_id=rid,
        is_finished=lambda: finished,
    )


def _tm(*, chunk_frames: int = 8, left_context: int = 4):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        put_req_chunk=defaultdict(int),
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": chunk_frames,
                    "codec_left_context_frames": left_context,
                }
            }
        ),
    )


def test_qwen3_omni_async_chunk_accepts_multi_frame_codebook_major_output():
    tm = _tm(chunk_frames=3, left_context=2)
    codes = torch.arange(16 * 3, dtype=torch.long).reshape(16, 3)

    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={"codes": {"audio": codes}},
        request=_req("r"),
    )

    assert payload is not None
    assert payload["meta"]["left_context_size"] == 0
    assert payload["meta"]["finished"].item() is False
    assert payload["codes"]["audio"] == codes.reshape(-1).tolist()
    assert len(tm.code_prompt_token_ids["r"]) == 3
    assert {len(frame) for frame in tm.code_prompt_token_ids["r"]} == {16}


def test_qwen3_omni_async_chunk_keeps_frames_rectangular_after_variable_outputs():
    tm = _tm(chunk_frames=8, left_context=4)
    long_codes = torch.arange(16 * 321, dtype=torch.long).reshape(16, 321)
    one_frame = torch.arange(16, dtype=torch.long).reshape(16, 1) + 1

    assert (
        talker2code2wav_async_chunk(
            transfer_manager=tm,
            pooling_output={"codes": {"audio": long_codes}},
            request=_req("r"),
        )
        is None
    )
    assert (
        talker2code2wav_async_chunk(
            transfer_manager=tm,
            pooling_output={"codes": {"audio": one_frame}},
            request=_req("r"),
        )
        is None
    )

    assert len(tm.code_prompt_token_ids["r"]) == 322
    assert {len(frame) for frame in tm.code_prompt_token_ids["r"]} == {16}


def test_qwen3_omni_async_chunk_flushes_tail_on_finished_without_new_codes():
    tm = _tm(chunk_frames=8, left_context=4)
    tm.code_prompt_token_ids["r"] = [[q + f * 16 for q in range(16)] for f in range(10)]

    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output=None,
        request=_req("r", finished=True),
        is_finished=True,
    )

    assert payload is not None
    assert payload["meta"]["finished"].item() is True
    assert payload["meta"]["left_context_size"] == 4
    assert len(payload["codes"]["audio"]) == 16 * 6
