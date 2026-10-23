import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_talker import (
    MingFlashOmniTalkerForConditionalGeneration,
)
from vllm_omni.model_executor.models.ming_flash_omni.talker_module import (
    _seq_length_to_int,
)
from vllm_omni.model_executor.stage_input_processors.ming_flash_omni import (
    thinker2talker,
    thinker2talker_async_chunk,
)
from vllm_omni.transformers_utils.processors.ming import (
    process_ming_images,
    process_ming_videos,
)


class FakeQwen2VLImageProcessor:
    def __call__(self, **kwargs):
        assert "videos" not in kwargs
        images = kwargs["images"]
        if isinstance(images, list):
            frame_count = sum(int(getattr(item, "shape", [1])[0]) for item in images)
        else:
            frame_count = int(images.shape[0]) if len(images.shape) >= 4 else 1
        return BatchFeature(
            data={
                "pixel_values": torch.zeros((frame_count * 196, 1536)),
                "image_grid_thw": torch.tensor([[1, 14, 14]] * frame_count),
            }
        )


class FakeAsyncTokenizer:
    def decode(self, token_ids, **kwargs):
        return "".join(chr(token_id) for token_id in token_ids)


class FakeTransferManager:
    def __init__(self):
        self._ming_async_tokenizer = FakeAsyncTokenizer()


class FakeRequest:
    def __init__(
        self,
        text,
        finished=False,
        *,
        request_id="internal-req-1",
        external_req_id="req-1",
        additional_information=None,
    ):
        self.request_id = request_id
        self.external_req_id = external_req_id
        self.additional_information = additional_information
        self.output_token_ids = [ord(ch) for ch in text]
        self._finished = finished

    def is_finished(self):
        return self._finished


def test_process_ming_images_uses_legacy_images_entrypoint():
    processor = FakeQwen2VLImageProcessor()
    image = np.zeros((224, 224, 3), dtype=np.uint8)

    out = process_ming_images(processor, image)

    assert "pixel_values" in out
    assert out["image_grid_thw"].shape == (1, 3)


def test_process_ming_videos_renames_and_merges_frame_grid():
    processor = FakeQwen2VLImageProcessor()
    video_a = np.zeros((3, 224, 224, 3), dtype=np.uint8)
    video_b = np.zeros((5, 224, 224, 3), dtype=np.uint8)

    out = process_ming_videos(processor, [video_a, video_b])

    assert "pixel_values" not in out
    assert out["pixel_values_videos"].shape == (8 * 196, 1536)
    assert out["video_grid_thw"].tolist() == [[3, 14, 14], [5, 14, 14]]


def test_seq_length_to_int_accepts_scalar_tensor():
    assert _seq_length_to_int(7) == 7
    assert _seq_length_to_int(torch.tensor(11)) == 11


def test_thinker2talker_async_chunk_emits_sentence_delta():
    manager = FakeTransferManager()

    first = thinker2talker_async_chunk(
        manager,
        pooling_output=None,
        request=FakeRequest(
            "你好！",
            finished=False,
            additional_information={"ming_incremental": True},
        ),
        is_finished=False,
    )
    assert first is not None
    assert first["text"] == "你好！"
    assert first["request_id"] == "req-1"
    assert first["ming_incremental"] is True
    assert first["ming_async_source_finished"] is False
    assert first["codes"]["audio"] == [0]
    assert bool(first["meta"]["finished"].item()) is False

    second = thinker2talker_async_chunk(
        manager,
        pooling_output=None,
        request=FakeRequest("你好！欢迎。", finished=True),
        is_finished=True,
    )
    assert second is not None
    assert second["text"] == "欢迎。"
    assert second["request_id"] == "req-1"
    assert second["ming_async_source_finished"] is True
    assert bool(second["meta"]["finished"].item()) is True


def test_thinker2talker_sync_propagates_request_id_and_incremental_flag():
    source_output = type(
        "SourceOutput",
        (),
        {
            "external_req_id": "external-sync-1",
            "request_id": "internal-sync-1",
            "outputs": [type("Output", (), {"text": "你好。"})()],
        },
    )()
    prompt = type(
        "Prompt",
        (),
        {"additional_information": {"ming_incremental": True, "voice_name": "DB30"}},
    )()

    out = thinker2talker([source_output], prompt=[prompt])

    info = out[0]["additional_information"]
    assert info["request_id"] == "external-sync-1"
    assert info["ming_incremental"] is True
    assert info["text"] == "你好。"


def test_ming_incremental_idle_chunk_omits_finished_flag():
    talker = MingFlashOmniTalkerForConditionalGeneration.__new__(
        MingFlashOmniTalkerForConditionalGeneration
    )

    out = MingFlashOmniTalkerForConditionalGeneration._decode_incremental_latents(
        talker,
        [],
        state=object(),
        finished=None,
    )

    assert out.multimodal_outputs["incremental"].item() == 1
    assert "finished" not in out.multimodal_outputs
