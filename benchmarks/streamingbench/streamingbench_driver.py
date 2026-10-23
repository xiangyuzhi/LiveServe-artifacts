#!/usr/bin/env python3
"""StreamingBench workload driver for LiveServe/vllm-omni video streaming.

The driver talks to ``/v1/video/chat/stream`` over WebSocket and replays
StreamingBench-style videos with timestamped questions.  It supports:

* official StreamingBench JSON files (stream and offline forms);
* an example fixture for smoke tests without downloading the full dataset;
* text-only or text+audio output;
* optional question-audio input when StreamingBench ``audio_path`` files exist;
* client-side barge-in by issuing a second ``video.query`` mid-response;
* short multiple-choice or long spoken-answer prompts;
* optional client-side multi-turn dialogue context within each video session;
* concurrent video sessions and JSON/JSONL summaries for plotting.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import wave
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    import websockets
except ImportError as exc:  # pragma: no cover - environment sanity path
    raise SystemExit("websockets is required; use the omni_020 environment") from exc

_AUDIO_CONTINUITY_HELPER = (
    Path(__file__).resolve().parents[2]
    / "vllm-omni"
    / "vllm_omni"
    / "benchmarks"
    / "audio_continuity.py"
)
_AUDIO_CONTINUITY_SPEC = importlib.util.spec_from_file_location(
    "_liveserve_audio_continuity", _AUDIO_CONTINUITY_HELPER
)
if _AUDIO_CONTINUITY_SPEC is None or _AUDIO_CONTINUITY_SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot import audio continuity helper: {_AUDIO_CONTINUITY_HELPER}")
_audio_continuity = importlib.util.module_from_spec(_AUDIO_CONTINUITY_SPEC)
sys.modules[_AUDIO_CONTINUITY_SPEC.name] = _audio_continuity
_AUDIO_CONTINUITY_SPEC.loader.exec_module(_audio_continuity)

compute_continuity_stats_from_durations = _audio_continuity.compute_continuity_stats_from_durations
get_audio_continuity_threshold_s = _audio_continuity.get_audio_continuity_threshold_s
normalize_audio_chunk_durations = _audio_continuity.normalize_audio_chunk_durations


DEFAULT_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_WS_URL = "ws://127.0.0.1:8777/v1/video/chat/stream"

PROMPT_TEMPLATE = """You are an advanced video question-answering AI assistant. You have been provided with frames from a streaming video and a multiple-choice question related to the current time point. Carefully analyze the visible video context and answer with only the letter A, B, C, or D.

Question: {}

Options:
{}
{}
{}
{}

The best option is:"""

PROMPT_TEMPLATE_WITHOUT_OPTIONS = """You are an advanced video question-answering AI assistant. You have been provided with frames from a streaming video and a question related to the current time point. Give a concise answer.

Question: {}

Answer:"""

LONG_ANSWER_PROMPT_TEMPLATE = """You are an advanced video question-answering AI assistant. You have been provided with frames from a streaming video and a multiple-choice question related to the current time point. Carefully analyze the visible video context.

First state the best option letter, then explain the evidence in 2 to 3 complete spoken sentences. Keep the explanation grounded in what is visible or audible in the provided stream segment.

Question: {}

Options:
{}
{}
{}
{}

Answer:"""

LONG_ANSWER_PROMPT_TEMPLATE_WITHOUT_OPTIONS = """You are an advanced video question-answering AI assistant. You have been provided with frames from a streaming video and a question related to the current time point.

Answer in 2 to 3 complete spoken sentences. Keep the explanation grounded in what is visible or audible in the provided stream segment.

Question: {}

Answer:"""

DETAILED_ANSWER_PROMPT_TEMPLATE = """You are an advanced video question-answering AI assistant. You have been provided with frames from a streaming video and a multiple-choice question related to the current time point. Carefully analyze the visible video context.

First state the best option letter. Then give a detailed spoken explanation in 5 to 7 complete sentences. Use natural conversational prose, avoid bullet lists, and mention concrete visual or audio evidence from the stream. If the evidence is ambiguous, say what is uncertain and why the chosen option is still most likely.

Question: {}

Options:
{}
{}
{}
{}

Answer:"""

DETAILED_ANSWER_PROMPT_TEMPLATE_WITHOUT_OPTIONS = """You are an advanced video question-answering AI assistant. You have been provided with frames from a streaming video and a question related to the current time point.

Give a detailed spoken answer in 5 to 7 complete sentences. Use natural conversational prose, avoid bullet lists, and mention concrete visual or audio evidence from the stream. If the evidence is ambiguous, say what is uncertain.

Question: {}

Answer:"""

EXTENDED_ANSWER_PROMPT_TEMPLATE = """You are an advanced video question-answering AI assistant. You have been provided with frames from a streaming video and a multiple-choice question related to the current time point. Carefully analyze the visible video context.

First state the best option letter in the first sentence. Then continue with a long spoken explanation of 9 to 12 complete sentences and at least 180 words. Do not stop after the option letter, do not use bullet lists, and do not give a terse exam-style answer. Explain the temporal sequence, concrete visual or audio evidence, why the selected option fits, why at least two other options are less likely, and what uncertainty remains. Finish with a short concluding sentence.

Question: {}

Options:
{}
{}
{}
{}

Answer:"""

EXTENDED_ANSWER_PROMPT_TEMPLATE_WITHOUT_OPTIONS = """You are an advanced video question-answering AI assistant. You have been provided with frames from a streaming video and a question related to the current time point.

Give a long spoken answer of 9 to 12 complete sentences and at least 180 words. Do not use bullet lists, and do not give a terse answer. Explain the temporal sequence, concrete visual or audio evidence, what conclusion follows, and what uncertainty remains. Finish with a short concluding sentence.

Question: {}

Answer:"""

SCRIPTED_LONG_ANSWER_PROMPT_TEMPLATE = """You are an advanced video question-answering AI assistant. You have been provided with frames from a streaming video and a multiple-choice question related to the current time point. Carefully analyze the visible video context.

Give a long natural spoken answer in exactly four short paragraphs and at least 240 words. Do not use bullets or numbered lists. Paragraph 1 must begin with "The best answer is <letter>" and briefly state the decision. Paragraph 2 must describe concrete visual or audio evidence from the current stream segment. Paragraph 3 must explain the temporal sequence before and after this moment. Paragraph 4 must compare the selected option against at least two weaker options, mention any uncertainty, and end with a complete concluding sentence. The response is invalid if it stops after one sentence or before all four paragraphs are complete.

Question: {}

Options:
{}
{}
{}
{}

Answer:"""

SCRIPTED_LONG_ANSWER_PROMPT_TEMPLATE_WITHOUT_OPTIONS = """You are an advanced video question-answering AI assistant. You have been provided with frames from a streaming video and a question related to the current time point.

Give a long natural spoken answer in exactly four short paragraphs and at least 240 words. Do not use bullets or numbered lists. Paragraph 1 must directly answer the question. Paragraph 2 must describe concrete visual or audio evidence from the current stream segment. Paragraph 3 must explain the temporal sequence before and after this moment. Paragraph 4 must mention uncertainty or alternate interpretations and end with a complete concluding sentence. The response is invalid if it stops after one sentence or before all four paragraphs are complete.

Question: {}

Answer:"""

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech. For multiple-choice video questions, answer with only "
    "the option letter."
)

LONG_ANSWER_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech. For video questions, produce a natural spoken answer. "
    "For multiple-choice questions, begin with the option letter and then give "
    "a concise explanation."
)

DETAILED_ANSWER_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech. For video questions, produce a detailed natural spoken "
    "answer. For multiple-choice questions, begin with the option letter and "
    "then explain the supporting evidence in several complete sentences."
)

EXTENDED_ANSWER_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech. For video questions, produce a long natural spoken "
    "answer. For multiple-choice questions, begin with the option letter, then "
    "continue with a multi-sentence explanation; never stop after only the "
    "letter or one short sentence. A complete long answer is required."
)

SCRIPTED_LONG_ANSWER_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech. For this benchmark, every answer must be a complete long "
    "spoken response with four short paragraphs: decision, evidence, temporal "
    "sequence, and alternatives or uncertainty. Never stop after only an option "
    "letter, one sentence, or a brief acknowledgment."
)

THINKER_SP = {
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 1,
    "seed": 42,
    "repetition_penalty": 1.05,
    "stop_token_ids": [151645],
}
TALKER_SP = {
    "temperature": 0.9,
    "top_k": 50,
    "max_tokens": 4096,
    "seed": 42,
    "detokenize": False,
    "repetition_penalty": 1.05,
    "stop_token_ids": [2150],
}
CODE2WAV_SP = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "max_tokens": 65536,
    "seed": 42,
    "detokenize": True,
    "repetition_penalty": 1.1,
}


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


@dataclass
class BenchQuestion:
    question_id: str
    question: str
    timestamp_sec: float
    prompt: str
    answer: str | None = None
    options: list[str] = field(default_factory=list)
    audio_path: str | None = None
    task_type: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoGroup:
    group_id: str
    video_path: str
    resolved_video_path: str | None
    questions: list[BenchQuestion]


@dataclass
class WsResponse:
    request_id: str
    text: str = ""
    started_at_sec: float | None = None
    first_text_sec: float | None = None
    first_audio_sec: float | None = None
    text_done_sec: float | None = None
    audio_done_sec: float | None = None
    last_message_sec: float | None = None
    text_done: bool = False
    audio_done: bool = False
    audio_chunks: int = 0
    audio_duration_sec: float = 0.0
    audio_bytes: int = 0
    audio_chunk_arrival_times_sec: list[float] = field(default_factory=list)
    audio_chunk_durations_sec: list[float] = field(default_factory=list)
    audio_timeline: list[dict[str, float | int]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def update(self, msg: dict[str, Any], now_sec: float, record_audio_timeline: bool = False) -> None:
        msg_type = msg.get("type")
        self.last_message_sec = now_sec
        if msg_type == "response.start":
            if self.started_at_sec is None:
                self.started_at_sec = now_sec
        elif msg_type == "response.text.delta":
            delta = str(msg.get("delta") or "")
            if delta and self.first_text_sec is None:
                self.first_text_sec = now_sec
            self.text += delta
        elif msg_type == "response.text.done":
            done_text = msg.get("text")
            if isinstance(done_text, str) and done_text:
                self.text = done_text
            self.text_done = True
            self.text_done_sec = now_sec
        elif msg_type == "response.audio.delta":
            data_b64 = msg.get("data") or ""
            try:
                audio = base64.b64decode(data_b64)
                self.audio_bytes += len(audio)
                self.audio_chunks += 1
                dur = wav_duration(audio)
                if dur is not None:
                    self.audio_duration_sec += dur
                    self.audio_chunk_arrival_times_sec.append(now_sec)
                    self.audio_chunk_durations_sec.append(dur)
                    if record_audio_timeline:
                        self.audio_timeline.append(
                            {
                                "chunk": self.audio_chunks,
                                "recv_sec": now_sec,
                                "audio_duration_sec": dur,
                                "bytes": len(audio),
                            }
                        )
                if self.first_audio_sec is None:
                    self.first_audio_sec = now_sec
            except Exception as exc:
                self.errors.append(f"audio decode failed: {exc}")
        elif msg_type == "response.audio.done":
            self.audio_done = True
            self.audio_done_sec = now_sec
        elif msg_type == "error":
            self.errors.append(str(msg.get("message") or "unknown error"))

    def is_done(self, expect_audio: bool) -> bool:
        if not self.text_done:
            return False
        if not expect_audio:
            return True
        return self.audio_done or self.audio_chunks == 0


@dataclass
class RequestResult:
    session_id: str
    request_id: str
    video_id: str
    video_path: str
    question_id: str
    question_index: int
    timestamp_sec: float
    mode: str
    concurrency: int
    issued_at: float
    frames_sent: int = 0
    input_audio_sent: bool = False
    status: str = "ok"
    error: str | None = None
    text: str = ""
    answer: str | None = None
    prediction: str | None = None
    correct: bool | None = None
    text_ttft_sec: float | None = None
    audio_ttfp_sec: float | None = None
    e2e_sec: float | None = None
    audio_chunks: int = 0
    audio_duration_sec: float = 0.0
    audio_stream_span_sec: float | None = None
    audio_stream_rtf: float | None = None
    e2e_audio_rtf: float | None = None
    interrupted: bool = False
    interrupt_request_id: str | None = None
    interrupt_triggered_at_sec: float | None = None
    pre_interrupt_text_chars: int = 0
    pre_interrupt_audio_chunks: int = 0
    pre_interrupt_audio_duration_sec: float = 0.0
    audio_underrun_sec: float | None = None
    audio_underrun_event_count: int | None = None
    audio_continuity_ok: bool | None = None
    audio_continuity_threshold_sec: float | None = None
    audio_continuity_evaluated: bool = False
    audio_continuity_excluded_reason: str = ""
    audio_timeline: list[dict[str, float | int]] = field(default_factory=list)
    frame_extract_sec: float | None = None
    frame_send_sec: float | None = None
    input_audio_send_sec: float | None = None
    query_collect_sec: float | None = None
    client_question_sec: float | None = None
    prompt_chars: int = 0
    response_text_chars: int = 0
    kv_metadata_enabled: bool = False
    kv_committed_prefix_hash: str | None = None
    kv_history_prefix_hash: str | None = None
    kv_prefix_token_estimate: int | None = None
    kv_history_tokens_before: int | None = None
    kv_history_tokens_after: int | None = None
    prefetch_requested: bool = False
    prefetch_ok: bool | None = None
    prefetch_status: str | None = None
    prefetch_request_id: str | None = None
    prefetch_scope: str | None = None
    prefetch_wall_sec: float | None = None
    prefetch_skip_reason: str | None = None
    raw_response_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        if len(data.get("text") or "") > 500:
            data["text"] = data["text"][:500] + "..."
        if not data.get("audio_timeline"):
            data.pop("audio_timeline", None)
        return data


def parse_timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" not in text:
        return float(text)
    total = 0.0
    for part in text.split(":"):
        total = total * 60.0 + float(part)
    return total


def format_prompt(q: dict[str, Any], prompt_style: str = "mcq_letter") -> tuple[str, list[str]]:
    question = str(q.get("question") or q.get("ground_truth_output") or "")
    options = list(q.get("options") or [])
    if options:
        labels = ["A", "B", "C", "D"]
        normalized = []
        for i, option in enumerate(options[:4]):
            text = str(option)
            if i < 4 and not re.match(r"^[A-D][.)]\s*", text):
                text = f"{labels[i]}. {text}"
            normalized.append(text)
        while len(normalized) < 4:
            normalized.append(f"{labels[len(normalized)]}.")
        if prompt_style == "scripted_long_answer":
            return SCRIPTED_LONG_ANSWER_PROMPT_TEMPLATE.format(question, *normalized[:4]), normalized[:4]
        if prompt_style == "extended_answer":
            return EXTENDED_ANSWER_PROMPT_TEMPLATE.format(question, *normalized[:4]), normalized[:4]
        if prompt_style == "detailed_answer":
            return DETAILED_ANSWER_PROMPT_TEMPLATE.format(question, *normalized[:4]), normalized[:4]
        if prompt_style == "long_answer":
            return LONG_ANSWER_PROMPT_TEMPLATE.format(question, *normalized[:4]), normalized[:4]
        return PROMPT_TEMPLATE.format(question, *normalized[:4]), normalized[:4]
    if prompt_style == "scripted_long_answer":
        return SCRIPTED_LONG_ANSWER_PROMPT_TEMPLATE_WITHOUT_OPTIONS.format(question), []
    if prompt_style == "extended_answer":
        return EXTENDED_ANSWER_PROMPT_TEMPLATE_WITHOUT_OPTIONS.format(question), []
    if prompt_style == "detailed_answer":
        return DETAILED_ANSWER_PROMPT_TEMPLATE_WITHOUT_OPTIONS.format(question), []
    if prompt_style == "long_answer":
        return LONG_ANSWER_PROMPT_TEMPLATE_WITHOUT_OPTIONS.format(question), []
    return PROMPT_TEMPLATE_WITHOUT_OPTIONS.format(question), []


def resolve_path(path_text: str | None, data_root: Path | None, metadata_dir: Path) -> str | None:
    if not path_text:
        return None
    path = Path(path_text)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        clean = Path(str(path_text).lstrip("./"))
        if data_root is not None:
            candidates.append(data_root / clean)
            candidates.append(data_root / path)
        candidates.append(metadata_dir / clean)
        candidates.append(metadata_dir / path)
        candidates.append(Path.cwd() / clean)
        candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def _records_from_streamingbench(data: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not isinstance(data, list):
        return records
    for item in data:
        if isinstance(item, dict):
            records.append(item)
        elif isinstance(item, list):
            for sub in item:
                if isinstance(sub, dict):
                    records.append(sub)
    return records


def load_groups(
    metadata_path: Path,
    data_root: Path | None,
    start_video_index: int,
    max_videos: int | None,
    max_questions_per_video: int | None,
    prompt_style: str = "mcq_letter",
) -> list[VideoGroup]:
    start_video_index = max(0, start_video_index)
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metadata_dir = metadata_path.parent
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for record in _records_from_streamingbench(data):
        video_path = str(record.get("video_path") or "")
        if not video_path:
            continue
        grouped.setdefault(video_path, [])
        for q in record.get("questions") or []:
            if isinstance(q, dict):
                grouped[video_path].append(q)

    groups: list[VideoGroup] = []
    for idx, (video_path, questions_raw) in enumerate(grouped.items()):
        if idx < start_video_index:
            continue
        questions: list[BenchQuestion] = []
        for q_idx, q in enumerate(sorted(questions_raw, key=lambda x: parse_timestamp(x.get("time_stamp", 0)))):
            if max_questions_per_video is not None and len(questions) >= max_questions_per_video:
                break
            prompt, options = format_prompt(q, prompt_style)
            questions.append(
                BenchQuestion(
                    question_id=f"v{idx:04d}-q{q_idx:02d}",
                    question=str(q.get("question") or ""),
                    timestamp_sec=parse_timestamp(q.get("time_stamp", 0)),
                    prompt=prompt,
                    answer=str(q.get("answer")).strip().upper() if q.get("answer") is not None else None,
                    options=options,
                    audio_path=resolve_path(q.get("audio_path"), data_root, metadata_dir),
                    task_type=q.get("task_type"),
                    raw=q,
                )
            )
        if questions:
            groups.append(
                VideoGroup(
                    group_id=f"video-{idx:04d}",
                    video_path=video_path,
                    resolved_video_path=resolve_path(video_path, data_root, metadata_dir),
                    questions=questions,
                )
            )
        if max_videos is not None and len(groups) >= max_videos:
            break
    return groups


def extract_answer_letter(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"\b([ABCD])\b", text.upper())
    if match:
        return match.group(1)
    match = re.search(r"([ABCD])\s*[.)]", text.upper())
    return match.group(1) if match else None


def wav_duration(audio_bytes: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return None


def load_audio_pcm16_16k(path: str) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to convert question audio")
    proc = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            path,
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout


def extract_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    target_fps: float,
    max_frames: int,
    resize_max_side: int,
    jpeg_quality: int,
) -> list[tuple[float, bytes]]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for video frame extraction") from exc

    if end_sec <= start_sec:
        end_sec = start_sec + 0.5
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = frame_count / video_fps if frame_count > 0 else end_sec
    start_sec = max(0.0, min(start_sec, max(duration - 0.1, 0.0)))
    end_sec = max(start_sec + 0.1, min(end_sec, duration))

    step = 1.0 / max(target_fps, 1e-3)
    times: list[float] = []
    t = start_sec
    while t <= end_sec + 1e-6:
        times.append(t)
        t += step
    if not times:
        times = [end_sec]
    if len(times) > max_frames:
        if max_frames <= 1:
            times = [times[-1]]
        else:
            stride = (len(times) - 1) / float(max_frames - 1)
            times = [times[round(i * stride)] for i in range(max_frames)]

    frames: list[tuple[float, bytes]] = []
    for ts in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        longest = max(h, w)
        if resize_max_side > 0 and longest > resize_max_side:
            scale = resize_max_side / float(longest)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        if ok:
            frames.append((ts, buf.tobytes()))
    cap.release()
    return frames


def build_sampling_params(
    with_audio_output: bool,
    max_tokens: int,
    min_tokens: int = 0,
    ignore_eos: bool = False,
    disable_thinker_stop_token_ids: bool = False,
) -> list[dict[str, Any]]:
    max_tokens = max(1, int(max_tokens))
    min_tokens = max(0, min(int(min_tokens), max_tokens - 1))
    thinker = {**THINKER_SP, "max_tokens": max_tokens}
    if disable_thinker_stop_token_ids:
        thinker.pop("stop_token_ids", None)
    if min_tokens > 0:
        thinker["min_tokens"] = min_tokens
    if ignore_eos:
        thinker["ignore_eos"] = True
    if with_audio_output:
        return [thinker, dict(TALKER_SP), dict(CODE2WAV_SP)]
    return [thinker]


TRANSIENT_WS_ERRORS = (
    "timed out during opening handshake",
    "no close frame received or sent",
    "connection closed",
    "keepalive ping timeout",
    "opening handshake failed",
)


class SessionAttemptError(RuntimeError):
    def __init__(self, exc: Exception, results: list[RequestResult]):
        super().__init__(str(exc))
        self.original = exc
        self.results = results

    @property
    def results_count(self) -> int:
        return len(self.results)


def is_transient_ws_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(pattern in text for pattern in TRANSIENT_WS_ERRORS)


def is_server_unavailable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if "server health check failed" in text:
        return True
    if "service restart" in text or "received 1012" in text:
        return True
    return (
        "websocket connect failed" in text
        and (
            "connect call failed" in text
            or "connection refused" in text
            or "errno 111" in text
        )
    )


def completed_question_blocks(results: list[RequestResult], turns_per_question: int) -> int:
    """Count only complete primary+follow-up blocks when resuming a session."""
    if turns_per_question <= 0:
        return 0
    indices = sorted(
        r.question_index
        for r in results
        if isinstance(r.question_index, int) and r.question_index >= 0
    )
    completed = 0
    while all((completed * turns_per_question + offset) in indices for offset in range(turns_per_question)):
        completed += 1
    return completed


def dedupe_retry_results(results: list[RequestResult]) -> list[RequestResult]:
    deduped: OrderedDict[tuple[str, str], RequestResult] = OrderedDict()
    passthrough: list[RequestResult] = []
    for result in results:
        if result.question_id == "fatal" or result.question_index < 0:
            passthrough.append(result)
            continue
        key = (result.video_id, result.question_id)
        if key in deduped:
            del deduped[key]
        deduped[key] = result
    return list(deduped.values()) + passthrough


def derive_health_url(ws_url: str) -> str | None:
    parsed = urlparse(ws_url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
        return None
    scheme = "https" if parsed.scheme == "wss" else "http"
    return f"{scheme}://{parsed.netloc}/health"


def derive_prefetch_url(ws_url: str) -> str | None:
    parsed = urlparse(ws_url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
        return None
    scheme = "https" if parsed.scheme == "wss" else "http"
    return f"{scheme}://{parsed.netloc}/v1/omni/prefetch_session"


def check_server_health(health_url: str, timeout_sec: float) -> tuple[bool, str]:
    try:
        req = Request(health_url, method="GET")
        with urlopen(req, timeout=max(timeout_sec, 0.1)) as resp:
            code = int(resp.getcode())
        return 200 <= code < 300, f"HTTP {code}"
    except HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except (OSError, TimeoutError, URLError) as exc:
        return False, str(exc)


def estimate_text_tokens(text: str) -> int:
    # Coarse but stable enough for scheduling hints.
    return max(1, int(math.ceil(len(text) / 4.0))) if text else 0


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        total += estimate_text_tokens(str(msg.get("role") or ""))
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_text_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += estimate_text_tokens(str(part.get("text") or ""))
    return total


def committed_prefix_hash(messages: list[dict[str, Any]]) -> str:
    blob = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def to_omni_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append(
                {
                    "role": msg.get("role", "user"),
                    "content": [{"type": "text", "text": content}],
                }
            )
        else:
            out.append({"role": msg.get("role", "user"), "content": content})
    return out


def post_prefetch_sync(url: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "ok": False,
        "request_id": payload.get("request_id"),
        "prefetch_scope": (payload.get("metadata") or {}).get("prefetch_scope"),
    }
    try:
        with urlopen(req, timeout=max(timeout_sec, 0.1)) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            out["http_status"] = int(resp.getcode())
        data = json.loads(text) if text else {}
        out.update(data)
        out["ok"] = data.get("status") == "SUCCESS"
    except HTTPError as exc:
        out["http_status"] = exc.code
        try:
            out["error"] = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            out["error"] = str(exc)
    except (OSError, TimeoutError, URLError, json.JSONDecodeError) as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    out["wall_s"] = time.perf_counter() - t0
    return out


def recent_history_prefix_messages(
    system_prompt: str,
    server_history_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prefix: list[dict[str, Any]] = []
    if system_prompt:
        prefix.append({"role": "system", "content": system_prompt})
    prefix.extend(server_history_messages[-2:])
    return prefix


def build_kv_live_metadata(
    args: argparse.Namespace,
    *,
    session_id: str,
    request_index: int,
    timestamp_sec: float,
    query_prompt: str,
    prefix_messages: list[dict[str, Any]],
    next_gap_sec: float | None,
    is_terminal_turn: bool,
) -> dict[str, Any]:
    history_tokens = estimate_message_tokens(prefix_messages)
    query_tokens = estimate_text_tokens(query_prompt)
    prefix_hash = committed_prefix_hash(prefix_messages) if len(prefix_messages) > 1 else None
    metadata: dict[str, Any] = {
        "omni_round_index": request_index,
        "omni_trace_timestamp": timestamp_sec,
        "omni_query_length": query_tokens,
        "omni_response_length": max(1, int(args.max_tokens)),
        "omni_history_tokens_before": history_tokens,
        "omni_history_tokens_after": history_tokens + query_tokens + max(1, int(args.max_tokens)),
        "omni_is_terminal_turn": bool(is_terminal_turn),
    }
    if prefix_hash:
        metadata.update(
            {
                "omni_session_id": session_id,
                "committed_prefix_hash": prefix_hash,
                "omni_history_prefix_hash": prefix_hash,
                "omni_prefix_token_estimate": history_tokens,
                "omni_history_prefix_token_estimate": history_tokens,
            }
        )
    if next_gap_sec is not None:
        metadata["omni_expected_next_gap_sec"] = max(0.0, float(next_gap_sec))
        metadata["omni_expected_next_use_ts"] = float(timestamp_sec) + max(0.0, float(next_gap_sec))
    return metadata


async def maybe_prefetch_history(
    args: argparse.Namespace,
    *,
    session_id: str,
    request_index: int,
    prefix_messages: list[dict[str, Any]],
    prefix_hash: str | None,
    prefix_token_estimate: int,
) -> dict[str, Any] | None:
    if not args.enable_kv_prefetch or not prefix_hash or len(prefix_messages) <= 1:
        return None
    url = args.prefetch_url or derive_prefetch_url(args.ws_url)
    if not url:
        return {
            "ok": False,
            "status": "SKIPPED",
            "skip_reason": "no_prefetch_url",
            "prefetch_scope": args.prefetch_scope,
        }
    rid = f"kvpref-video-{session_id}-r{request_index}-{random.randint(0, 0xFFFFFF):06x}"
    payload = {
        "model": args.model,
        "messages": to_omni_messages(prefix_messages),
        "request_id": rid,
        "session_id": session_id,
        "committed_prefix_hash": prefix_hash,
        "prefix_token_estimate": prefix_token_estimate,
        "max_tokens": max(1, int(args.prefetch_max_tokens)),
        "lead_ms": float(args.prefetch_lead_ms),
        "stream_options": {"include_usage": True},
        "metadata": {
            "prefetch_include_current_user": False,
            "prefetch_scope": args.prefetch_scope,
            "workload": "streamingbench_multiturn_video",
        },
    }
    return await asyncio.to_thread(
        post_prefetch_sync,
        url,
        payload,
        args.prefetch_timeout_sec,
    )


async def send_frames(ws: Any, frames: list[tuple[float, bytes]], realtime_speedup: float) -> None:
    prev_ts: float | None = None
    for ts, frame in frames:
        if realtime_speedup > 0 and prev_ts is not None:
            await asyncio.sleep(max(0.0, ts - prev_ts) / realtime_speedup)
        prev_ts = ts
        await ws.send(
            json.dumps(
                {
                    "type": "video.frame",
                    "data": base64.b64encode(frame).decode("ascii"),
                }
            )
        )


async def send_audio(ws: Any, pcm: bytes, chunk_sec: float = 1.0) -> None:
    chunk_size = max(1, int(16000 * 2 * chunk_sec))
    for off in range(0, len(pcm), chunk_size):
        await ws.send(
            json.dumps(
                {
                    "type": "audio.chunk",
                    "data": base64.b64encode(pcm[off : off + chunk_size]).decode("ascii"),
                }
            )
        )


async def connect_with_retries(args: argparse.Namespace, config: dict[str, Any], session_id: str) -> Any:
    last_exc: Exception | None = None
    for attempt in range(args.websocket_connect_retries + 1):
        ws = None
        try:
            ws = await websockets.connect(
                args.ws_url,
                max_size=args.websocket_max_size,
                ping_interval=None,
                open_timeout=args.websocket_open_timeout,
                close_timeout=args.websocket_close_timeout,
                proxy=None,
            )
            await ws.send(json.dumps(config))
            return ws
        except Exception as exc:
            last_exc = exc
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass
            if attempt >= args.websocket_connect_retries:
                break
            sleep_sec = min(args.websocket_retry_backoff_sec * (2**attempt), 15.0)
            print(
                f"[WARN] websocket connect failed for {session_id} "
                f"(attempt {attempt + 1}/{args.websocket_connect_retries + 1}): {exc}; "
                f"retrying in {sleep_sec:.1f}s",
                file=sys.stderr,
            )
            await asyncio.sleep(sleep_sec)
    raise RuntimeError(f"websocket connect failed for {session_id}: {last_exc}")


def _new_accumulators(primary_id: str) -> dict[str, WsResponse]:
    return {primary_id: WsResponse(request_id=primary_id)}


def make_session_id(args: argparse.Namespace, session_index: int) -> str:
    base = f"s{session_index:04d}"
    prefix = str(getattr(args, "request_id_prefix", "") or "").strip()
    if not prefix:
        return base
    return f"{prefix}-{base}"


def build_system_prompt(args: argparse.Namespace) -> str:
    if args.prompt_style == "scripted_long_answer":
        return SCRIPTED_LONG_ANSWER_SYSTEM_PROMPT
    if args.prompt_style == "extended_answer":
        return EXTENDED_ANSWER_SYSTEM_PROMPT
    if args.prompt_style == "detailed_answer":
        return DETAILED_ANSWER_SYSTEM_PROMPT
    if args.prompt_style == "long_answer":
        return LONG_ANSWER_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def truncate_for_history(text: str, max_chars: int) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    return clean[: max(0, max_chars - 3)].rstrip() + "..."


def build_query_prompt(
    question: BenchQuestion,
    dialogue_history: list[tuple[str, str]],
    args: argparse.Namespace,
) -> str:
    if args.dialogue_mode != "multiturn" or not dialogue_history:
        return question.prompt

    turns = dialogue_history[-max(1, args.dialogue_history_turns) :]
    history_lines: list[str] = []
    for i, (prev_question, prev_answer) in enumerate(turns, start=1):
        history_lines.append(f"Previous turn {i} user: {truncate_for_history(prev_question, 240)}")
        history_lines.append(
            f"Previous turn {i} assistant: "
            f"{truncate_for_history(prev_answer, args.dialogue_history_answer_chars)}"
        )

    return (
        "Continue the same streaming-video conversation. Use the previous turns "
        "only as dialogue context; answer the latest question using the current "
        "video context.\n\n"
        + "\n".join(history_lines)
        + "\n\nLatest turn:\n"
        + question.prompt
    )


def make_followup_question(
    base: BenchQuestion,
    followup_index: int,
    prompt_style: str,
    followup_profile: str,
) -> BenchQuestion:
    if prompt_style == "scripted_long_answer":
        sentence_target = (
            "Speak in exactly three short paragraphs, at least 220 words, "
            "and include evidence, temporal sequence, and alternatives or uncertainty."
        )
    elif prompt_style == "extended_answer":
        sentence_target = "Speak in 8 to 10 complete sentences and at least 140 words."
    elif prompt_style == "detailed_answer":
        sentence_target = "Speak in 5 to 7 complete sentences."
    else:
        sentence_target = "Speak in 3 to 4 complete sentences."

    if followup_profile == "varied":
        templates = [
            (
                "The user asks for clarification. Continue the same video discussion "
                "and give a richer spoken explanation for the previous answer, focusing "
                "on concrete evidence, temporal context, and uncertainty. Do not change "
                "the selected option unless the video evidence clearly requires it."
            ),
            (
                "The user interrupts to ask why the other choices are weaker. Continue "
                "with a natural spoken comparison between the selected answer and the "
                "most plausible alternatives, grounded in the current video context."
            ),
            (
                "The user asks what is likely happening before and after this moment. "
                "Continue with a spoken explanation of the event sequence, what changed "
                "over time, and which visible details support that interpretation."
            ),
        ]
        detail = templates[followup_index % len(templates)]
    else:
        detail = (
            "Please continue the same video discussion. Give a richer spoken explanation "
            "for the previous answer, focusing on concrete evidence, temporal context, "
            "and any uncertainty. Do not change the selected option unless the video "
            "evidence clearly requires it."
        )

    detail += (
        f" {sentence_target} Do not answer with only a single sentence or a brief acknowledgment. "
        "Do not merely acknowledge or restate the user's request; give the complete "
        "answer directly and finish the final sentence. If you already gave the option "
        "letter, keep expanding with grounded spoken analysis instead of stopping."
    )
    prompt = f"Follow-up request: {detail}\n\nOriginal question: {base.question}\n\nAnswer:"
    return BenchQuestion(
        question_id=f"{base.question_id}-followup{followup_index + 1}",
        question=f"Follow-up {followup_index + 1}: {base.question}",
        timestamp_sec=base.timestamp_sec,
        prompt=prompt,
        answer=None,
        options=[],
        audio_path=None,
        task_type=base.task_type,
        raw={"followup_of": base.question_id, "followup_index": followup_index + 1},
    )


async def query_and_collect(
    ws: Any,
    question: BenchQuestion,
    query_prompt: str,
    session_id: str,
    question_index: int,
    args: argparse.Namespace,
    rng: random.Random,
    kv_metadata: dict[str, Any] | None = None,
) -> tuple[WsResponse, WsResponse | None, float | None, str | None]:
    primary_id = f"{session_id}-q{question_index:02d}"
    payload = {"type": "video.query", "text": query_prompt, "request_id": primary_id}
    if kv_metadata:
        payload.update(kv_metadata)
    await ws.send(json.dumps(payload))
    t0 = time.monotonic()
    accs = _new_accumulators(primary_id)

    interrupt_id: str | None = None
    interrupt_at: float | None = None
    interrupt_delay: float | None = None
    allow_interrupt = args.mode in ("bargein", "interrupt") and rng.random() <= args.interrupt_prob
    if allow_interrupt:
        jitter = rng.uniform(-args.interrupt_jitter_sec, args.interrupt_jitter_sec)
        interrupt_delay = max(0.0, args.interrupt_after_seconds + jitter)
        if args.interrupt_time_anchor == "request":
            interrupt_at = interrupt_delay

    final_id = primary_id
    post_text_deadline: float | None = None
    deadline = t0 + args.timeout_sec
    next_health_check = t0 + args.health_check_interval_sec

    while True:
        now = time.monotonic()
        if now >= deadline:
            accs[final_id].errors.append("timeout waiting for response")
            break

        primary = accs[primary_id]
        primary_still_active = not primary.text_done or (args.with_audio_output and not primary.audio_done)
        if allow_interrupt and interrupt_id is None and interrupt_delay is not None and primary_still_active:
            anchor_sec: float | None
            if args.interrupt_time_anchor == "request":
                anchor_sec = 0.0
            elif args.interrupt_time_anchor == "text":
                anchor_sec = primary.first_text_sec
            else:
                anchor_sec = primary.first_audio_sec
            if anchor_sec is not None:
                interrupt_at = anchor_sec + interrupt_delay
        if (
            allow_interrupt
            and interrupt_id is None
            and interrupt_at is not None
            and (now - t0) >= interrupt_at
            and primary_still_active
        ):
            interrupt_id = f"{primary_id}-intr"
            final_id = interrupt_id
            accs[interrupt_id] = WsResponse(request_id=interrupt_id)
            interrupt_prompt = (
                "A new user turn arrives while the previous audio is playing. Cancel the "
                "previous response internally and now give a complete natural spoken answer "
                "to this latest turn only. Follow all length and evidence instructions below; "
                "do not merely acknowledge the interruption or restate the user's request.\n\n"
                + query_prompt
            )
            interrupt_payload = {"type": "video.query", "text": interrupt_prompt, "request_id": interrupt_id}
            if kv_metadata:
                interrupt_payload.update(kv_metadata)
            await ws.send(json.dumps(interrupt_payload))

        timeout = min(0.25, max(0.01, deadline - now))
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            if args.health_url and args.health_check_interval_sec > 0 and now >= next_health_check:
                ok, detail = await asyncio.to_thread(
                    check_server_health,
                    args.health_url,
                    args.health_timeout_sec,
                )
                next_health_check = now + args.health_check_interval_sec
                if not ok:
                    raise RuntimeError(f"server health check failed: {detail}")
            final = accs[final_id]
            if final.text_done and not args.with_audio_output:
                break
            if final.text_done and args.with_audio_output:
                if final.audio_done:
                    break
                if post_text_deadline is None:
                    post_text_deadline = time.monotonic() + args.post_text_audio_grace_sec
                elif time.monotonic() >= post_text_deadline:
                    break
            continue

        msg = json.loads(raw)
        msg_type = msg.get("type")
        rid = msg.get("request_id")
        if msg_type == "session.done":
            break
        if msg_type == "error" and rid is None:
            accs[final_id].errors.append(str(msg.get("message") or "unknown error"))
            continue
        if not rid:
            rid = final_id
        if rid not in accs:
            accs[rid] = WsResponse(request_id=rid)
        accs[rid].update(
            msg,
            time.monotonic() - t0,
            record_audio_timeline=args.record_audio_timeline,
        )

        final = accs[final_id]
        if final.text_done and not args.with_audio_output:
            break
        if final.text_done and args.with_audio_output and final.audio_done:
            break

    return accs[final_id], accs.get(primary_id) if interrupt_id else None, interrupt_at, interrupt_id


def response_to_result(
    session_id: str,
    response: WsResponse,
    interrupted_response: WsResponse | None,
    interrupt_at: float | None,
    interrupt_id: str | None,
    group: VideoGroup,
    question: BenchQuestion,
    question_index: int,
    issued_at: float,
    frames_sent: int,
    input_audio_sent: bool,
    query_prompt: str,
    frame_extract_sec: float | None,
    frame_send_sec: float | None,
    input_audio_send_sec: float | None,
    query_collect_sec: float | None,
    client_question_sec: float | None,
    args: argparse.Namespace,
    kv_metadata: dict[str, Any] | None = None,
    prefetch_result: dict[str, Any] | None = None,
) -> RequestResult:
    audio_chunk_durations_sec = normalize_audio_chunk_durations(
        response.audio_chunk_durations_sec,
        mode=args.audio_continuity_chunk_mode,
    )
    audio_duration_sec = sum(audio_chunk_durations_sec) if audio_chunk_durations_sec else response.audio_duration_sec

    e2e_candidates = [
        x
        for x in [response.audio_done_sec, response.text_done_sec, response.last_message_sec]
        if isinstance(x, (int, float))
    ]
    e2e_sec = max(e2e_candidates) if e2e_candidates else None
    audio_span = None
    if response.first_audio_sec is not None:
        end = response.audio_done_sec or response.last_message_sec
        if end is not None:
            audio_span = max(0.0, end - response.first_audio_sec)
    audio_stream_rtf = None
    if audio_span is not None and audio_duration_sec > 0:
        audio_stream_rtf = audio_span / audio_duration_sec
    e2e_audio_rtf = None
    if e2e_sec is not None and audio_duration_sec > 0:
        e2e_audio_rtf = e2e_sec / audio_duration_sec

    audio_underrun_sec: float | None = None
    audio_underrun_event_count: int | None = None
    audio_continuity_ok: bool | None = None
    audio_continuity_evaluated = False
    audio_continuity_excluded_reason = ""
    if not args.with_audio_output:
        audio_continuity_excluded_reason = "audio_disabled"
    elif interrupt_id is not None and not args.continuity_include_interrupted:
        audio_continuity_excluded_reason = "interrupted"
    elif not response.audio_chunk_arrival_times_sec or not audio_chunk_durations_sec:
        audio_continuity_excluded_reason = "no_audio_chunks"
    elif len(response.audio_chunk_arrival_times_sec) != len(audio_chunk_durations_sec):
        audio_continuity_excluded_reason = "invalid_audio_chunks"
    else:
        stats = compute_continuity_stats_from_durations(
            response.audio_chunk_arrival_times_sec,
            audio_chunk_durations_sec,
            threshold_s=args.audio_continuity_threshold_sec,
            chunk_duration_mode="fragment",
        )
        audio_underrun_sec = stats.audio_underrun_s
        audio_underrun_event_count = stats.audio_underrun_event_count
        audio_continuity_ok = stats.audio_continuity_ok
        audio_continuity_evaluated = True

    pred = extract_answer_letter(response.text)
    correct = None
    if question.answer in {"A", "B", "C", "D"}:
        correct = pred == question.answer

    status = "ok"
    error = None
    errors = list(response.errors)
    if errors:
        status = "error"
        error = "; ".join(errors[:3])

    pre_text_chars = 0
    pre_audio_chunks = 0
    pre_audio_duration = 0.0
    if interrupted_response is not None:
        pre_text_chars = len(interrupted_response.text)
        pre_audio_chunks = interrupted_response.audio_chunks
        pre_audio_duration = sum(
            normalize_audio_chunk_durations(
                interrupted_response.audio_chunk_durations_sec,
                mode=args.audio_continuity_chunk_mode,
            )
        ) or interrupted_response.audio_duration_sec

    return RequestResult(
        session_id=session_id,
        request_id=response.request_id,
        video_id=group.group_id,
        video_path=group.video_path,
        question_id=question.question_id,
        question_index=question_index,
        timestamp_sec=question.timestamp_sec,
        mode=args.mode,
        concurrency=args.concurrency,
        issued_at=issued_at,
        frames_sent=frames_sent,
        input_audio_sent=input_audio_sent,
        status=status,
        error=error,
        text=response.text,
        answer=question.answer,
        prediction=pred,
        correct=correct,
        text_ttft_sec=response.first_text_sec,
        audio_ttfp_sec=response.first_audio_sec,
        e2e_sec=e2e_sec,
        audio_chunks=response.audio_chunks,
        audio_duration_sec=audio_duration_sec,
        audio_stream_span_sec=audio_span,
        audio_stream_rtf=audio_stream_rtf,
        e2e_audio_rtf=e2e_audio_rtf,
        interrupted=interrupt_id is not None,
        interrupt_request_id=interrupt_id,
        interrupt_triggered_at_sec=interrupt_at,
        pre_interrupt_text_chars=pre_text_chars,
        pre_interrupt_audio_chunks=pre_audio_chunks,
        pre_interrupt_audio_duration_sec=pre_audio_duration,
        audio_underrun_sec=audio_underrun_sec,
        audio_underrun_event_count=audio_underrun_event_count,
        audio_continuity_ok=audio_continuity_ok,
        audio_continuity_threshold_sec=args.audio_continuity_threshold_sec,
        audio_continuity_evaluated=audio_continuity_evaluated,
        audio_continuity_excluded_reason=audio_continuity_excluded_reason,
        audio_timeline=response.audio_timeline if args.record_audio_timeline else [],
        frame_extract_sec=frame_extract_sec,
        frame_send_sec=frame_send_sec,
        input_audio_send_sec=input_audio_send_sec,
        query_collect_sec=query_collect_sec,
        client_question_sec=client_question_sec,
        prompt_chars=len(query_prompt),
        response_text_chars=len(response.text),
        kv_metadata_enabled=bool(kv_metadata),
        kv_committed_prefix_hash=(
            str(kv_metadata.get("committed_prefix_hash"))
            if kv_metadata and kv_metadata.get("committed_prefix_hash")
            else None
        ),
        kv_history_prefix_hash=(
            str(kv_metadata.get("omni_history_prefix_hash"))
            if kv_metadata and kv_metadata.get("omni_history_prefix_hash")
            else None
        ),
        kv_prefix_token_estimate=(
            int(kv_metadata["omni_prefix_token_estimate"])
            if kv_metadata and isinstance(kv_metadata.get("omni_prefix_token_estimate"), int)
            else None
        ),
        kv_history_tokens_before=(
            int(kv_metadata["omni_history_tokens_before"])
            if kv_metadata and isinstance(kv_metadata.get("omni_history_tokens_before"), int)
            else None
        ),
        kv_history_tokens_after=(
            int(kv_metadata["omni_history_tokens_after"])
            if kv_metadata and isinstance(kv_metadata.get("omni_history_tokens_after"), int)
            else None
        ),
        prefetch_requested=bool(prefetch_result),
        prefetch_ok=(
            bool(prefetch_result.get("ok"))
            if isinstance(prefetch_result, dict) and prefetch_result.get("ok") is not None
            else None
        ),
        prefetch_status=(
            str(prefetch_result.get("status"))
            if isinstance(prefetch_result, dict) and prefetch_result.get("status") is not None
            else None
        ),
        prefetch_request_id=(
            str(prefetch_result.get("request_id"))
            if isinstance(prefetch_result, dict) and prefetch_result.get("request_id")
            else None
        ),
        prefetch_scope=(
            str(prefetch_result.get("prefetch_scope"))
            if isinstance(prefetch_result, dict) and prefetch_result.get("prefetch_scope")
            else None
        ),
        prefetch_wall_sec=(
            float(prefetch_result.get("wall_s"))
            if isinstance(prefetch_result, dict) and isinstance(prefetch_result.get("wall_s"), (int, float))
            else None
        ),
        prefetch_skip_reason=(
            str(prefetch_result.get("skip_reason") or prefetch_result.get("error"))
            if isinstance(prefetch_result, dict)
            and (prefetch_result.get("skip_reason") is not None or prefetch_result.get("error") is not None)
            else None
        ),
        raw_response_errors=errors,
    )


async def run_video_session(
    session_index: int,
    group: VideoGroup,
    args: argparse.Namespace,
    write_lock: asyncio.Lock,
    jsonl_file: Any,
) -> list[RequestResult]:
    base_session_id = make_session_id(args, session_index)
    all_results: list[RequestResult] = []
    start_question_index = 0
    prev_timestamp = 0.0
    max_questions = len(group.questions)
    if args.max_questions_per_video is not None:
        max_questions = min(max_questions, args.max_questions_per_video)

    for attempt in range(args.session_retries + 1):
        session_id = base_session_id if attempt == 0 else f"{base_session_id}-retry{attempt}"
        try:
            res = await run_video_session_attempt(
                session_index,
                session_id,
                group,
                args,
                write_lock,
                jsonl_file,
                start_question_index,
                prev_timestamp,
            )
            all_results.extend(res)
            return all_results
        except SessionAttemptError as exc:
            if exc.results:
                all_results.extend(exc.results)
                turns_per_question = args.followup_turns_per_question + 1
                completed_questions = completed_question_blocks(exc.results, turns_per_question)
                start_question_index += completed_questions
                if start_question_index > 0:
                    prev_timestamp = group.questions[start_question_index - 1].timestamp_sec

            if start_question_index >= max_questions and is_transient_ws_error(exc.original):
                print(
                    f"[WARN] session close failed after all questions for {session_id}: {exc}; "
                    "keeping completed results",
                    file=sys.stderr,
                )
                return all_results

            can_retry = attempt < args.session_retries and is_transient_ws_error(exc.original)
            if not can_retry:
                raise exc.original
            sleep_sec = min(args.session_retry_backoff_sec * (2**attempt), 30.0)
            print(
                f"[WARN] session failed after {exc.results_count} result(s) for {session_id}; "
                f"resume_at_q={start_question_index}: {exc}; "
                f"retrying in {sleep_sec:.1f}s",
                file=sys.stderr,
            )
            await asyncio.sleep(sleep_sec)
    raise RuntimeError(f"unreachable session retry state for {base_session_id}")


async def run_video_session_attempt(
    session_index: int,
    session_id: str,
    group: VideoGroup,
    args: argparse.Namespace,
    write_lock: asyncio.Lock,
    jsonl_file: Any,
    start_question_index: int = 0,
    prev_timestamp: float = 0.0,
) -> list[RequestResult]:
    rng = random.Random(args.seed + session_index)
    results: list[RequestResult] = []
    if not group.resolved_video_path:
        for q_idx, question in enumerate(group.questions):
            result = RequestResult(
                session_id=session_id,
                request_id=f"{session_id}-q{q_idx:02d}",
                video_id=group.group_id,
                video_path=group.video_path,
                question_id=question.question_id,
                question_index=q_idx,
                timestamp_sec=question.timestamp_sec,
                mode=args.mode,
                concurrency=args.concurrency,
                issued_at=time.time(),
                status="missing_video",
                error=f"video not found: {group.video_path}",
            )
            results.append(result)
        return results

    system_prompt = build_system_prompt(args)
    config = {
        "type": "session.config",
        "model": args.model,
        "modalities": ["text", "audio"] if args.with_audio_output else ["text"],
        "max_frames": args.max_buffer_frames,
        "num_frames": args.num_sample_frames,
        "system_prompt": system_prompt,
        "enable_frame_filter": not args.no_evs,
        "frame_filter_threshold": args.evs_threshold,
        "use_audio_in_video": args.send_input_audio != "never",
        "sampling_params_list": build_sampling_params(
            args.with_audio_output,
            args.max_tokens,
            args.min_tokens,
            args.ignore_eos,
            args.disable_thinker_stop_token_ids,
        ),
        "include_request_ids": True,
    }
    ws = await connect_with_retries(args, config, session_id)
    body_exc: Exception | None = None
    dialogue_history: list[tuple[str, str]] = []
    server_history_messages: list[dict[str, Any]] = []
    max_questions = len(group.questions)
    if args.max_questions_per_video is not None:
        max_questions = min(max_questions, args.max_questions_per_video)
    try:
        for q_idx, question in enumerate(group.questions):
            question_t0 = time.monotonic()
            if args.max_questions_per_video is not None and q_idx >= args.max_questions_per_video:
                break
            if q_idx < start_question_index:
                continue
            if args.stream_segments:
                start_sec = max(0.0, prev_timestamp)
            else:
                start_sec = max(0.0, question.timestamp_sec - args.context_time)
            end_sec = max(start_sec + 0.1, question.timestamp_sec)
            extract_t0 = time.monotonic()
            frames = extract_frames(
                group.resolved_video_path,
                start_sec,
                end_sec,
                args.fps,
                args.frames_per_query,
                args.resize_max_side,
                args.jpeg_quality,
            )
            frame_extract_sec = time.monotonic() - extract_t0
            frame_send_t0 = time.monotonic()
            await send_frames(ws, frames, args.realtime_speedup)
            frame_send_sec = time.monotonic() - frame_send_t0

            input_audio_sent = False
            input_audio_send_sec = 0.0
            if args.send_input_audio != "never" and question.audio_path:
                try:
                    audio_t0 = time.monotonic()
                    pcm = load_audio_pcm16_16k(question.audio_path)
                    await send_audio(ws, pcm, args.audio_chunk_sec)
                    input_audio_send_sec = time.monotonic() - audio_t0
                    input_audio_sent = True
                except Exception as exc:
                    print(f"[WARN] failed to send audio {question.audio_path}: {exc}", file=sys.stderr)

            query_prompt = build_query_prompt(question, dialogue_history, args)
            issued_at = time.time()
            query_t0 = time.monotonic()
            request_index = q_idx * (args.followup_turns_per_question + 1)
            next_question_gap = (
                max(0.0, group.questions[q_idx + 1].timestamp_sec - question.timestamp_sec)
                if q_idx + 1 < max_questions
                else None
            )
            query_next_gap = 0.1 if args.followup_turns_per_question > 0 else next_question_gap
            query_terminal = args.followup_turns_per_question <= 0 and next_question_gap is None
            prefix_messages = recent_history_prefix_messages(system_prompt, server_history_messages)
            kv_metadata = None
            prefetch_result = None
            if args.enable_kv_metadata or args.enable_kv_prefetch:
                candidate_kv_metadata = build_kv_live_metadata(
                    args,
                    session_id=session_id,
                    request_index=request_index,
                    timestamp_sec=question.timestamp_sec,
                    query_prompt=query_prompt,
                    prefix_messages=prefix_messages,
                    next_gap_sec=query_next_gap,
                    is_terminal_turn=query_terminal,
                )
                prefix_hash = candidate_kv_metadata.get("committed_prefix_hash")
                explicit_kv_metadata = bool(
                    getattr(args, "kv_metadata_explicit", args.enable_kv_metadata)
                )
                if explicit_kv_metadata or prefix_hash:
                    kv_metadata = candidate_kv_metadata
                prefetch_result = await maybe_prefetch_history(
                    args,
                    session_id=session_id,
                    request_index=request_index,
                    prefix_messages=prefix_messages,
                    prefix_hash=prefix_hash,
                    prefix_token_estimate=int(
                        candidate_kv_metadata.get("omni_prefix_token_estimate") or 0
                    ),
                )
            response, interrupted_response, interrupt_at, interrupt_id = await query_and_collect(
                ws,
                question,
                query_prompt,
                session_id,
                request_index,
                args,
                rng,
                kv_metadata,
            )
            query_collect_sec = time.monotonic() - query_t0
            client_question_sec = time.monotonic() - question_t0
            result = response_to_result(
                session_id,
                response,
                interrupted_response,
                interrupt_at,
                interrupt_id,
                group,
                question,
                request_index,
                issued_at,
                len(frames),
                input_audio_sent,
                query_prompt,
                frame_extract_sec,
                frame_send_sec,
                input_audio_send_sec,
                query_collect_sec,
                client_question_sec,
                args,
                kv_metadata,
                prefetch_result,
            )
            results.append(result)
            async with write_lock:
                jsonl_file.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
                jsonl_file.flush()
            if result.status == "ok":
                dialogue_history.append((question.question, response.text))
                server_history_messages.append({"role": "user", "content": query_prompt})
                server_history_messages.append({"role": "assistant", "content": response.text})

            for followup_index in range(args.followup_turns_per_question):
                followup = make_followup_question(
                    question,
                    followup_index,
                    args.prompt_style,
                    args.followup_profile,
                )
                followup_t0 = time.monotonic()
                followup_prompt = build_query_prompt(followup, dialogue_history, args)
                followup_issued_at = time.time()
                followup_query_t0 = time.monotonic()
                followup_request_index = request_index + followup_index + 1
                is_last_followup = followup_index == args.followup_turns_per_question - 1
                followup_next_gap = next_question_gap if is_last_followup else 0.1
                followup_terminal = is_last_followup and next_question_gap is None
                followup_prefix_messages = recent_history_prefix_messages(
                    system_prompt,
                    server_history_messages,
                )
                followup_kv_metadata = None
                followup_prefetch_result = None
                if args.enable_kv_metadata or args.enable_kv_prefetch:
                    candidate_followup_kv_metadata = build_kv_live_metadata(
                        args,
                        session_id=session_id,
                        request_index=followup_request_index,
                        timestamp_sec=followup.timestamp_sec,
                        query_prompt=followup_prompt,
                        prefix_messages=followup_prefix_messages,
                        next_gap_sec=followup_next_gap,
                        is_terminal_turn=followup_terminal,
                    )
                    followup_prefix_hash = candidate_followup_kv_metadata.get(
                        "committed_prefix_hash"
                    )
                    explicit_kv_metadata = bool(
                        getattr(args, "kv_metadata_explicit", args.enable_kv_metadata)
                    )
                    if explicit_kv_metadata or followup_prefix_hash:
                        followup_kv_metadata = candidate_followup_kv_metadata
                    followup_prefetch_result = await maybe_prefetch_history(
                        args,
                        session_id=session_id,
                        request_index=followup_request_index,
                        prefix_messages=followup_prefix_messages,
                        prefix_hash=followup_prefix_hash,
                        prefix_token_estimate=int(
                            candidate_followup_kv_metadata.get("omni_prefix_token_estimate") or 0
                        ),
                    )
                followup_response, followup_interrupted, followup_interrupt_at, followup_interrupt_id = (
                    await query_and_collect(
                        ws,
                        followup,
                        followup_prompt,
                        session_id,
                        followup_request_index,
                        args,
                        rng,
                        followup_kv_metadata,
                    )
                )
                followup_query_collect_sec = time.monotonic() - followup_query_t0
                followup_client_question_sec = time.monotonic() - followup_t0
                followup_result = response_to_result(
                    session_id,
                    followup_response,
                    followup_interrupted,
                    followup_interrupt_at,
                    followup_interrupt_id,
                    group,
                    followup,
                    followup_request_index,
                    followup_issued_at,
                    0,
                    False,
                    followup_prompt,
                    0.0,
                    0.0,
                    0.0,
                    followup_query_collect_sec,
                    followup_client_question_sec,
                    args,
                    followup_kv_metadata,
                    followup_prefetch_result,
                )
                results.append(followup_result)
                async with write_lock:
                    jsonl_file.write(json.dumps(followup_result.to_dict(), ensure_ascii=False) + "\n")
                    jsonl_file.flush()
                if followup_result.status == "ok":
                    dialogue_history.append((followup.question, followup_response.text))
                    server_history_messages.append({"role": "user", "content": followup_prompt})
                    server_history_messages.append({"role": "assistant", "content": followup_response.text})
            prev_timestamp = question.timestamp_sec

        await ws.send(json.dumps({"type": "video.done"}))
        try:
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                if msg.get("type") == "session.done":
                    break
        except Exception:
            pass
    except Exception as exc:
        body_exc = exc
    finally:
        try:
            await ws.close()
        except Exception as exc:
            if body_exc is None and not results:
                body_exc = exc
            elif not is_transient_ws_error(exc):
                print(f"[WARN] websocket close failed for {session_id}: {exc}", file=sys.stderr)
    if body_exc is not None:
        raise SessionAttemptError(body_exc, results)
    return results


def percentile(values: list[float], p: float) -> float | None:
    clean = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    k = (len(clean) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return clean[int(k)]
    return clean[lo] * (hi - k) + clean[hi] * (k - lo)


def dist(values: list[float]) -> dict[str, float | None]:
    clean = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return {
        "count": len(clean),
        "mean": statistics.mean(clean) if clean else None,
        "p50": percentile(clean, 0.50),
        "p90": percentile(clean, 0.90),
        "p95": percentile(clean, 0.95),
    }


def summarize(results: list[RequestResult], wall_time_sec: float, args: argparse.Namespace) -> dict[str, Any]:
    ok = [r for r in results if r.status == "ok"]
    answerable = [r for r in ok if r.correct is not None]
    correct = [r for r in answerable if r.correct]
    rtf_values = [r.audio_stream_rtf for r in ok if r.audio_stream_rtf is not None]
    rtf_pass = [x for x in rtf_values if x is not None and x <= args.rtf_threshold]
    continuity_evaluated = [r for r in ok if r.audio_continuity_evaluated]
    continuity_ok = [r for r in continuity_evaluated if r.audio_continuity_ok]
    continuity_excluded_reasons: dict[str, int] = {}
    for result in ok:
        if result.audio_continuity_evaluated:
            continue
        reason = result.audio_continuity_excluded_reason or "not_evaluated"
        continuity_excluded_reasons[reason] = continuity_excluded_reasons.get(reason, 0) + 1

    return {
        "meta": {
            "mode": args.mode,
            "ws_url": args.ws_url,
            "model": args.model,
            "concurrency": args.concurrency,
            "request_id_prefix": args.request_id_prefix,
            "start_video_index": args.start_video_index,
            "with_audio_output": args.with_audio_output,
            "prompt_style": args.prompt_style,
            "dialogue_mode": args.dialogue_mode,
            "dialogue_history_turns": args.dialogue_history_turns,
            "followup_profile": args.followup_profile,
            "followup_turns_per_question": args.followup_turns_per_question,
            "max_tokens": args.max_tokens,
            "min_tokens": args.min_tokens,
            "ignore_eos": args.ignore_eos,
            "disable_thinker_stop_token_ids": args.disable_thinker_stop_token_ids,
            "rtf_threshold": args.rtf_threshold,
            "audio_continuity_threshold_sec": args.audio_continuity_threshold_sec,
            "audio_continuity_chunk_mode": args.audio_continuity_chunk_mode,
            "continuity_include_barge_in": args.continuity_include_interrupted,
            "continuity_include_interrupted": args.continuity_include_interrupted,
            "record_audio_timeline": args.record_audio_timeline,
            "enable_kv_metadata": args.enable_kv_metadata,
            "enable_kv_preload": args.enable_kv_prefetch,
            "enable_kv_prefetch": args.enable_kv_prefetch,
            "preload_scope": args.prefetch_scope,
            "preload_lead_ms": args.prefetch_lead_ms,
            "preload_max_tokens": args.prefetch_max_tokens,
            "prefetch_scope": args.prefetch_scope,
            "prefetch_lead_ms": args.prefetch_lead_ms,
            "prefetch_max_tokens": args.prefetch_max_tokens,
            "barge_in_time_anchor": args.interrupt_time_anchor,
            "interrupt_time_anchor": args.interrupt_time_anchor,
            "metadata": str(args.metadata),
            "data_root": str(args.data_root) if args.data_root else None,
        },
        "totals": {
            "requests": len(results),
            "ok": len(ok),
            "errors": len(results) - len(ok),
            "wall_time_sec": wall_time_sec,
            "throughput_qps": (len(ok) / wall_time_sec) if wall_time_sec > 0 else None,
            "answerable": len(answerable),
            "correct": len(correct),
            "accuracy": (len(correct) / len(answerable)) if answerable else None,
            "interrupted": sum(1 for r in ok if r.interrupted),
        },
        "latency": {
            "text_ttft_sec": dist([r.text_ttft_sec for r in ok if r.text_ttft_sec is not None]),
            "audio_ttfp_sec": dist([r.audio_ttfp_sec for r in ok if r.audio_ttfp_sec is not None]),
            "e2e_sec": dist([r.e2e_sec for r in ok if r.e2e_sec is not None]),
        },
        "audio": {
            "chunks": dist([float(r.audio_chunks) for r in ok]),
            "short_chunk_count": sum(1 for r in ok if r.audio_chunks <= 1),
            "short_chunk_rate": (sum(1 for r in ok if r.audio_chunks <= 1) / len(ok)) if ok else None,
            "duration_sec": dist([r.audio_duration_sec for r in ok if r.audio_duration_sec > 0]),
            "stream_rtf": dist([r.audio_stream_rtf for r in ok if r.audio_stream_rtf is not None]),
            "e2e_rtf": dist([r.e2e_audio_rtf for r in ok if r.e2e_audio_rtf is not None]),
            "rtf_pass_count": len(rtf_pass),
            "rtf_pass_rate": (len(rtf_pass) / len(rtf_values)) if rtf_values else None,
            "streaming_continuity": {
                "threshold_sec": args.audio_continuity_threshold_sec,
                "chunk_mode": args.audio_continuity_chunk_mode,
                "include_interrupted": args.continuity_include_interrupted,
                "n_total_ok": len(ok),
                "n_evaluated": len(continuity_evaluated),
                "n_ok": len(continuity_ok),
                "ok_rate": (len(continuity_ok) / len(continuity_evaluated)) if continuity_evaluated else None,
                "audio_underrun_sec": dist(
                    [
                        r.audio_underrun_sec
                        for r in continuity_evaluated
                        if r.audio_underrun_sec is not None
                    ]
                ),
                "audio_underrun_event_count": dist(
                    [
                        float(r.audio_underrun_event_count)
                        for r in continuity_evaluated
                        if r.audio_underrun_event_count is not None
                    ]
                ),
                "goodput_rps": (len(continuity_ok) / wall_time_sec) if wall_time_sec > 0 else None,
                "excluded_reasons": continuity_excluded_reasons,
            },
        },
        "text": {
            "response_chars": dist([float(r.response_text_chars or len(r.text or "")) for r in ok]),
        },
        "client": {
            "frame_extract_sec": dist([r.frame_extract_sec for r in ok if r.frame_extract_sec is not None]),
            "frame_send_sec": dist([r.frame_send_sec for r in ok if r.frame_send_sec is not None]),
            "input_audio_send_sec": dist(
                [r.input_audio_send_sec for r in ok if r.input_audio_send_sec is not None]
            ),
            "query_collect_sec": dist([r.query_collect_sec for r in ok if r.query_collect_sec is not None]),
            "client_question_sec": dist([r.client_question_sec for r in ok if r.client_question_sec is not None]),
            "prompt_chars": dist([float(r.prompt_chars) for r in ok if r.prompt_chars > 0]),
        },
        "interrupt": {
            "pre_interrupt_text_chars": dist([float(r.pre_interrupt_text_chars) for r in ok if r.interrupted]),
            "pre_interrupt_audio_chunks": dist([float(r.pre_interrupt_audio_chunks) for r in ok if r.interrupted]),
            "pre_interrupt_audio_duration_sec": dist(
                [r.pre_interrupt_audio_duration_sec for r in ok if r.interrupted]
            ),
        },
        "kv": {
            "metadata_enabled_requests": sum(1 for r in results if r.kv_metadata_enabled),
            "with_committed_prefix": sum(1 for r in results if r.kv_committed_prefix_hash),
            "prefix_token_estimate": dist(
                [
                    float(r.kv_prefix_token_estimate)
                    for r in results
                    if r.kv_prefix_token_estimate is not None
                ]
            ),
            "history_tokens_before": dist(
                [
                    float(r.kv_history_tokens_before)
                    for r in results
                    if r.kv_history_tokens_before is not None
                ]
            ),
            "prefetch": {
                "attempts": sum(1 for r in results if r.prefetch_requested),
                "ok": sum(1 for r in results if r.prefetch_ok is True),
                "failed_or_skipped": sum(1 for r in results if r.prefetch_requested and r.prefetch_ok is not True),
                "status_counts": {
                    status: sum(1 for r in results if r.prefetch_status == status)
                    for status in sorted({r.prefetch_status for r in results if r.prefetch_status})
                },
                "skip_reason_counts": {
                    reason: sum(1 for r in results if r.prefetch_skip_reason == reason)
                    for reason in sorted({r.prefetch_skip_reason for r in results if r.prefetch_skip_reason})
                },
                "wall_sec": dist(
                    [r.prefetch_wall_sec for r in results if r.prefetch_wall_sec is not None]
                ),
            },
        },
    }


async def run(args: argparse.Namespace) -> int:
    if not args.health_url:
        args.health_url = derive_health_url(args.ws_url)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    groups = load_groups(
        Path(args.metadata),
        Path(args.data_root) if args.data_root else None,
        args.start_video_index,
        args.max_videos,
        args.max_questions_per_video,
        args.prompt_style,
    )
    if args.repeat > 1:
        base = groups
        groups = []
        for rep in range(args.repeat):
            for group in base:
                groups.append(
                    VideoGroup(
                        group_id=f"{group.group_id}-r{rep}",
                        video_path=group.video_path,
                        resolved_video_path=group.resolved_video_path,
                        questions=group.questions,
                    )
                )
    if not groups:
        print("[fatal] no benchmark groups loaded", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"loaded groups={len(groups)}")
        for group in groups[:3]:
            print(
                json.dumps(
                    {
                        "group_id": group.group_id,
                        "video_path": group.video_path,
                        "resolved": group.resolved_video_path,
                        "questions": [q.question_id for q in group.questions],
                    },
                    indent=2,
                )
            )
        return 0

    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    stop_event = asyncio.Event()
    all_results: list[RequestResult] = []
    t0 = time.monotonic()
    jsonl_path = result_dir / "per_request.jsonl"

    async def _run_one(i: int, group: VideoGroup, jsonl_file: Any) -> None:
        async with sem:
            if stop_event.is_set():
                return
            try:
                res = await run_video_session(i, group, args, write_lock, jsonl_file)
                all_results.extend(res)
            except Exception as exc:
                session_id = make_session_id(args, i)
                if is_server_unavailable_error(exc):
                    stop_event.set()
                    print(
                        f"[fatal] server unavailable while opening {session_id}; "
                        "stopping queued sessions",
                        file=sys.stderr,
                    )
                err = RequestResult(
                    session_id=session_id,
                    request_id=f"{session_id}-fatal",
                    video_id=group.group_id,
                    video_path=group.video_path,
                    question_id="fatal",
                    question_index=-1,
                    timestamp_sec=0.0,
                    mode=args.mode,
                    concurrency=args.concurrency,
                    issued_at=time.time(),
                    status="error",
                    error=str(exc),
                )
                all_results.append(err)
                async with write_lock:
                    jsonl_file.write(json.dumps(err.to_dict(), ensure_ascii=False) + "\n")
                    jsonl_file.flush()

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        tasks = [_run_one(i, group, jsonl_file) for i, group in enumerate(groups)]
        await asyncio.gather(*tasks)

    wall = time.monotonic() - t0
    # A websocket session can fail after it has already written successful
    # per-question rows. Summarize the durable JSONL records so summary.json
    # matches the data used by plot/report scripts.
    all_results = []
    with jsonl_path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if line.strip():
                all_results.append(RequestResult(**json.loads(line)))
    raw_results_count = len(all_results)
    all_results = dedupe_retry_results(all_results)
    summary = summarize(all_results, wall, args)
    summary["totals"]["raw_requests"] = raw_results_count
    summary["totals"]["deduped_requests"] = len(all_results)
    with (result_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (result_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump({"argv": sys.argv, "summary": summary["meta"]}, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["totals"]["errors"] == 0 else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--metadata", default=str(here / "example_questions.json"))
    parser.add_argument("--data-root", default=str(Path("/workspace/xyz/test/video/StreamingBench/src/data")))
    parser.add_argument("--result-dir", default=str(Path("results/video/streamingbench_smoke")))
    parser.add_argument("--ws-url", default=os.environ.get("LIVESERVE_VIDEO_WS_URL", DEFAULT_WS_URL))
    parser.add_argument("--model", default=os.environ.get("LIVESERVE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--mode", choices=["baseline", "bargein", "barge_in", "interrupt"], default="baseline")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--start-video-index",
        type=int,
        default=0,
        help="Skip this many grouped videos before applying --max-videos.",
    )
    parser.add_argument(
        "--request-id-prefix",
        default=os.environ.get("STREAMINGBENCH_REQUEST_ID_PREFIX", ""),
        help="Optional namespace prepended to all client request IDs.",
    )
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-questions-per-video", type=int, default=None)
    parser.add_argument("--context-time", type=float, default=60.0)
    parser.add_argument("--stream-segments", action="store_true", help="Send only the segment since the previous query.")
    parser.add_argument(
        "--prompt-style",
        choices=["mcq_letter", "long_answer", "detailed_answer", "extended_answer", "scripted_long_answer"],
        default=os.environ.get("STREAMINGBENCH_PROMPT_STYLE", "mcq_letter"),
        help=(
            "mcq_letter preserves official short-answer behavior; long_answer "
            "uses 2-3 spoken sentences; detailed_answer uses 5-7 spoken sentences; "
            "extended_answer uses 9-12 spoken sentences and asks for option comparison; "
            "scripted_long_answer enforces four spoken paragraphs for barge-in stress."
        ),
    )
    parser.add_argument(
        "--dialogue-mode",
        choices=["single", "multiturn"],
        default=os.environ.get("STREAMINGBENCH_DIALOGUE_MODE", "single"),
        help="When multiturn, include previous turns from the same video session in the next query prompt.",
    )
    parser.add_argument("--dialogue-history-turns", type=int, default=3)
    parser.add_argument("--dialogue-history-answer-chars", type=int, default=600)
    parser.add_argument(
        "--followup-profile",
        choices=["evidence", "varied"],
        default=os.environ.get("STREAMINGBENCH_FOLLOWUP_PROFILE", "evidence"),
        help="How no-new-frame follow-up turns are phrased. varied cycles clarification, alternatives, and timeline prompts.",
    )
    parser.add_argument(
        "--followup-turns-per-question",
        type=int,
        default=int(os.environ.get("STREAMINGBENCH_FOLLOWUP_TURNS_PER_QUESTION", "0")),
        help="Append this many no-new-frame follow-up turns after each timestamped video question.",
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--frames-per-query", type=int, default=32)
    parser.add_argument("--max-buffer-frames", type=int, default=64)
    parser.add_argument("--num-sample-frames", type=int, default=16)
    parser.add_argument("--resize-max-side", type=int, default=448)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--realtime-speedup", type=float, default=0.0, help="0 means send frames as fast as possible.")
    parser.add_argument("--with-audio-output", action="store_true")
    parser.add_argument("--send-input-audio", choices=["auto", "never"], default="auto")
    parser.add_argument("--audio-chunk-sec", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=int(os.environ.get("STREAMINGBENCH_MIN_TOKENS", "0")),
        help="Minimum thinker output tokens. Useful for preventing long-answer prompts from collapsing to one audio chunk.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        default=env_flag("STREAMINGBENCH_IGNORE_EOS", False),
        help="Ignore thinker EOS and rely on --max-tokens for benchmark-only long-answer stress tests.",
    )
    parser.add_argument(
        "--disable-thinker-stop-token-ids",
        action="store_true",
        default=env_flag("STREAMINGBENCH_DISABLE_THINKER_STOP_TOKEN_IDS", False),
        help=(
            "Remove thinker stop_token_ids from sampling params. This is a "
            "benchmark-only control used with --ignore-eos and --min-tokens "
            "to reduce output-length skew between variants."
        ),
    )
    parser.add_argument("--barge-in-prob", "--interrupt-prob",
                        dest="interrupt_prob", type=float, default=1.0)
    parser.add_argument("--barge-in-after-seconds", "--interrupt-after-seconds",
                        dest="interrupt_after_seconds", type=float, default=2.0)
    parser.add_argument("--barge-in-jitter-sec", "--interrupt-jitter-sec",
                        dest="interrupt_jitter_sec", type=float, default=0.0)
    parser.add_argument(
        "--barge-in-time-anchor",
        "--interrupt-time-anchor",
        dest="interrupt_time_anchor",
        choices=["request", "text", "audio"],
        default=os.environ.get(
            "STREAMINGBENCH_BARGE_IN_TIME_ANCHOR",
            os.environ.get("STREAMINGBENCH_INTERRUPT_TIME_ANCHOR", "request"),
        ),
        help=(
            "request triggers after query send; text triggers after first text delta; "
            "audio triggers after first audio delta. Audio anchoring is useful for "
            "barge-in workloads where the user interrupts playback, not prompt prefill."
        ),
    )
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--post-text-audio-grace-sec", type=float, default=5.0)
    parser.add_argument("--rtf-threshold", type=float, default=1.0)
    parser.add_argument(
        "--audio-continuity-threshold-sec",
        type=float,
        default=env_float(
            "STREAMINGBENCH_AUDIO_CONTINUITY_THRESHOLD_SEC",
            get_audio_continuity_threshold_s(),
        ),
        help=(
            "Maximum tolerated streamed-audio underrun in seconds. "
            "Defaults to STREAMINGBENCH_AUDIO_CONTINUITY_THRESHOLD_SEC, "
            "LIVESERVE_AUDIO_CONTINUITY_THRESHOLD_S, or 0.1."
        ),
    )
    parser.add_argument(
        "--audio-continuity-chunk-mode",
        choices=["fragment", "cumulative", "auto"],
        default=os.environ.get("STREAMINGBENCH_AUDIO_CONTINUITY_CHUNK_MODE", "fragment"),
        help="How to interpret WAV duration in response.audio.delta chunks.",
    )
    parser.add_argument(
        "--continuity-include-barge-in",
        "--continuity-include-interrupted",
        dest="continuity_include_interrupted",
        action="store_true",
        default=env_flag(
            "STREAMINGBENCH_CONTINUITY_INCLUDE_BARGE_IN",
            env_flag("STREAMINGBENCH_CONTINUITY_INCLUDE_INTERRUPTED", False),
        ),
        help="Include barge-in requests in audio continuity evaluation.",
    )
    parser.add_argument(
        "--record-audio-timeline",
        action="store_true",
        default=env_flag("STREAMINGBENCH_RECORD_AUDIO_TIMELINE", False),
        help="Store per-audio-chunk receive timestamps and durations in per_request.jsonl.",
    )
    parser.add_argument(
        "--enable-kv-metadata",
        action="store_true",
        default=env_flag("STREAMINGBENCH_ENABLE_KV_METADATA", False),
        help="Attach interaction-aware KV metadata to video.query messages.",
    )
    parser.add_argument(
        "--enable-kv-preload",
        "--enable-kv-prefetch",
        dest="enable_kv_prefetch",
        action="store_true",
        default=env_flag(
            "STREAMINGBENCH_ENABLE_KV_PRELOAD",
            env_flag("STREAMINGBENCH_ENABLE_KV_PREFETCH", False),
        ),
        help="Issue /v1/omni/prefetch_session for committed video dialogue history.",
    )
    parser.add_argument(
        "--preload-url",
        "--prefetch-url",
        dest="prefetch_url",
        default=os.environ.get("STREAMINGBENCH_PREFETCH_URL", ""),
        help="Override /v1/omni/prefetch_session URL; defaults to the ws host.",
    )
    parser.add_argument(
        "--preload-scope",
        "--prefetch-scope",
        dest="prefetch_scope",
        default=os.environ.get(
            "STREAMINGBENCH_PRELOAD_SCOPE",
            os.environ.get("STREAMINGBENCH_PREFETCH_SCOPE", "history_preload"),
        ),
        choices=["history_preload", "current_turn_prefill", "full_prompt"],
    )
    parser.add_argument(
        "--preload-lead-ms",
        "--prefetch-lead-ms",
        dest="prefetch_lead_ms",
        type=float,
        default=env_float("STREAMINGBENCH_PRELOAD_LEAD_MS", env_float("STREAMINGBENCH_PREFETCH_LEAD_MS", 5000.0)),
    )
    parser.add_argument(
        "--preload-timeout-sec",
        "--prefetch-timeout-sec",
        dest="prefetch_timeout_sec",
        type=float,
        default=env_float("STREAMINGBENCH_PRELOAD_TIMEOUT_SEC", env_float("STREAMINGBENCH_PREFETCH_TIMEOUT_SEC", 5.0)),
    )
    parser.add_argument(
        "--preload-max-tokens",
        "--prefetch-max-tokens",
        dest="prefetch_max_tokens",
        type=int,
        default=int(os.environ.get("STREAMINGBENCH_PRELOAD_MAX_TOKENS", os.environ.get("STREAMINGBENCH_PREFETCH_MAX_TOKENS", "1"))),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-evs", action="store_true")
    parser.add_argument("--evs-threshold", type=float, default=0.95)
    parser.add_argument("--websocket-max-size", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--websocket-open-timeout", type=float, default=45.0)
    parser.add_argument("--websocket-close-timeout", type=float, default=1.0)
    parser.add_argument("--websocket-connect-retries", type=int, default=5)
    parser.add_argument("--websocket-retry-backoff-sec", type=float, default=1.0)
    parser.add_argument("--session-retries", type=int, default=10)
    parser.add_argument("--session-retry-backoff-sec", type=float, default=2.0)
    parser.add_argument(
        "--health-url",
        default=None,
        help="Health endpoint used to fail fast when the server stays up but its engine is dead.",
    )
    parser.add_argument("--health-check-interval-sec", type=float, default=5.0)
    parser.add_argument("--health-timeout-sec", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.mode == "barge_in":
        args.mode = "bargein"
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.followup_turns_per_question < 0:
        raise SystemExit("--followup-turns-per-question must be >= 0")
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    if args.audio_continuity_threshold_sec < 0:
        raise SystemExit("--audio-continuity-threshold-sec must be >= 0")
    args.kv_metadata_explicit = bool(args.enable_kv_metadata)
    if args.enable_kv_prefetch:
        args.enable_kv_metadata = True
    if args.prefetch_max_tokens < 1:
        raise SystemExit("--preload-max-tokens must be >= 1")
    if args.prefetch_timeout_sec <= 0:
        raise SystemExit("--preload-timeout-sec must be > 0")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
