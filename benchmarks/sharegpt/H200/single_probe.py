"""Send one ShareGPT sample to a running vllm-omni server.

The helper supports text-only and text+audio modes for the Qwen3-Omni
three-stage pipeline.

Usage:
  # Default text-only mode.
  python3 single_probe.py

  # Enable text+audio output.
  python3 single_probe.py --with-audio

  # Non-streaming text+audio.
  python3 single_probe.py --with-audio --no-stream

  # Select one sample or walk the first N shuffled samples.
  python3 single_probe.py --with-audio --index 3
  python3 single_probe.py --with-audio --walk 5

  # Use a custom prompt instead of loading the dataset.
  python3 single_probe.py --with-audio \\
      --custom-prompt "请用中文解释一下量子纠缠，答案控制在30字以内。"

  # Save audio to a custom directory.
  python3 single_probe.py --with-audio --audio-dir /workspace/xyz/output_audio
"""

import argparse
import base64
import io
import json
import os
import random
import sys
import time
import wave

import requests

import os as _os
_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_DATASET = _os.environ.get(
    "LIVESERVE_SHAREGPT_DATASET",
    _os.path.join(_REPO_ROOT, "data", "ShareGPT_Chinese_English_90k.json"),
)
DEFAULT_URL = "http://127.0.0.1:8666/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_AUDIO_DIR = _os.environ.get(
    "LIVESERVE_SHAREGPT_AUDIO_DIR",
    _os.path.join(_os.path.dirname(__file__), "output_audio"),
)

# Qwen3-Omni system prompt from the official online serving examples.
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, "
    "Alibaba Group, capable of perceiving auditory and visual inputs, "
    "as well as generating text and speech."
)

# Default three-stage sampling params from the official Qwen3-Omni example.
THINKER_SP = {
    "temperature": 0.4,
    "top_p": 0.9,
    "top_k": 1,
    "seed": 42,
    "repetition_penalty": 1.05,
    "stop_token_ids": [151645],  # <|im_end|>
}
TALKER_SP = {
    "temperature": 0.9,
    "top_k": 50,
    "max_tokens": 4096,
    "seed": 42,
    "detokenize": False,
    "repetition_penalty": 1.05,
    "stop_token_ids": [2150],  # TALKER_CODEC_EOS
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


def load_samples(path, seed=0, shuffle=True):
    """Mirror vLLM ShareGPTDataset ordering so probes match benchmark input."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = [e for e in data if "conversations" in e and len(e["conversations"]) >= 2]
    if shuffle:
        random.seed(seed)
        random.shuffle(data)
    return data


def build_payload(prompt, args):
    """Build an OpenAI-compatible omni chat payload.

    - text_only=True triggers only the thinker through modalities=["text"].
    - text_only=False runs text+audio with a three-stage sampling_params_list.
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    payload = {
        "model": args.model,
        "messages": messages,
        "stream": args.stream,
    }
    if args.stream:
        payload["stream_options"] = {"include_usage": True}

    if args.text_only:
        # Text-only path: thinker only, with max_tokens at the top level.
        payload["modalities"] = ["text"]
        payload["max_tokens"] = args.max_tokens
        payload["temperature"] = 0.7
        payload["top_p"] = 0.8
        payload["top_k"] = 20
    else:
        # Text input + speech output through the full three-stage pipeline.
        payload["modalities"] = ["text", "audio"]
        thinker = {**THINKER_SP, "max_tokens": args.max_tokens}
        # Official order: [thinker, talker, code2wav].
        payload["sampling_params_list"] = [thinker, TALKER_SP, CODE2WAV_SP]

    return payload


def save_wav(audio_bytes, out_dir, idx, tag=""):
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    path = os.path.join(out_dir, f"probe_{idx}{suffix}.wav")
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return path


def parse_wav_info(audio_bytes):
    """Return (duration_s, sample_rate), or (None, None) on parse failure."""
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate, rate
    except Exception:
        return None, None


def handle_stream(idx, resp, args, t0):
    """Parse SSE chunks, using obj["modality"] to separate text/audio.

    vllm-omni streaming protocol:
      - text chunk:  {"modality": "text",  "choices": [{"delta": {"content": "..."}}]}
      - audio chunk: {"modality": "audio", "choices": [{"delta": {"content": "<base64 wav>"}}]}
      each audio chunk contains a playable WAV fragment.
    """
    total_text = ""
    audio_chunks = []
    text_ttft = 0.0
    audio_ttfp = 0.0
    usage = None

    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8", errors="replace")
        if not decoded.startswith("data: "):
            continue
        chunk = decoded[6:]
        if chunk == "[DONE]":
            break
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue

        if obj.get("usage"):
            usage = obj["usage"]

        choices = obj.get("choices") or []
        if not choices:
            continue

        modality = obj.get("modality", "text")
        delta = choices[0].get("delta") or {}
        content = delta.get("content") or ""

        if modality == "text" and content:
            if text_ttft == 0.0:
                text_ttft = time.time() - t0
            total_text += content
            sys.stdout.write(content)
            sys.stdout.flush()
        elif modality == "audio" and content:
            if audio_ttfp == 0.0:
                audio_ttfp = time.time() - t0
                print(f"\n[{idx}] audio first chunk at {audio_ttfp:.2f}s")
            try:
                audio_chunks.append(base64.b64decode(content))
                sys.stdout.write(f"\r[{idx}] audio chunks received: {len(audio_chunks)}")
                sys.stdout.flush()
            except Exception as e:
                print(f"\n[{idx}] audio b64 decode error: {e}")

    elapsed = time.time() - t0

    print(f"\n\n--- [{idx}] Summary ({'text+audio' if not args.text_only else 'text-only'}) ---")
    print(f"  elapsed        : {elapsed:.2f}s")
    if text_ttft > 0:
        print(f"  text TTFT      : {text_ttft:.3f}s")
    print(f"  text chars     : {len(total_text)}")
    if usage:
        print(f"  usage          : {usage}")

    if audio_chunks:
        print(f"  audio TTFP     : {audio_ttfp:.3f}s")
        print(f"  audio chunks   : {len(audio_chunks)}")
        # Each audio chunk is a full WAV. Save the last chunk, and save the
        # first chunk as a useful sanity comparison when multiple chunks arrive.
        last_path = save_wav(audio_chunks[-1], args.audio_dir, idx, "last")
        print(f"  saved (last)   : {last_path} ({len(audio_chunks[-1])/1024:.1f} KB)")
        if len(audio_chunks) > 1:
            first_path = save_wav(audio_chunks[0], args.audio_dir, idx, "first")
            print(f"  saved (first)  : {first_path} ({len(audio_chunks[0])/1024:.1f} KB)")

        dur, sr = parse_wav_info(audio_chunks[-1])
        if dur:
            print(f"  audio duration : {dur:.2f}s @ {sr}Hz (last chunk)")
            print(f"  audio RTF      : {elapsed/dur:.2f}x  (elapsed / duration)")
    elif not args.text_only:
        print("  [WARN] no audio chunks received; check the server stage config")

    return True


def handle_non_stream(idx, resp, args, t0):
    elapsed = time.time() - t0
    data = resp.json()
    choices = data.get("choices") or []
    usage = data.get("usage", {})

    print(f"\n--- [{idx}] Non-stream Summary ---")
    print(f"  elapsed : {elapsed:.2f}s")
    print(f"  usage   : {usage}")

    for ci, choice in enumerate(choices):
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        audio_obj = msg.get("audio")
        finish = choice.get("finish_reason")
        print(f"\n  choice[{ci}] finish_reason={finish}")

        if content:
            preview = content[:300].replace("\n", " ")
            print(f"    text ({len(content)} chars): {preview!r}")

        if audio_obj and audio_obj.get("data"):
            audio_bytes = base64.b64decode(audio_obj["data"])
            path = save_wav(audio_bytes, args.audio_dir, idx, f"c{ci}")
            print(f"    audio saved: {path} ({len(audio_bytes)/1024:.1f} KB)")
            dur, sr = parse_wav_info(audio_bytes)
            if dur:
                print(f"    audio duration: {dur:.2f}s @ {sr}Hz  RTF={elapsed/dur:.2f}x")
        elif not args.text_only:
            print(f"    [WARN] no audio in response")

    return True


def send_one(idx, prompt, args):
    mode = "text+audio (3-stage)" if not args.text_only else "text-only (thinker)"
    preview = prompt[:100].replace("\n", " ")
    print(f"\n{'=' * 60}")
    print(f"[{idx}] mode={mode} prompt_chars={len(prompt)} preview={preview!r}")
    print(f"{'=' * 60}")

    if args.max_prompt_chars > 0:
        prompt = prompt[: args.max_prompt_chars]

    payload = build_payload(prompt, args)
    t0 = time.time()

    # Explicitly bypass proxies. requests honors upper/lowercase proxy env vars,
    # and stale proxy settings can make a probe appear to run without the server
    # seeing anything beyond health checks.
    session = requests.Session()
    session.trust_env = False
    session.proxies = {"http": "", "https": ""}

    print(f"[{idx}] POST {args.url} (proxies bypassed)")
    try:
        resp = session.post(
            args.url,
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=args.stream,
            timeout=args.timeout,
            proxies={"http": "", "https": ""},
        )
    except Exception as e:
        print(f"[{idx}] request raised: {type(e).__name__}: {e}")
        return False

    print(f"[{idx}] status={resp.status_code} connect_time={time.time()-t0:.2f}s")
    if resp.status_code != 200:
        print(f"[{idx}] body: {resp.text[:500]}")
        return False

    if args.stream:
        return handle_stream(idx, resp, args, t0)
    return handle_non_stream(idx, resp, args, t0)


def main():
    ap = argparse.ArgumentParser(
        description="Qwen3-Omni single sample probe (text-only / text+audio)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR, help="audio output directory")

    # Sample selection.
    ap.add_argument("--index", type=int, default=0, help="run only this sample index")
    ap.add_argument("--walk", type=int, default=0, help="run the first N samples sequentially")
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed matching the benchmark")
    ap.add_argument("--no-shuffle", action="store_true", help="disable dataset shuffle")
    ap.add_argument("--custom-prompt", type=str, default=None,
                    help="use a custom prompt instead of loading the dataset")

    # Generation settings.
    ap.add_argument("--max-tokens", type=int, default=256, help="thinker max_tokens")
    ap.add_argument("--max-prompt-chars", type=int, default=0,
                    help="truncate prompt to this many characters when >0")
    ap.add_argument("--timeout", type=int, default=600)

    # Streaming mode.
    ap.add_argument("--stream", action="store_true", default=True, help="use SSE streaming")
    ap.add_argument("--no-stream", dest="stream", action="store_false", help="disable streaming")

    # Modality selection.
    ap.add_argument("--text-only", action="store_true", default=True,
                    help="request only text through modalities=['text']")
    ap.add_argument("--with-audio", dest="text_only", action="store_false",
                    help="request text+audio through the full three-stage path")

    args = ap.parse_args()

    print(f"Mode       : {'text+audio (3-stage)' if not args.text_only else 'text-only'}")
    print(f"Stream     : {args.stream}")
    print(f"URL        : {args.url}")
    print(f"Model      : {args.model}")
    print(f"Audio dir  : {args.audio_dir if not args.text_only else '(N/A)'}")

    # Custom prompt mode.
    if args.custom_prompt:
        print(f"Using custom prompt (len={len(args.custom_prompt)})")
        send_one(0, args.custom_prompt, args)
        return

    # Dataset mode.
    samples = load_samples(args.dataset, seed=args.seed, shuffle=not args.no_shuffle)
    print(f"Loaded {len(samples)} samples from {args.dataset}")

    if args.walk > 0:
        for i in range(min(args.walk, len(samples))):
            prompt = samples[i]["conversations"][0]["value"]
            ok = send_one(i, prompt, args)
            if not ok:
                print(f"!!! Stopped at index {i}")
                break
            time.sleep(0.3)
    else:
        prompt = samples[args.index]["conversations"][0]["value"]
        send_one(args.index, prompt, args)


if __name__ == "__main__":
    main()
