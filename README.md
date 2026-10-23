# LiveServe

LiveServe is a serving system for interactive multimodal models, built on
[vLLM-Omni](https://github.com/vllm-project/vllm-omni). It coordinates text and
speech generation across model stages, manages session KV caches, and provides
clients for measuring latency, throughput, and streaming audio continuity.

The primary quickstart uses **Qwen3-Omni-30B-A3B-Instruct**. Model integrations
and benchmark clients for **Ming** are also available.

## Features

- **Interaction-aware scheduling:** control how far text and audio generation
  run ahead, with optional first-chunk protection and resumption policies.
- **Session cache management:** prefetch conversation history, route subsequent
  requests toward prepared replicas, and offload KV caches together with the
  intermediate features needed to reuse them correctly.
- **Configurable cache eviction:** evaluate retention policies with separate
  observation and enforcement modes.
- **Streaming inference:** generate text and audio incrementally, with request
  cancellation propagated through the serving pipeline.
- **Benchmark clients:** run ShareGPT, multi-turn interaction, and video
  workloads, and collect per-request latency and audio measurements.

For Qwen3-Omni, inference follows this pipeline:

```text
Request → Thinker → Talker → Code2Wav → Streaming audio
              └────────────────────→ Streaming text
```

## Requirements

The commands below target the following GPU environment:

| Component | Configuration |
|---|---|
| Operating system | Linux x86_64, glibc 2.35 or newer |
| Python | 3.12 |
| vLLM | 0.20.0 |
| PyTorch / CUDA runtime | 2.11.0 / CUDA 13.0 |
| GPUs | 4 × NVIDIA RTX PRO 6000 Blackwell Server, 96 GB each |
| Host memory | At least 96 GiB available for the example deployment |
| Tools | Git, Bash, curl, and an NVIDIA driver compatible with CUDA 13.0 |

The default [vLLM 0.20.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.20.0)
uses PyTorch 2.11 and CUDA 13.0. Use matching wheels and drivers when adapting
these instructions to another platform.

The supplied four-GPU configuration places two Thinker replicas on GPUs 0–1
and two Talker/Code2Wav replicas on GPUs 2–3. It reserves 16 GiB of CPU KV
storage and 2 GiB of sidecar storage per configured connector. Start with
idle GPUs and sufficient host memory. Other GPU layouts require corresponding
changes to the deployment YAML.

## Quickstart

Run the following commands from the repository root unless stated otherwise.

### 1. Install

```bash
git clone https://github.com/xiangyuzhi/LiveServe-artifacts.git
cd LiveServe-artifacts

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

VLLM_OMNI_TARGET_DEVICE=cuda \
VLLM_OMNI_VERSION_OVERRIDE=0.0.dev20261023+liveserve \
  python -m pip install "vllm==0.20.0" -e ./vllm-omni \
    -r requirements/dev.txt

python -m pip check
python -c "import torch, vllm, vllm_omni; print('vLLM:', vllm.__version__, 'torch:', torch.__version__, 'CUDA:', torch.version.cuda)"
vllm-omni --help
```

Use the local runtime under `vllm-omni/` to include LiveServe's scheduling and
cache-management changes. The version override supplies a consistent package
version for editable installation.

### 2. Download the model and start the server

```bash
export MODEL_ID=Qwen/Qwen3-Omni-30B-A3B-Instruct
hf download "$MODEL_ID"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm-omni serve "$MODEL_ID" \
  --omni \
  --deploy-config deploy/Blackwell/qwen3_omni_4gpu_smoke.yaml \
  --stage-init-timeout 1800 \
  --host 127.0.0.1 --port 8666
```

Model weights are downloaded to the Hugging Face cache. Set `HF_HOME` before
downloading and starting the server if you need a different cache location.
The deployment file is [qwen3_omni_4gpu_smoke.yaml](deploy/Blackwell/qwen3_omni_4gpu_smoke.yaml).

In another terminal, activate the same environment and wait for the health
endpoint to return HTTP 200:

```bash
source .venv/bin/activate
curl --fail http://127.0.0.1:8666/health
```

### 3. Send a request

Send a streaming text request:

```bash
curl --fail --silent --show-error --no-buffer --max-time 600 \
  http://127.0.0.1:8666/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Explain prefix caching in two sentences."}],
    "modalities": ["text"],
    "stream": true,
    "max_tokens": 64
  }'
```

To exercise all three stages and inspect audio output, use the probe client:

```bash
python benchmarks/sharegpt/H200/single_probe.py \
  --url http://127.0.0.1:8666/v1/chat/completions \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --with-audio \
  --custom-prompt "Say hello and introduce yourself in one sentence." \
  --max-tokens 64 \
  --audio-dir results/audio_probe
```

Check the printed HTTP status, audio chunk count, and positive WAV duration.
The probe prints generated text and saves the last WAV chunk in
`results/audio_probe/`, plus the first chunk when multiple chunks arrive.
These files contain individual chunks. Audio
responses use SSE with Base64-encoded WAV data, so saving the raw HTTP response
as a `.wav` file does not produce a playable recording.

## Run a ShareGPT benchmark

Keep the server running and use a second terminal with the environment
activated.

### Prepare the dataset

```bash
LIVESERVE_PYBIN="$(command -v python)" bash scripts/prepare_sharegpt.sh
```

This downloads `shareAI/ShareGPT-Chinese-English-90k` and creates
`data/ShareGPT_Chinese_English_90k.json`. To use an existing normalized dataset,
pass its path to the driver's `--dataset` argument.

### Check the workload

Start with four requests at concurrency one:

```bash
python benchmarks/sharegpt/H200/sharegpt_driver.py \
  --dataset data/ShareGPT_Chinese_English_90k.json \
  --url http://127.0.0.1:8666/v1/chat/completions \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --mode audio \
  --arrival-process closed_loop \
  --num-requests 4 --concurrency 1 \
  --max-tokens 64 --max-prompt-chars 500 \
  --seed 42 --policy none \
  --result-dir results/sharegpt_audio_smoke

python -m json.tool results/sharegpt_audio_smoke/summary.json
```

For this workload, check that `totals.completed_ok` is 4, `totals.errors` is 0,
all four requests have positive decoded audio durations, and
`audio.streaming_continuity.n_evaluated` is 4. Request success alone does not
establish valid audio output. These four requests are a functional check and
warmup, not a performance result.

### Sweep concurrency

After the functional check, run the same sampled workload at different
concurrency levels:

```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
for concurrency in 1 2 4 8; do
  python benchmarks/sharegpt/H200/sharegpt_driver.py \
    --dataset data/ShareGPT_Chinese_English_90k.json \
    --url http://127.0.0.1:8666/v1/chat/completions \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --mode audio --arrival-process closed_loop \
    --num-requests 64 --concurrency "$concurrency" \
    --max-tokens 128 --max-prompt-chars 500 \
    --seed 42 --policy none \
    --result-dir "results/${RUN_ID}/audio_c${concurrency}"
done
```

Use `--mode text` for text-only inference. `--max-tokens` bounds Thinker
output; Talker and Code2Wav retain their own output budgets. Use the same
dataset, prompt limits, output budgets, and random seed when comparing runs.
This exploratory sweep shares the server's prefix cache across runs. For a
controlled comparison, restart the server and apply the same warmup workload
before measuring each concurrency level.

Each result directory contains:

| File | Contents |
|---|---|
| `meta.json` | Client configuration |
| `partial_per_request.jsonl` | Results appended as requests finish |
| `per_request.jsonl` | Final per-request records, sorted by request index |
| `summary.json` | Aggregate latency, throughput, audio, and error statistics |

The main metrics are `totals.throughput_rps`, `latency.ttft_ms`,
`latency.audio_ttfp_ms`, and `audio.streaming_continuity`. First-audio latency
is measured at the first non-empty audio SSE payload received by the client;
it is not a playback timestamp. The benchmark decodes audio to measure its
duration but does not save WAV recordings.

Repeat measurements before drawing performance conclusions. Record the Git
commit, model revision, dataset checksum, deployment YAML, environment
variables, and `python -m pip freeze` output alongside results. Include errors,
actual output lengths, and prefetch's own work in comparisons. The default
LiveServe configuration includes runtime changes; for an upstream baseline,
use unmodified vLLM-Omni with matching model, hardware, and workload settings.

## Configure scheduling and caching

Deployment YAML controls GPU placement, replicas, chunk sizes, prefix caching,
and CPU offload. The four-GPU example already enables prefix caching and CPU
KV offload with sidecar features. Start with its default scheduling behavior,
then change one policy at a time and rerun the same workload.

Set optional controls in the server's environment **before starting it**:

| Setting | Purpose | Default |
|---|---|---|
| `OMNI_TALKER_LEAD_MAX` | Limit estimated Talker lead in codec frames | Disabled |
| `OMNI_THINKER_LEAD_MAX` | Limit estimated Thinker lead in output tokens | Disabled |
| `OMNI_BP_FIRST_CHUNK_PROTECT_MS` | Protect requests awaiting their first codec chunk | `0` |
| `OMNI_PREFETCH_ADMISSION` | Apply admission control to session prefetch requests | `0` |
| `OMNI_PREFETCH_REPLICA_AFFINITY` | Prefer replicas associated with completed prefetches | `0` |
| `OMNI_KV_EVICT_SHADOW` | Observe alternative eviction choices | `0` |
| `OMNI_KV_EVICT_ENFORCE_POLICY` | Select a custom eviction policy | Disabled |

Lead is estimated from generated progress and elapsed time, rather than
measured client playback. First-chunk protection also bounds the deferral
period. Scheduling policies are experimental and remain disabled by default;
measure both latency and throughput before adopting a policy for your workload.
Prefetch can reduce foreground work while increasing total work, so account
for both phases when evaluating it.

Session prefetch uses `POST /v1/omni/prefetch_session`. A completed prefetch
can guide later routing, but it does not guarantee that its KV blocks remain
resident. See the [prefetch API implementation](vllm-omni/vllm_omni/entrypoints/openai/api_server.py)
and [replica routing](vllm-omni/vllm_omni/engine/stage_pool.py) for the request
fields and completion rules.

## Other workloads

| Workload | Entry point |
|---|---|
| Multi-turn interaction and interruption | [trace_replay_driver.py](benchmarks/interactive/H200/trace_replay_driver.py) |
| Streaming video | [streamingbench_driver.py](benchmarks/streamingbench/streamingbench_driver.py) |
| Ming text/audio | [ming_stream_benchmark.py](benchmarks/ming/H200/ming_stream_benchmark.py) |
| H200 deployment configurations | [deploy/H200/](deploy/H200/) |

Use each client's `--help` to inspect its dataset format and service arguments.
These workloads require their own models and input data. Hardware and workload
settings should be kept separate when comparing them with the Qwen quickstart.

## Development

Run the CPU regression suite with Python 3.11 or 3.12 and Bash:

```bash
python scripts/validate_cpu_tests.py
```

The 121 tests cover launcher behavior, scheduling, prefetch routing and
completion, cancellation, tracing, and model input-processing helpers. They
require no Python packages beyond the standard library. GitHub Actions runs
this suite on both Python versions. GPU serving and audio-quality validation
require a running model and representative workloads.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines, and use
[GitHub Issues](https://github.com/xiangyuzhi/LiveServe-artifacts/issues) for bug
reports. Include the command, deployment configuration, package versions,
and relevant logs when reporting a problem.

## License

LiveServe is distributed under the [MIT License](LICENSE). The vLLM-Omni
runtime retains its [Apache-2.0 license](vllm-omni/LICENSE) and upstream copyright
notices. Models and datasets are subject to their respective licenses.
