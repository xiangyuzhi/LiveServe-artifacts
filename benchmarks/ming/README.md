# Ming Benchmarks

Ming-flash-omni uses a two-stage `sampling_params_list`, so it has dedicated
benchmark clients instead of reusing the Qwen three-stage audio drivers.

Current H200 entrypoints:

| path | purpose |
|---|---|
| [`H200/ming_stream_benchmark.py`](H200/ming_stream_benchmark.py) | Latency and throughput smoke/comparison client. Records first text, first audio, E2E, text length, audio chunks, and audio bytes. |
| [`H200/ming_quality_probe.py`](H200/ming_quality_probe.py) | Small quality probe for streamed text/audio corruption signals before using async-chunk results. |
| [`H200/ming_cbound_driver.py`](H200/ming_cbound_driver.py) | Qwen-style c-bound adapter for Ming `sharegpt`, `interactive`, and `mix` workload lanes. |

Manual stream/quality clients and the integrated c-bound validation runner now
default to the 4GPU TP2/talker-DP2 path on port 8098.  Override the URL
explicitly only when testing a manual compatibility launcher on another port:

```text
LIVESERVE_MING_URL=http://127.0.0.1:8098/v1/chat/completions
LIVESERVE_MODEL=/path/to/Jonathan1909-Ming-flash-omni-2.0
```

Run static validation without GPUs:

```bash
bash scripts/validate_ming_no_gpu.sh
```

Deployment and reproduction details are documented in
[`../../docs/experiments/ming_tp_reproduction.md`](../../docs/experiments/ming_tp_reproduction.md).
