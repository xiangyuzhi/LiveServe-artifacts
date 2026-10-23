# LiveServe artifacts

A source snapshot of LiveServe, a research serving stack for interactive
multimodal models. It contains interaction-aware scheduling, session KV
prefetch and replica routing, KV eviction, model integration fixes, benchmark
clients, and deployment examples. Qwen3-Omni is the primary evaluated path;
Ming and older H200 workflows are also included.

This is a **single repository**: `vllm-omni/` contains the patched runtime as
ordinary source files, not a Git submodule. The original top-level `docs/`
and `profiling/` trees, experiment outputs, model weights, and datasets are
not included. Some historical wrappers still need those omitted research
files or machine-specific inputs; their presence is not a claim that every
workflow runs from this snapshot.

## Get the source and run CPU checks

```bash
git clone https://github.com/xiangyuzhi/LiveServe-artifacts.git
cd LiveServe-artifacts
python scripts/validate_cpu_tests.py
```

Use Python **3.11 or 3.12** and Bash. The selected suite contains **121 tests**
(10 repository tests and 111 runtime tests) and requires only the Python
standard library: no package installation, torch, model download, CUDA, or
GPU service. It is a focused CPU regression suite, not the full upstream
runtime test suite. It does not establish model quality or GPU performance.

## Install for GPU serving

Prepare a **Linux** environment with **vLLM 0.20.0** and a compatible
PyTorch/CUDA/driver combination first. The historical Qwen environment used
Python 3.12.3 and torch 2.11.0+cu130 on four RTX PRO 6000 Blackwell Server
96 GB GPUs; this records that environment, not a compatibility guarantee for
other hardware. H200 examples use different placements and memory budgets.

Then install the included runtime in that environment:

```bash
VLLM_OMNI_TARGET_DEVICE=cuda \
VLLM_OMNI_VERSION_OVERRIDE=0.0.dev20261023+liveserve \
  python -m pip install -e ./vllm-omni
vllm-omni --help
```

The explicit version is a package label for this source snapshot, not an
upstream release claim. It avoids deriving the runtime version from the new
repository's unrelated Git history. `VLLM_OMNI_TARGET_DEVICE` selects the
runtime's platform dependencies. These commands can download large packages;
they do not provide model weights. The historical `scripts/setup_env.sh`
helper is not a complete vLLM/PyTorch/CUDA installer.

The serving entry point is `vllm-omni serve`. For example, after preparing
sufficient resources for the supplied four-GPU Blackwell configuration:

```bash
vllm-omni serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --deploy-config deploy/Blackwell/qwen3_omni_4gpu_smoke.yaml \
  --host 127.0.0.1 --port 8666
```

Select the model path, visible GPUs, deployment file, and port explicitly for
your machine. Deployment examples are hardware-specific. Experimental
scheduler and routing switches are opt-in; enabling all of them together is
not a recommended configuration. No new GPU experiment was performed for
this artifact publication.

## Source layout and entry points

| Location | Purpose |
|---|---|
| [vllm-omni/](vllm-omni/) | Patched runtime, upstream source, and runtime tests |
| [deploy/Blackwell/](deploy/Blackwell/) and [deploy/H200/](deploy/H200/) | Hardware-specific deployment examples |
| [ShareGPT client](benchmarks/sharegpt/H200/sharegpt_driver.py) | Text/audio workload requests; specify dataset, URL, model, and result directory |
| [Interactive replay](benchmarks/interactive/H200/trace_replay_driver.py) | Multi-turn replay and interruption workflows |
| [StreamingBench client](benchmarks/streamingbench/streamingbench_driver.py) | Video workflows requiring explicit local data and service inputs |
| [Ming client](benchmarks/ming/H200/ming_stream_benchmark.py) | Ming-specific streamed text/audio requests |
| [scripts/](scripts/) | CPU validation, dataset preparation, and legacy launch helpers |

Consult each client's `--help` after installing its dependencies. Benchmark
clients have additional dependencies beyond the standard-library CPU suite.
Historical aggregate validation and reproduction wrappers may refer to omitted
`docs/`, `profiling/`, or external result trees; use the CPU command above for
this artifact's selected checks.

## Provenance and interpretation

The source is derived from LiveServe parent revision
`f5dcf1540b994dfff647812001b0851c3d777824` and runtime revision
`3440334b6b02e3a63de2fbddbae5d97a1dbd3084`, with the 2026-10-21 code
simplification and publication packaging changes. Those revisions describe
the source bases, not an assertion that this snapshot is byte-identical to an
older measured run.

Historical first-codec protection/re-entry candidates did not pass the joint
latency/throughput screen. Prefetch can move work out of the foreground while
increasing total work; comparisons must include prefetch's own cost. API
success does not prove cache residency, and cancellation return does not
prove GPU drain. This trimmed source snapshot does not contain the complete
historical evidence needed to reproduce those conclusions.

## License

LiveServe's outer code is distributed under the [MIT License](LICENSE).
The included runtime retains its [Apache-2.0 license](vllm-omni/LICENSE) and
upstream copyright notices. Model weights and external datasets have their
own terms and are not distributed here. See [CONTRIBUTING.md](CONTRIBUTING.md)
for contribution and validation expectations.
