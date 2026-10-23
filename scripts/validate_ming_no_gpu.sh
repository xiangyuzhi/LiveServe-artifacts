#!/usr/bin/env bash
# Run no-GPU validation for Ming launchers, deploy configs, and benchmark clients.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

source "${ROOT}/scripts/env_defaults.sh"
PYBIN="$(liveserve_resolve_pybin "${ROOT}")"
liveserve_require_command "${PYBIN}" "python"
liveserve_export_pycache_prefix

run() {
  printf '\n[validate-ming-no-gpu] +'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

for script in \
  scripts/env_defaults.sh \
  scripts/H200/start_ming_4gpu_dp2_baseline.sh \
  scripts/H200/start_ming_4gpu_dp2_async.sh \
  scripts/H200/start_ming_tp4_4gpu_baseline.sh \
  scripts/H200/start_ming_tp4_4gpu_async.sh \
  scripts/validate_ming_integration_no_gpu.sh \
  profiling/scheduler_analysis/scripts/gpu_preflight_common.sh \
  profiling/scheduler_analysis/scripts/run_ming_cbound_smoke.sh \
  profiling/scheduler_analysis/scripts/run_ming_4gpu_tuning_sweep.sh; do
  run bash -n "${script}"
done

run "${PYBIN}" scripts/validate_ming_runner_wiring.py

run "${PYBIN}" -m py_compile \
  benchmarks/ming/H200/ming_stream_benchmark.py \
  benchmarks/ming/H200/ming_quality_probe.py \
  benchmarks/ming/H200/ming_cbound_driver.py \
  profiling/scheduler_analysis/scripts/summarize_ming_4gpu_tuning.py \
  scripts/validate_ming_runner_wiring.py

run "${PYBIN}" - <<'PY'
from pathlib import Path

root = Path.cwd()

public_roots = [
    root / "README.md",
    root / "docs",
    root / "benchmarks" / "ming",
    root / "deploy" / "H200",
    root / "scripts" / "H200",
]
stale_needles = [
    "ming_async_" + "b" + "pv3",
    "BP" + "v3",
    "BP" + "-v3",
    "BP" + "-style",
]
hits: list[str] = []
for base in public_roots:
    files = [base] if base.is_file() else sorted(base.rglob("*"))
    for path in files:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for needle in stale_needles:
            if needle in text:
                hits.append(f"{path.relative_to(root)}: {needle}")
assert not hits, "stale Ming public naming: " + "; ".join(hits[:20])
print("PASS Ming public naming avoids stale async/BP labels")
PY

run "${PYBIN}" - <<'PY'
from pathlib import Path

root = Path.cwd()
cases = [
    ("scripts/H200/start_ming_4gpu_dp2_baseline.sh", "4,5,6,7", "ming_flash_omni_4gpu_tp2_talker_dp2.yaml", False),
    ("scripts/H200/start_ming_4gpu_dp2_async.sh", "4,5,6,7", "ming_flash_omni_4gpu_tp2_talker_dp2_async.yaml", True),
    ("scripts/H200/start_ming_tp4_4gpu_baseline.sh", "4,5,6,7", "ming_flash_omni_4gpu.yaml", False),
    ("scripts/H200/start_ming_tp4_4gpu_async.sh", "4,5,6,7", "ming_flash_omni_4gpu_async.yaml", True),
]

for rel, cuda_default, deploy_name, is_async in cases:
    text = (root / rel).read_text(encoding="utf-8")
    assert f'CUDA_DEVS="${{CUDA_VISIBLE_DEVICES:-{cuda_default}}}"' in text, rel
    assert deploy_name in text, rel
    assert "compgen -e OMNI_" in text, rel
    assert "compgen -e SCHEDULER_POLICY_" in text, rel
    assert 'export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"' in text, rel
    assert 'INIT_TIMEOUT="${LIVESERVE_INIT_TIMEOUT:-3000}"' in text, rel
    assert 'STAGE_INIT_TIMEOUT="${LIVESERVE_STAGE_INIT_TIMEOUT:-1800}"' in text, rel
    assert '--init-timeout "${INIT_TIMEOUT}"' in text, rel
    assert '--stage-init-timeout "${STAGE_INIT_TIMEOUT}"' in text, rel
    assert "MING_CUDA_WHEEL_DIR" in text, rel
    assert "MING_NVRTC_LINK_DIR" in text, rel
    assert "libnvrtc.so" in text, rel
    if is_async:
        assert 'CHUNK_STEPS="${OMNI_MING_INCREMENTAL_CHUNK_STEPS:-4}"' in text, rel
        assert 'L_HI="${OMNI_TALKER_LEAD_MAX:-12}"' in text, rel
        assert 'L_LO="${OMNI_TALKER_LEAD_LO:-6}"' in text, rel
        assert "export OMNI_MING_INCREMENTAL=1" in text, rel
        assert "export OMNI_TALKER_STAGE_IDS=1" in text, rel
        assert "export OMNI_THINKER_STAGE_IDS=0" in text, rel
    else:
        assert "export OMNI_MING_INCREMENTAL=1" not in text, rel
        assert "OMNI_*/SCHEDULER_POLICY_*   = cleared" in text, rel
print(f"PASS Ming launcher defaults: {len(cases)} checked")
PY

run "${PYBIN}" - <<'PY'
from pathlib import Path

import yaml

root = Path.cwd()
cases = [
    ("deploy/H200/ming_flash_omni_4gpu.yaml", False, 4, "0,1,2,3", "3"),
    ("deploy/H200/ming_flash_omni_4gpu_async.yaml", True, 4, "0,1,2,3", "3"),
    ("deploy/H200/ming_flash_omni_4gpu_tp2_talker_dp2.yaml", False, 2, "0,1", "2,3"),
    ("deploy/H200/ming_flash_omni_4gpu_tp2_talker_dp2_async.yaml", True, 2, "0,1", "2,3"),
]

for rel, async_chunk, tp, thinker_devices, talker_devices in cases:
    path = root / rel
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["pipeline"] == "ming_flash_omni", rel
    assert bool(data.get("async_chunk")) is async_chunk, rel
    stages = data.get("stages") or []
    assert len(stages) == 2, rel
    assert stages[0].get("stage_id") == 0, rel
    assert stages[1].get("stage_id") == 1, rel
    assert int(stages[0].get("tensor_parallel_size")) == tp, rel
    assert str(stages[0].get("devices")) == thinker_devices, rel
    assert str(stages[1].get("devices")) == talker_devices, rel
    if talker_devices == "2,3":
        assert int(stages[1].get("num_replicas")) == 2, rel
    if async_chunk:
        assert data.get("connectors"), rel
        assert stages[0].get("output_connectors"), rel
        assert stages[1].get("input_connectors"), rel
    else:
        assert not data.get("connectors"), rel
print(f"PASS Ming deploy configs: {len(cases)} checked")
PY

run "${PYBIN}" - <<'PY'
import argparse
from pathlib import Path
import importlib.util
import sys

root = Path.cwd()
stream_path = root / "benchmarks/ming/H200/ming_stream_benchmark.py"
quality_path = root / "benchmarks/ming/H200/ming_quality_probe.py"
cbound_path = root / "benchmarks/ming/H200/ming_cbound_driver.py"

def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

stream = load(stream_path, "_ming_stream_benchmark_validate")
quality = load(quality_path, "_ming_quality_probe_validate")
cbound = load(cbound_path, "_ming_cbound_driver_validate")
assert stream.DEFAULT_URL == quality.DEFAULT_URL == cbound.DEFAULT_URL == "http://127.0.0.1:8098/v1/chat/completions"
assert stream.DEFAULT_MODEL == quality.DEFAULT_MODEL == cbound.DEFAULT_MODEL == "/workspace/models/Jonathan1909-Ming-flash-omni-2.0"
payload = stream.build_payload(
    "请介绍杭州西湖。",
    argparse.Namespace(
        model=stream.DEFAULT_MODEL,
        temperature=0.4,
        top_p=0.9,
        max_tokens=32,
    ),
)
assert payload["modalities"] == ["text", "audio"]
assert len(payload["sampling_params_list"]) == 2
assert payload["sampling_params_list"][0]["detokenize"] is True
assert payload["sampling_params_list"][1]["max_tokens"] == 1
assert cbound.DEFAULT_TRACE.name == "total_traces_part1.txt"
assert cbound.DEFAULT_SHAREGPT_DATASET.name == "ShareGPT_Chinese_English_90k.json"
print("PASS Ming benchmark defaults and payload shape")
PY

run git diff --check -- \
  benchmarks/ming/H200 \
  deploy/H200/ming_flash_omni_4gpu.yaml \
  deploy/H200/ming_flash_omni_4gpu_async.yaml \
  deploy/H200/ming_flash_omni_4gpu_tp2_talker_dp2.yaml \
  deploy/H200/ming_flash_omni_4gpu_tp2_talker_dp2_async.yaml \
  profiling/scheduler_analysis/scripts/run_ming_cbound_smoke.sh \
  profiling/scheduler_analysis/scripts/run_ming_4gpu_tuning_sweep.sh \
  profiling/scheduler_analysis/scripts/summarize_ming_4gpu_tuning.py \
  scripts/H200/start_ming_4gpu_dp2_baseline.sh \
  scripts/H200/start_ming_4gpu_dp2_async.sh \
  scripts/H200/start_ming_tp4_4gpu_baseline.sh \
  scripts/H200/start_ming_tp4_4gpu_async.sh \
  scripts/validate_ming_integration_no_gpu.sh \
  scripts/validate_ming_runner_wiring.py \
  scripts/validate_ming_no_gpu.sh

echo
echo "[validate-ming-no-gpu] PASS"
