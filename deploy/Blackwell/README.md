# Qwen3-Omni on four Blackwell GPUs

[`qwen3_omni_4gpu_smoke.yaml`](qwen3_omni_4gpu_smoke.yaml) is derived from the
20 October 2026 functional smoke on four RTX PRO 6000 Blackwell Server 96 GB
GPUs. It allocates two Thinker replicas to devices 0–1 and two Talker/codec
replicas to devices 2–3. The bounded diagnostic owner requires at least 90 GiB
free on each GPU and 96 GiB available host memory before launch.

The stage-0 offload configuration budgets 16 GiB of CPU KV storage and 2 GiB
of sidecar storage per configured connector, with 4,096 GPU blocks. These are
hardware-specific experimental settings, not a capacity recommendation for
other GPUs. Model weights and compatible CUDA packages must be supplied.

All deployment values match the measured input except removal of the
machine-local `omni_sidecar_audit_path`. For a mechanism audit, add a writable
absolute output path under that stage's `kv_connector_extra_config` and seal
the resulting YAML. The original measured YAML SHA256 is
`8f5d986a97b670b46a835622104556b27f2a531a6aacc93cde6cad74d064574b`.
Changing the audit destination does not establish a new measured result.

The [diagnostic guide](../../profiling/interaction_optimization/README.md)
describes explicit model/runtime/lock arguments, phase inputs and bounded
cleanup. Supply complete original requests for a reproduction; this example
does not distribute historical workloads or raw evidence. Feature overrides
are separate from deployment. First-codec scheduling and GPU pinning remain
disabled by default; do not enable all experimental options together.

See the [validation report](../../docs/experiments/release_validation_20261020.md)
for the smoke's exact code revisions and limits. It established normal audio,
actual live cancellation and successful prefetch, not a performance gain.
