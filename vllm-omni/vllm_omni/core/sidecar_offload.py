"""Identity-safe Qwen sidecar mirror for vLLM 0.20 CPU KV offload.

Preserves the existing KV admission, lookup and eviction algorithms. Supports
one full-attention group and a 1:1 CPU/GPU block ratio, failing closed otherwise.
The extra CPU tensor storage is bounded independently from the KV budget.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.reuse_manager import FilterReusedOffloadingManager
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

from vllm_omni.data_entry_keys import flatten_payload

logger = init_logger(__name__)
HIDDEN = "__hidden_states__"
REQUIRED_MM = {"hidden_states.layer_0", "hidden_states.layer_24"}


class SidecarContractError(RuntimeError):
    pass


class IdentityCPUOffloadingManager(CPUOffloadingManager):
    """Keep upstream allocation/eviction; transport its full content keys."""

    def prepare_load(self, keys, req_context):
        return super().prepare_load(list(keys), req_context)

    def _get_load_store_spec(self, keys, blocks):
        keys = list(keys)
        spec = super()._get_load_store_spec(keys, blocks)
        # Full OffloadKey, produced by vLLM from the chained token/block hash,
        # including multimodal, LoRA, and cache-salt identity. Never request ID.
        spec.omni_sidecar_keys = tuple(bytes(key) for key in keys)
        return spec


@dataclass
class Snapshot:
    key: bytes
    generation: int
    tensors: dict[str, torch.Tensor]
    checksums: dict[str, str]


class SidecarPool:
    """Bounded CPU-slot snapshots with generation and in-flight ownership.

    KV allocation remains authoritative. Stores snapshot after the preceding
    model step and commit only with successful KV DMA. Loads copy and validate
    all destinations before the delegate can report completion to the scheduler.
    """

    def __init__(self, cache, num_cpu_blocks, audit_path=None, request_for_job=None, byte_limit=2**31):
        if num_cpu_blocks <= 0:
            raise SidecarContractError("CPU offload has no slots")
        self.cache = cache
        self.capacity = num_cpu_blocks
        self.audit_path = audit_path
        self.request_for_job = request_for_job or (lambda job: None)
        self.generations = {}
        self.committed = {}
        self.valid = {}
        self.reserved = {}
        self.loading = {}
        self.committed_bytes = 0
        self.pending_bytes = 0
        self.byte_limit = int(byte_limit)
        self.lock = threading.RLock()
        self.peak_bytes = 0
        original = cache.update_omni_tensor_prefix_cache
        signature = inspect.signature(original)

        def update(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            result = original(*args, **kwargs)
            self.mark_updated(**bound.arguments)
            return result

        cache.update_omni_tensor_prefix_cache = update

    def tensors(self):
        return {HIDDEN: self.cache.hidden_states_cache, **self.cache.mm_outputs_cache}

    def mark_updated(
        self, hidden_states, multimodal_outputs, num_tokens_unpadded, slot_mapping, num_tokens_padded=None
    ):
        tensors = self.tensors()
        flat = flatten_payload(multimodal_outputs or {})
        if hidden_states is not None:
            flat[HIDDEN] = hidden_states
        padded = num_tokens_unpadded if num_tokens_padded is None else num_tokens_padded
        slots = slot_mapping[:num_tokens_unpadded].to(device="cpu", dtype=torch.long)
        with self.lock:
            for key, tensor in tensors.items():
                if key not in self.valid:
                    self.valid[key] = torch.zeros(tensor.shape[:2], dtype=torch.bool)
                valid = self.valid[key]
                good = isinstance(flat.get(key), torch.Tensor) and (key == HIDDEN or flat[key].shape[0] == padded)
                valid.view(-1)[slots] = good

    @staticmethod
    def digest(tensor):
        return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    @staticmethod
    def snapshot_bytes(snapshot):
        return sum(t.numel() * t.element_size() for t in snapshot.tensors.values())

    def emit(self, event, job, rows, success=True, transfer_result=None):
        if not self.audit_path:
            return
        path = Path(self.audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(
            event=event,
            job_id=job,
            request_id=self.request_for_job(job),
            pid=os.getpid(),
            success=success,
            wall_time=time.time(),
            direction=["GPU", "CPU"] if event.startswith("store") else ["CPU", "GPU"],
            blocks=[self.describe(*row) for row in rows],
            committed_slots=len(self.committed),
            capacity_slots=self.capacity,
            committed_bytes=self.committed_bytes,
            pending_bytes=self.pending_bytes,
            peak_sidecar_bytes=self.peak_bytes,
            byte_limit=self.byte_limit,
        )
        if event == "load_restore":
            data["destination_checksums_verified"] = True
        if transfer_result is not None:
            data.update(
                transfer_size=transfer_result.transfer_size,
                transfer_time=transfer_result.transfer_time,
                transfer_type=transfer_result.transfer_type,
            )
        raw = (json.dumps(data, separators=(",", ":")) + "\n").encode()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, raw)
        finally:
            os.close(fd)

    def mapping(self, spec, store):
        gpu, cpu = spec if store else (spec[1], spec[0])
        if not isinstance(gpu, GPULoadStoreSpec) or not isinstance(cpu, CPULoadStoreSpec):
            raise SidecarContractError("Expected CPU/GPU transfer spec")
        ids = [int(x) for x in cpu.block_ids]
        gids = [int(x) for x in gpu.block_ids]
        keys = getattr(cpu, "omni_sidecar_keys", None)
        if keys is None or len(keys) != len(ids) or len(ids) != len(gids):
            raise SidecarContractError("Missing or mismatched offload identities")
        if len(gpu.group_sizes) != 1 or int(gpu.group_sizes[0]) != len(gids):
            raise SidecarContractError("Only one KV group is supported")
        if len(set(ids)) != len(ids) or len(set(gids)) != len(gids):
            raise SidecarContractError("Duplicate transfer slots")
        if any(i < 0 or i >= self.capacity for i in ids):
            raise SidecarContractError("CPU slot outside configured capacity")
        if any(i < 0 or i >= self.cache.num_blocks for i in gids):
            raise SidecarContractError("GPU slot outside sidecar allocation")
        return list(zip(ids, gids, [bytes(k) for k in keys]))

    def describe(self, cpu, gpu, snapshot):
        return dict(
            cpu_slot=cpu,
            gpu_slot=gpu,
            generation=snapshot.generation,
            offload_key=snapshot.key.hex(),
            tensors={
                k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=snapshot.checksums.get(k))
                for k, v in snapshot.tensors.items()
            },
        )

    def stage_store(self, job, spec):
        with self.lock:
            mapping = self.mapping(spec, True)
            tensors = self.tensors()
            if not REQUIRED_MM.issubset(tensors):
                raise SidecarContractError("Qwen layer 0/24 sidecar not initialized")
            # Validate the whole operation before reserving any CPU slot.
            for cpu, gpu, key in mapping:
                if cpu in self.reserved or self.loading.get(cpu, 0):
                    raise SidecarContractError("CPU slot has an unfinished transfer")
                for name, tensor in tensors.items():
                    if tensor.device.type != "cpu" or tensor.ndim != 3:
                        raise SidecarContractError("Unsupported sidecar tensor layout")
                    if name not in self.valid or not bool(self.valid[name][gpu].all()):
                        raise SidecarContractError("Incomplete/stale source sidecar block: " + name)
            per_block_bytes = sum(t[0].numel() * t.element_size() for t in tensors.values())
            if per_block_bytes * self.capacity > self.byte_limit:
                raise SidecarContractError("Sidecar capacity exceeds the explicit additional host-memory budget")
            staged = []
            for cpu, gpu, key in mapping:
                generation = self.generations.get(cpu, 0) + 1
                self.generations[cpu] = generation
                self.reserved[cpu] = (job, generation)
                old = self.committed.pop(cpu, None)  # Original KV manager reassigned the slot.
                if old is not None:
                    self.committed_bytes -= self.snapshot_bytes(old)
                    del old
                copies = {name: tensor[gpu].clone() for name, tensor in tensors.items()}
                checksums = {name: self.digest(value) for name, value in copies.items()} if self.audit_path else {}
                snapshot = Snapshot(key, generation, copies, checksums)
                self.pending_bytes += self.snapshot_bytes(snapshot)
                self.peak_bytes = max(self.peak_bytes, self.committed_bytes + self.pending_bytes)
                staged.append((cpu, gpu, snapshot))
            self.emit("store_snapshot", job, staged)
            return staged

    def finish_store(self, job, staged, success, transfer_result=None):
        with self.lock:
            for cpu, gpu, snapshot in staged:
                if (
                    self.reserved.get(cpu) != (job, snapshot.generation)
                    or self.generations.get(cpu) != snapshot.generation
                ):
                    raise SidecarContractError("Stale store completion cannot overwrite a newer CPU-slot identity")
            for cpu, gpu, snapshot in staged:
                self.reserved.pop(cpu)
                self.pending_bytes -= self.snapshot_bytes(snapshot)
                if success:
                    self.committed[cpu] = snapshot
                    self.committed_bytes += self.snapshot_bytes(snapshot)
                else:
                    self.committed.pop(cpu, None)
            if len(self.committed) > self.capacity:
                raise SidecarContractError("Unbounded sidecar pool")
            self.emit("store_commit" if success else "store_failed", job, staged, success, transfer_result)

    def restore(self, job, spec):
        with self.lock:
            mapping = self.mapping(spec, False)
            targets = self.tensors()
            restore = []
            # Full-job prevalidation prevents partially writing a batch of destinations.
            for cpu, gpu, key in mapping:
                snap = self.committed.get(cpu)
                if (
                    cpu in self.reserved
                    or snap is None
                    or snap.key != key
                    or snap.generation != self.generations.get(cpu)
                ):
                    raise SidecarContractError("Missing or mismatched CPU sidecar identity")
                if set(snap.tensors) != set(targets):
                    raise SidecarContractError("Dynamic sidecar key set differs; cannot restore incomplete output")
                for name, value in snap.tensors.items():
                    if value.shape != targets[name].shape[1:] or value.dtype != targets[name].dtype:
                        raise SidecarContractError("Sidecar shape/dtype changed")
                    if snap.checksums and self.digest(value) != snap.checksums[name]:
                        raise SidecarContractError("Sidecar checksum mismatch")
                restore.append((cpu, gpu, snap))
            for cpu, gpu, snap in restore:
                for name, value in snap.tensors.items():
                    targets[name][gpu].copy_(value)
                    if self.audit_path and self.digest(targets[name][gpu]) != snap.checksums[name]:
                        raise SidecarContractError("Restored destination sidecar checksum mismatch")
                    if name not in self.valid:
                        self.valid[name] = torch.zeros(targets[name].shape[:2], dtype=torch.bool)
                    self.valid[name][gpu] = True
            for cpu, gpu, snap in restore:
                self.loading[cpu] = self.loading.get(cpu, 0) + 1
            self.emit("load_restore", job, restore)
            return restore

    def finish_load(self, job, rows, success, transfer_result=None):
        with self.lock:
            for cpu, gpu, snapshot in rows:
                if self.loading.get(cpu, 0) <= 0:
                    raise SidecarContractError("Untracked sidecar load completion")
                self.loading[cpu] -= 1
                if not self.loading[cpu]:
                    self.loading.pop(cpu)
            if success:
                self.emit("load_complete", job, rows, transfer_result=transfer_result)

    def invalidate_destinations(self, spec):
        try:
            mapping = self.mapping(spec, False)
        except SidecarContractError:
            return
        with self.lock:
            for cpu, gpu, key in mapping:
                for valid in self.valid.values():
                    valid[gpu] = False


class SidecarTransferHandler(OffloadingHandler):
    """Couple sidecar transactions to the real upstream DMA completion paths."""

    def __init__(self, delegate, pool, store):
        self.delegate = delegate
        self.pool = pool
        self.store = store
        self.pending = {}
        self.ready = []

    def transfer_async(self, job_id, spec):
        if job_id in self.pending:
            raise SidecarContractError("Duplicate active transfer job")
        try:
            rows = self.pool.stage_store(job_id, spec) if self.store else self.pool.restore(job_id, spec)
        except BaseException:
            if not self.store:
                self.pool.invalidate_destinations(spec)
            raise
        self.pending[job_id] = (spec, rows)
        try:
            success = self.delegate.transfer_async(job_id, spec)
        except BaseException:
            self._fail(job_id)
            raise
        if not success:
            self._fail(job_id)
            raise SidecarContractError("KV transfer submission failed; sidecar not committed")
        return True

    def _fail(self, job):
        spec, rows = self.pending.pop(job)
        if self.store:
            self.pool.finish_store(job, rows, False)
        else:
            self.pool.finish_load(job, rows, False)
            self.pool.invalidate_destinations(spec)

    def _drain(self):
        for result in self.delegate.get_finished():
            if result.job_id not in self.pending:
                raise SidecarContractError("Completion has no matching sidecar transaction")
            spec, rows = self.pending.pop(result.job_id)
            if self.store:
                self.pool.finish_store(result.job_id, rows, result.success, result)
            else:
                self.pool.finish_load(result.job_id, rows, result.success, result)
                if not result.success:
                    self.pool.invalidate_destinations(spec)
            if not result.success:
                raise SidecarContractError("KV transfer failed; request must not continue")
            self.ready.append(result)

    def get_finished(self):
        self._drain()
        ready = self.ready
        self.ready = []
        return ready

    def wait(self, job_ids):
        self.delegate.wait(job_ids)
        self._drain()
        if any(job in self.pending for job in job_ids):
            raise SidecarContractError("KV wait returned before sidecar transaction completed")

    def shutdown(self):
        self.delegate.shutdown()
        for job in list(self.pending):
            self._fail(job)
        self.ready.clear()


class OmniCPUOffloadingSpec(CPUOffloadingSpec):
    def __init__(self, vllm_config, kv_cache_config):
        super().__init__(vllm_config, kv_cache_config)
        groups = kv_cache_config.kv_cache_groups
        if (
            len(groups) != 1
            or type(groups[0].kv_cache_spec) is not FullAttentionSpec
            or self.block_size_factor != 1
            or vllm_config.parallel_config.world_size != 1
        ):
            raise SidecarContractError(
                "Omni sidecar offload requires TP1, one full-attention KV group and block-size factor 1"
            )
        self.sidecar_pool = None

    def get_manager(self):
        if self._manager is None:
            events = self.vllm_config.kv_events_config
            manager = IdentityCPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,
                enable_events=events is not None and events.enable_kv_cache_events,
            )
            threshold = int(self.extra_config.get("store_threshold", 0))
            if threshold >= 2:
                manager = FilterReusedOffloadingManager(
                    backing=manager,
                    store_threshold=threshold,
                    max_tracker_size=int(self.extra_config.get("max_tracker_size", 64000)),
                )
            self._manager = manager
        return self._manager

    def bind_sidecar(self, cache, worker):
        if self.sidecar_pool is not None:
            raise SidecarContractError("Sidecar already bound")
        limit = int(self.extra_config.get("omni_sidecar_cpu_bytes_limit", 2**31))
        minimum_bytes = (
            self.num_blocks * cache.hidden_states_cache[0].numel() * cache.hidden_states_cache.element_size() * 3
        )
        if minimum_bytes > limit:
            raise SidecarContractError("Required Qwen hidden/layer0/layer24 sidecar exceeds its host-memory budget")
        self.sidecar_pool = SidecarPool(
            cache,
            self.num_blocks,
            self.extra_config.get("omni_sidecar_audit_path"),
            request_for_job=lambda job: worker._jobs.get(job, (None, None))[0],
            byte_limit=limit,
        )
        logger.info(
            "[OmniSidecarOffload] bound spec=OmniCPUOffloadingSpec module=vllm_omni.core.sidecar_offload "
            "cpu_slots=%d block_size=%d byte_limit=%d required_mm=layer_0,layer_24",
            self.num_blocks,
            cache.block_size,
            self.sidecar_pool.byte_limit,
        )

    def get_handlers(self, kv_caches):
        if self.sidecar_pool is None:
            raise SidecarContractError("Sidecar must bind before registering KV handlers")
        for src, dst, handler in super().get_handlers(kv_caches):
            yield src, dst, SidecarTransferHandler(handler, self.sidecar_pool, store=src.medium() == "GPU")


def bind_sidecar_to_offloading_connector(cache):
    from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group

    if not has_kv_transfer_group():
        return
    connector = get_kv_transfer_group()
    worker = getattr(connector, "connector_worker", None)
    spec = getattr(worker, "spec", None)
    if isinstance(spec, OmniCPUOffloadingSpec):
        spec.bind_sidecar(cache, worker)
