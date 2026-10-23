#!/usr/bin/env python3
"""CPU-only regression suite; actual offload worker/manager, no CUDA handlers."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import OffloadingConnectorWorker
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.kv_offload.abstract import ReqContext, make_offload_key
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.worker.worker import OffloadingHandler, TransferResult

from vllm_omni.core.prefix_cache import OmniTensorPrefixCache
from vllm_omni.core.sidecar_offload import (
    HIDDEN,
    IdentityCPUOffloadingManager,
    OmniCPUOffloadingSpec,
    SidecarContractError,
    SidecarPool,
    SidecarTransferHandler,
)
from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
from vllm_omni.request import OmniRequest


class FakeDMA(OffloadingHandler):
    def __init__(self):
        self.active = {}
        self.completed = []
        self.submit_ok = True

    def transfer_async(self, job_id, spec):
        self.active[job_id] = spec
        return self.submit_ok

    def complete(self, job, success=True):
        self.active.pop(job)
        self.completed.append(TransferResult(job, success))

    def get_finished(self):
        result = self.completed
        self.completed = []
        return result

    def wait(self, ids):
        for job in ids:
            if job in self.active:
                self.complete(job)

    def shutdown(self):
        self.active.clear()


def spec(cpu, gpu, keys, store=True):
    c = CPULoadStoreSpec(cpu)
    c.omni_sidecar_keys = tuple(keys)
    g = GPULoadStoreSpec(gpu, [len(gpu)], [0])
    return (g, c) if store else (c, g)


class SidecarTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache = OmniTensorPrefixCache(num_blocks=8, block_size=16, hidden_size=2, hs_dtype=torch.bfloat16)
        self.pool = SidecarPool(self.cache, 4, self.root / "audit.jsonl")
        self.sd = FakeDMA()
        self.ld = FakeDMA()
        self.store = SidecarTransferHandler(self.sd, self.pool, True)
        self.load = SidecarTransferHandler(self.ld, self.pool, False)
        self.write([1, 2], 1)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, blocks, value, extra=False, omit24=False):
        n = len(blocks) * 16
        v = torch.full((n, 2), value, dtype=torch.bfloat16)
        layers = {"0": v.clone()}
        if not omit24:
            layers["24"] = v.clone()
        mm = {"hidden_states": {"layers": layers}}
        if extra:
            mm["extra"] = v.clone()
        slots = torch.tensor([b * 16 + i for b in blocks for i in range(16)])
        self.cache.update_omni_tensor_prefix_cache(v, mm, n, slots, n)

    def commit(self, job=1, cpu=[0, 1], gpu=[1, 2], keys=[b"A0", b"A1"]):
        self.store.transfer_async(job, spec(cpu, gpu, keys))
        self.sd.complete(job)
        self.store.get_finished()

    def assert_values(self, block, value):
        for tensor in self.pool.tensors().values():
            self.assertTrue(torch.all(tensor[block] == value))

    def test_same_and_different_reused_gpu_blocks(self):
        self.commit()
        self.write([1, 2, 5, 6], 9)
        for job, gpu in [(2, [1, 2]), (3, [5, 6])]:
            self.load.transfer_async(job, spec([0, 1], gpu, [b"A0", b"A1"], False))
            for block in gpu:
                self.assert_values(block, 1)
            self.ld.complete(job)
            self.assertEqual([r.job_id for r in self.load.get_finished()], [job])

    def test_partial_mixed_local_and_external_preserves_local(self):
        self.commit()
        self.write([4, 5, 6], 8)
        self.load.transfer_async(2, spec([1], [5], [b"A1"], False))
        self.assert_values(4, 8)
        self.assert_values(5, 1)
        self.assert_values(6, 8)

    def test_whole_load_validates_before_any_destination_write(self):
        self.commit()
        self.write([5, 6], 8)
        with self.assertRaises(SidecarContractError):
            self.load.transfer_async(2, spec([0, 1], [5, 6], [b"A0", b"WRONG"], False))
        self.assert_values(5, 8)
        self.assert_values(6, 8)
        self.assertFalse(self.ld.active)
        self.assertFalse(self.pool.valid[HIDDEN][5].any())

    def test_failed_store_never_commits_and_retry_new_identity(self):
        self.store.transfer_async(1, spec([0], [1], [b"A"]))
        self.sd.complete(1, False)
        with self.assertRaises(SidecarContractError):
            self.store.get_finished()
        self.assertNotIn(0, self.pool.committed)
        self.commit(2, [0], [2], [b"B"])
        self.assertEqual(self.pool.committed[0].key, b"B")

    def test_cpu_slot_reuse_and_late_completion(self):
        old = self.pool.stage_store(1, spec([0], [1], [b"A"]))
        with self.assertRaises(SidecarContractError):
            self.pool.stage_store(2, spec([0], [2], [b"B"]))
        self.pool.finish_store(1, old, False)
        new = self.pool.stage_store(2, spec([0], [2], [b"B"]))
        self.pool.finish_store(2, new, True)
        with self.assertRaises(SidecarContractError):
            self.pool.finish_store(1, old, True)
        self.assertEqual(self.pool.committed[0].key, b"B")
        with self.assertRaises(SidecarContractError):
            self.load.transfer_async(3, spec([0], [6], [b"A"], False))

    def test_dynamic_keys_dtype_and_missing_source_rejected(self):
        self.commit()
        self.write([3], 2, extra=True)
        with self.assertRaises(SidecarContractError):
            self.load.transfer_async(2, spec([0], [5], [b"A0"], False))
        self.write([1], 1, extra=True)
        self.commit(3, [0], [1], [b"NEW"])
        self.write([5], 9, extra=True)
        self.load.transfer_async(4, spec([0], [5], [b"NEW"], False))
        self.assert_values(5, 1)
        self.cache.mm_outputs_cache["extra"] = self.cache.mm_outputs_cache["extra"].float()
        with self.assertRaises(SidecarContractError):
            self.load.transfer_async(5, spec([0], [6], [b"NEW"], False))
        self.write([1], 7, omit24=True)
        with self.assertRaises(SidecarContractError):
            self.store.transfer_async(6, spec([1], [1], [b"MISSING"]))

    def test_checksum_corruption_and_capacity_rejected(self):
        self.commit()
        self.pool.committed[0].tensors[HIDDEN][0, 0] = 42
        with self.assertRaises(SidecarContractError):
            self.load.transfer_async(2, spec([0], [5], [b"A0"], False))
        with self.assertRaises(SidecarContractError):
            self.store.transfer_async(3, spec([4], [1], [b"OUT"]))

    def test_load_submission_and_completion_fail_closed(self):
        self.commit()
        self.ld.submit_ok = False
        with self.assertRaises(SidecarContractError):
            self.load.transfer_async(2, spec([0], [5], [b"A0"], False))
        self.assertFalse(self.pool.valid[HIDDEN][5].any())
        self.ld.submit_ok = True
        self.load.transfer_async(3, spec([0], [6], [b"A0"], False))
        self.ld.complete(3, False)
        with self.assertRaises(SidecarContractError):
            self.load.get_finished()
        self.assertFalse(self.pool.valid[HIDDEN][6].any())

    def test_wait_commits_before_return_and_preserves_completion(self):
        self.store.transfer_async(1, spec([0], [1], [b"A"]))
        self.store.wait({1})
        self.assertIn(0, self.pool.committed)
        self.assertEqual([r.job_id for r in self.store.get_finished()], [1])
        self.assertEqual(self.store.get_finished(), [])

    def test_actual_worker_deferred_store_samples_after_model_update(self):
        worker = OffloadingConnectorWorker(SimpleNamespace())
        worker.worker.register_handler(GPULoadStoreSpec, CPULoadStoreSpec, self.store)
        worker.worker.register_handler(CPULoadStoreSpec, GPULoadStoreSpec, self.load)
        self.pool.request_for_job = lambda job: worker._jobs[job][0]
        worker.prepare_store_kv(OffloadingConnectorMetadata({}, {"A": spec([0], [1], [b"A"])}))
        self.assertFalse(self.pool.committed)
        self.write([1], 7)
        worker.handle_preemptions(OffloadingConnectorMetadata({}, {}, {"A"}))
        self.assertTrue(torch.all(self.pool.committed[0].tensors[HIDDEN] == 7))
        worker.get_finished({"A"})
        worker.start_kv_transfers(OffloadingConnectorMetadata({"A-repeat": spec([0], [5], [b"A"], False)}, {}))
        self.assert_values(5, 7)
        self.ld.complete(1)
        self.assertEqual(worker.get_finished(set())[1], {"A-repeat"})
        events = [json.loads(line) for line in (self.root / "audit.jsonl").read_text().splitlines()]
        self.assertEqual(next(e for e in events if e["event"] == "store_commit")["request_id"], "A")
        self.assertEqual(next(e for e in events if e["event"] == "load_complete")["request_id"], "A-repeat")

    def test_shared_content_two_concurrent_load_jobs_and_cancel(self):
        self.commit()
        self.load.transfer_async(2, spec([0], [5], [b"A0"], False))
        self.load.transfer_async(3, spec([0], [6], [b"A0"], False))
        self.ld.complete(3)
        self.ld.complete(2)
        self.assertEqual({r.job_id for r in self.load.get_finished()}, {2, 3})
        self.store.transfer_async(4, spec([2], [1], [b"cancel"]))
        self.store.shutdown()
        self.assertNotIn(2, self.pool.committed)

    def test_byte_accounting_pending_replacement_failure_and_bound(self):
        block_bytes = 16 * 2 * 2 * 3
        self.store.transfer_async(1, spec([0, 1], [1, 2], [b"A0", b"A1"]))
        self.assertEqual(self.pool.pending_bytes, 2 * block_bytes)
        self.assertEqual(self.pool.peak_bytes, 2 * block_bytes)
        self.sd.complete(1)
        self.store.get_finished()
        self.assertEqual(self.pool.committed_bytes, 2 * block_bytes)
        self.assertEqual(self.pool.pending_bytes, 0)
        self.store.transfer_async(2, spec([0], [1], [b"B"]))
        self.assertEqual(self.pool.committed_bytes, block_bytes)
        self.assertEqual(self.pool.pending_bytes, block_bytes)
        self.sd.complete(2, False)
        with self.assertRaises(SidecarContractError):
            self.store.get_finished()
        self.assertEqual(self.pool.committed_bytes, block_bytes)
        self.assertEqual(self.pool.pending_bytes, 0)
        self.pool.byte_limit = block_bytes
        with self.assertRaises(SidecarContractError):
            self.store.transfer_async(3, spec([0], [1], [b"C"]))
        self.assertFalse(self.pool.reserved)

    def test_active_load_holds_cpu_slot_until_completion(self):
        self.commit()
        self.load.transfer_async(2, spec([0], [5], [b"A0"], False))
        with self.assertRaises(SidecarContractError):
            self.store.transfer_async(3, spec([0], [2], [b"B"]))
        self.ld.complete(2)
        self.load.get_finished()
        self.commit(4, [0], [2], [b"B"])
        self.assertEqual(self.pool.committed[0].key, b"B")

    def test_full_prompt_last_token_recompute_and_partial_block_mm(self):
        self.write([3], 1)

        class BT:
            block_tables = [1]

            def __getitem__(self, i):
                return SimpleNamespace(block_table=SimpleNamespace(cpu=torch.tensor([[1, 2, 3]])))

        for computed, scheduled in [(15, 1), (16, 7), (31, 1), (32, 7), (47, 1), (17, 7), (0, 1)]:
            batch = SimpleNamespace(
                req_ids=["full"], req_id_to_index={"full": 0}, num_computed_tokens_cpu=[computed], block_table=BT()
            )
            self.cache.add_prefix_cached_new_req_id("full")
            tail = torch.full((scheduled, 2), 3)
            got = self.cache.get_merged_hidden_states(
                query_start_loc=[0], input_batch=batch, hidden_states=tail, num_scheduled_tokens={"full": scheduled}
            )["full"]
            self.assertEqual(got.shape[0], computed + scheduled)
            self.assertTrue(torch.all(got[:computed] == 1))
            self.assertTrue(torch.all(got[computed:] == 3))
            mm = self.cache.get_merged_multimodal_states(
                query_start_loc=[0],
                input_batch=batch,
                multimodal_outputs={"hidden_states": {"layers": {"0": tail, "24": tail}}},
                num_scheduled_tokens={"full": scheduled},
            )
            for key in ["hidden_states.layer_0", "hidden_states.layer_24"]:
                self.assertTrue(torch.equal(got, mm[key]["full"]))

    def test_resident_hit_and_miss_merge_unchanged(self):
        class BT:
            block_tables = [1]

            def __getitem__(self, i):
                return SimpleNamespace(block_table=SimpleNamespace(cpu=torch.tensor([[1]])))

        batch = SimpleNamespace(
            req_ids=["resident"], req_id_to_index={"resident": 0}, num_computed_tokens_cpu=[16], block_table=BT()
        )
        self.cache.add_prefix_cached_new_req_id("resident")
        got = self.cache.get_merged_hidden_states(
            query_start_loc=[0],
            input_batch=batch,
            hidden_states=torch.full((1, 2), 3),
            num_scheduled_tokens={"resident": 1},
        )["resident"]
        self.assertTrue(torch.all(got[:16] == 1))
        self.assertTrue(torch.all(got[16] == 3))
        self.cache.reset_prefix_cached_new_req_ids()
        miss = self.cache.get_merged_hidden_states(
            query_start_loc=[0],
            input_batch=batch,
            hidden_states=torch.full((1, 2), 3),
            num_scheduled_tokens={"resident": 1},
        )["resident"]
        self.assertEqual(miss.shape[0], 1)


class ActualRunnerSlotMappingTests(unittest.TestCase):
    """Real AR update hook, with separate CPU buffers standing in for GPU/host.

    No CUDA work is performed. The authoritative-buffer selection and copy are
    real; GPU computation of the mapping itself is covered by the pressure gate.
    """

    def setUp(self):
        from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

        self.hook = GPUARModelRunner._maybe_update_prefix_cache
        self.cache = OmniTensorPrefixCache(8, 16, 2, torch.bfloat16)
        self.pool = SidecarPool(self.cache, 4)
        self.mirror = SimpleNamespace(
            cpu=torch.zeros(80, dtype=torch.long), gpu=torch.full((80,), -1, dtype=torch.long)
        )
        self.runner = SimpleNamespace(
            omni_prefix_cache=self.cache,
            input_batch=SimpleNamespace(block_table=[SimpleNamespace(slot_mapping=self.mirror)]),
        )
        self.sd = FakeDMA()
        self.store = SidecarTransferHandler(self.sd, self.pool, True)

    def update(self, slots, values, padded=None):
        n = len(slots)
        padded = n if padded is None else padded
        self.mirror.gpu.fill_(-1)
        self.mirror.gpu[:n] = torch.tensor(slots)
        value = torch.tensor(values, dtype=torch.bfloat16).reshape(n, 1).expand(n, 2).clone()
        if padded > n:
            value = torch.cat([value, torch.full((padded - n, 2), 99, dtype=torch.bfloat16)])
        with patch(
            "vllm_omni.worker.gpu_ar_model_runner.get_pp_group", return_value=SimpleNamespace(is_last_rank=True)
        ):
            self.hook(
                self.runner, value, {"hidden_states": {"layers": {"0": value.clone(), "24": value.clone()}}}, n, padded
            )

    def test_actual_hook_ignores_stale_cpu_mirror_and_graph_padding(self):
        self.update(list(range(16, 48)), [1] * 32, padded=64)
        for key, tensor in self.pool.tensors().items():
            self.assertTrue(torch.all(tensor[1:3] == 1))
            self.assertTrue(self.pool.valid[key][1:3].all())
            self.assertFalse(self.pool.valid[key][0].any())
            self.assertFalse(self.pool.valid[key][7].any())
        self.assertTrue(torch.all(self.mirror.cpu == 0))
        self.store.transfer_async(1, spec([0, 1], [1, 2], [b"A0", b"A1"]))
        self.sd.complete(1)
        self.store.get_finished()
        self.assertEqual(set(self.pool.committed), {0, 1})

    def test_mixed_partial_and_new_request_slots_preserve_local_prefix(self):
        self.update(list(range(16, 48)), [1] * 32)
        self.update([31, *range(80, 96)], [3, *([7] * 16)], padded=32)
        for tensor in self.pool.tensors().values():
            self.assertTrue(torch.all(tensor[1, :15] == 1))
            self.assertTrue(torch.all(tensor[1, 15] == 3))
            self.assertTrue(torch.all(tensor[2] == 1))
            self.assertTrue(torch.all(tensor[5] == 7))
        self.store.transfer_async(1, spec([0, 1], [1, 5], [b"A-new", b"B"]))
        self.sd.complete(1)
        self.store.get_finished()
        load = SidecarTransferHandler(FakeDMA(), self.pool, False)
        load.transfer_async(2, spec([1], [6], [b"B"], False))
        for tensor in self.pool.tensors().values():
            self.assertTrue(torch.all(tensor[6] == 7))

    def test_invalid_unpadded_slot_rejected_before_cache_mutation(self):
        for bad in [-1, 128]:
            with self.subTest(slot=bad), self.assertRaisesRegex(ValueError, "Invalid physical slot"):
                self.update([16, bad], [1, 1], padded=16)
            self.assertFalse(self.pool.valid)

    def test_real_worker_flush_snapshots_prior_step_before_next_mapping(self):
        worker = OffloadingConnectorWorker(SimpleNamespace())
        worker.worker.register_handler(GPULoadStoreSpec, CPULoadStoreSpec, self.store)
        worker.prepare_store_kv(OffloadingConnectorMetadata({}, {"A": spec([0, 1], [1, 2], [b"A0", b"A1"])}))
        self.update(list(range(16, 48)), [1] * 32)
        # The next batch reuses GPU input buffers. Stored sidecar rows must not
        # alias this buffer, and the real worker flushes before its next forward.
        self.mirror.gpu.fill_(80)
        worker.handle_preemptions(OffloadingConnectorMetadata({}, {"B": spec([2], [5], [b"B"])}, {"A"}))
        self.assertTrue(torch.all(self.pool.committed[0].tensors[HIDDEN] == 1))
        self.assertTrue(torch.all(self.pool.committed[1].tensors[HIDDEN] == 1))
        self.assertNotIn(2, self.pool.committed)


class IdentityTests(unittest.TestCase):
    def test_real_hashes_lora_mm_salt_and_generator_transport(self):
        hf = get_hash_fn_by_name("sha256")
        init_none_hash(hf)
        hasher = get_request_block_hasher(16, hf)

        def key(salt=None, lora=None, mm=None):
            core = build_engine_core_request_from_tokens(
                "R", dict(prompt_token_ids=[10] * 17), SamplingParams(max_tokens=1), mm_features=mm
            )
            core.cache_salt = salt
            core.lora_request = lora
            return make_offload_key(OmniRequest.from_engine_core_request(core, hasher).block_hashes[0], 0)

        keys = [
            key(),
            key(salt="S1"),
            key(salt="S2"),
            key(lora=LoRARequest("A", 1, "/not-loaded/A")),
            key(lora=LoRARequest("B", 2, "/not-loaded/B")),
            key(
                mm=[
                    MultiModalFeatureSpec(
                        data=None,
                        modality="image",
                        identifier="real-mm-A",
                        mm_position=PlaceholderRange(offset=0, length=16),
                    )
                ]
            ),
            key(
                mm=[
                    MultiModalFeatureSpec(
                        data=None,
                        modality="image",
                        identifier="real-mm-B",
                        mm_position=PlaceholderRange(offset=0, length=16),
                    )
                ]
            ),
        ]
        self.assertEqual(len(set(keys)), len(keys))
        manager = IdentityCPUOffloadingManager(16)
        context = ReqContext(kv_transfer_params=None)
        self.assertIsNotNone(manager.prepare_store(iter(keys), context))
        manager.complete_store(keys)
        load = manager.prepare_load((k for k in keys), context)
        self.assertEqual(load.omni_sidecar_keys, tuple(keys))
        self.assertEqual(len(load.block_ids), len(keys))
        metadata = OffloadingConnectorMetadata(
            {"req": (load, GPULoadStoreSpec(list(range(len(keys))), [len(keys)], [0]))}, {}
        )
        buffers = []
        wire = pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL, buffer_callback=buffers.append)
        restored = pickle.loads(wire, buffers=buffers)
        self.assertEqual(restored.reqs_to_load["req"][0].omni_sidecar_keys, tuple(keys))
        # The actual tensor multiprocessing transport uses ForkingPickler.
        from multiprocessing.reduction import ForkingPickler

        restored = ForkingPickler.loads(ForkingPickler.dumps(metadata))
        self.assertEqual(restored.reqs_to_load["req"][0].omni_sidecar_keys, tuple(keys))
        from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

        with patch.dict("os.environ", {"VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}):
            restored = MsgpackDecoder(OffloadingConnectorMetadata).decode(MsgpackEncoder().encode(metadata))
        self.assertEqual(restored.reqs_to_load["req"][0].omni_sidecar_keys, tuple(keys))

    def test_actual_spec_factory_budget_and_unsupported_layout(self):
        group = KVCacheGroupSpec(
            ["layer"], FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=2, dtype=torch.bfloat16)
        )
        cache = KVCacheConfig(
            num_blocks=8, kv_cache_tensors=[KVCacheTensor(size=8192, shared_by=["layer"])], kv_cache_groups=[group]
        )
        extra = dict(
            spec_name="OmniCPUOffloadingSpec",
            spec_module_path="vllm_omni.core.sidecar_offload",
            cpu_bytes_to_use=16384,
            eviction_policy="lru",
            store_threshold=0,
        )
        cfg = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
            cache_config=SimpleNamespace(block_size=16),
            parallel_config=SimpleNamespace(world_size=1),
            kv_events_config=None,
        )
        spec = OffloadingSpecFactory.create_spec(cfg, cache)
        self.assertIsInstance(spec, OmniCPUOffloadingSpec)
        self.assertEqual(spec.num_blocks, 16)
        self.assertIsInstance(spec.get_manager(), IdentityCPUOffloadingManager)
        self.assertEqual(spec.extra_config["cpu_bytes_to_use"], 16384)
        with self.assertRaises(SidecarContractError):
            list(spec.get_handlers(None))
        from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

        from vllm_omni.core.sidecar_offload import bind_sidecar_to_offloading_connector

        sidecar = OmniTensorPrefixCache(num_blocks=8, block_size=16, hidden_size=2, hs_dtype=torch.bfloat16)
        worker = SimpleNamespace(spec=spec, _jobs={})
        with (
            patch("vllm.distributed.kv_transfer.has_kv_transfer_group", return_value=True),
            patch(
                "vllm.distributed.kv_transfer.get_kv_transfer_group",
                return_value=SimpleNamespace(connector_worker=worker),
            ),
        ):
            bind_sidecar_to_offloading_connector(sidecar)
        self.assertIs(spec.sidecar_pool.cache, sidecar)
        with patch.object(
            CPUOffloadingSpec,
            "get_handlers",
            return_value=iter(
                [(GPULoadStoreSpec, CPULoadStoreSpec, FakeDMA()), (CPULoadStoreSpec, GPULoadStoreSpec, FakeDMA())]
            ),
        ):
            handlers = list(spec.get_handlers(None))
        self.assertEqual([x[2].store for x in handlers], [True, False])
        self.assertTrue(all(x[2].pool is spec.sidecar_pool for x in handlers))

        cfg.parallel_config.world_size = 2
        with self.assertRaises(SidecarContractError):
            OffloadingSpecFactory.create_spec(cfg, cache)
        cfg.parallel_config.world_size = 1
        extra["block_size"] = 32
        with self.assertRaises(SidecarContractError):
            OffloadingSpecFactory.create_spec(cfg, cache)
        extra.pop("block_size")
        cache.kv_cache_groups = [group, group]
        with self.assertRaises(SidecarContractError):
            OffloadingSpecFactory.create_spec(cfg, cache)


if __name__ == "__main__":
    unittest.main(verbosity=2)
