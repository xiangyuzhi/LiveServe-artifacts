"""Exercise real pin TTL/refcount lifecycle without importing vLLM or a GPU."""
from __future__ import annotations

import ast
import math
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def scheduler_class():
    path = Path(__file__).resolve().parents[2] / "vllm_omni/core/sched/omni_scheduler_mixin.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OmniSchedulerMixin")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    namespace = {"os": os, "math": math, "logger": Mock(), "RequestStatus": SimpleNamespace(FINISHED_ABORTED="aborted")}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["OmniSchedulerMixin"], namespace


class PrefetchPinTests(unittest.TestCase):
    def test_nonfinite_request_or_environment_TTL_still_expires(self):
        for request_ttl, env_ttl, expected in ((float("inf"), "250", .25),
                                               (None, "inf", 5),
                                               (float("-inf"), "nan", 5),
                                               (float("nan"), "400", .4),
                                               (200, "inf", .2)):
            with self.subTest(request_ttl=request_ttl, env_ttl=env_ttl):
                cls, namespace = scheduler_class()
                scheduler = cls()
                now = [100.0]
                namespace["time"] = SimpleNamespace(monotonic=lambda: now[0])
                block = SimpleNamespace(is_null=False, block_hash=b"cached", ref_cnt=1)
                def touch(blocks):
                    for value in blocks:
                        value.ref_cnt += 1
                def free(blocks):
                    for value in blocks:
                        value.ref_cnt -= 1
                pool = SimpleNamespace(touch=Mock(side_effect=touch), free_blocks=Mock(side_effect=free))
                scheduler.kv_cache_manager = SimpleNamespace(block_pool=pool, get_blocks=lambda rid: SimpleNamespace(blocks=[[block]]))
                request = SimpleNamespace(request_id="prefetch", status="stopped", kv_transfer_params={
                    "omni_scheduler_prefetch": True, "omni_prefetch_gpu_pin": True,
                    "omni_prefetch_session_id": "s", "omni_prefetch_committed_prefix_hash": "p",
                    "omni_prefetch_gpu_pin_ttl_ms": request_ttl})
                with patch.dict(os.environ, {"OMNI_PREFETCH_GPU_PIN": "1", "OMNI_PREFETCH_GPU_PIN_TTL_MS": env_ttl}, clear=True):
                    scheduler._omni_maybe_pin_prefetch_request(request)
                lease = scheduler._omni_gpu_pin_leases["s:p"]
                self.assertAlmostEqual(lease["expires_monotonic"], 100 + expected)
                self.assertTrue(math.isfinite(lease["expires_monotonic"]))
                self.assertEqual(block.ref_cnt, 2)
                now[0] = 100 + expected + .01
                self.assertEqual(scheduler._omni_release_expired_gpu_pins(), 1)
                self.assertEqual(block.ref_cnt, 1)
                self.assertEqual(scheduler._omni_gpu_pin_leases, {})
                self.assertEqual(scheduler._omni_release_expired_gpu_pins(), 0)
                pool.free_blocks.assert_called_once_with([block])

    def test_default_off_does_not_touch_blocks_or_create_pin_state(self):
        cls, _ = scheduler_class()
        scheduler = cls()
        scheduler.kv_cache_manager = Mock()
        request = SimpleNamespace(kv_transfer_params={"omni_scheduler_prefetch": True, "omni_prefetch_gpu_pin": True})
        with patch.dict(os.environ, {}, clear=True):
            scheduler._omni_maybe_pin_prefetch_request(request)
        self.assertFalse(hasattr(scheduler, "_omni_gpu_pin_leases"))
        scheduler.kv_cache_manager.get_blocks.assert_not_called()


if __name__ == "__main__":
    unittest.main()
