"""Exercise the shared HTTP/video policy contract without loading GPU deps."""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINTS = ROOT / "vllm_omni/entrypoints/openai"
spec = importlib.util.spec_from_file_location("omni_kv_policy_cpu", ENTRYPOINTS / "omni_kv_policy.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def entrypoint_method(filename: str):
    """Compile the real entrypoint method; leave unrelated GPU imports unloaded."""
    path = ENTRYPOINTS / filename
    tree = ast.parse(path.read_text())
    method = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_apply_omni_kv_policy"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    namespace = {
        "apply_omni_kv_policy": policy.apply_omni_kv_policy,
        "omni_kv_policy_is_active": policy.omni_kv_policy_is_active,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method.name]


class KVPolicyContractTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict("os.environ", {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.chat = entrypoint_method("serving_chat.py")
        self.video = entrypoint_method("serving_video_stream.py")

    def test_same_metadata_different_comprehension_stage_and_no_shared_mutation(self):
        shared = {"other": 7, "kv_transfer_params": {"existing": "kept"}}
        chat_params = [SimpleNamespace(extra_args=shared), SimpleNamespace(extra_args=shared)]
        video_params = [SimpleNamespace(extra_args=shared)]
        metadata = {
            "enabled": True,
            "session_id": "session-2",
            "committed_prefix_hash": "history-v3",
            "matched_prefix_hash": "history-v2",
            "skip_offload": True,
            "prefetch_hit_used": False,
            "eviction_metadata": {"omni_round_index": 3, "ignored": 99, "omni_null": None},
        }
        self.chat(
            SimpleNamespace(_get_comprehension_stage_index=lambda: 1),
            chat_params,
            SimpleNamespace(omni_kv_policy=metadata),
        )
        self.video(None, video_params, metadata)
        expected = {
            "existing": "kept",
            "omni_skip_offloading_connector": True,
            "omni_prefetch_hit_used": False,
            "omni_session_id": "session-2",
            "omni_committed_prefix_hash": "history-v3",
            "omni_matched_prefix_hash": "history-v2",
            "omni_round_index": 3,
        }
        self.assertEqual(chat_params[1].extra_args["kv_transfer_params"], expected)
        self.assertEqual(video_params[0].extra_args, chat_params[1].extra_args)
        self.assertIs(chat_params[0].extra_args, shared)
        self.assertEqual(shared, {"other": 7, "kv_transfer_params": {"existing": "kept"}})
        self.assertEqual(metadata["eviction_metadata"]["ignored"], 99)

    def test_disabled_or_missing_policy_does_not_access_stage(self):
        for metadata in (None, {}, {"enabled": False}, {"eviction_metadata": {"omni_round_index": 1}}):
            with self.subTest(metadata=metadata):
                self.video(None, [], metadata)
                self.chat(None, [], SimpleNamespace(omni_kv_policy=metadata))

    def test_shadow_metadata_does_not_enable_offload_or_fake_hit(self):
        metadata = {
            "enabled": False,
            "skip_offload": True,
            "prefetch_hit_used": True,
            "lookup_state": "candidate",
            "session_id": "s",
            "eviction_metadata": {"omni_round_index": 2},
        }
        with patch.dict("os.environ", {"OMNI_KV_EVICT_SHADOW": "true"}):
            target = SimpleNamespace(extra_args=None)
            self.video(None, [target], metadata)
        self.assertEqual(target.extra_args, {"kv_transfer_params": {"omni_session_id": "s", "omni_round_index": 2}})

    def test_enforcement_gate_and_existing_stage_error_behavior(self):
        metadata = {"eviction_metadata": {"omni_round_index": 2}}
        for name in ("", "off", "none", "lru", "actual_lru"):
            with patch.dict("os.environ", {"OMNI_KV_EVICT_ENFORCE_POLICY": name}):
                self.assertFalse(policy.omni_kv_policy_is_active(metadata))
        with patch.dict("os.environ", {"OMNI_KV_EVICT_ENFORCE_POLICY": "response_len"}):
            self.assertTrue(policy.omni_kv_policy_is_active(metadata))
        self.chat(
            SimpleNamespace(_get_comprehension_stage_index=lambda: 5),
            [],
            SimpleNamespace(omni_kv_policy={"enabled": True}),
        )
        with self.assertRaises(IndexError):
            self.video(None, [], {"enabled": True})


if __name__ == "__main__":
    unittest.main()
