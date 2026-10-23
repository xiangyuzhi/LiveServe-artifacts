# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared KV-policy metadata for HTTP chat and video requests.

The entrypoint chooses its comprehension stage. These helpers only decide
whether a policy is active and merge its fields into that stage's parameters.
"""

import os
from typing import Any


def omni_kv_policy_is_active(policy: Any) -> bool:
    if not isinstance(policy, dict):
        return False
    policy_enabled = bool(policy.get("enabled"))
    eviction_metadata = policy.get("eviction_metadata")
    eviction_metadata_mode = str(os.environ.get("OMNI_KV_EVICT_SHADOW", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    } or str(os.environ.get("OMNI_KV_EVICT_ENFORCE_POLICY", "")).strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
        "none",
        "lru",
        "actual_lru",
    }
    return policy_enabled or (
        eviction_metadata_mode and isinstance(eviction_metadata, dict) and bool(eviction_metadata)
    )


def apply_omni_kv_policy(params: Any, policy: dict[str, Any]) -> None:
    """Merge an active policy without mutating shared parameter dictionaries."""
    policy_enabled = bool(policy.get("enabled"))
    eviction_metadata = policy.get("eviction_metadata")
    extra_args = dict(getattr(params, "extra_args", {}) or {})
    kv_transfer_params = dict(extra_args.get("kv_transfer_params") or {})
    if policy_enabled and policy.get("skip_offload"):
        kv_transfer_params["omni_skip_offloading_connector"] = True
    if policy_enabled:
        kv_transfer_params["omni_prefetch_hit_used"] = bool(policy.get("prefetch_hit_used"))
    lookup_state = policy.get("lookup_state")
    if policy_enabled and isinstance(lookup_state, str):
        kv_transfer_params["omni_prefetch_lookup_state"] = lookup_state
    session_id = policy.get("session_id")
    if isinstance(session_id, str):
        kv_transfer_params["omni_session_id"] = session_id
    prefix_hash = policy.get("committed_prefix_hash")
    if isinstance(prefix_hash, str):
        kv_transfer_params["omni_committed_prefix_hash"] = prefix_hash
    matched_prefix_hash = policy.get("matched_prefix_hash")
    if isinstance(matched_prefix_hash, str):
        kv_transfer_params["omni_matched_prefix_hash"] = matched_prefix_hash
    lookup_key_kind = policy.get("lookup_key_kind")
    if isinstance(lookup_key_kind, str):
        kv_transfer_params["omni_prefetch_lookup_key_kind"] = lookup_key_kind
    prefix_tokens = policy.get("prefix_token_estimate")
    if isinstance(prefix_tokens, int):
        kv_transfer_params["omni_prefix_token_estimate"] = prefix_tokens
    if isinstance(eviction_metadata, dict):
        for key, value in eviction_metadata.items():
            if key.startswith("omni_") and value is not None:
                kv_transfer_params[key] = value
    extra_args["kv_transfer_params"] = kv_transfer_params
    params.extra_args = extra_args
