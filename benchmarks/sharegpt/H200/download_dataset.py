"""Download and normalize shareAI/ShareGPT-Chinese-English-90k.

The vLLM ShareGPTDataset loader expects this schema:
[
  {
    "id": "...",
    "conversations": [
      {"from": "human",  "value": "..."},
      {"from": "gpt",    "value": "..."},
      ...
    ]
  },
  ...
]
"""

import os
import json
from huggingface_hub import snapshot_download

REPO_ID = "shareAI/ShareGPT-Chinese-English-90k"
# Default dataset paths can be overridden by env vars to support multi-host setups (H20/H200).
#   LIVESERVE_DATA_ROOT          — root data dir (default: <repo>/data, gitignored)
#   LIVESERVE_SHAREGPT_RAW_DIR   — ShareGPT raw dir (default: $LIVESERVE_DATA_ROOT/ShareGPT_raw)
#   LIVESERVE_SHAREGPT_DATASET   — ShareGPT cleaned dataset (default: $LIVESERVE_DATA_ROOT/ShareGPT_Chinese_English_90k.json)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DATA_ROOT = os.environ.get("LIVESERVE_DATA_ROOT", os.path.join(_REPO_ROOT, "data"))
LOCAL_DIR = os.environ.get(
    "LIVESERVE_SHAREGPT_RAW_DIR",
    os.path.join(_DATA_ROOT, "ShareGPT_raw"),
)
OUTPUT_PATH = os.environ.get(
    "LIVESERVE_SHAREGPT_DATASET",
    os.path.join(_DATA_ROOT, "ShareGPT_Chinese_English_90k.json"),
)

os.makedirs(LOCAL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# 1) Fetch JSON/JSONL files directly and avoid dataset schema casting.
local_path = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=LOCAL_DIR,
    allow_patterns=["*.json", "*.jsonl"],
)
print(f"Downloaded to: {local_path}")

# 2) Collect all JSON/JSONL data files.
data_files = []
for root, _, files in os.walk(local_path):
    for fn in files:
        if fn.endswith(".json") or fn.endswith(".jsonl"):
            # Skip dataset metadata files.
            if fn in ("dataset_infos.json",):
                continue
            data_files.append(os.path.join(root, fn))
print(f"Found {len(data_files)} data files: {data_files}")

# 3) Parse and convert fields: {human, assistant} -> {from, value}.
converted = []
for fp in data_files:
    with open(fp, "r", encoding="utf-8") as f:
        # Try JSON first, then fall back to JSONL.
        try:
            items = json.load(f)
            if isinstance(items, dict):
                items = [items]
        except json.JSONDecodeError:
            f.seek(0)
            items = [json.loads(line) for line in f if line.strip()]

    for item in items:
        # Some source files use singular "conversation" with human/assistant keys.
        turns = item.get("conversation") or item.get("conversations") or []
        new_convs = []
        for turn in turns:
            if "human" in turn and turn["human"]:
                new_convs.append({"from": "human", "value": turn["human"]})
            if "assistant" in turn and turn["assistant"]:
                new_convs.append({"from": "gpt", "value": turn["assistant"]})

        # vLLM ShareGPTDataset requires at least one user/assistant pair.
        if len(new_convs) < 2:
            continue

        converted.append({
            "id": str(item.get("id", item.get("conversation_id", ""))),
            "conversations": new_convs,
        })

# 4) Save the cleaned dataset.
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)

print(f"Saved {len(converted)} entries to {OUTPUT_PATH}")
