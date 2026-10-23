import json
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
dataset_path = os.environ.get(
    "LIVESERVE_SHAREGPT_DATASET",
    os.path.join(_REPO_ROOT, "data", "ShareGPT_Chinese_English_90k.json"),
)

with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")
print(f"First entry keys: {data[0].keys()}")
print(f"First conversation: {data[0]['conversations'][:2]}")

# Validate the minimum ShareGPT conversation shape.
valid_count = sum(1 for entry in data if "conversations" in entry and len(entry["conversations"]) >= 2)
print(f"Valid entries (>=2 turns): {valid_count}")
