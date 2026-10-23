# ShareGPT workload

> Single-turn baseline workload plus synthetic length-aligned pressure tests.

## Workspace

| Subdir | Owner | Purpose |
|---|---|---|
| [`H200/`](H200/) | Current H200 | ShareGPT original, synthetic len1024/2048/4096 clients, and plotting helpers. |

Prepare the dataset with:

```bash
bash scripts/prepare_sharegpt.sh
```

See [`../../docs/README.md`](../../docs/README.md) for download and
replay details.
