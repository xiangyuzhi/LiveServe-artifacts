# Interactive-conversation workload

> Multi-turn trace replay with optional barge-in, plus synthetic length-aligned
> prompt/response variants.

## Workspace

| Subdir | Owner | Purpose |
|---|---|---|
| [`H200/`](H200/) | Current H200 | Current trace replay driver, barge-in clients, chain scripts, and len1024/2048/4096 wrapper. |

The H200 folder is self-contained for current validation. Historical H20
drivers and notes were removed from this branch to keep the project focused on
the active H200 path.

## Status

Current H200 scripts include `trace_replay_driver.py`, barge-in client helpers,
scheduler compatibility wrappers, Step 1/2 chains, aggregators, plots, and
`run_interactive-len1024.sh`.
Prefer `scheduler_run_one_cell.sh`, `step2_scheduler_chain*.sh`, and
`run_bargein3gpu_matrix.sh` for new reproductions.  Older compatibility
filenames remain only for historical artifacts.
Trace `part1` is committed; longer trace parts should be copied from project
storage only for long sweeps.

For current workload download and replay details see
[`../../docs/README.md`](../../docs/README.md).
