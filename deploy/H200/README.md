# H200 Deploy Configs

This directory contains the active H200/L20X deploy YAMLs used by current
LiveServe profiling.  Historical H20 templates were removed from this branch.

## Rules

- Keep deploy files generic and configurable; do not bake one experiment's GPU
  allocation into code.
- Runners should select devices through `CUDA_VISIBLE_DEVICES`, `TEXT_GPUS`, or
  `VIDEO_GPUS`.
- If a deploy changes memory split, prefix caching, chunk size, or offload
  behavior, record the intended variant in the nearest runner README.
- Use no-offload deploys for strict zero-optimization baseline comparisons and
  for ShareGPT non-retained scheduler-only reproduction, where offload is not
  part of the promoted result.

Current text/audio and video runners default to H200 deploys through environment
variables such as `LIVESERVE_DEPLOY_YAML`, `LIVESERVE_DEPLOY_YAML_2GPU`, and
`LIVESERVE_DEPLOY_YAML_NO_OFFLOAD`.
