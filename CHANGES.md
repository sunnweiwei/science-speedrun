# Changes on the `grace_period` branch

Base: upstream `main` at `0d7d405` ("Add accepted DrugCLIP task with BEDROC v2 and shared runner
integration").

This branch adds three things to the upstream repository. Two are optional runner settings that  
are off by default. The third is a bug fix for the fruit-fly task on H100.


| Change                                           | Setting            | Default                           |
| ------------------------------------------------ | ------------------ | --------------------------------- |
| Claude Code reasoning effort                     | `CLAUDE_REASONING` | empty (Claude Code's own default) |
| Submission-only window after the research time   | `GRACE_SECONDS`    | `0` (no window)                   |
| Fruit-fly point-sampling library builds for H100 | none               | always on                         |




## 1. Claude Code reasoning effort

The runner had an effort setting for Codex (`CODEX_REASONING`) but none for Claude Code.
`CLAUDE_REASONING=low|medium|high|xhigh|max` now passes Harbor's existing `reasoning_effort`
option to the Claude Code agent. Our Claude Code runs used `xhigh`.

Files: `runner/config.sh`.

## 2. Submission-only window after the research time ("grace period")

`GRACE_SECONDS=300` gives the agent a separate, hard-limited window after `HOURS`, in which it is
told to stop experimenting and publish its best package. We ran every trial with a five-minute
window.

How it works, inside the task container:

1. Research ends after `HOURS`. The end is now enforced by the agent's own script rather than by
  Harbor's timeout, because the window has to continue the same session.
2. Every process of the agent user is stopped, so nothing started during research keeps running
  into the window.
3. The package is left alone for one snapshot interval plus 15 s, so `snapshot.sh` keeps a copy of
  it as it stood when research ended.
4. The same session is continued once with the submit prompt, for at most `GRACE_SECONDS`.
5. Everything is stopped again, and `/logs/agent/grace-period.json` records the times and the
  package digest at the end of research and after the window.

The window is not research time, so its result is kept apart:

- `reward.json` and `submission/` stay the package from the end of research.
- If the window changed the package, that package is kept and scored separately as
`submission-grace/`, `reward-grace.json` and `report-grace.json`.
- `grace-period.json` in the results folder says which case applied: `unchanged`, `split`,
`no_package_at_research_end` or `research_end_package_not_captured`.
- `score_snapshots.sh` leaves versions published in the window out of the reward-over-time curve.

Files: `runner/harbor_ext/trial_options.py` (new: the prompt and the in-container script),
`runner/grace_result.py` (new: separates the two packages), `runner/run_trial.sh`, `runner/run.sh`,
`runner/score_snapshots.sh`, the three `runner/harbor_ext/*_continue.py` agents, `runner/README.md`.

## 3. Fruit-fly: the point-sampling library cannot run on H100



### The problem

The fruit-fly task declares H100s (`task.toml`: `gpus = 8`, `gpu_types = ["H100"]`), but the
library its SHAPE stage needs, `point-ops/pathfinder_fps.so`, is built for an older GPU generation
only (sm_86; `RUNTIME.md` says it was built for an A5000). It carries no portable fallback code, so
on an H100 the first call fails:

```
CUDA kernel failed : no kernel image is available for execution on the device
```

`infer_shape.py` then exits with status 255 and the verifier reports an infrastructure failure  
about three hours into an evaluation. The library is not built by the user: it ships prebuilt in  
the task's prepared data release (`development/runtime/point-ops/`), and the verifier mounts it  
from there. So a stock run of this task on H100 fails whenever the SHAPE stage has a candidate to  
score. It also affects research, since the agent's development copy is the same file.

### What this branch changes

`build_point_ops.py` and `audit_fps_build.py` defaulted to `TORCH_CUDA_ARCH_LIST=8.6`. They now
default to `8.6;9.0`, so a rebuild works on both the development GPUs and H100. `RUNTIME.md` has a
paragraph on the failure.

### What still has to be done upstream

The code change does not fix the released file. The maintainers need to:

1. rebuild `pathfinder_fps.so` with the new default;
2. re-release it in the prepared data, with its new hash in `runtime-manifest.json`;
3. update `compiled_cuda_arch='8.6'` in `model_package.py`, which describes the bundled build.

