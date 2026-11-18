# Changes

## 2026-11-18: metric V4, submission window, Claude Code effort, fruit-fly on H100

Base: `main` at `0d7d405` ("Add accepted DrugCLIP task with BEDROC v2 and shared runner integration").
It merges the `grace_period` branch (`4cf8206`) and brings the tasks to the metric-V4 verifiers.

| Change | Where | Default |
| --- | --- | --- |
| Metric-V4 verifiers for the 20 tasks | `tasks/*` | always |
| Sealed verifier files, supplied at scoring time | `runner/stage_sealed.sh`, `SEALED_DIR` | needed by 3 tasks |
| Submission-only window after the research time | `GRACE_SECONDS` | `0` (no window) |
| Claude Code reasoning effort | `CLAUDE_REASONING` | empty (Claude Code's default) |
| Fruit-fly point-sampling library for H100 | `tasks/fruit-fly-neuron-reconstruction` | always |
| SSI index code, its tests and the metric documents | `harness/`, `docs/` | |

### Reproducing the reported runs

The runs of the `grace_period` branch used a five-minute window for every trial and `xhigh` effort for
Claude Code:

```bash
export EVAL_ROOT=/persistent/speedrun CODEX_PKG=/opt/codex-0.160.1 MOUNT_ROOTS="/data /opt/codex-0.160.1"
export SEALED_DIR=/path/to/ai4sci-tasks        # private task repository, see section 2
GRACE_SECONDS=300 AGENT_KIND=codex HOURS=48 runner/run.sh TASK
GRACE_SECONDS=300 CLAUDE_REASONING=xhigh AGENT_KIND=claude-code HOURS=48 runner/run.sh TASK
```

`runner/selftest.sh` checks an installation in about two minutes (see Validation below).

## 1. Metric V4

The verifiers, scoring code and instructions are those that scored the SSI V4 results. Per task:

| Task | V4 change |
| --- | --- |
| atomistic-materials | Score is the arithmetic mean of the eight suites; the MOF zero-force anchor is a parameter (`atom_zero_force_anchors.json`) with a measurement tool, `tests/measure_mof_zero_force_d3_anchor.py`. |
| brain-computer-interface-decoding | FACED removed; pretrained weights that overlap the test subjects (BIOT) blocked; inference isolated per subject and session; four-track subject/session balanced-accuracy composite. |
| crystal-structure-generation | Score `(SUN + 0.5 × MetaSUN) / 2500`. |
| earth-observation-pretraining-transfer | AWF scored on a frozen 205-window spatially disjoint test split (sealed roster); those windows are removed from the agent's development data; 20-NN probe and LFMC test MAE. |
| fluid-flow-control | Sealed blind flow seeds; policy seed masking; OpenMPI runtime settings and a multi-GPU device fix. |
| fruit-fly-neuron-reconstruction | Metric unchanged (on hold); H100 library fix (section 4). |
| genome-sequence-to-function | Score `0.5 × tracks + 0.5 × variants`; CADD v1.7 excluded from pure-sequence anchors. |
| mass-spectrometry-molecule-retrieval | Streaming one-spectrum-at-a-time inference (`serve.sh`) with a cold-start parity check against cross-query leakage; the 373-batch mode stays as a fallback. |
| molecular-pretraining-transfer | Score `1 / (1 + mean of four groups' MA-RAE)` everywhere. |
| natural-disease-progression | `score`/`reward` keys aligned; a TabPFN clinical comparator replay. |
| pde-surrogate | Stale `normalised_score` field removed. |
| probabilistic-global-weather | Ten-variable score in `grade.py`; the evaluator is unchanged. |
| protein-design | One combined score, `unconditional / 51 + (50/153) × (small-molecule binder + PPI binder + enzyme)`; a trivial straight-helix control. |
| protein-folding-prediction | Inference and grading in separate containers: the inference container mounts no ground truth, the grader runs no submitted code. |
| protein-sequence-modeling | PPI predictions averaged over both input orders. |
| single-cell-perturbation-prediction | Two-pass query-independence check: predictions for a query must not depend on the other queries in the request. |
| surface-code-decoding | Keyed HMAC-SHA256 sample ids and a keyed shot-order shuffle. |
| target-based-drug-screening | MMseqs2 coverage mode 1 (target coverage); swapped-pocket counterfactual diagnostic. |
| timeseries-forecasting | Partial-coverage interpolation applied exactly once; the 6.4M container starter is the baseline. |
| tokamak-magnetic-control | 105 sealed episodes (7 cases × 15 seeds), score = successful episodes / 105; seed masked in the reset context. |

`harness/ssi_index_v3.py` computes the index from the verifiers' receipts (`harness/test_ssi_index_v4.py` checks
the V4 changes); `docs/SSI_BENCHMARK_V2.md` and `docs/METRIC_VALIDITY_V2.md` describe the metric.

Task tests that use the index code now find `harness/` from either repository layout (`tasks/<task>` here,
`<task>` in the private repository). `atomistic-materials/tests/test_mof_zero_force_anchor.py` gained the
`copy()` its fake atoms need under the pinned ASE 3.26.

## 2. Sealed verifier files

Three verifiers score against held-out seeds or rosters that must not be public:

| Task | Sealed file |
| --- | --- |
| earth-observation-pretraining-transfer | `tests/sealed/spatial-roster.json`, `tests/sealed/spatial-roster-audit.json` |
| fluid-flow-control | `tests/sealed/blind_flow_seeds.json` |
| tokamak-magnetic-control | `tests/sealed/perturbations.json` (V4 seeds; the V3 file published earlier is removed) |

This repository carries only `tests/sealed/SHA256SUMS` and a README for each. Set `SEALED_DIR` to a checkout of
the private task repository: `run_verify.sh` calls `stage_sealed.sh`, which copies the files in and stops the
verification if a file is missing or differs from its hash. Each task's `tests/test.sh` checks the hashes again
inside the verifier, and the unit tests that need the files skip without them.

## 3. Submission-only window after the research time ("grace period")

`GRACE_SECONDS=300` gives the agent a separate, hard-limited window after `HOURS`, in which it is told to stop
experimenting and publish its best package.

How it works, inside the task container:

1. Research ends after `HOURS`. The end is enforced by the agent's own script rather than by Harbor's timeout,
   because the window has to continue the same session.
2. Every process of the agent user that the session started is stopped, so nothing started during research
   keeps running into the window.
3. The package is left alone for one snapshot interval plus 15 s, so `snapshot.sh` keeps a copy of it as it
   stood when research ended.
4. The same session is continued once with the submit prompt, for at most `GRACE_SECONDS`.
5. Everything is stopped again, and `/logs/agent/grace-period.json` records the times and the package digest at
   the end of research and after the window.

The window is not research time, so its result is kept apart:

- `reward.json` and `submission/` stay the package from the end of research.
- If the window changed the package, that package is kept and scored separately as `submission-grace/`,
  `reward-grace.json` and `report-grace.json`.
- `grace-period.json` in the results folder says which case applied: `unchanged`, `split`,
  `no_package_at_research_end` or `research_end_package_not_captured`.
- `score_snapshots.sh` leaves versions published in the window out of the reward-over-time curve.

**Fix in this merge.** The branch stopped every process of the agent user. Under Harbor's Docker environment the
agent runs as root in 19 of the 20 tasks, and in natural-disease-progression as the same user as the container's
keep-alive process (`sleep infinity`). Stopping that process ended the container: the trial failed with exit 137
when research ended and no window ran. Step 2 now stops only processes that started after the session script
(by their start time in `/proc`).

Files: `runner/harbor_ext/trial_options.py` (the prompt and the in-container script), `runner/grace_result.py`
(separates the two packages), `runner/run_trial.sh`, `runner/run.sh`, `runner/score_snapshots.sh`, the three
`runner/harbor_ext/*_continue.py` agents, `runner/README.md`.

## 4. Claude Code reasoning effort

`CLAUDE_REASONING=low|medium|high|xhigh|max` passes Harbor's `reasoning_effort` option to the Claude Code agent
(`--effort`). Codex already had `CODEX_REASONING`. File: `runner/config.sh`.

## 5. Fruit-fly: the point-sampling library on H100

The task declares H100s, but `point-ops/pathfinder_fps.so` in the prepared data release was built for sm_86 only
and carries no PTX. On an H100 the first call fails with `CUDA kernel failed : no kernel image is available for
execution on the device`; `infer_shape.py` exits with status 255 and the verifier reports an infrastructure failure
about three hours in. The verifier copies this library from the operator's runtime, so every SHAPE evaluation on
H100 failed.

`build_point_ops.py` and `audit_fps_build.py` now default to `TORCH_CUDA_ARCH_LIST=8.6;9.0`, and
`model_package.py` records that list. The library was rebuilt that way with CUDA 12.4 in the task's own runtime
(Python 3.11.13, the locked torch 2.6.0+cu124): `upstream/pathfinder/runtime/fps-build.Dockerfile`, receipt
`upstream/pathfinder/receipts/fps-sm86-sm90-rebuild.json`. It holds sm_86 and sm_90 code; on an RTX A6000 its FPS
indices equal the released library's in 120 cases. It has not yet run on an H100.

The rebuilt library and the runtime manifest that lists it reach operators as an overlay of the release:
`download_hf.py` reads an optional `overlay` entry in `environment/data/download-source.json` (a second HF
repository at a pinned commit) and replaces those two files of the verified release, checked by size and SHA-256.
**Not pinned yet:** the overlay repository is still to be published, and until `download-source.json` carries it,
`prepare.sh` installs the released sm_86-only library.

## Validation

- Task unit tests, run file by file on CPU: the outcomes match the V4 development tree file for file, except that
  the sealed-file tests skip here.
- `runner/selftest.sh` with Codex 0.160.1 and with Claude Code (`CLAUDE_REASONING=high`), each with
  `GRACE_SECONDS=120`: reward 1 for the research package and for the window package (`split`), several resumed
  turns. Without the window (Codex): reward 1.
