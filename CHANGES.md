# Changes

## 2026-11-18: metric V4, submission window, Claude Code effort, fruit-fly on H100

Base: `main` at `0d7d405` ("Add accepted DrugCLIP task with BEDROC v2 and shared runner integration").
It merges the `grace_period` branch (`4cf8206`) and brings the tasks to the metric-V4 verifiers.

| Change | Where | Default |
| --- | --- | --- |
| Metric-V4 verifiers for the 20 tasks | `tasks/*` | always |
| Sealed verifier files, published and hash-checked at scoring time | `runner/stage_sealed.sh`, `tests/sealed/SHA256SUMS` | 3 tasks |
| Submission-only window after the research time | `GRACE_SECONDS` | `0` (no window) |
| Claude Code reasoning effort | `CLAUDE_REASONING` | empty (Claude Code's default) |
| GPUs for scoring, apart from the session's (one-GPU sessions, eight-GPU verifiers) | `VERIFY_GPU_IDS` | `GPU_IDS` |
| Fruit-fly point-sampling library for H100 | `tasks/fruit-fly-neuron-reconstruction` | always |
| SSI index code, its tests and the metric documents | `harness/`, `docs/` | |

### Reproducing the reported runs

The published results (42 histories in the HF dataset `s42chen/sci-run`) gave every session one H100 for 48 hours,
then a submission-only window of at most five minutes. 16 tasks declare eight GPUs; they ran with 22 CPUs and
224 GiB. brain-computer-interface-decoding, natural-disease-progression, tokamak-magnetic-control and
earth-observation-pretraining-transfer ran with 64 CPUs and 128 GiB. Codex ran `gpt-6-astra` at xhigh reasoning;
Claude Code 2.1.319 ran `claude-fable-5-2` at xhigh effort. The runs were on Beaker with adapted isolation, and some
were interrupted and resumed (see the dataset's README).

```bash
export EVAL_ROOT=/persistent/speedrun CODEX_PKG=/opt/codex-0.160.1 MOUNT_ROOTS="/data /opt/codex-0.160.1"
export HOURS=48 GRACE_SECONDS=300 GPUS=1 GPU_IDS=0 AGENT_CPUS=22 AGENT_MEMORY_MB=229376 VERIFY_GPU_IDS=
CODEX_MODEL=openai/gpt-6-astra CODEX_REASONING=xhigh AGENT_KIND=codex runner/run.sh TASK
CLAUDE_MODEL=anthropic/claude-fable-5-2 CLAUDE_REASONING=xhigh AGENT_KIND=claude-code runner/run.sh TASK
```

`GPU_IDS` picks the session's GPU (one per parallel session). `VERIFY_GPU_IDS` picks the GPUs scoring may use;
empty means any. [EVALUATING.md](EVALUATING.md) is the guide for running another agent or model.
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

Three verifiers score against held-out seeds or rosters:

| Task | Sealed file |
| --- | --- |
| earth-observation-pretraining-transfer | `tests/sealed/spatial-roster.json`, `tests/sealed/spatial-roster-audit.json` |
| fluid-flow-control | `tests/sealed/blind_flow_seeds.json` |
| tokamak-magnetic-control | `tests/sealed/perturbations.json` (V4 seeds; the V3 seeds moved to a legacy test fixture) |

They are published with the tasks. The agent's development image never contains `tests/`, and sessions run offline,
so the agent cannot read them. `tests/sealed/SHA256SUMS` lists them by hash. `run_verify.sh` calls `stage_sealed.sh`,
which stops the verification if a file is missing or differs; it can fill a missing file from `SEALED_DIR`. Each
task's `tests/test.sh` checks the hashes again inside the verifier.

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

The rebuilt library and the runtime manifest that lists it ship with the task as an overlay of the release
(`environment/data/overlay/`, 360 KB). `download-source.json` pins them by hash, and `download_hf.py` replaces those
two files of the verified release, checked by size and SHA-256. Checked against the real release manifest: the
new runtime manifest lists exactly the overlaid runtime files, as the verifier requires. Re-run
`prepare.sh fruit-fly-neuron-reconstruction` on data prepared earlier to install it.

## Validation

- Task unit tests, run file by file on CPU: the outcomes match the V4 development tree file for file, except that
  the sealed-file tests skip here.
- `runner/selftest.sh` with Codex 0.160.1 and with Claude Code (`CLAUDE_REASONING=high`), each with
  `GRACE_SECONDS=120`: reward 1 for the research package and for the window package (`split`), several resumed
  turns. Without the window (Codex): reward 1.
