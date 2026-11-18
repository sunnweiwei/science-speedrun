# Verifier

`test.sh` is the Harbor entry point (inference time limit via
`PDE_SURROGATE_INFERENCE_TIMEOUT_SECONDS`, default 14400; manifest via
`PDE_SURROGATE_SPLIT_MANIFEST`, default `/tests/split.json`, with
`/tests/split_smoke.json` baked in for plumbing runs). It runs
`main_verifier.py`, which stages the held-out cases from the pinned data (mounted
read-only under `/private`, a root-only directory, so a package running as the
runner uid cannot regenerate the truth from it; `/tests` is root-only for the
same reason) and the manifest, replays `bash infer.sh INPUT_DIR OUTPUT_DIR` from a sealed copy of the
submission under a wall-clock limit, scores every case with
`environment/evaluation/score.py`, and writes
`/logs/verifier/{reward.json, report.json, status.json, infer.log}`.
`reward.json` carries the unified scalar `score` and `reward` matching
`ssi_index_v3.pde()` (the legacy `normalised_score` field has been removed in V4),
while `report.json` additionally records `inference_wall_seconds` as an
unpenalized runtime diagnostic and `status.json` records the `isolation` mode.
After scoring it reruns the package on three staged cases under the same
isolation and records the maximum absolute difference in `status.json`
(`determinism`); a nonzero difference is reported, not penalized, since the
contract requires deterministic inference. Inference and the rerun share the one
time budget. Each call gets an empty `HOME`/`TMPDIR` that is removed
afterwards. In uid mode every call then ends with `isolation.after_uid_call`:
every process of the runner uid is killed (a detached child would otherwise
outlive the call), the call's output and home go back to root with mode 0700,
everything the runner owns in the directories it can write (found once by a
scan run as the runner: `/tmp`, `/var/tmp`, `/dev/shm`, its passwd home and
whatever else the image leaves writable) is removed, and so are its SysV IPC
objects. So the rerun can neither replay nor read what the first call produced.
The sealed copy of the package loses group/other write and set-id bits, and the
logs are root-only (0600).

The grader reads package output with the verifier's privileges, outside every
isolation mode, so it only reads regular files whose `prediction` is a plain
dataset (`score.open_package_h5`): a symlink, FIFO or device, an external or
soft link, a virtual dataset or external raw storage is a failed case, since any
of them could point the grader at the truth. A `model_manifest.json` that is not
a JSON object is a rule violation (reward 0).

- Model failures (missing, malformed, non-finite or wrongly shaped
  predictions; non-zero exit; timeout) are scored at VRMSE 10 per case, listed
  with reasons in `report.json` and `status.json`, and never abort the run.
- A malformed package (wrong contract, no `infer.sh`, unreadable manifest,
  declared pretrained assets, a symlink that escapes the package) scores 0 with
  the violation named in `reward.json` and `status.json`, exit 0.
- Setup or grader errors exit with status 2 and a `status.json` naming the
  stage; a non-finite value in the staged truth is one of them, never a 0.
- Isolation (`isolation.py`): inference runs inside a Bubblewrap namespace
  holding only the runtime, the sealed package, `INPUT_DIR` read-only and
  `OUTPUT_DIR` writable: no truth directory, no work directory, no network, no
  host `/tmp`. Docker's default seccomp profile denies the user namespace this
  needs, so `docker-compose.yaml` grants it (`security_opt`); Docker also
  refuses the network namespace (`bwrap: loopback: Failed RTM_NEWADDR`), in
  which case the namespace is built without it and the container's
  `network_mode: none` keeps the package offline (`status.json` records
  `isolation_detail`). Where no namespace works and the verifier is root, the
  package runs as uid 10001 while the truth, `/private` and `/tests` stay
  root-only (the boundary protein-sequence-modeling's verifier uses; the integrator
  ran it under Docker). `status.json` says which (`"bwrap"` or `"uid"`). With neither
  available the verifier refuses to score unless the operator sets
  `PDE_SURROGATE_ALLOW_NO_ISOLATION=1`, and then marks the run untrusted,
  because an unsandboxed replay proves much less than it looks like it does.
  `test_isolation.py` proves the namespace with a package that walks the
  filesystem hunting for the targets: it finds them without isolation and
  nothing with it.
- `docker-compose.yaml` is the Harbor overlay for the separate verifier: the
  `main` service built from `Dockerfile` with the task directory as context,
  the verifier data mounted read-only at `/private/data`, no network, the
  namespace grant, and a healthcheck. `main_verifier.py --phase stage|infer|score` splits a run into
  phases for operators who want the truth staged by a different process.

Tests are pytest-only (`bash run_unit_tests.sh`, about 25 s, no data or GPU;
`python -m unittest discover` finds nothing; as root, the end-to-end tests open
pytest's temporary tree to the runner uid themselves): `test_task.py` (grader, split,
readers), `test_verifier_failures.py` (end-to-end packages on synthetic data:
good, crashing, hanging, malformed, symlinked; the image file layout),
`test_isolation.py`, `test_release.py`, `test_score_bounds.py`,
`test_docs_scripts.py`.

Local use without Docker (the runtime interpreter must provide torch, h5py,
numpy; for the starter that is the task virtual environment):

```bash
PATH=/path/to/venv/bin:$PATH bash pde-surrogate/tests/run_unit_tests.sh
PATH=/path/to/venv/bin:$PATH python3 pde-surrogate/tests/main_verifier.py \
  --submission /path/to/submission --data "$TASK_PRIVATE_DIR" \
  --manifest pde-surrogate/environment/data/split_smoke.json --output /tmp/verifier-logs
```

The verifier image (`Dockerfile`) has not been built on the contributor's
machine, which has no Docker; it bakes the same files the local driver uses
(`main_verifier.py`, `isolation.py`, the grader modules, both manifests,
`score_bounds.json`), installs bubblewrap and the runner uid, and its pins were
checked to resolve.

---

## Scientific & Literature Justification of Verifier Design

### 1. Primary Metric & Upstream Paper Provenance

- **Metric Definition**: On each of the `1,085` held-out test rollouts (`850` across 16 deduplicated `ape2d/*` PDE families at $K = 26$ steps, `200` on `pdearena/navier_stokes_2d` at $K = 10$ steps, `26` on `well/active_matter` at $K = 30$ steps, and `9` on `well/turbulent_radiative_layer_2D` at $K = 30$ steps), rollout accuracy per predicted step $t \in \{1,\dots,K\}$ and channel $c \in \{1,\dots,C\}$ is measured using variance-normalized root-mean-square error with unbiased spatial variance (`ddof=1`, $\varepsilon = 10^{-7}$), clamped to $10.0$:
  $$\text{VRMSE}_c(t) = \min\!\left(10.0,\; \sqrt{\frac{\operatorname{mean}_{x,y}\!\bigl(\hat{u}_{t,c}(x,y) - u_{t,c}(x,y)\bigr)^2}{\operatorname{var}_{x,y}\!\bigl(u_{t,c}(x,y)\bigr) + 10^{-7}}}\right)$$
  Averaging equally over channels and rollout steps yields each case's VRMSE, averaging over cases yields each dataset's VRMSE, and datasets aggregate into four equal-weighted benchmark groups $g \in \{\texttt{ape2d}, \texttt{pdearena}, \texttt{well/active\_matter}, \texttt{well/turbulent\_radiative\_layer\_2D}\}$:
  $$S_{11} = \frac{1}{4} \sum_{g} \frac{1 - \log_{10}\!\bigl(\operatorname{clip}(\bar{v}_g,\, 0.01,\, 10.0)\bigr)}{3} \in [0, 1]$$
  Every scheduled case stays in the denominator (failed/missing predictions receive `VRMSE = 10.0`, mapping to `0.0`).
- **Upstream Literature Provenance**:
  - **Per-channel spatial VRMSE**: Adopted directly from *The Well* (Ohana et al., *NeurIPS* 2024 Datasets & Benchmarks, arXiv:`2412.00568`, commit `ad50de0e`), where normalizing by instantaneous per-channel spatial variance prevents high-magnitude background density/pressure channels from masking small-scale vorticity or concentration errors across multi-physics fields.
  - **Equation-conditioned corpus & multi-family metadata**: Inherited directly from *APEBench* (Koehler et al., *NeurIPS* 2024, arXiv:`2411.00180`) and *PDE-Transformer* (Koehler et al.), alongside *PDEArena* (Gupta & Brandstetter, *ICLR* 2023 / *TMLR*), where physical metadata (`pde`, `fields`, `field_scheme`, `dt`, `domain_extent`, `boundary_conditions`, `constants`) are part of the canonical operator-conditioning problem definition.

### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted

- **No Ad-Hoc Inference Speed Penalty Mixed into the Scalar Score**: McGreivy & Hakim (*Nature Machine Intelligence* 2024, DOI: `10.1038/s42256-024-00897-5`) rightly demonstrate that accuracy-vs-cost Pareto comparisons against coarse numerical solvers are essential when claiming neural PDE speedups. However, inside a containerized 48-hour benchmark evaluation on shared single-GPU hardware, wall-clock inference latency conflates Python/HDF5 I/O batching overheads with intrinsic operator complexity, and none of the upstream multi-equation surrogate benchmarks (*APEBench*, *PDEArena*, *The Well*) fold arbitrary seconds-per-step multipliers into rollout VRMSE. Instead, V4 enforces a uniform 4-hour hard inference budget (`14,400 s`), records `inference_wall_seconds` directly in `report.json`, and publishes an explicit **Observation-grid Coarse Numerical Solver** mechanistic control (`docs/validation/numerical_controls/score.json`, overall `0.5359`: Exponax spectral/ETDRK stepping on `ape2d` `0.5822`, PhiFlow 2.1 staggered-grid continuation on `pdearena` `1.0000`, and persistence on the partially observed/subgrid *Well* transfer suites where governing Observation-grid closures are underdetermined from 4 frames alone) alongside all neural surrogate results.
- **No Stripping of Upstream Equation Metadata**: Stripping `pde`, `constants`, `dt`, or `boundary_conditions` attributes from `cases/<case_id>.h5` would turn well-posed operator learning into an ill-posed system-identification puzzle from 4 frames—contradicting *PDE-Transformer*, *Poseidon*, and *DPOT* conditioning protocols—while `contract.md` already forbids switching between separately trained per-dataset checkpoints by name.

### 3. Diagnostic Signals Retained in `report.json`

While `reward.json` exports the pure 4-group rollout score (`score` and `reward`), `report.json` (and `status.json`) retains rich physical and operational diagnostics:
- **Runtime & Determinism Audits**: `inference_wall_seconds` in `report.json`, plus `status.json["inference"]` (`exit_code`, `timed_out`, `isolation`, `wall_seconds`) and `status.json["determinism"]` (`max_abs_difference` across a 3-case isolated cold rerun).
- **PDE-Transformer Joint-Channel nRMSE**: `nrmse_joint` per case and per dataset (`sqrt(mean_{c,x,y}(pred - target)^2 / (mean_{c,x,y} target^2 + 1e-7))`), allowing direct cross-paper comparison with *PDE-Transformer* tables.
- **Short / Medium / Long Rollout Horizon Windows**: `vrmse_steps_1_5`, `vrmse_steps_6_12`, and `vrmse_steps_13_30` per case and per dataset (matching *The Well*'s temporal error-accumulation breakdown), plus full `vrmse_by_step` and `vrmse_by_channel` curves.
