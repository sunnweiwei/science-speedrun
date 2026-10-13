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
`reward.json` carries the raw `reward`, the `normalised_score` between the
track's floor and ceiling (`evaluation/score_bounds.json`, baked into the image
beside the grader) and the `isolation` mode. After scoring it reruns the package
on three staged cases under the same isolation and records the maximum absolute
difference in `status.json` (`determinism`); a nonzero difference is reported,
not penalized, since the contract requires deterministic inference. Inference
and the rerun share the one time budget. Each call gets an empty `HOME`/`TMPDIR` that is removed
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
  ran it under Docker). `reward.json` says which (`"bwrap"` or `"uid"`). With neither
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
checked to resolve. A test imports `main_verifier.py` from a copy of that layout
so the bounds file is found where the image puts it.
