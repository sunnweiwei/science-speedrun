# Tests

- `test.sh`: Harbor entry point; runs `main_verifier.py` on `/workspace/submission` with data at `/private/data` (the verifier overlay mounts `gifteval/` there; `/private` and `/tests` are root-only, so a package running as the runner uid can regenerate no label) and the split at `/tests/split.json` (`TIMESERIES_SPLIT_MANIFEST=/tests/split_smoke.json` for a plumbing run), writes `/logs/verifier/{report,reward,status}.json`, `results.csv` and `infer.log`. `reward.json` carries the raw reward, the normalised score (`evaluation/score_bounds.json`, baked beside the grader) and the isolation mode.
- `main_verifier.py`: stages input/truth from the pinned data (both root-only: a call is given one cohort's copy under `work/cohorts/<c>/input`, never the whole split, whose later cohorts' contexts hold earlier cohorts' labels), runs the sealed package once per cutoff cohort inside a Bubblewrap namespace (`isolation.py`: runtime, package, input, output; no data, labels, workspace or network) under one timeout, each call with an empty `HOME`/`TMPDIR` that is removed afterwards. In uid mode every call then ends with `isolation.after_uid_call`: every runner-uid process is killed (a detached child would otherwise outlive its call), the call's output and home go back to root with mode 0700, everything the runner owns in the directories it can write (found once by a scan run as the runner) is removed, and so are its SysV IPC objects; so no call, the determinism rerun included, sees another's files, processes or outputs. The sealed package loses group/other write and set-id bits; `infer.log` is root-only and each call's output is copied into it as capped bytes. Merges the calls, reruns the smallest cohort for determinism under the same isolation, scores. Package output is read only from regular files whose `mean` and `quantiles` are plain datasets (`ts_io.open_package_h5`): an external or soft link, a virtual dataset (the review mapped one onto the labels and scored 0.999), external storage, a symlink or a FIFO is a failed config. A malformed package (wrong contract, a manifest that is not a JSON object, no `infer.sh`, declared pretrained assets, a symlink escaping the package) scores 0 with the violation named. `--phase stage|infer|score` splits it across processes for operators; the infer phase re-applies the root-only mode to the staged input and checks, in uid mode, that the runner cannot reach the data, the labels or the full input.
- `isolation.py`: the namespace (built without the network namespace where Docker refuses it, `status.json` says so; the container's `network_mode: none` keeps the package offline), and the fallback when no namespace works: with the verifier running as root the package runs as uid 10001 while the labels, `/private` and `/tests` stay root-only (`"uid"`); only with `TIMESERIES_ALLOW_NO_ISOLATION=1` does it run a plain child process, marked untrusted. `docker-compose.yaml` grants what namespaces the host allows (`security_opt`).
- `check_local_isolation.py`: runs a canary package through the verifier and asserts it saw no data, truth, workspace or network (`docs/validation/isolation_probe.json`).
- `Dockerfile`: verifier image (`pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime`, PEP 668 override, the runtime pins under `constraints.txt`), with bubblewrap, the runner uid, the grader, the normalizer, both manifests and the bounds baked in, and `/private` and `/tests` root-only. Built with the task directory as context (`docker-compose.yaml`). Not built on the contributor's machine (no Docker); the pde-surrogate twin of this image was built and run by the integrator.
- `docker-compose.yaml`: the Harbor overlay for the separate verifier: `main` built from `Dockerfile`, `${TASK_DATA_DIR}/gifteval` read-only at `/private/data/gifteval`, no network, the namespace grant, a healthcheck.
- Unit tests, pytest only (`bash run_unit_tests.sh`, about 15 s, no data or GPU; `unittest discover` finds nothing; as root the end-to-end tests open pytest's temporary tree to the runner uid themselves): `test_task.py` (protocol tables, the frozen manifest, masked seasonal error, hand-computed metrics, the zero-seasonal-error rule, quantile sorting, reward bounds), `test_verifier_failures.py` (missing / mis-shaped / NaN / unreadable predictions are model failures; malformed packages score 0; partial coverage; symlinks), `test_isolation.py` (the namespace hides planted labels, mode selection, the uid drop when root), `test_release.py`, `test_score_bounds.py`, `test_docs_scripts.py`.

Two notes for maintainers. The trusted grader is imported in process by `main_verifier.py` rather
than run as a separate `grader.py`: the submission is never imported, only executed through
`isolation.py` in its own namespace, and the verifier image bakes its own copy of `score.py`, so a
package cannot reach the scoring code. And the sealed copy dereferences symlinks, but only those that stay inside the package: a link that
escapes it (to the verifier's truth directory, to `/data`, anywhere on the host) is a rule violation
worth reward 0, because `copytree` would otherwise copy the target into the package. Packages
therefore carry their own `ts_io.py` rather than linking to `environment/evaluation/`, and a unit
test checks the copies have not drifted.

Local replay without Docker:

```
python3 tests/main_verifier.py --submission /path/to/package --data /large-disk/timeseries \
    --manifest environment/data/split.json --output /tmp/verify-out --work /tmp/verify-work

# the seven-config smoke split instead, about a minute
python3 tests/main_verifier.py --submission /path/to/package --data /large-disk/timeseries \
    --manifest environment/data/split_smoke.json --output /tmp/verify-out --work /tmp/verify-work
```
