#!/usr/bin/env python3
"""Independent replay of a timeseries-forecasting submission.

    python3 main_verifier.py --submission /workspace/submission --data /private/data \
        --manifest /tests/split.json --output /logs/verifier [--split test] [--timeout 14400]

1. Check the package (infer.sh, model_manifest.json with the right contract, no pretrained assets declared).
2. Stage INPUT_DIR (per-instance contexts) and a hidden TRUTH_DIR (labels, seasonal errors) from the pinned
   GIFT-Eval data and the split manifest.
3. Copy the package into a fresh sealed directory and run `bash infer.sh INPUT OUTPUT` once per cutoff cohort
   (instances whose labels are not contained in any other instance's context of the same call; see make_cases.py),
   under one overall wall-clock limit; merge the cohort outputs. The package never sees TRUTH_DIR or the data.
4. Score with the trusted grader against the Seasonal Naive reference; write report.json, results.csv (leaderboard
   column names), reward.json and status.json.
Model failures (missing or bad predictions, non-zero exit, timeout) are scored, listed and never abort the run.
Setup or grader errors exit non-zero with a status.json that names the stage.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_EVAL = HERE / "evaluation" if (HERE / "evaluation").exists() else HERE.parent / "environment" / "evaluation"
sys.path.insert(0, str(_EVAL))
CONTRACT = "timeseries-forecast/v1"
sys.path.insert(0, str(HERE))
import isolation  # noqa: E402

LOG_CAP = 2 * 1024 * 1024   # bytes of one call's stdout/stderr kept in infer.log


# the bounds travel with the grader modules: /tests/evaluation in the verifier image, environment/evaluation in a checkout
BOUNDS = _EVAL / "score_bounds.json"


def harbor_reward(values):
    """Harbor parses reward.json as a flat object of finite numbers; strings, None and nested details
    belong in report.json / status.json."""
    import math as _math
    return {k: v for k, v in values.items() if isinstance(v, (int, float)) and _math.isfinite(v)}


def normalised(reward):
    """Map the raw reward onto 0-1 between this track's reference floor and ceiling.

    Returns None when no bounds file is present, so a checkout without one still scores.
    """
    if not BOUNDS.exists():
        return None
    b = json.loads(BOUNDS.read_text())
    lo, hi = b["lower"]["reward"], b["upper"]["reward"]
    if not hi > lo:
        raise ValueError(f"{BOUNDS}: upper bound {hi} must exceed lower bound {lo}")
    return {"normalised_score": min(1.0, max(0.0, (reward - lo) / (hi - lo))), "lower": lo, "upper": hi}


def write(path, obj):
    """Write JSON with non-finite floats as null, so every artifact parses under a strict JSON reader."""
    sys.path.insert(0, str(_EVAL))
    from score import json_safe  # noqa: E402

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), indent=1, allow_nan=False) + "\n")


def seal_package(sub: Path, sealed: Path):
    """Copy the package for execution. Symlinks are dereferenced (that is what makes a package portable) but only if
    they stay inside the package: a link to the truth directory, the data or anywhere else on the host is a rule
    violation, not a shortcut. Returns None on success or the violation string."""
    root = sub.resolve()
    try:
        for p in sub.rglob("*"):
            if p.is_symlink():
                try:
                    target = p.resolve(strict=True)
                except OSError as exc:
                    return f"broken symlink in the package: {p.relative_to(sub)} ({exc.strerror})"
                if not (target == root or target.is_relative_to(root)):
                    return f"symlink escapes the package: {p.relative_to(sub)} -> {target}"
    except (RecursionError, OSError) as exc:
        return f"package tree could not be walked: {type(exc).__name__}"
    if sealed.exists():
        isolation._rm_rf(sealed)
    try:
        shutil.copytree(sub, sealed, symlinks=False)
    except Exception as exc:  # noqa: BLE001 - an unreadable package is the package's problem
        return f"package could not be copied: {type(exc).__name__}: {str(exc)[:300]}"
    # the copy keeps the package's mode bits: nothing in it may stay writable by the runner uid, or carry set-id or
    # sticky bits (one sealed copy serves all 82 calls and the rerun, so uid mode would let a call change the next)
    subprocess.run(["chmod", "-R", "go-w,ug-s,a-t", "--", str(sealed)], capture_output=True, timeout=600)
    return None


def cohort_inputs(inp: Path, work: Path, cohort: int):
    """Stage the instances of one cohort of every config into work/cohorts/<c>/input; returns (dir, [(config_id, source indices)])."""
    import ts_io  # noqa: E402

    index = json.loads((inp / "index.json").read_text())
    cdir = work / "cohorts" / f"{cohort:03d}"
    if cdir.exists():
        isolation._rm_rf(cdir)
    (cdir / "input" / "cases").mkdir(parents=True)
    members, staged = [], []
    for cfg in index["configs"]:
        case = ts_io.CaseFile(inp / "cases" / f"{cfg['config_id']}.h5")
        idx = np.nonzero(case.cohort == cohort)[0]
        if len(idx):
            ts_io.write_case_subset(case, idx, cdir / "input" / "cases" / f"{cfg['config_id']}.h5")
            members.append((cfg["config_id"], idx, (case.n_variates, case.prediction_length)))
            staged.append({**cfg, "n_instances": int(len(idx))})
        case.close()
    (cdir / "input" / "index.json").write_text(json.dumps({**index, "cohort": cohort, "configs": staged}))
    (cdir / "output").mkdir()
    return cdir, members


def run_package(sealed: Path, inp: Path, outp: Path, env, timeout: float, log, work: Path):
    """bash infer.sh INPUT OUTPUT inside the isolation namespace (tests/isolation.py); returns (exit code or None, seconds).

    Every call gets an empty HOME/TMPDIR (a sibling of its output directory) that is removed afterwards. In uid mode
    run_isolated also kills every runner process, gives the output back to root (0700) and empties every
    runner-writable directory and the runner's IPC objects (isolation.after_uid_call), so nothing a package writes
    or leaves running in one call reaches a later one, the determinism rerun included. Bubblewrap gives each call
    its own namespace anyway."""
    home = Path(outp).parent / f"home-{Path(outp).name}"
    isolation._rm_rf(home)
    try:
        rc, secs, mode = isolation.run_isolated(sealed, inp, outp, timeout, log, protect=(work,),
                                                fallback_env={**env, "HOME": str(home), "TMPDIR": str(home)})
    finally:
        isolation._rm_rf(home)
    ISOLATION.setdefault("modes", set()).add(mode)
    return rc, secs


def copy_call_log(call_log: Path, log) -> None:
    """Append one call's stdout/stderr to infer.log as bytes, capped: the package decides what it prints, so the
    copy must neither decode it (one invalid UTF-8 byte used to end the run with exit 2) nor hold it all in memory."""
    with open(call_log, "rb") as f:
        data = f.read(LOG_CAP + 1)
    if len(data) > LOG_CAP:
        data = data[:LOG_CAP] + b"\n[... call log truncated at %d bytes]\n" % LOG_CAP
    log.write(data); log.flush()


def _private_log(path: Path, binary: bool = False):
    """A log only root can read: a later call running as the runner must not read what an earlier one printed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb" if binary else "w")


def _scalar(v):
    """A manifest value as recorded in status.json: numbers and short strings as they are, anything else as a short
    string (a deeply nested value would otherwise break the JSON writer)."""
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return v[:300]
    try:
        return repr(v)[:300]
    except Exception:  # noqa: BLE001 - e.g. RecursionError on a deeply nested value
        return f"<{type(v).__name__}>"


ISOLATION = {}


def merge_cohort(cdir: Path, members, merged: dict):
    """Collect a cohort's predictions into per-config arrays keyed by source index; missing/bad files leave gaps."""
    import ts_io  # noqa: E402

    for config_id, idx, shape in members:
        p = cdir / "output" / "cases" / f"{config_id}.h5"
        try:
            mean, quant = ts_io.read_prediction_file(p, expect=(len(idx),) + shape)
        except Exception as exc:  # noqa: BLE001 - a model failure for these instances
            merged.setdefault(config_id, {"errors": []})["errors"].append(f"cohort {cdir.name}: {type(exc).__name__}: {exc}")
            continue
        m = merged.setdefault(config_id, {"errors": []})
        m.setdefault("parts", []).append((idx, mean, quant))


def write_merged(merged: dict, inp: Path, outp: Path):
    """Assemble full prediction files; a config with any gap gets no file (the grader scores it as failed)."""
    import ts_io  # noqa: E402

    index = json.loads((inp / "index.json").read_text())
    (outp / "cases").mkdir(parents=True, exist_ok=True)
    problems, coverage = {}, {}
    for cfg in index["configs"]:
        cid, n = cfg["config_id"], cfg["n_instances"]
        m = merged.get(cid, {"errors": ["no cohort produced this config"]})
        parts = m.get("parts", [])
        n_covered = sum(len(idx) for idx, _, _ in parts)
        if m["errors"]:
            problems[cid] = list(m["errors"]) + ([f"{n_covered}/{n} instances predicted"] if n_covered != n else [])
        if not parts:
            continue                      # nothing usable: the grader scores the config as failed
        try:
            mean = np.zeros((n,) + parts[0][1].shape[1:], np.float32)
            quant = np.zeros((n,) + parts[0][2].shape[1:], np.float32)
            covered = np.zeros(n, bool)
            for idx, pm, pq in parts:
                mean[idx] = pm; quant[idx] = pq; covered[idx] = True
            ts_io.write_prediction_file(outp / "cases" / f"{cid}.h5", mean, quant, {"merged_from_cohorts": len(parts)})
            if not covered.all():         # a cut-short run: the grader scores what is there and penalizes the rest
                coverage[cid] = covered.tolist()
                if not any("instances predicted" in e for e in problems.get(cid, [])):
                    problems.setdefault(cid, []).append(f"{int(covered.sum())}/{n} instances predicted")
        except Exception as exc:  # noqa: BLE001 - inconsistent parts are a model failure, not a grader failure
            problems.setdefault(cid, []).append(f"merge failed: {type(exc).__name__}: {exc}")
            (outp / "cases" / f"{cid}.h5").unlink(missing_ok=True)
    return problems, coverage


def determinism_check(sealed, work, cohort_input: Path, members, first_output: Path, env, timeout):
    """Rerun the package on one cohort and compare with its first outputs bit for bit."""
    import ts_io  # noqa: E402

    out2 = work / "output_det"
    isolation._rm_rf(out2)
    out2.mkdir()
    rc, secs = run_package(sealed, cohort_input, out2, env, timeout, subprocess.DEVNULL, work)
    if rc is None:
        return {"checked_cohort": cohort_input.parent.name, "rerun_timed_out": True}
    diffs, missing = [], 0
    for config_id, *_ in members:
        a, b = first_output / "cases" / f"{config_id}.h5", out2 / "cases" / f"{config_id}.h5"
        if not (a.exists() and b.exists()):
            missing += 1; continue
        try:
            ma, qa = ts_io.read_prediction_file(a); mb, qb = ts_io.read_prediction_file(b)
            diffs.append(max(float(np.max(np.abs(ma.astype(np.float64) - mb))), float(np.max(np.abs(qa.astype(np.float64) - qb)))))
        except Exception:  # noqa: BLE001
            missing += 1
    return {"checked_cohort": cohort_input.parent.name, "configs": [m[0] for m in members], "rerun_exit_code": rc, "seconds": round(secs, 1),
            "unavailable": missing, "max_abs_difference": max(diffs) if diffs else None, "deterministic": bool(diffs) and max(diffs) == 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--split", default="test"); ap.add_argument("--timeout", type=float, default=14400)
    ap.add_argument("--work", default="/tmp/timeseries-forecasting-verify"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--configs", default="")
    ap.add_argument("--baseline", default="", help="Seasonal Naive reference JSON (default: the grader's bundled file)")
    ap.add_argument("--phase", choices=["all", "stage", "infer", "score"], default="all",
                    help="two-container mode: 'stage' writes input/truth, 'infer' runs the package on input only, 'score' grades")
    ap.add_argument("--determinism-configs", type=int, default=1, help="1: rerun the smallest cohort and compare bit for bit; 0 disables")
    args = ap.parse_args()
    out, work = Path(args.output), Path(args.work)
    status = {"contract": CONTRACT, "stage": "setup", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        sub = Path(args.submission)
        inp, truth, outp, sealed = work / "input", work / "truth", work / "output", work / "sealed"
        if args.phase in ("all", "infer"):
            violation = None
            try:
                manifest = json.loads((sub / "model_manifest.json").read_text())
            except Exception as exc:  # noqa: BLE001 - a malformed package is the package's fault, not the harness's
                manifest, violation = {}, f"model_manifest.json unreadable: {type(exc).__name__}: {str(exc)[:300]}"
            if violation is None and not isinstance(manifest, dict):
                manifest, violation = {}, f"model_manifest.json must be a JSON object, not {type(manifest).__name__}"
            if violation is None and manifest.get("contract") != CONTRACT:
                violation = f"model_manifest.json contract {str(manifest.get('contract'))[:100]!r} != {CONTRACT!r}"
            if violation is None and not (sub / "infer.sh").exists():
                violation = "infer.sh missing"
            if violation is not None:
                status.update({"stage": "done", "reward": 0.0, "rule_violation": violation})
                write(out / "status.json", status); write(out / "reward.json", {"reward": 0.0})
                (out / "reward.txt").write_text("0.0\n")
                print(json.dumps({"reward": 0.0, "rule_violation": violation}))
                return
            status["model_manifest"] = {k: _scalar(manifest.get(k)) for k in ("name", "parameters", "pretrained_assets", "sources_used")}
            if violation is None and manifest.get("pretrained_assets"):
                violation = "model_manifest.json declares pretrained assets; the task forbids them (instruction.md)"
                status.update({"stage": "done", "reward": 0.0, "rule_violation": violation})
                write(out / "status.json", status); write(out / "reward.json", {"reward": 0.0})
                (out / "reward.txt").write_text("0.0\n")
                print(json.dumps({"reward": 0.0, "rule_violation": status["rule_violation"]}))
                return
        if args.phase == "all" and work.exists():
            isolation._rm_rf(work)                   # a fresh run: no stale input, truth, seal or output
        if args.phase in ("all", "infer"):
            work.mkdir(parents=True, exist_ok=True)
            violation = seal_package(sub, sealed)   # before the labels are staged: a package is never copied beside them
            if violation is not None:
                status.update({"stage": "done", "reward": 0.0, "rule_violation": violation})
                write(out / "status.json", status); write(out / "reward.json", {"reward": 0.0})
                (out / "reward.txt").write_text("0.0\n")
                print(json.dumps({"reward": 0.0, "rule_violation": violation}))
                return
        if args.phase in ("all", "stage"):
            from make_cases import main as make_cases_main  # noqa: E402
            argv = sys.argv
            sys.argv = ["make_cases.py", "--data", args.data, "--manifest", args.manifest, "--split", args.split,
                        "--input", str(inp), "--truth", str(truth)]
            if args.limit:
                sys.argv += ["--limit", str(args.limit)]
            if args.configs:
                sys.argv += ["--configs", args.configs]
            make_cases_main(); sys.argv = argv
            # root-only: a call sees one cohort's copy under work/cohorts/<c>/input, never the whole split
            # (later cohorts' contexts hold earlier cohorts' labels)
            os.chmod(truth, 0o700); os.chmod(inp, 0o700)
            if args.phase == "stage":
                write(out / "status.json", {**status, "stage": "staged", "n_configs": len(json.loads((inp / "index.json").read_text())["configs"])})
                return
        index = json.loads((inp / "index.json").read_text())
        status["n_configs"] = len(index["configs"]); status["n_instances"] = sum(c["n_instances"] for c in index["configs"])
        status["stage"] = "inference"
        if args.phase in ("all", "infer"):
            for p in (inp, truth):          # root-only also when a separate stage container wrote them
                if p.exists():
                    os.chmod(p, 0o700)
            mode = isolation.select_mode()
            if mode == "subprocess":
                if not os.environ.get("TIMESERIES_ALLOW_NO_ISOLATION"):
                    raise RuntimeError("neither Bubblewrap (user namespaces) nor a uid drop (root) is available: the package "
                                       "would run with access to the truth directory and the data. Run the verifier as root, "
                                       "give the container user namespaces (security_opt seccomp/apparmor unconfined, see "
                                       "tests/docker-compose.yaml), or set TIMESERIES_ALLOW_NO_ISOLATION=1 to run anyway and "
                                       "mark the result untrusted.")
                status["isolation_warning"] = "ran with the verifier's own privileges; the score is not trustworthy"
            if mode == "uid":
                # the uid drop hides only what root-only permissions hide: the data the labels were staged
                # from must be out of the runner's reach as well, or a package can rebuild every target
                exposed = [str(p) for p in (Path(args.data), truth, inp) if isolation.runner_can_reach(p)]
                if exposed:
                    raise RuntimeError(
                        "inference would run as the runner uid, which can read " + ", ".join(exposed) + "; mount the "
                        "verifier data under a root-only directory (tests/docker-compose.yaml uses /private/data "
                        "inside a 0700 /private)")
        t0 = time.time()
        env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "HOME": str(work / "home"),
               "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8"), "HF_HUB_OFFLINE": "1"}
        for k in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
            if k in os.environ:
                env[k] = os.environ[k]
        out.mkdir(parents=True, exist_ok=True)
        if args.phase in ("all", "infer") and outp.exists():
            isolation._rm_rf(outp)        # never score a previous run's predictions
        outp.mkdir(exist_ok=True)
        if args.phase in ("all", "infer"):
            n_cohorts = int(index.get("n_cohorts", 1))
            invocations, merged, deadline = [], {}, time.time() + args.timeout
            smallest = None
            with _private_log(out / "infer.log", binary=True) as log:
                for c in range(n_cohorts):
                    cdir, members = cohort_inputs(inp, work, c)
                    if not members:
                        continue
                    n_inst = sum(len(m[1]) for m in members)
                    remaining = deadline - time.time()
                    log.write(f"=== cohort {c}: {len(members)} configs, {n_inst} instances, {remaining:.0f} s left\n".encode()); log.flush()
                    # a fresh log per call: the child can reopen its own stdout through /proc, so it must not be a
                    # stream shared with the other calls (cross-call state) or with the verifier's own record
                    call_log = work / f"infer-cohort-{c:03d}.log"
                    with _private_log(call_log) as child_log:
                        rc, secs = run_package(sealed, cdir / "input", cdir / "output", env, max(1.0, remaining), child_log, work)
                    copy_call_log(call_log, log); call_log.unlink(missing_ok=True)
                    invocations.append({"cohort": c, "configs": len(members), "instances": n_inst, "exit_code": rc, "seconds": round(secs, 1)})
                    merge_cohort(cdir, members, merged)
                    if smallest is None or n_inst < smallest[1]:
                        smallest = (c, n_inst, cdir, members)
                    if rc is None:
                        status["inference_note"] = f"time limit reached at cohort {c}; later cohorts not run"
                        break
                    isolation._rm_rf(cdir / "input")  # keep only the outputs
            problems, coverage = write_merged(merged, inp, outp)
            cut_short = any(i["exit_code"] is None for i in invocations) or "inference_note" in status
            if not cut_short:
                coverage = {}      # gaps the package chose are failures, not partial credit
                for cid in list(problems):
                    (outp / "cases" / f"{cid}.h5").unlink(missing_ok=True)
            write(work / "coverage.json", coverage)
            status["isolation"] = sorted(ISOLATION.get("modes", []))
            status["inference"] = {"exit_code": max((i["exit_code"] if i["exit_code"] is not None else 124) for i in invocations) if invocations else None,
                                   "timed_out": any(i["exit_code"] is None for i in invocations), "seconds": round(time.time() - t0, 1),
                                   "invocations": len(invocations), "n_cohorts": n_cohorts, "cohort_seconds_max": max((i["seconds"] for i in invocations), default=0),
                                   "configs_with_gaps": problems}
            write(out / "invocations.json", invocations)
            status["stage"] = "determinism"
            if smallest is not None and args.determinism_configs > 0:
                c, _, cdir, members = smallest
                kept = work / "det_first_output"
                if kept.exists():
                    isolation._rm_rf(kept)
                shutil.move(str(cdir / "output"), str(kept))
                cdir, members = cohort_inputs(inp, work, c)  # its input was removed after the run: restage it
                left = max(0.0, deadline - time.time())
                status["determinism"] = (determinism_check(sealed, work, cdir / "input", members, kept, env, min(left, 1800))
                                         if left > 30 and not status["inference"]["timed_out"] else {"checked_cohort": None, "skipped": "no budget left"})
            else:
                status["determinism"] = {"checked_cohort": None}
            write(work / "inference-status.json", status)
            if args.phase == "infer":
                write(out / "status.json", status)
                return
        elif (work / "inference-status.json").exists():
            status.update(json.loads((work / "inference-status.json").read_text()))
        status["stage"] = "scoring"
        status["output_manifest_present"] = (outp / "model_manifest.json").exists() or any((work / "cohorts").glob("*/output/model_manifest.json"))
        from score import score, write_results_csv  # noqa: E402
        baseline = json.loads(Path(args.baseline).read_text())["configs"] if args.baseline else None
        coverage = json.loads((work / "coverage.json").read_text()) if (work / "coverage.json").exists() else {}
        report = score(truth, outp, baseline, coverage=coverage)
        write(out / "report.json", report)
        reward = report["reward"]
        payload = {"reward": reward, "isolation": status.get("isolation")}
        bounds = normalised(reward)
        if bounds is not None:
            payload["normalised_score"] = bounds["normalised_score"]
            payload["bounds"] = {"lower": bounds["lower"], "upper": bounds["upper"]}
        write(out / "reward.json", harbor_reward(payload))
        (out / "reward.txt").write_text(f"{report['reward']}\n")
        write_results_csv(report, out / "results.csv", status.get("model_manifest", {}).get("name") or "submission")
        status.update({"stage": "done", "reward": report["reward"], "skill_score": report["skill_score"],
                       "rel_mase_gmean": report["rel_mase_gmean"], "rel_wql_gmean": report["rel_wql_gmean"],
                       "n_failed": report["n_failed"], "n_partial": report["n_partial"],
                       "n_instances_scored": report["n_instances_scored"],
                       "failure_reasons": sorted({f["reason"] for f in report["failures"]}),
                       "groups": report["groups"]})
        write(out / "status.json", status)
        print(json.dumps({k: status[k] for k in ("reward", "skill_score", "rel_mase_gmean", "rel_wql_gmean", "n_configs", "n_failed", "inference") if k in status}))
    except Exception as exc:  # noqa: BLE001 - setup/scorer failure, distinguishable from model failure
        status.update({"error": f"{type(exc).__name__}: {str(exc)[:1000]}"})
        try:
            write(out / "status.json", status)
        except Exception:  # noqa: BLE001 - never lose the record of a setup error
            (out / "status.json").write_text(json.dumps({"stage": str(status.get("stage")), "error": status["error"]}) + "\n")
        print(json.dumps({"stage": str(status.get("stage")), "error": status["error"]}), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
