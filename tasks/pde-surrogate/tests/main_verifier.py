#!/usr/bin/env python3
"""Independent replay of a pde-surrogate submission.

    python3 main_verifier.py --submission /workspace/submission --data /private/data \
        --manifest /tests/split.json --output /logs/verifier [--split test] [--timeout 14400]

1. Check the package (infer.sh, model_manifest.json with the right contract).
2. Stage INPUT_DIR (context frames + metadata) and a hidden TRUTH_DIR from the
   pinned data and the split manifest.
3. Copy the package into a fresh sealed directory and run
   `bash infer.sh INPUT_DIR OUTPUT_DIR` with a wall-clock limit, isolated from
   TRUTH_DIR (tests/isolation.py: a Bubblewrap namespace, else an unprivileged
   uid); the determinism rerun goes through the same isolation.
4. Score every staged case with the trusted grader; write report.json,
   reward.json (raw reward, normalised score, isolation mode) and status.json.
Model failures (bad or missing predictions, non-zero exit, timeout) are scored,
listed and never abort the run; a malformed package (wrong contract, no
infer.sh, declared pretrained assets, escaping symlinks) scores 0 with the
violation named. Setup or grader errors exit non-zero with a status.json that
names the stage, so they are never mistaken for a measured 0.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# grader modules: baked into /tests/evaluation in the verifier image, or the repository's environment/evaluation
_EVAL = HERE / "evaluation" if (HERE / "evaluation").exists() else HERE.parent / "environment" / "evaluation"
sys.path.insert(0, str(_EVAL))
CONTRACT = "pde-surrogate-rollout/v1"


sys.path.insert(0, str(HERE))
import isolation  # noqa: E402

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


def _json_safe(obj):
    """Non-finite floats become null so every artifact parses under a strict JSON reader."""
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(obj), indent=1, allow_nan=False) + "\n")


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


def _private_log(path: Path):
    """A log file only root can read: a later call running as the runner must not read an earlier call's output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w")


def refuse(out, status, violation):
    """A malformed package is the package's fault: reward 0 with the violation named, exit 0."""
    status.update({"stage": "done", "reward": 0.0, "rule_violation": violation})
    write(out / "status.json", status)
    write(out / "reward.json", {"reward": 0.0, "rule_violation": 1.0})
    (out / "reward.txt").write_text("0.0\n")
    print(json.dumps({"reward": 0.0, "rule_violation": violation}))


def seal_package(sub: Path, sealed: Path):
    """Copy the package for execution. Symlinks are dereferenced (that is what makes a package portable) but
    only if they stay inside the package: a link to the truth directory, the data or anywhere else on the
    host is a rule violation, not a shortcut. Returns None on success or the violation string."""
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
    # the copy keeps the package's mode bits: nothing in it may stay writable by the runner uid, or carry
    # set-id or sticky bits (uid mode would otherwise let one call change the code the next call runs)
    subprocess.run(["chmod", "-R", "go-w,ug-s,a-t", "--", str(sealed)], capture_output=True, timeout=600)
    return None


def run_call(sealed, inp, outp, work, env, timeout, log, mode, name):
    """One isolated `infer.sh` call with its own empty HOME/TMPDIR (work/home-<name>), removed afterwards. In uid
    mode run_isolated also kills every runner process, gives the output back to root (0700) and empties the
    runner-writable directories and IPC objects (isolation.after_uid_call). Without this the determinism rerun
    could read or replay what the first call produced and pass without being deterministic."""
    home = work / f"home-{name}"
    isolation._rm_rf(home)
    try:
        return isolation.run_isolated(sealed, inp, outp, timeout, log, fallback_env={**env, "HOME": str(home), "TMPDIR": str(home)},
                                      mode=mode, protect=(work,))
    finally:
        isolation._rm_rf(home)


def determinism_check(sealed, inp, outp, work, env, n_cases, timeout, mode="subprocess"):
    """Rerun the package on a few staged cases, under the same isolation, and compare bit for bit."""
    if n_cases <= 0:
        return {"checked": 0}
    import numpy as np  # noqa: E402
    import h5py  # noqa: E402
    index = json.loads((inp / "index.json").read_text())
    subset = index["cases"][:n_cases]
    inp2, out2 = work / "input_det", work / "output_det"
    (inp2 / "cases").mkdir(parents=True, exist_ok=True); out2.mkdir(exist_ok=True)
    for c in subset:
        shutil.copy(inp / "cases" / f"{c['case_id']}.h5", inp2 / "cases" / f"{c['case_id']}.h5")
    (inp2 / "index.json").write_text(json.dumps({"contract": CONTRACT, "cases": subset}))
    with _private_log(work / "determinism.log") as log:
        rc, _, used = run_call(sealed, inp2, out2, work, env, timeout, log, mode, "determinism")
    if rc is None:
        return {"checked": len(subset), "rerun_timed_out": True, "isolation": used}
    diffs, missing = [], 0
    for c in subset:
        a, b = outp / "cases" / f"{c['case_id']}.h5", out2 / "cases" / f"{c['case_id']}.h5"
        if not (a.exists() and b.exists()):
            missing += 1; continue
        try:
            from score import open_package_h5  # noqa: E402
            with open_package_h5(a, ["prediction"]) as fa, open_package_h5(b, ["prediction"]) as fb:
                diffs.append(float(np.max(np.abs(fa["prediction"][()].astype(np.float64) - fb["prediction"][()].astype(np.float64)))))
        except Exception:  # noqa: BLE001
            missing += 1
    return {"checked": len(subset), "rerun_exit_code": rc, "isolation": used, "unavailable": missing,
            "max_abs_difference": max(diffs) if diffs else None, "deterministic": bool(diffs) and max(diffs) == 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--split", default="test"); ap.add_argument("--timeout", type=float, default=14400)
    ap.add_argument("--work", default="/tmp/pde-surrogate-verify"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--context-frames", type=int, default=4)
    ap.add_argument("--phase", choices=["all", "stage", "infer", "score"], default="all",
                    help="two-container mode: 'stage' writes input/truth, 'infer' runs the package on input only, 'score' grades")
    ap.add_argument("--determinism-cases", type=int, default=3,
                    help="rerun infer.sh on this many staged cases and report the max abs difference (0 disables)")
    args = ap.parse_args()
    out, work = Path(args.output), Path(args.work)
    status = {"contract": CONTRACT, "stage": "setup", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        sub = Path(args.submission)
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
            if violation is None and manifest.get("pretrained_assets"):
                violation = "model_manifest.json declares pretrained assets; the task forbids them (instruction.md)"
            if violation is not None:
                refuse(out, status, violation)
                return
            status["model_manifest"] = {k: _scalar(manifest.get(k)) for k in ("name", "parameters", "context_frames", "pretrained_assets")}
        inp, truth, outp, sealed = work / "input", work / "truth", work / "output", work / "sealed"
        if args.phase in ("all", "stage"):
            if work.exists() and args.phase == "all":
                isolation._rm_rf(work)
            from make_cases import main as make_cases_main  # noqa: E402
            argv = sys.argv
            sys.argv = ["make_cases.py", "--data", args.data, "--manifest", args.manifest, "--split", args.split,
                        "--input", str(inp), "--truth", str(truth), "--context-frames", str(args.context_frames)]
            if args.limit:
                sys.argv += ["--limit", str(args.limit)]
            make_cases_main(); sys.argv = argv
            os.chmod(truth, 0o700)
            if args.phase == "stage":
                write(out / "status.json", {**status, "stage": "staged", "n_cases": len(json.loads((inp / "index.json").read_text())["cases"])})
                return
        status["n_cases"] = len(json.loads((inp / "index.json").read_text())["cases"])
        if args.phase in ("all", "infer"):
            violation = seal_package(sub, sealed)
            if violation is not None:
                refuse(out, status, violation)
                return
        status["stage"] = "inference"
        t0 = time.time()
        env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "HOME": str(work / "home-inference"),
               "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8"),
               "HF_HUB_OFFLINE": "1", "PIP_NO_INDEX": "1"}
        for k in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
            if k in os.environ:
                env[k] = os.environ[k]
        outp.mkdir(exist_ok=True); out.mkdir(parents=True, exist_ok=True)
        if args.phase in ("all", "infer"):
            # The truth directory sits outside INPUT_DIR, but a package running as the same user could
            # simply scan for it. Bubblewrap gives the package a namespace holding only the runtime, the
            # sealed package, the input read-only and the output writable - no truth, no work directory,
            # no network. Without it the run is scored but marked untrusted, and only if the operator
            # opts in, because an unsandboxed replay proves much less than it appears to.
            mode = isolation.select_mode()
            if mode == "subprocess" and not os.environ.get("PDE_SURROGATE_ALLOW_NO_ISOLATION"):
                raise RuntimeError(
                    "neither Bubblewrap (user namespaces) nor a uid drop (root) is available, so inference cannot be "
                    "isolated from the held-out targets. Run the verifier as root, give the container user namespaces "
                    "(security_opt seccomp/apparmor unconfined, see tests/docker-compose.yaml), or set "
                    "PDE_SURROGATE_ALLOW_NO_ISOLATION=1 to score an untrusted run.")
            if mode == "uid":
                # the uid drop hides only what root-only permissions hide: the data the truth was staged
                # from must be out of the runner's reach as well, or a package can rebuild every target
                exposed = [str(p) for p in (Path(args.data), truth) if isolation.runner_can_reach(p)]
                if exposed:
                    raise RuntimeError(
                        "inference would run as the runner uid, which can read " + ", ".join(exposed) + "; mount the "
                        "verifier data under a root-only directory (tests/docker-compose.yaml uses /private/data "
                        "inside a 0700 /private)")
            deadline = time.time() + args.timeout     # inference and the determinism rerun share one budget
            with _private_log(out / "infer.log") as log:
                rc, seconds, mode = run_call(sealed, inp, outp, work, env, args.timeout, log, mode, "inference")
            status["inference"] = {"exit_code": rc, "timed_out": rc is None, "isolation": mode, "seconds": round(seconds, 1),
                                   "isolation_detail": isolation.bwrap_detail() if mode == "bwrap" else
                                   ("package ran as uid %d; /private/data, /tests and the truth are root-only" % isolation.RUNNER_UID if mode == "uid" else "none")}
            if mode == "subprocess":
                status["inference"]["untrusted"] = "inference ran with the verifier's own privileges and could have read the held-out targets"
            status["stage"] = "determinism"
            left = deadline - time.time()
            status["determinism"] = (determinism_check(sealed, inp, outp, work, env, args.determinism_cases, min(left, 3600), mode)
                                     if left > 30 and rc is not None else
                                     {"checked": 0, "skipped": "inference timed out" if rc is None else "no budget left"})
            write(work / "inference-status.json", status)
            if args.phase == "infer":
                write(out / "status.json", status)
                return
        elif (work / "inference-status.json").exists():
            status.update(json.loads((work / "inference-status.json").read_text()))
        status["stage"] = "scoring"
        from score import score  # noqa: E402
        report = score(inp, truth, outp)
        write(out / "report.json", report)
        reward = report["reward"]
        payload = {"reward": reward, "isolation": status.get("inference", {}).get("isolation")}
        bounds = normalised(reward)
        if bounds is not None:
            payload["normalised_score"] = bounds["normalised_score"]
            payload["bounds"] = {"lower": bounds["lower"], "upper": bounds["upper"]}
        write(out / "reward.json", harbor_reward(payload))
        (out / "reward.txt").write_text(f"{reward}\n")
        status.update({"stage": "done", "reward": report["reward"], "mean_vrmse": report["mean_vrmse"],
                       "n_failed": report["n_failed"], "failure_reasons": sorted({f["reason"] for f in report["failures"]}),
                       "groups": report["groups"]})
        write(out / "status.json", status)
        print(json.dumps({k: status[k] for k in ("reward", "mean_vrmse", "n_cases", "n_failed", "inference")}))
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
