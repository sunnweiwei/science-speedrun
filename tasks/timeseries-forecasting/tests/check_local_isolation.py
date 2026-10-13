#!/usr/bin/env python3
"""Probe the boundary the verifier gives a package: run a canary package through main_verifier.py and report what it could reach.

    python3 tests/check_local_isolation.py --data /large-disk/timeseries --out /tmp/isolation-probe [--manifest environment/data/split_smoke.json]
    python3 tests/check_local_isolation.py ... --expect-leak     # control run without isolation: must detect the leak

The canary's infer.sh writes valid (zero) forecasts and a probe.json recording, for each target, whether the package
could reach it (search permission on a directory, read permission on a file). The targets are baked into the package
before it is sealed, as absolute paths on the verifier's side, because a package receives no environment from the
verifier:

- in every mode: the data directory, the labels (work/truth), the full staged input (work/input), the previous call's
  output, and whether the sealed package itself is writable;
- in Bubblewrap mode also the verifier's work directory, the host's /tmp, /workspace, the task repository and the
  network. uid mode has no mount or
  network namespace, so there those are the container's to restrict (/tests is 0700 and network_mode is none in the
  compose overlay), and they are recorded but not asserted.

It exits 0 when every asserted target was out of reach (with --expect-leak: when the data or the labels were reachable,
which is what a run without isolation must show), and writes the result to OUT/isolation_probe.json.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANARY = r'''#!/usr/bin/env python3
import json, os, socket, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, ts_io
inp, out = sys.argv[1], sys.argv[2]
targets = json.load(open(os.path.join(HERE, "probe_targets.json")))
index = json.load(open(os.path.join(inp, "index.json")))
cohort = int(index.get("cohort", 0))
if cohort > 0:
    targets["previous_call_output"] = os.path.join(targets["work"], "cohorts", f"{cohort - 1:03d}", "output")
def reach(path):
    if not path or not os.path.lexists(path):
        return False
    return os.access(path, os.X_OK if os.path.isdir(path) else os.R_OK)
probe = {"cohort": cohort, "cwd": os.getcwd(), "uid": os.getuid(), "targets": {k: {"path": v, "reachable": reach(v)} for k, v in targets.items()}}
try:
    s = socket.create_connection(("1.1.1.1", 53), timeout=3); s.close(); probe["network"] = "reachable"
except Exception as exc:  # noqa: BLE001
    probe["network"] = f"blocked: {type(exc).__name__}"
try:
    open(os.path.join(HERE, "canary-write-test"), "w").write("x"); probe["package_writable"] = True
except Exception:  # noqa: BLE001
    probe["package_writable"] = False
for cfg in index["configs"]:
    case = ts_io.CaseFile(os.path.join(inp, "cases", cfg["config_id"] + ".h5"))
    n, d, h = case.n, case.n_variates, case.prediction_length
    ts_io.write_prediction_file(os.path.join(out, "cases", cfg["config_id"] + ".h5"), np.zeros((n, d, h), np.float32), np.zeros((n, d, 9, h), np.float32))
    case.close()
json.dump(probe, open(os.path.join(out, "probe.json"), "w"), indent=1)
'''
EVERY_MODE = ("data", "truth", "staged_input", "previous_call_output")
BWRAP_ONLY = ("work", "host_tmp_marker", "workspace", "task_repo")   # uid mode must traverse work/ to reach its own call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "environment" / "data" / "split_smoke.json"))
    ap.add_argument("--expect-leak", action="store_true", help="control run without isolation: succeed only if the probe sees the data or labels")
    args = ap.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    pkg = out / "canary"; pkg.mkdir(exist_ok=True)
    (pkg / "predict.py").write_text(CANARY)
    (pkg / "infer.sh").write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec python3 "$(dirname "$(readlink -f "$0")")/predict.py" "$1" "$2"\n')
    (pkg / "ts_io.py").write_text((ROOT / "environment" / "evaluation" / "ts_io.py").read_text())
    (pkg / "model_manifest.json").write_text(json.dumps({"name": "isolation-canary", "contract": "timeseries-forecast/v1", "parameters": 0,
                                                         "pretrained_assets": [], "sources_used": []}))
    marker = Path("/tmp/timeseries-isolation-marker")
    marker.unlink(missing_ok=True)   # another user's file in sticky /tmp cannot be reopened, even by root (protected_regular)
    marker.write_text("host tmp\n"); os.chmod(marker, 0o644)
    targets = {"data": str(Path(args.data).resolve()), "truth": str(work / "truth"), "staged_input": str(work / "input"),
               "work": str(work), "host_tmp_marker": str(marker), "workspace": "/workspace", "task_repo": str(ROOT)}
    (pkg / "probe_targets.json").write_text(json.dumps(targets, indent=1))
    if os.geteuid() == 0:        # the runner uid must be able to traverse to the package and the work directory
        for p in [out, *out.parents]:
            if str(p) in ("/", "/tmp"):
                break
            os.chmod(p, os.stat(p).st_mode | 0o011)
    proc = subprocess.run([sys.executable, str(ROOT / "tests" / "main_verifier.py"), "--submission", str(pkg), "--data", args.data,
                           "--manifest", args.manifest, "--output", str(out / "verifier"), "--work", str(work), "--timeout", "1800",
                           "--determinism-configs", "0"], capture_output=True, text=True)
    status = json.loads((out / "verifier" / "status.json").read_text())
    probes = [json.loads(p.read_text()) for p in sorted((work / "cohorts").glob("*/output/probe.json"))]
    if not probes:
        print(proc.stdout[-2000:], proc.stderr[-2000:]); sys.exit("no probe output: the canary did not run")
    modes = status.get("isolation") or []
    asserted = EVERY_MODE + (BWRAP_ONLY if modes == ["bwrap"] else ())
    reached = sorted({k for p in probes for k, t in p["targets"].items() if t["reachable"] and k in asserted})
    result = {"isolation_modes": modes, "verifier_exit": proc.returncode, "reward": status.get("reward"), "n_probe_calls": len(probes),
              "asserted_targets": list(asserted), "reached": reached,
              "reachable_by_target": {k: any(p["targets"].get(k, {}).get("reachable") for p in probes) for k in (*EVERY_MODE, *BWRAP_ONLY)},
              "network": sorted({p["network"] for p in probes}), "package_writable": any(p["package_writable"] for p in probes),
              "runner_uid": sorted({p["uid"] for p in probes})}
    network_ok = modes != ["bwrap"] or all(n.startswith("blocked") for n in result["network"])
    result["passed"] = not reached and not result["package_writable"] and network_ok
    if args.expect_leak:
        result["control"] = "a run without isolation must reach the data or the labels"
        result["passed"] = bool({"data", "truth"} & set(reached))
    (out / "isolation_probe.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
