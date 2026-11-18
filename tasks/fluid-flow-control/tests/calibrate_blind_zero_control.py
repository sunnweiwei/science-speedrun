"""Generate sealed blind perturbation seeds and calibrate paired zero-control anchors on GPU.

CPU mode (`--generate-seeds`):
  Generates 3 distinct private perturbation seeds (!= 123) per case using cryptographic randomness
  and writes `tests/sealed/blind_flow_seeds.json` (`calibrated: false` until GPU replay runs).

GPU calibration mode (`--calibrate --data /data --output /tmp/flow-zero-calibration`):
  Builds a temporary zero-action `hydrogym-policy/v1` submission package, replays the 26-case
  verifier suite (or `--case all`) across the 3 blind seeds per case using `evaluate_native`,
  verifies finite multi-seed stability, updates `zero_control_anchors` + `calibrated: true` in
  `blind_flow_seeds.json`, and writes an `ssi_index_v4_inputs` flow_cases patch receipt.
"""
import argparse
import json
import math
import os
from pathlib import Path
import secrets
import sys
import tempfile

TASK_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = TASK_ROOT / "environment" / "evaluation"
sys.path.insert(0, str(EVAL_DIR))

from aggregate import anchors as load_default_anchors  # noqa: E402
from case_spec import VERIFIER_CASES, load_release, select_cases  # noqa: E402
from evaluate import write_json  # noqa: E402
from evaluate_native import evaluate_native  # noqa: E402

DEFAULT_BLIND_SEEDS = TASK_ROOT / "tests" / "sealed" / "blind_flow_seeds.json"
DEV_SEED = 123


def generate_blind_seed_roster(seeds_per_case=3, rng=None):
    if seeds_per_case < 1:
        raise ValueError("seeds_per_case must be >= 1")
    rng = rng or secrets.SystemRandom()
    release = load_release()
    used_globally = {DEV_SEED}
    cases = {}
    for row in sorted(release["cases"], key=lambda r: r["id"]):
        case_seeds = []
        while len(case_seeds) < seeds_per_case:
            candidate = rng.randrange(100_000, 2_000_000_000)
            if candidate not in used_globally:
                used_globally.add(candidate)
                case_seeds.append(candidate)
        cases[row["id"]] = case_seeds
    return {
        "schema": "hydrogym-blind-seeds/v1",
        "seeds_per_case": seeds_per_case,
        "calibrated": False,
        "cases": cases,
        "zero_control_anchors": {},
    }


def build_zero_action_package(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "README.md").write_text(
        "# Zero-Action Calibration Controller\n\n"
        "Emits exact native zero actions (clamped into [action_low, action_high]) for zero-control calibration.\n"
    )
    (dest / "policy.py").write_text(
        "import json, sys\n"
        "low = high = None\n"
        "for line in sys.stdin:\n"
        "    req = json.loads(line)\n"
        "    op = req['op']\n"
        "    rep = {'request_id': req['request_id']}\n"
        "    if op == 'reset':\n"
        "        low, high = req['action_low'], req['action_high']\n"
        "        rep['ready'] = True\n"
        "    elif op == 'act':\n"
        "        rep['action'] = [max(lo, min(hi, 0.0)) for lo, hi in zip(low, high)]\n"
        "    elif op == 'close':\n"
        "        rep['closed'] = True\n"
        "        print(json.dumps(rep), flush=True)\n"
        "        break\n"
        "    print(json.dumps(rep), flush=True)\n"
    )
    infer = dest / "infer.sh"
    infer.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec python "$(dirname "$0")/policy.py"\n')
    infer.chmod(0o755)
    return dest


def calibrate_zero_controls(data: Path, seeds_file: Path, output: Path, case_names=None):
    if not seeds_file.is_file():
        raise FileNotFoundError(f"Blind seeds file not found: {seeds_file}")
    case_names = case_names or ["verifier"]
    os.environ["HYDROGYM_BLIND_SEEDS_FILE"] = str(seeds_file.resolve())
    _, selected_cases = select_cases(case_names, blind_seeds_path=seeds_file)

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zero-policy-") as pkg_tmp:
        zero_pkg = build_zero_action_package(Path(pkg_tmp) / "submission")
        replay_out = output / "replay"
        result = evaluate_native(data.resolve(), zero_pkg, replay_out, case_names)

    if result.get("status") != "ok":
        raise RuntimeError(f"Zero-control blind replay did not succeed cleanly: {result.get('status')}")

    prev_anchors = load_default_anchors()
    roster = json.loads(seeds_file.read_text())
    calibrated = dict(roster.get("zero_control_anchors") or {})
    flow_cases_patch = {}
    stability_rows = {}

    for case in selected_cases:
        cid = case["id"]
        row = result["cases"][cid]
        pm = row["primary_metric"]
        if pm.get("status") != "ok" or not math.isfinite(float(pm["value"])):
            raise RuntimeError(f"Nonfinite zero-control primary metric on {cid}: {pm}")
        val = float(pm["value"])
        ep_vals = [float(ep["primary_metric"]["value"]) for ep in row["episodes"]]
        calibrated[cid] = val
        prev_z = float(prev_anchors[cid]["zero"])
        rel_shift = abs(val - prev_z) / abs(prev_z) if prev_z != 0 else abs(val)
        stability_rows[cid] = {
            "blind_seeds": case["seeds"],
            "per_seed_values": ep_vals,
            "blind_mean_zero": val,
            "dev_seed123_zero": prev_z,
            "relative_shift_vs_seed123": rel_shift,
        }
        if cid in VERIFIER_CASES:
            flow_cases_patch[cid] = {
                "direction": pm["direction"],
                "metric": pm["id"],
                "zero": val,
            }

    roster["zero_control_anchors"] = calibrated
    if set(VERIFIER_CASES).issubset(calibrated):
        roster["calibrated"] = True
    write_json(seeds_file, roster)
    receipt = {
        "schema": "hydrogym-blind-zero-calibration/v1",
        "calibrated_verifier_complete": bool(roster["calibrated"]),
        "seeds_file": str(seeds_file),
        "stability": stability_rows,
        "flow_cases": dict(sorted(flow_cases_patch.items())),
    }
    write_json(output / "blind_zero_calibration_receipt.json", receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds-file", type=Path, default=DEFAULT_BLIND_SEEDS)
    p.add_argument("--seeds-per-case", type=int, default=3)
    p.add_argument("--generate-seeds", action="store_true",
                   help="Generate fresh cryptographic blind seeds and write --seeds-file")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow overwriting an existing --seeds-file when generating seeds")
    p.add_argument("--calibrate", action="store_true",
                   help="Run GPU zero-control replay across blind seeds and populate zero_control_anchors")
    p.add_argument("--data", type=Path, default=Path("/data"))
    p.add_argument("--output", type=Path, default=Path("/tmp/blind-zero-calibration"))
    p.add_argument("--case", action="append", help="Default: verifier (26 cases); or repeat / use all")
    args = p.parse_args()

    if not args.generate_seeds and not args.calibrate:
        p.error("Specify --generate-seeds and/or --calibrate")

    if args.generate_seeds:
        if args.seeds_file.exists() and not args.overwrite:
            raise FileExistsError(f"{args.seeds_file} already exists; pass --overwrite to regenerate")
        args.seeds_file.parent.mkdir(parents=True, exist_ok=True)
        roster = generate_blind_seed_roster(seeds_per_case=args.seeds_per_case)
        write_json(args.seeds_file, roster)
        print(json.dumps({
            "status": "generated",
            "seeds_file": str(args.seeds_file),
            "case_count": len(roster["cases"]),
            "seeds_per_case": roster["seeds_per_case"],
        }, indent=2))

    if args.calibrate:
        receipt = calibrate_zero_controls(args.data, args.seeds_file, args.output, args.case)
        print(json.dumps({
            "status": "calibrated",
            "calibrated_verifier_complete": receipt["calibrated_verifier_complete"],
            "cases_calibrated": len(receipt["stability"]),
        }, indent=2))


if __name__ == "__main__":
    main()
