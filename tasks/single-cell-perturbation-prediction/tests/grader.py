#!/usr/bin/env python3
"""Trusted scorer. Reads sealed predictions + private responses; never executes submission code.

Writes to --output:
  reward.json          {"reward": mean over test contexts of the FINAL-subset score, ...}
  report.json          full per-context / per-subset / per-stratum report (operator-facing)
  agent_feedback.json  leaderboard-subset scores only (what the runner may return to the agent)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cellscore import cell_eval_version, score_context, sha256


def harbor_reward(values):
    """Harbor parses reward.json as a flat object of finite numbers; strings, None and nested details
    belong in report.json / status.json."""
    import math as _math
    return {k: v for k, v in values.items() if isinstance(v, (int, float)) and _math.isfinite(v)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed", required=True, type=lambda x: Path(x).resolve())
    ap.add_argument("--benchmark", required=True, type=lambda x: Path(x).resolve())
    ap.add_argument("--private", required=True, type=lambda x: Path(x).resolve())
    ap.add_argument("--output", required=True, type=lambda x: Path(x).resolve())
    ap.add_argument("--wait-seconds", type=int, default=14400)
    ap.add_argument("--threads", type=int)
    args = ap.parse_args()
    ready = args.sealed.parent / "READY"
    deadline = time.monotonic() + args.wait_seconds
    while not ready.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("prediction snapshot was not published")
        time.sleep(2)
    snapshot_path = args.sealed / "snapshot.json"
    if ready.read_text().strip() != sha256(snapshot_path):
        raise ValueError("snapshot ready hash mismatch")
    snapshot = json.loads(snapshot_path.read_text())
    contexts = sorted(p.name for p in args.private.iterdir() if (p / "responses.h5ad").exists())
    report = {"schema": "virtual-cell-zeroshot-report/v1", "cell_eval_version": cell_eval_version(),
              "evaluation_mode": "isolated_model_replay", "contexts": {}}
    finals, boards = [], []
    scratch = args.output / "scratch"
    for ctx in contexts:
        priv = args.private / ctx
        pm = json.loads((priv / "manifest.json").read_text())
        if pm["responses_sha256"] != sha256(priv / "responses.h5ad"):
            raise ValueError(f"{ctx}: private response checksum mismatch")
        public = args.benchmark / ctx / "public"
        entry = snapshot["contexts"].get(ctx)
        if pm["requests_sha256"] != sha256(public / "requests.jsonl") or pm["public_basal_sha256"] != sha256(public / "basal.h5ad"):
            raise ValueError(f"{ctx}: public inputs differ from the private manifest")
        if not entry or entry.get("failed"):
            report["contexts"][ctx] = {"status": "failed", "error": (entry or {}).get("error", "missing"),
                                      "tier": pm["tier"], "final_score": 0.0, "leaderboard_score": 0.0}
            finals.append(0.0); boards.append(0.0)
            continue
        pred = args.sealed / ctx / "predictions.h5ad"
        if entry["predictions_sha256"] != sha256(pred) or entry["requests_sha256"] != sha256(public / "requests.jsonl") \
                or entry["basal_sha256"] != sha256(public / "basal.h5ad"):
            raise ValueError(f"{ctx}: sealed snapshot provenance mismatch")
        subsets = {"leaderboard": [r["request_id"] for r in pm["requests"] if r["subset"] == "leaderboard"],
                   "final": [r["request_id"] for r in pm["requests"] if r["subset"] == "final"]}
        groups = {"stratum": {r["request_id"]: r["stratum"] for r in pm["requests"]},
                  "target_seen": {r["request_id"]: ("withheld" if r["withheld"] else ("seen" if r["seen_in_training"] else "unseen"))
                                  for r in pm["requests"]},
                  "final_by_stratum": {r["request_id"]: r["stratum"] for r in pm["requests"] if r["subset"] == "final"}}
        baseline_dir = priv / "noeffect" if (priv / "noeffect").exists() else None
        res = score_context(pred, priv / "basal.h5ad", priv / "responses.h5ad", public / "requests.jsonl",
                            scratch / ctx, subsets=subsets, groups=groups, threads=args.threads, baseline_dir=baseline_dir)
        final = res["subsets"]["final"]["score"]; board = res["subsets"]["leaderboard"]["score"]
        report["contexts"][ctx] = {"status": "ok", "tier": pm["tier"], "chemistry": pm["chemistry"],
                                   "final_score": final, "leaderboard_score": board, "all_score": res["all"]["score"],
                                   "final": res["subsets"]["final"], "leaderboard": res["subsets"]["leaderboard"],
                                   "all": res["all"], "groups": res["groups"], "inference_seconds": entry.get("seconds")}
        finals.append(final); boards.append(board)
    reward = float(np.mean(finals)) if finals else 0.0
    report["reward"] = reward
    report["leaderboard_reward"] = float(np.mean(boards)) if boards else 0.0
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "agent_feedback.json").write_text(json.dumps({
        "schema": "virtual-cell-zeroshot-feedback/v1", "cell_eval_version": report["cell_eval_version"],
        "leaderboard_reward": report["leaderboard_reward"],
        "contexts": {c: {"status": v["status"], "tier": v["tier"], "leaderboard_score": v["leaderboard_score"],
                         "leaderboard": v.get("leaderboard"), "error": v.get("error")} for c, v in report["contexts"].items()},
    }, indent=2, sort_keys=True) + "\n")
    (args.output / "reward.json").write_text(json.dumps(harbor_reward({
        "reward": reward, "leaderboard_reward": report["leaderboard_reward"],
        **{f"{c}_final": v["final_score"] for c, v in report["contexts"].items()},
        **{f"{c}_leaderboard": v["leaderboard_score"] for c, v in report["contexts"].items()},
    }), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
