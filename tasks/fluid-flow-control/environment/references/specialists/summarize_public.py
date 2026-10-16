"""Read-only progress and fixed-window physical comparisons for manual runs."""

import argparse
import json
from pathlib import Path
import statistics


def grouping_key(config, protocol, step):
    """Pool only distinct seeds of the same learner and physical protocol."""
    training = {k: v for k, v in config.items() if k != "seed"}
    physics = {k: v for k, v in protocol.items() if k not in ("seed", "source_path", "status")}
    return json.dumps({"config": training, "physics": physics, "evaluation_step": step}, sort_keys=True)


def records(path):
    if not path.exists():
        return []
    rows = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A live append can end with an incomplete final record.
            if index != len(lines) - 1:
                raise
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--zero", type=Path, required=True)
    args = parser.parse_args()
    zero = records(args.zero / "episodes.jsonl")[-1]
    if "last_half_mean_drag" not in zero:
        parser.error("Zero-control record lacks the fixed last-half window; use a fully logged zero-control run")
    zero_protocol = json.loads((args.zero / "protocol.json").read_text())
    grouped = {}
    for run in args.runs:
        cfg_path = run / "config.json"
        if not cfg_path.exists():
            print(json.dumps({"run": str(run), "status": "config-not-yet-written"}))
            continue
        cfg = json.loads(cfg_path.read_text())
        protocol = json.loads((run / "protocol.json").read_text())
        for field in ("environment", "effective_substeps", "effective_omega", "horizon"):
            if protocol[field] != zero_protocol[field]:
                raise ValueError(f"Zero-control comparison uses different {field}: {run}")
        episodes = records(run / "episodes.jsonl")
        training = [r for r in episodes if r["phase"] == "training"]
        evaluations = [r for r in episodes if r["phase"] == "evaluation"]
        latest = evaluations[-1] if evaluations else None
        best = max(evaluations, key=lambda r: r["return"]) if evaluations else None
        report = {"run": run.name, "algo": cfg["algorithm"], "recipe": cfg.get("recipe", "public"), "seed": cfg["seed"],
                  "completed": (run / "completed.json").exists(),
                  "training_episodes": len(training), "training_steps": sum(r["steps"] for r in training)}
        for label, row in (("latest", latest), ("best_development", best)):
            if row is not None:
                metrics = {"training_step": row["training_step"], "raw_return": row["return"],
                           "reward_per_action": row["return"] / row["steps"],
                           "mean_drag": row["mean_drag"], "rms_lift": row["rms_lift"],
                           "full_window_drag_reduction": 1 - row["mean_drag"] / zero["mean_drag"],
                           "last_half_drag_reduction": 1 - row["last_half_mean_drag"] / zero["last_half_mean_drag"]}
                report[label] = metrics
        print(json.dumps(report, allow_nan=False))
        if latest:
            grouped.setdefault(grouping_key(cfg, protocol, latest["training_step"]), []).append(report)
    for encoded, reports in grouped.items():
        key = json.loads(encoded)
        rows = [report["latest"] for report in reports]
        if len(rows) < 2:
            continue
        seeds = [report["seed"] for report in reports]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Repeated training seeds cannot be pooled as independent replication")
        # Same checkpoint budget only. These are training-seed statistics, not CFD reset seeds.
        summary = {"algorithm": key["config"]["algorithm"], "recipe": key["config"].get("recipe", "public"),
                   "training_step": key["evaluation_step"], "seeds": seeds, "n_runs": len(rows)}
        for key in ("raw_return", "full_window_drag_reduction", "last_half_drag_reduction"):
            values = [row[key] for row in rows]
            summary[key] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}
        print(json.dumps({"training_seed_summary": summary}, allow_nan=False))


if __name__ == "__main__":
    main()
