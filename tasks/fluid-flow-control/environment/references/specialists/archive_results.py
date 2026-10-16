"""Archive completed manual-run evidence; never certify performance automatically."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil


FILES = ("config.json", "protocol.json", "author-source.json", "launch.json",
         "completed.json", "workflow-completed.json", "episodes.jsonl",
         "evaluation-steps.jsonl", "updates.jsonl", "candidate.pt", "best.pt")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_final_metrics(final, independent):
    metrics = ("return", "mean_drag", "rms_lift") if "mean_drag" in final else ("return",)
    for key in metrics:
        for row in (final, independent):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Nonfinite or invalid final metric: {key}")
        if abs(final[key] - independent[key]) > 1e-5:
            raise ValueError(f"Independent final replay disagrees: {key}")


def archive(runs, output):
    # Validate all sources before creating the destination. These workflows only
    # publish completion after fresh-process final/best CFD evaluation exits.
    summaries = []
    for run in runs:
        cfg = json.loads((run / "config.json").read_text())
        workflow = json.loads((run / "workflow-completed.json").read_text())
        if workflow["status"] not in ("trained-and-independently-evaluated-not-paper-certified",
                                      "cavity-trained-and-replayed-not-paper-matched"):
            raise ValueError(f"Incomplete independent workflow: {run}")
        rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
        training = [row for row in rows if row["phase"] == "training"]
        if sum(row["steps"] for row in training) != cfg["training_steps"]:
            raise ValueError(f"Training-step evidence mismatch: {run}")
        for name in ("candidate.pt", "best.pt"):
            if digest(run / name) not in workflow["policy_sha256"]:
                raise ValueError(f"Unevaluated checkpoint: {run / name}")
        final = [row for row in rows if row["phase"] == "evaluation"][-1]
        independent = [json.loads(line) for line in (run / "independent-final/episodes.jsonl").read_text().splitlines()][-1]
        validate_final_metrics(final, independent)
        summaries.append({"run": run.name, "algorithm": cfg["algorithm"], "seed": cfg["seed"],
                          "recipe": cfg.get("recipe", "public"), "training_steps": cfg["training_steps"],
                          "training_episodes": len(training), "final_evaluation": final,
                          "paper_performance": "not-matched"})
    if output.exists():
        raise ValueError("Archive destination exists; choose a new version instead of overwriting")
    output.mkdir(parents=True)
    checksums = {}
    for run in runs:
        target = output / run.name
        target.mkdir()
        selected = [run / name for name in FILES]
        for directory in ("independent-final", "independent-best"):
            folder = run / directory
            if folder.exists():
                selected += [folder / name for name in ("completed.json", "episodes.jsonl",
                                                         "evaluation-steps.jsonl", "protocol.json")]
        source = run / "source"
        for path in sorted(source.rglob("*")):
            if "__pycache__" in path.relative_to(source).parts or not path.is_file():
                continue
            if not path.resolve().is_relative_to(source.resolve()):
                raise ValueError(f"Source snapshot has an escaping link: {path}")
            selected.append(path)
        for path in selected:
            relative = path.relative_to(run)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            before = digest(path)
            shutil.copyfile(path, destination)
            if digest(destination) != before or digest(path) != before:
                raise ValueError(f"Source changed while archiving: {path}")
            checksums[destination.relative_to(output).as_posix()] = before
    manifest = {"status": "completed-manual-runs-not-accepted-paper-baselines",
                "resource_revision": "cf5ef6748443ae72e9647f398487e98d421a7243",
                "runs": summaries, "sha256": checksums}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return {"runs": len(runs), "files": len(checksums), "path": str(output), "status": manifest["status"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps(archive(args.runs, args.output)))


if __name__ == "__main__":
    main()
