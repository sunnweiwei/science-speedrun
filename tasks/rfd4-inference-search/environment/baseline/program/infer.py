"""Runnable single-shot RFD4 baseline for a request of any supported monomer length."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def baseline_command(request, source, run):
    count = request["num_outputs"]
    model = Path(os.environ["RFD4_MODEL"])
    package = Path(__file__).resolve().parent
    if (package / "reference_policy.py").is_file() and (package / "policy.json").is_file():
        # This adapter also replays packages saved in the original policy format.
        import yaml
        from reference_policy import load_policy, native_config

        length = request["input"]["length"]
        plan = load_policy(package, [length])[length]
        width = plan["width"]
        native_slots = ((count + width - 1) // width) * width
        cfg = native_config(plan, length, native_slots, request["seed"], model.parent, run,
                            input_file=source)
        config_path = run.parent / "reference.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        command = ["rfd4-generate", "--config-dir", str(run.parent), "--config-name", "reference"]
        priorities = {p["step"]: p["priority"] for p in plan["checkpoints"] if "priority" in p}
        if priorities:
            (run.parent / "priorities.json").write_text(json.dumps(priorities))
            command[:1] = [sys.executable, str(package / "native_worker.py")]
        if not plan["checkpoints"]:
            command.append("~metrics")
        return command, native_slots
    preset_file = package / "reference-preset.json"
    level = json.loads(preset_file.read_text()) if preset_file.exists() else "medium"
    if level not in ("low", "medium", "high"):
        raise ValueError("Unknown one-shot reference preset")
    return (["rfd4-generate", f"recycle_strategy={level}/front_loaded",
             f"inputs={source}", f"out_dir={run}", f"hf_ckpt_path={model}",
             f"num_replicates={count}", f"seed={request['seed']}", f"batch_size={min(count, 5)}",
             "compile_lengths=null", "num_workers=0", "skip_existing=false", "~metrics"], count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if request["version"] != 1 or request["kind"] != "unconditional_monomer":
        parser.exit(64, "Unsupported request version or kind\n")
    count = request["num_outputs"]
    output = args.output.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        source = (args.request.parent / request["input"]["file"]).resolve()
        run = Path(tmp) / "generation"
        command, native_slots = baseline_command(request, source, run)
        subprocess.run(command, cwd=tmp, check=True)
        with (run / "inference_metrics/per_example_metrics.csv").open() as f:
            paths = sorted(Path(row["designed_structure_path"]) for row in csv.DictReader(f))
        if len(paths) != native_slots or len(set(paths)) != native_slots:
            raise RuntimeError("Incomplete generation manifest")
        for idx, path in enumerate(paths[:count]):
            shutil.copyfile(path, output / f"slot-{idx:06d}.cif")


if __name__ == "__main__":
    main()
