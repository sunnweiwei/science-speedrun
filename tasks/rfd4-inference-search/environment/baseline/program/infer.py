"""Runnable RFD4 baseline for unified conditioned design (all 4 Figure 2 tracks) and monomer requests."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

AA_COMPS = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "UNK",
})
DNA_COMPS = frozenset({"DA", "DC", "DG", "DT", "DI", "DU"})


def detect_s8_schedule(source: Path, level: str) -> str:
    """Select the official S8.3 recycling schedule from input mmCIF chemical composition."""
    import gemmi
    import zstandard

    if source.name.endswith(".zst"):
        with source.open("rb") as f:
            with zstandard.ZstdDecompressor().stream_reader(f) as reader:
                text = reader.read().decode("utf-8")
    else:
        text = source.read_text(encoding="utf-8")
    block = gemmi.cif.read_string(text).sole_block()
    comps = {gemmi.cif.as_string(v) for v in block.find_values("_atom_site.label_comp_id")}
    if comps & DNA_COMPS:
        return "high/flat" if level == "high" else f"{level}/front_loaded"
    non_polymer_ligands = {c for c in comps if c not in AA_COMPS and "-" not in c}
    if non_polymer_ligands:
        return f"{level}/midpeak"
    return f"{level}/front_loaded"


def baseline_command(request, source, run):
    count = request["num_outputs"]
    model = Path(os.environ["RFD4_MODEL"])
    package = Path(__file__).resolve().parent
    if request["kind"] == "unconditional_monomer" and (package / "reference_policy.py").is_file() and (package / "policy.json").is_file():
        # This adapter also replays packages saved in the original monomer policy format.
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
    if request["kind"] == "conditioned_design":
        schedule = detect_s8_schedule(source, level)
        bsize = min(count, 10)
    else:
        schedule = f"{level}/front_loaded"
        bsize = min(count, 5)
    return (["rfd4-generate", f"recycle_strategy={schedule}",
             f"inputs={source}", f"out_dir={run}", f"hf_ckpt_path={model}",
             f"num_replicates={count}", f"seed={request['seed']}", f"batch_size={bsize}",
             "compile_lengths=null", "num_workers=0", "skip_existing=false", "~metrics"], count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if request.get("version") != 1 or request.get("kind") not in ("unconditional_monomer", "conditioned_design"):
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
