"""Run fixed physical calculations against a model's ASE calculator.

This is a development evaluator, not yet a sandbox for untrusted submissions.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import time

from common import PROTOCOL, digest, indexed, read_rows, with_oom_retry, write_json
from physics import CALCULATIONS


def load_calculator(factory: str, artifact: Path | None, device: str):
    if factory == "uma":
        if artifact is None or not artifact.is_file():
            raise ValueError("UMA requires --artifact pointing to a local checkpoint")
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.calculate.pretrained_mlip import load_predict_unit

        unit = load_predict_unit(
            str(artifact),
            device=device,
            inference_settings="default",
        )
        return FAIRChemCalculator(unit, task_name="omat")
    if factory == "submission":
        # Contract: ARTIFACT is the submission directory; calculator.py defines
        # make_calculator(submission_dir, device) returning an ASE calculator (E/F/S).
        spec = importlib.util.spec_from_file_location("submission_calculator", Path(artifact) / "calculator.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.make_calculator(Path(artifact), device)
    if factory == "emt":
        from ase.calculators.emt import EMT

        return EMT()  # plumbing-only baseline; not a universal materials model
    module, function = factory.split(":", 1)
    return getattr(importlib.import_module(module), function)(artifact, device)


def run(
    inputs: Path,
    output: Path,
    factory,
    artifact=None,
    device="cpu",
    energy_audit=None,
    calculator=None,
    verbose=True,
):
    cases = read_rows(inputs)
    indexed(cases)
    if output.exists():
        raise FileExistsError(f"Refusing to mix/overwrite predictions: {output}")
    output.mkdir(parents=True)
    versions = {}
    for package in ("numpy", "ase", "pymatgen", "phonopy", "fairchem-core", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = {
        "protocol": PROTOCOL,
        "inputs_sha256": digest(inputs),
        "factory": factory,
        "checkpoint_sha256": digest(artifact) if artifact and artifact.exists() else None,
        "device": device,
        "platform": platform.platform(),
        "versions": versions,
        "cases": len(cases),
        "complete": False,
    }
    if energy_audit is not None:
        audit = json.loads(Path(energy_audit).read_text())
        if audit.get("checkpoint_sha256") != metadata["checkpoint_sha256"]:
            raise ValueError("Energy-convention audit belongs to another checkpoint")
        metadata["energy_audit"] = audit
        metadata["energy_audit_sha256"] = digest(Path(energy_audit))
    write_json(output / "run.json", metadata)
    if calculator is None:
        calculator = load_calculator(factory, artifact, device)
    with (output / "predictions.jsonl").open("w") as stream:
        for case in cases:
            started = time.monotonic()
            record = {"id": case["id"], "suite": case["suite"]}
            try:
                def calculate():
                    calculator.reset()
                    return CALCULATIONS[case["suite"]](case, calculator)
                record.update(with_oom_retry(calculate))
                record["status"] = "ok"
                json.dumps(record, allow_nan=False)
            except Exception as error:
                # Preserve the failure as a record. Scoring rejects incomplete runs.
                record = {
                    "id": case["id"],
                    "suite": case["suite"],
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            record["seconds"] = time.monotonic() - started
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            if verbose:
                print(
                    f"{case['id']}: {record['status']} ({record['seconds']:.1f}s)",
                    flush=True,
                )
    metadata["complete"] = True
    metadata["predictions_sha256"] = digest(output / "predictions.jsonl")
    write_json(output / "run.json", metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factory", default="uma", help="uma, emt, or module:function")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--energy-audit",
        type=Path,
        help="Operator-reviewed training provenance for MP-compatible raw energies",
    )
    args = parser.parse_args()
    run(
        args.inputs,
        args.output,
        args.factory,
        args.artifact,
        args.device,
        args.energy_audit,
    )
