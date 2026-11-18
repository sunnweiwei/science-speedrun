#!/usr/bin/env python3
"""Measure the exact unrounded ZeroForce + D3(BJ) bulk-modulus MAPE anchor on the MOF panel.

Why this script exists:
In `environment/evaluation/extra.py` (lines 191-196), the MOF mechanics driver wraps every
submitted calculator with:
    SumCalculator([
        calc,
        TorchDFTD3Calculator(device=..., damping="bj", xc="pbe", cutoff=40.0 * units.Bohr),
    ])
Consequently, a trivial zero-force MLIP (`E = 0`, `F = 0`, `stress = 0`) does NOT predict
`B_0 = 0 GPa` (which would give `100.0%` MAPE); instead, its total energy equals pure PBE-D3(BJ)
dispersion `E_D3(V)`, producing non-zero curvature / EOS bulk moduli (or capped 200% failures
where pure attractive dispersion collapses without short-range repulsion). Measuring the exact
unrounded `bulk_modulus_MAPE_pct` of `ZeroForceCalculator + D3(BJ)` on the held-out 60-MOF test
panel ensures that `ZeroForceCalculator` receives exact `0.0` (`|term| < 1e-15`) on the
`Metal–organic frameworks` component in both `aggregate.py` and `harness/ssi_index_v3.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
TASK_ROOT = TESTS_DIR.parent
EVALUATION = Path("/evaluation") if Path("/evaluation").is_dir() else TASK_ROOT / "environment" / "evaluation"
if str(EVALUATION) not in sys.path:
    sys.path.insert(0, str(EVALUATION))

import aggregate  # noqa: E402
import score_extra  # noqa: E402

RECEIPT_SCHEMA = "mof-zero-force-d3-anchor-receipt/v1"
D3_CUTOFF_BOHR = 40.0
D3_XC = "pbe"
D3_DAMPING = "bj"

try:
    from ase.calculators.calculator import Calculator as _ASECalculatorBase
    from ase.calculators.calculator import all_changes as _ALL_CHANGES
except ImportError:  # Host CPU unit-test fallback when ASE is only inside /opt/venv

    class _ASECalculatorBase:  # type: ignore[no-redef]
        implemented_properties: list[str] = []

        def __init__(self, **kwargs: Any) -> None:
            self.results: dict[str, Any] = {}
            self.atoms: Any = None

        def reset(self) -> None:
            self.results = {}

        def calculate(self, atoms=None, properties=None, system_changes=None) -> None:
            self.atoms = atoms

    _ALL_CHANGES = ["positions", "numbers", "cell", "pbc", "initial_charges", "initial_magmoms"]


class ZeroForceCalculator(_ASECalculatorBase):
    """Exact zero-energy, zero-force, zero-stress ASE calculator."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def calculate(
        self,
        atoms: Any = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        if properties is None:
            properties = self.implemented_properties
        if system_changes is None:
            system_changes = _ALL_CHANGES
        super().calculate(atoms, properties, system_changes)
        n_atoms = len(atoms) if atoms is not None else 0
        self.results = {
            "energy": 0.0,
            "free_energy": 0.0,
            "forces": np.zeros((n_atoms, 3), dtype=np.float64),
            "stress": np.zeros(6, dtype=np.float64),
        }


def build_zero_force_d3_calculator(
    device: str = "cuda",
    *,
    d3_calculator_factory: Callable[[str], Any] | None = None,
) -> Any:
    """Construct `SumCalculator([ZeroForceCalculator(), TorchDFTD3Calculator(...)])` matching extra.py."""
    zero_calc = ZeroForceCalculator()
    if d3_calculator_factory is not None:
        return d3_calculator_factory(device, zero_calc)
    from ase import units
    from ase.calculators.mixing import SumCalculator
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

    d3_calc = TorchDFTD3Calculator(
        device=device,
        damping=D3_DAMPING,
        xc=D3_XC,
        cutoff=D3_CUTOFF_BOHR * units.Bohr,
    )
    return SumCalculator([zero_calc, d3_calc])


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False, allow_nan=False)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def verify_zero_force_mof_anchor(mof_mape_pct: float) -> dict[str, Any]:
    """Verify that setting `mof_bulk_modulus_MAPE_pct = mof_mape_pct` gives exact 0.0 MOF skill."""
    if not (math.isfinite(mof_mape_pct) and mof_mape_pct > 0.0):
        raise ValueError(f"Invalid measured MOF zero-force MAPE (%): {mof_mape_pct!r}")
    anchors = aggregate.load_atom_zero_force_anchors(
        overrides={"mof_bulk_modulus_MAPE_pct": float(mof_mape_pct)}
    )
    zero_metrics = {
        "discovery_F1": aggregate.DISCOVERY_F1_BASE,
        **anchors,
        "mof_bulk_modulus_MAPE_pct": float(mof_mape_pct),
    }
    scored = aggregate.ssi_v4_score(zero_metrics, anchors=anchors)
    mof_term = float(scored["components"]["Metal\u2013organic frameworks"])
    signed_total = float(scored["signed_score"])
    total = float(scored["score"])
    if abs(mof_term) >= 1e-15:
        raise AssertionError(f"MOF zero-force component {mof_term!r} is not within 1e-15 of 0.0")
    if abs(signed_total) >= 1e-15 or abs(total) >= 1e-15:
        raise AssertionError(f"All-zero-force total score {signed_total!r} is not within 1e-15 of 0.0")
    return {
        "mof_component_at_measured_anchor": mof_term,
        "all_zero_force_signed_score": signed_total,
        "all_zero_force_clipped_score": total,
        "abs_mof_component_lt_1e_15": abs(mof_term) < 1e-15,
    }


def _default_mof_items(inputs_root: Path) -> list[tuple[str, Path]]:
    manifest = json.loads((inputs_root / "manifest.json").read_text())
    files = manifest["files"]
    return [(k, inputs_root / "structures" / v) for k, v in sorted(files.items())]


def _run_one_mof(
    key: str,
    structure_path: Path,
    calc: Any,
    mof_runner: Callable[[Path, Any], dict[str, Any]],
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        if hasattr(calc, "reset"):
            calc.reset()
        out = mof_runner(structure_path, calc)
        rec: dict[str, Any] = {"id": key, "status": "ok", **out}
    except Exception as exc:
        rec = {
            "id": key,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:],
        }
    rec["seconds"] = time.monotonic() - started
    return rec


def _gpu_worker(gpu_id: str, queue: Any, result_queue: Any, inputs_root: Path, device: str) -> None:
    if device.startswith("cuda") and gpu_id != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        worker_device = "cuda"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        worker_device = "cpu"
    import extra

    calc = build_zero_force_d3_calculator(device=worker_device)
    todo = dict(_default_mof_items(inputs_root))
    while (key := queue.get()) is not None:
        result_queue.put(_run_one_mof(key, todo[key], calc, extra.run_mof))


def measure_mof_zero_force_anchor(
    inputs: Path,
    references: Path,
    *,
    device: str = "cuda",
    gpus: str = "",
    workers_per_gpu: int = 1,
    d3_calculator_factory: Callable[[str], Any] | None = None,
    mof_runner: Callable[[Path, Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run ZeroForceCalculator + D3(BJ) across all MOF inputs and compute exact unrounded MAPE."""
    todo = _default_mof_items(inputs)
    if mof_runner is None:
        import extra

        mof_runner = extra.run_mof

    gpu_list = [g.strip() for g in gpus.split(",") if g.strip()] if gpus else []
    use_parallel = (
        len(gpu_list) > 0
        and (len(gpu_list) * max(1, workers_per_gpu) > 1)
        and d3_calculator_factory is None
        and mof_runner.__name__ == "run_mof"
    )

    preds: dict[str, dict[str, Any]] = {}
    t0 = time.monotonic()
    if use_parallel:
        ctx = mp.get_context("spawn")
        in_q = ctx.Queue()
        out_q = ctx.Queue()
        slots = [g for g in gpu_list for _ in range(max(1, workers_per_gpu))]
        for key, _ in todo:
            in_q.put(key)
        for _ in slots:
            in_q.put(None)
        procs = [ctx.Process(target=_gpu_worker, args=(g, in_q, out_q, inputs, device)) for g in slots]
        for proc in procs:
            proc.start()
        for _ in todo:
            rec = out_q.get()
            preds[rec["id"]] = rec
        for proc in procs:
            proc.join()
    else:
        calc = build_zero_force_d3_calculator(device=device, d3_calculator_factory=d3_calculator_factory)
        for key, structure_path in todo:
            preds[key] = _run_one_mof(key, structure_path, calc, mof_runner)

    wall_seconds = time.monotonic() - t0
    scored = score_extra.score_mof(preds, references)
    mape_pct = float(scored["bulk_modulus_MAPE_pct"])
    verification = verify_zero_force_mof_anchor(mape_pct)

    ref_file = references / "references.json"
    ref_map = {r["id"]: float(r["B0_GPa"]) for r in score_extra.load(ref_file)}
    per_item = {}
    for key in sorted(ref_map):
        y = ref_map[key]
        p = preds.get(key)
        b = p.get("B_GPa") if score_extra.ok(p) else None
        capped_ape = (
            score_extra.MOF_CAP
            if (b is None or not np.isfinite(b))
            else float(min(abs(b - y) / abs(y), score_extra.MOF_CAP))
        )
        per_item[key] = {
            "status": p.get("status", "missing") if p else "missing",
            "B_GPa": float(b) if (b is not None and np.isfinite(b)) else None,
            "ref_B0_GPa": y,
            "capped_ape": capped_ape,
            "v0": p.get("v0") if p else None,
            "atoms": p.get("atoms") if p else None,
            "seconds": round(float(p.get("seconds", 0.0)), 4) if p else None,
            **({"error": p["error"]} if p and "error" in p else {}),
        }

    return {
        "schema": RECEIPT_SCHEMA,
        "calculator": "SumCalculator([ZeroForceCalculator(), TorchDFTD3Calculator(damping='bj', xc='pbe', cutoff=40.0*Bohr)])",
        "d3_config": {
            "calculator": "TorchDFTD3Calculator",
            "xc": D3_XC,
            "damping": D3_DAMPING,
            "cutoff_bohr": D3_CUTOFF_BOHR,
            "device": device,
        },
        "inputs_manifest_sha256": _file_sha256(inputs / "manifest.json"),
        "references_sha256": _file_sha256(ref_file),
        "items": int(scored["items"]),
        "failed": int(scored["failed"]),
        "wall_seconds": round(wall_seconds, 3),
        "mof_bulk_modulus_MAPE_pct": mape_pct,
        "verification": verification,
        "per_item": per_item,
    }


def update_anchor_files(
    receipt: dict[str, Any],
    *,
    anchors_json_path: Path | None = None,
    harness_inputs_path: Path | None = None,
) -> None:
    """Atomically write the measured unrounded MOF MAPE (%) to container and harness JSON configs."""
    mape_pct = float(receipt["mof_bulk_modulus_MAPE_pct"])
    verify_zero_force_mof_anchor(mape_pct)

    if anchors_json_path is not None:
        current = json.loads(anchors_json_path.read_text()) if anchors_json_path.is_file() else {
            "schema": "atomistic-materials-zero-force-anchors/v4",
            "discovery_positive_prevalence": aggregate.DISCOVERY_POSITIVE_PREVALENCE,
            "discovery_f1_base": aggregate.DISCOVERY_F1_BASE,
            "atom_zero_force_anchors": dict(aggregate.DEFAULT_ATOM_ZERO_FORCE_ANCHORS),
        }
        current.setdefault("atom_zero_force_anchors", dict(aggregate.DEFAULT_ATOM_ZERO_FORCE_ANCHORS))
        current["atom_zero_force_anchors"]["mof_bulk_modulus_MAPE_pct"] = mape_pct
        current["mof_d3_dispersion_config"] = {
            **receipt["d3_config"],
            "status": "measured",
            "items": receipt["items"],
            "failed": receipt["failed"],
            "inputs_manifest_sha256": receipt["inputs_manifest_sha256"],
            "references_sha256": receipt["references_sha256"],
        }
        _atomic_write_json(anchors_json_path, current)

    if harness_inputs_path is not None:
        harness_data = json.loads(harness_inputs_path.read_text())
        atom_anchors = dict(harness_data.get("atom_zero_force_anchors", {}))
        atom_anchors["mof_bulk_modulus_MAPE_pct"] = mape_pct
        harness_data["atom_zero_force_anchors"] = atom_anchors
        _atomic_write_json(harness_inputs_path, harness_data)


def export_zero_force_submission(target_dir: Path) -> None:
    """Export a valid `/workspace/submission`-compatible ZeroForceCalculator package."""
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "calculator.py").write_text(
        '"""Trivial zero-force ASE calculator for negative-control verification."""\n'
        "import numpy as np\n"
        "from ase.calculators.calculator import Calculator, all_changes\n\n\n"
        "class ZeroForceCalculator(Calculator):\n"
        '    implemented_properties = ["energy", "free_energy", "forces", "stress"]\n\n'
        "    def calculate(self, atoms=None, properties=None, system_changes=all_changes):\n"
        "        super().calculate(atoms, properties, system_changes)\n"
        "        n = len(atoms) if atoms is not None else 0\n"
        "        self.results = {\n"
        '            "energy": 0.0,\n'
        '            "free_energy": 0.0,\n'
        '            "forces": np.zeros((n, 3), dtype=np.float64),\n'
        '            "stress": np.zeros(6, dtype=np.float64),\n'
        "        }\n\n\n"
        "def make_calculator(submission_dir, device):\n"
        "    return ZeroForceCalculator()\n",
        encoding="utf-8",
    )
    (target_dir / "README.md").write_text(
        "# ZeroForceCalculator negative control\n\n"
        "Returns zero energy, zero forces and zero stress on every structure.\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inputs", type=Path, help="Path to test/mof/inputs directory")
    parser.add_argument("--references", type=Path, help="Path to test/mof/references directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mof_zero_force_d3_receipt.json"),
        help="Output path for receipt JSON",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpus", default="", help="Optional comma-separated GPU IDs for parallel workers")
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument(
        "--update-anchors",
        type=Path,
        default=None,
        help="Optional path to environment/evaluation/atom_zero_force_anchors.json to update atomically",
    )
    parser.add_argument(
        "--update-harness-inputs",
        type=Path,
        default=None,
        help="Optional path to harness/ssi_index_v3_inputs.json to update atomically",
    )
    parser.add_argument(
        "--export-submission",
        type=Path,
        default=None,
        help="Optional directory to write a submission-compatible ZeroForceCalculator package",
    )
    args = parser.parse_args(argv)

    if args.export_submission is not None:
        export_zero_force_submission(args.export_submission)
        if args.inputs is None and args.references is None:
            return 0

    if args.inputs is None or args.references is None:
        parser.error("--inputs and --references are required unless only --export-submission is used")

    receipt = measure_mof_zero_force_anchor(
        args.inputs,
        args.references,
        device=args.device,
        gpus=args.gpus,
        workers_per_gpu=args.workers_per_gpu,
    )
    _atomic_write_json(args.output, receipt)
    if args.update_anchors is not None or args.update_harness_inputs is not None:
        update_anchor_files(
            receipt,
            anchors_json_path=args.update_anchors,
            harness_inputs_path=args.update_harness_inputs,
        )
    print(
        json.dumps(
            {
                "mof_bulk_modulus_MAPE_pct": receipt["mof_bulk_modulus_MAPE_pct"],
                "items": receipt["items"],
                "failed": receipt["failed"],
                "verification": receipt["verification"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
