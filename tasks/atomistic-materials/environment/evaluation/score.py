"""Trusted, offline scoring; never imports a submitted model or calculator."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry

from common import (
    ELASTIC_FIELDS,
    PHONON_FIELDS,
    PROTOCOL,
    decode_atoms,
    digest,
    indexed,
    read_rows,
    write_json,
)


def corrected_formation(prediction, target):
    if version("pymatgen") != "2023.5.10":
        raise ValueError(
            "WBM scoring requires pymatgen 2023.5.10; use the environment/scoring project"
        )
    entry = ComputedStructureEntry.from_dict(target["entry"])
    structure = AseAtomsAdaptor.get_structure(decode_atoms(prediction["atoms"]))
    if entry.composition != structure.composition:
        raise ValueError("Predicted structure changed composition or atom count")
    entry._energy = float(prediction["energy"])
    entry._structure = structure
    # Pin the complete historical oxidation-state/correction implementation.
    # A newer pymatgen with strict_anions=no_check does not reproduce it.
    compatibility = MaterialsProject2020Compatibility()
    corrected = compatibility.process_entry(entry, clean=True)
    if corrected is None:
        raise ValueError(f"MP2020 rejected {entry.entry_id}")
    reference = sum(
        target["elemental_refs"][str(el)] * amount
        for el, amount in corrected.composition.items()
    )
    value = (corrected.energy - reference) / corrected.composition.num_atoms
    if not np.isfinite(value):
        raise ValueError("Nonfinite formation energy")
    return float(value)


def discovery_metrics(truth, predicted):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("Nonfinite hull distances")
    actual, positive = truth <= 0, predicted <= 0
    tp = int(np.sum(actual & positive))
    fp = int(np.sum(~actual & positive))
    fn = int(np.sum(actual & ~positive))
    tn = int(np.sum(~actual & ~positive))

    def divide(numerator, denominator):
        return numerator / denominator if denominator else None

    return {
        "n": len(truth),
        "threshold_eV_per_atom": 0.0,
        "F1": divide(2 * tp, 2 * tp + fp + fn),
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "hull_MAE_meV_per_atom": float(np.mean(np.abs(predicted - truth)) * 1000),
    }


FAILED_FORMATION_ERROR = 1.0  # eV/atom; a failed WBM case is predicted far above the hull


def _usable(pred, target, fields=()):
    if pred is None or pred.get("status") != "ok" or pred.get("suite") != target["suite"]:
        return False
    try:
        return all(np.isfinite(float(pred[f])) for f in fields)
    except (KeyError, TypeError, ValueError):
        return False


def score_rows(targets, predictions):
    """Every target is scored. A failed or nonfinite case predicts 0 for phonon/elastic fields and
    a +1 eV/atom formation-energy error for discovery; failures are counted per suite."""
    targets, predictions = indexed(targets), indexed(predictions)
    unexpected = predictions.keys() - targets.keys()
    if unexpected:
        raise ValueError(f"Unexpected prediction IDs: {sorted(unexpected)[:10]}")
    metrics = {}
    for suite, fields in (("phonons", PHONON_FIELDS), ("elasticity", ELASTIC_FIELDS)):
        ids = [key for key in targets if targets[key]["suite"] == suite]
        if not ids:
            continue
        ok = [_usable(predictions.get(key), targets[key], fields) for key in ids]
        metrics[suite] = {"n": len(ids), "failed": len(ids) - sum(ok)}
        for field, unit in fields.items():
            true = np.asarray([targets[key][field] for key in ids], dtype=float)
            pred = np.asarray([float(predictions[key][field]) if good else 0.0 for key, good in zip(ids, ok)])
            if not np.isfinite(true).all():
                raise ValueError(f"Nonfinite reference {suite}/{field}")
            metrics[suite][field] = {
                "MAE": float(np.mean(np.abs(pred - true))),
                "unit": unit,
            }
    ids = [key for key in targets if targets[key]["suite"] == "discovery"]
    if ids:
        true = np.array([targets[key]["e_hull"] for key in ids])
        errors, failed = [], 0
        for key in ids:
            pred = predictions.get(key)
            try:
                if not _usable(pred, targets[key], ("energy",)):
                    raise ValueError("failed case")
                errors.append(corrected_formation(pred, targets[key]) - targets[key]["e_form"])
            except ValueError as error:
                if "pymatgen 2023.5.10" in str(error):
                    raise
                failed += 1
                errors.append(FAILED_FORMATION_ERROR)
        pred_hull = true + np.array(errors)  # signed distance; never clamp to zero
        metrics["discovery"] = discovery_metrics(true, pred_hull)
        metrics["discovery"]["failed"] = failed
        unique = np.array([targets[key]["unique_prototype"] for key in ids])
        if unique.any():
            metrics["discovery"]["unique_prototypes"] = discovery_metrics(
                true[unique], pred_hull[unique]
            )
    return metrics


def score(references: Path, predictions: Path, output: Path):
    manifest = json.loads((references / "manifest.json").read_text())
    run = json.loads((predictions / "run.json").read_text())
    report = {
        "protocol": PROTOCOL,
        "valid": False,
        "preset": manifest["preset"],
        "leaderboard_comparable": False,
        "scoring_versions": {
            name: version(name) for name in ("numpy", "pymatgen", "ase")
        },
    }
    try:
        if manifest["protocol"] != PROTOCOL or run["protocol"] != PROTOCOL:
            raise ValueError("Protocol mismatch")
        if not run["complete"] or run["inputs_sha256"] != manifest["inputs_sha256"]:
            raise ValueError("Incomplete run or wrong evaluation inputs")
        if digest(references / "targets.jsonl") != manifest["targets_sha256"]:
            raise ValueError("Reference checksum mismatch")
        if digest(predictions / "predictions.jsonl") != run["predictions_sha256"]:
            raise ValueError("Prediction checksum mismatch")
        targets = read_rows(references / "targets.jsonl")
        rows = read_rows(predictions / "predictions.jsonl")
        report["metrics"] = score_rows(targets, rows)
        report["valid"] = True
        report["coverage"] = {"expected": len(targets),
                              "successful": sum(row.get("status") == "ok" for row in rows)}
        report["total_case_seconds"] = sum(row.get("seconds", 0.0) for row in rows)
        report["nonconverged_initial_relaxations"] = sum(
            not row.get("diagnostics", {}).get("converged", False) for row in rows
        )
        report["nonconverged_strained_relaxations"] = sum(
            not relaxed["converged"]
            for row in rows
            for relaxed in row.get("diagnostics", {}).get("strained_relaxations", [])
        )
    except (ValueError, KeyError, TypeError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = score(args.references, args.predictions, args.output)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["valid"] else 2)
