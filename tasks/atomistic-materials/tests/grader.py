#!/usr/bin/env python3
"""Trusted grader: check the sealed snapshot, then score every suite against the test references.

It never imports submission code. Physical suites use environment/evaluation/score.py in the
pinned MP2020 scoring runtime (/opt/scoring-venv); the other suites use score_extra.py.
All 15 raw metrics remain available; reward.json also includes the fixed eight-suite
Crystal Potential Score v2 (0–100) and reward = score / 100.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from verifier_common import sha256, tree_sha256, write_json

EVALUATION = Path("/evaluation") if Path("/evaluation").is_dir() else Path(__file__).resolve().parents[1] / "environment/evaluation"
sys.path.insert(0, str(EVALUATION))
from score_extra import SCORERS, score as score_extra  # noqa: E402
from aggregate import expanded_score  # noqa: E402

SCHEMA = "atomistic-materials-prediction-snapshot/v2"
SCORING_PYTHON = "/opt/scoring-venv/bin/python"
# Lower is better for every reward key except F1.
REWARD_KEYS = {
    ("physical", "discovery", "F1"): "discovery_F1",
    ("physical", "discovery", "hull_MAE_meV_per_atom"): "discovery_hull_MAE_meV_per_atom",
    ("physical", "phonons", "max_freq"): "phonon_max_freq_MAE_K",
    ("physical", "phonons", "avg_freq"): "phonon_avg_freq_MAE_K",
    ("physical", "phonons", "entropy"): "phonon_entropy_MAE",
    ("physical", "phonons", "heat_capacity"): "phonon_heat_capacity_MAE",
    ("physical", "phonons", "free_energy"): "phonon_free_energy_MAE",
    ("physical", "elasticity", "bulk_modulus_vrh"): "bulk_modulus_MAE_GPa",
    ("physical", "elasticity", "shear_modulus_vrh"): "shear_modulus_MAE_GPa",
    ("kappa", "kappa_SRME"): "kappa_SRME",
    ("mof", "bulk_modulus_MAPE_pct"): "mof_bulk_modulus_MAPE_pct",
    ("amorphous", "energy_RelRMSE_pct_mean"): "amorphous_energy_RelRMSE_pct",
    ("amorphous", "force_RelRMSE_pct_mean"): "amorphous_force_RelRMSE_pct",
    ("cleavage", "cleavage_relMAE_pct"): "cleavage_relMAE_pct",
    ("neb", "barrier_MAE_eV"): "neb_barrier_MAE_eV",
}


def harbor_reward(values):
    """Harbor parses reward.json as a flat object of finite numbers; strings, None and nested details
    belong in report.json / status.json."""
    import math as _math
    return {k: v for k, v in values.items() if isinstance(v, (int, float)) and _math.isfinite(v)}


def score_physical(predictions: Path, references: Path, output: Path) -> dict:
    proc = subprocess.run([SCORING_PYTHON, str(EVALUATION / "score.py"), "--references", str(references),
                           "--predictions", str(predictions), "--output", str(output)],
                          capture_output=True, text=True)
    if output.is_file():
        return json.loads(output.read_text())
    return {"valid": False, "error": (proc.stderr or proc.stdout)[-2000:]}


def flatten(report: dict) -> dict:
    reward = {}
    for path, name in REWARD_KEYS.items():
        value = report.get(path[0], {})
        value = value.get("metrics", value) if path[0] == "physical" else value
        for key in path[1:]:
            value = value.get(key, {}) if isinstance(value, dict) else {}
        if isinstance(value, dict):
            value = value.get("MAE")
        reward[name] = value if isinstance(value, (int, float)) else None
    return reward


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sealed", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True, help="test/<suite>/{inputs,references}")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wait-seconds", type=int, default=172800)
    a = p.parse_args()
    ready = a.sealed.parent / "READY"
    deadline = time.monotonic() + a.wait_seconds
    while not ready.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("prediction snapshot was not published")
        time.sleep(2)
    snapshot = json.loads((a.sealed / "snapshot.json").read_text())
    if ready.read_text().strip() != sha256(a.sealed / "snapshot.json") or snapshot.get("schema") != SCHEMA:
        raise ValueError("invalid sealed snapshot")
    a.output.mkdir(parents=True, exist_ok=True)
    report = {"schema": "atomistic-materials-report/v2", "split": "test", "runs": snapshot["runs"]}
    for suite, digest in snapshot["predictions"].items():
        pred = a.sealed / suite
        inputs, refs = a.labels / suite / "inputs", a.labels / suite / "references"
        try:
            if tree_sha256(pred) != digest:
                raise ValueError("sealed prediction hash mismatch")
            if tree_sha256(inputs) != snapshot["inputs"][suite]:
                raise ValueError("sealed predictions were made on other inputs")
            if suite == "physical":
                report[suite] = score_physical(pred, refs, a.output / "physical-score.json")
            elif (pred / "run.json").is_file():
                report[suite] = score_extra(suite, pred, inputs, refs)
            else:  # the run died before writing anything: every case is a failure
                report[suite] = {"suite": suite, **SCORERS[suite]({}, refs), "complete": False}
        except (ValueError, KeyError, OSError, SystemExit) as error:
            report[suite] = {"valid": False, "error": f"{type(error).__name__}: {error}"[:2000]}
    reward = flatten(report)
    report["aggregate"] = expanded_score(
        reward, valid=not any(r.get("valid") is False for r in report.values() if isinstance(r, dict))
    )
    reward["crystal_potential_score"] = report["aggregate"]["score"]
    reward["reward"] = report["aggregate"]["score"] / 100
    write_json(a.output / "report.json", report)
    write_json(a.output / "reward.json", harbor_reward(reward))


if __name__ == "__main__":
    main()
