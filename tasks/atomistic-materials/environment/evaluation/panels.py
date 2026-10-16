"""Small frozen, label-blind physical evaluation panels inside split v1."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from common import digest, write_json
from splits import DEFAULT_SPLITS, load_split

VERSION = "atomistic-physical-panels-v1"
DEFAULT_PANELS = Path(__file__).resolve().parents[2] / ".data/panels/v1/manifest.json"
COUNTS = {
    "val": {"discovery": 1000, "phonons": 100, "elasticity": 100},
    "test": {"discovery": 5000, "phonons": 500, "elasticity": 500},
}


def order(key):
    return hashlib.sha256(f"{VERSION}:{key}".encode()).hexdigest()


def stratum(row):
    size = next((n for n in [4, 8, 16, 32, 64, 128] if row["n_sites"] <= n), 99999)
    elements = min(4, len(row["composition"].split("|")))
    return f"sites<={size}:elements={elements}"


def select(rows, count):
    buckets = defaultdict(list)
    for row in rows:
        buckets[stratum(row)].append(row)
    if count < len(buckets) or count > len(rows):
        raise ValueError("Panel cannot cover occupied geometry strata")
    allocations = {key: 1 for key in buckets}
    # Largest proportional deficit allocation, with one per occupied stratum.
    while sum(allocations.values()) < count:
        key = max(
            (k for k in buckets if allocations[k] < len(buckets[k])),
            key=lambda k: (
                count * len(buckets[k]) / len(rows) - allocations[k],
                order(k),
            ),
        )
        allocations[key] += 1
    selected, groups = [], Counter()
    for key in sorted(buckets):
        candidates = sorted(buckets[key], key=lambda r: order(r["id"]))
        chosen = []
        # Prefer distinct composition groups, but never change split membership.
        for cap in (1, 2, count):
            for row in candidates:
                if row["id"] in chosen or groups[row["group"]] >= cap:
                    continue
                chosen.append(row["id"])
                groups[row["group"]] += 1
                if len(chosen) == allocations[key]:
                    break
            if len(chosen) == allocations[key]:
                break
        selected.extend(chosen)
    if len(set(selected)) != count:
        raise ValueError("Panel lost/duplicated IDs")
    return {
        "ids": sorted(selected),
        "cases": count,
        "composition_groups": len(groups),
        "strata": {
            k: {"population": len(buckets[k]), "selected": allocations[k]}
            for k in sorted(buckets)
        },
    }


def freeze(output, regression_predictions=None):
    if output.exists():
        raise FileExistsError(output)
    rows, _ = load_split()
    result = {
        "version": VERSION,
        "parent_split_manifest_sha256": digest(DEFAULT_SPLITS / "manifest.json"),
        "selection": "Geometry size and chemical-complexity strata; proportional allocation with one per occupied stratum; SHA256 ordering; prefer distinct composition groups; no target/model labels",
        "leaderboard_comparable": False,
        "panels": {},
    }
    for split, counts in COUNTS.items():
        result["panels"][split] = {
            suite: select(
                [r for r in rows if r["split"] == split and r["suite"] == suite], n
            )
            for suite, n in counts.items()
        }
    if regression_predictions:
        failed = [
            json.loads(line)
            for line in regression_predictions.open()
            if json.loads(line)["status"] == "failed"
        ]
        lookup = {r["id"]: r for r in rows}
        if any(lookup[r["id"]]["split"] != "val" for r in failed):
            raise ValueError("Regression cases must remain validation-only")
        result["regression"] = {
            "ids": sorted(r["id"] for r in failed),
            "selection": "Known pilot failures; diagnostic only, excluded from representative aggregate",
            "source_predictions_sha256": digest(regression_predictions),
        }
    write_json(output, result)
    return result


V2_VALIDATION = {"phonons": 300, "elasticity": 300}


def enlarge(v1, output):
    """Panels v2: v1 test panels and v1 discovery validation unchanged; larger phonon and
    elasticity validation panels drawn by the same label-blind stratified rule."""
    if output.exists():
        raise FileExistsError(output)
    result = json.loads(v1.read_text())
    rows, _ = load_split()
    if result["parent_split_manifest_sha256"] != digest(DEFAULT_SPLITS / "manifest.json"):
        raise ValueError("v1 panels belong to another split")
    for suite, n in V2_VALIDATION.items():
        result["panels"]["val"][suite] = select(
            [r for r in rows if r["split"] == "val" and r["suite"] == suite], n
        )
    result.update(version="atomistic-physical-panels-v2", derived_from_v1_sha256=digest(v1))
    write_json(output, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_PANELS)
    p.add_argument("--regression-predictions", type=Path)
    p.add_argument("--enlarge-v1", type=Path, help="v1 manifest; write panels v2 instead")
    args = p.parse_args()
    if args.enlarge_v1:
        r = enlarge(args.enlarge_v1, args.output)
    else:
        r = freeze(args.output, args.regression_predictions)
    print(
        json.dumps(
            {
                s: {k: v["cases"] for k, v in suites.items()}
                for s, suites in r["panels"].items()
            },
            indent=2,
        )
    )
