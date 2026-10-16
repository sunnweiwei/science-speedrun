"""Fail-closed fixed-cohort input audit; never reads numeric targets or models.

Missing completion receipts mean incomplete, not terminal or eligible for retry.
Slurm state must be checked separately before any recovery action.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_row(directory, site, observation, index, parent_hash, sources):
    child = read(directory / "manifest.json")
    expected = {
        "sites": [{**site, "observations": [observation]}],
        "parent_manifest_sha256": parent_hash,
        "parent_task_index": index,
    }
    require(child == expected, "Child manifest differs from fixed parent row")
    child_hash = digest(directory / "manifest.json")
    launch = read(directory / "launch.json")
    completion = read(directory / "completion.json")
    require(
        {k: v for k, v in completion.items() if k != "input_sha256"} == launch,
        "Launch/completion provenance mismatch",
    )
    require(completion["task_index"] == index, "Wrong task index")
    require(completion["parent_manifest_sha256"] == parent_hash, "Parent changed")
    require(completion["source_sha256"] == sources, "Acquisition sources changed")
    verification = read(directory / "verification.json")
    prepared = read(directory / "input/report.json")
    acquired = read(directory / "dataset/completion.json")
    for report in (completion, acquired, prepared, verification):
        require(
            report["target_used"] is False and report["model_used"] is False,
            "Unexpected target/model use",
        )
    for report in (completion, acquired, prepared):
        require(
            report["site_id"] == site["site_id"]
            and report["date"] == observation["date"],
            "Wrong site/date",
        )
    for report in (acquired, prepared):
        require(report["manifest_sha256"] == child_hash, "Child hash mismatch")
    require(verification["structural_checks_passed"] is True, "Raster checks failed")
    periods = verification["periods"]
    require(
        len(periods) == 24
        and {(p["modality"], p["index"]) for p in periods}
        == {(m, i) for m in ("sentinel1", "sentinel2") for i in range(12)},
        "Missing/duplicate raster verification periods",
    )
    require(
        prepared["verification_sha256"] == digest(directory / "verification.json"),
        "Raster verification receipt changed",
    )
    input_path = directory / "input/input.npz"
    input_hash = digest(input_path)
    require(
        prepared["input_sha256"] == completion["input_sha256"] == input_hash,
        "Saved input changed",
    )
    shapes = {
        "sentinel2_l2a": (32, 32, 12, 12),
        "sentinel1": (32, 32, 12, 2),
        "sentinel2_l2a_valid": (12, 3),
        "sentinel1_valid": (12, 1),
    }
    with np.load(input_path, allow_pickle=False) as arrays:
        require(set(arrays.files) == set(shapes), "Unexpected saved input fields")
        for name, shape in shapes.items():
            array = arrays[name]
            expected_dtype = "bool" if name.endswith("_valid") else "float32"
            require(array.shape == shape, f"Wrong {name} shape")
            require(str(array.dtype) == expected_dtype, f"Wrong {name} dtype")
            require(np.isfinite(array).all(), f"Nonfinite {name}")
            require(
                prepared["arrays"][name]
                == {"shape": list(shape), "dtype": expected_dtype},
                f"Wrong {name} metadata",
            )
    return input_hash


def reconcile(manifest_path, launch_path, root):
    manifest, launch = read(manifest_path), read(launch_path)
    parent_hash = digest(manifest_path)
    require(parent_hash == launch["manifest_sha256"], "Fixed manifest hash changed")
    rows = [(s, o) for s in manifest["sites"] for o in s["observations"]]
    require(len(rows) == launch["tasks"] > 0, "Wrong manifest row count")
    require(len(manifest["sites"]) == launch["sites"], "Wrong site count")
    require(
        len({(s["site_id"], o["date"]) for s, o in rows}) == len(rows),
        "Duplicate fixed site/date",
    )
    expected_dirs = {f"{i:04d}" for i in range(len(rows))}
    unexpected = sorted(
        p.name for p in root.iterdir() if p.is_dir() and p.name not in expected_dirs
    )
    records = []
    for index, (site, observation) in enumerate(rows):
        directory = root / f"{index:04d}"
        row = {"index": index, "site_id": site["site_id"], "date": observation["date"]}
        if (directory / "failure.json").exists():
            row.update(status="failure_record_present")
        elif not (directory / "completion.json").exists():
            row.update(status="incomplete")
        else:
            try:
                row["input_sha256"] = verify_row(
                    directory,
                    site,
                    observation,
                    index,
                    parent_hash,
                    launch["source_sha256"],
                )
                row["status"] = "verified"
            except (OSError, ValueError, KeyError, TypeError) as error:
                row.update(status="invalid", error=str(error))
        records.append(row)
    counts = {
        status: sum(r["status"] == status for r in records)
        for status in ("verified", "incomplete", "failure_record_present", "invalid")
    }
    return {
        "all_inputs_verified": counts["verified"] == len(rows) and not unexpected,
        "manifest_sha256": parent_hash,
        "launch_sha256": digest(launch_path),
        "target_used": False,
        "model_used": False,
        "counts": counts,
        "unexpected_directories": unexpected,
        "rows": records,
        "scope": "Saved input and provenance audit. Does not establish Slurm terminal state, reread raw rasters, lock scoring protocol, or demonstrate model benefit.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = reconcile(args.manifest, args.launch, args.root)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))
    raise SystemExit(0 if report["all_inputs_verified"] else 1)


if __name__ == "__main__":
    main()
