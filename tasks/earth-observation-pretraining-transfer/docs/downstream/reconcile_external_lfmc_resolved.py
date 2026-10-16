"""Audit an explicit fixed-row recovery ledger without hiding failed attempts.

This does not query Slurm, select successful retries, read targets, or run models.
The ledger lists every known attempt in order; only its final attempt is eligible.
"""

import argparse
import json
from pathlib import Path

from reconcile_external_lfmc_cohort import digest, read, require, verify_row


def reconcile_resolved(manifest_path, ledger_path):
    manifest, ledger = read(manifest_path), read(ledger_path)
    parent_hash = digest(manifest_path)
    require(parent_hash == ledger["manifest_sha256"], "Fixed manifest changed")
    rows = [(s, o) for s in manifest["sites"] for o in s["observations"]]
    require(len(rows) == ledger["tasks"] > 0, "Wrong fixed row count")
    require(len(manifest["sites"]) == ledger["sites"], "Wrong fixed site count")
    require(
        len({(s["site_id"], o["date"]) for s, o in rows}) == len(rows),
        "Duplicate site/date",
    )
    plans = ledger["rows"]
    require(
        [p["index"] for p in plans] == list(range(len(rows))),
        "Ledger must contain every fixed row exactly once in order",
    )
    launches = {}
    for key, entry in ledger["launches"].items():
        path = Path(entry["path"])
        require(digest(path) == entry["sha256"], "Launch receipt changed")
        launch = read(path)
        if "manifest_sha256" in launch:
            require(launch["manifest_sha256"] == parent_hash, "Launch parent changed")
        launches[key] = launch
    require("original" in launches, "Missing original acquisition receipt")
    original = launches["original"]
    require(original["tasks"] == len(rows), "Original acquisition scope changed")
    known_dirs = {key: set() for key in launches}
    records = []
    for index, ((site, observation), plan) in enumerate(zip(rows, plans)):
        attempts = plan["attempts"]
        require(
            attempts and attempts[0]["launch"] == "original",
            "Original attempt must be preserved",
        )
        keys = [a["launch"] for a in attempts]
        require(len(set(keys)) == len(keys), "Duplicate attempt")
        record = {
            "index": index,
            "site_id": site["site_id"],
            "date": observation["date"],
            "attempts": [],
        }
        try:
            for position, attempt in enumerate(attempts):
                key = attempt["launch"]
                launch = launches[key]
                allowed = ledger["launches"][key]["indices"]
                require(index in allowed, "Attempt outside declared launch scope")
                if "indices" in launch:
                    require(index in launch["indices"], "Wrong retry launch index")
                if "index" in launch:
                    require(index == launch["index"], "Wrong single retry index")
                directory = Path(launch["output_root"]) / f"{index:04d}"
                known_dirs[key].add(directory.name)
                detail = {"launch": key, "directory": str(directory)}
                record["attempts"].append(detail)
                failure_path = directory / "failure.json"
                if position < len(attempts) - 1:
                    require(
                        failure_path.exists(), "Replacement lacks preserved failure"
                    )
                    require(
                        digest(failure_path) == attempt["failure_sha256"],
                        "Historical failure changed",
                    )
                    require(
                        not (directory / "completion.json").exists(),
                        "Refusing replacement of completed attempt",
                    )
                    previous = read(directory / "launch.json")
                    require(
                        previous["task_index"] == index
                        and previous["parent_manifest_sha256"] == parent_hash
                        and previous["source_sha256"] == launch["source_sha256"],
                        "Historical attempt provenance mismatch",
                    )
                    detail.update(
                        status="preserved_failure", failure_sha256=digest(failure_path)
                    )
                    continue
                if failure_path.exists():
                    detail["status"] = record["status"] = "failure_record_present"
                elif not (directory / "completion.json").exists():
                    detail["status"] = record["status"] = "incomplete"
                else:
                    input_hash = verify_row(
                        directory,
                        site,
                        observation,
                        index,
                        parent_hash,
                        launch["source_sha256"],
                    )
                    detail["status"] = record["status"] = "verified"
                    record.update(
                        input_directory=str(directory), input_sha256=input_hash
                    )
        except (OSError, ValueError, KeyError, TypeError) as error:
            record.update(status="invalid", error=str(error))
        records.append(record)
    # Launches may share a root (e.g. separate geometry retry submissions).
    roots = {}
    for key, launch in launches.items():
        roots.setdefault(Path(launch["output_root"]), set()).update(known_dirs[key])
    unexpected = []
    for root, names in roots.items():
        if root.exists():
            unexpected.extend(
                str(p) for p in root.iterdir() if p.is_dir() and p.name not in names
            )
    counts = {
        status: sum(r["status"] == status for r in records)
        for status in ("verified", "incomplete", "failure_record_present", "invalid")
    }
    return {
        "all_inputs_verified": counts["verified"] == len(rows) and not unexpected,
        "manifest_sha256": parent_hash,
        "ledger_sha256": digest(ledger_path),
        "target_used": False,
        "model_used": False,
        "counts": counts,
        "unexpected_directories": sorted(unexpected),
        "rows": records,
        "scope": "Saved inputs and explicit attempt history only; no Slurm terminal-state check, raw-raster reread, model lock, target access, or benefit claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = reconcile_resolved(args.manifest, args.ledger)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))
    raise SystemExit(0 if report["all_inputs_verified"] else 1)


if __name__ == "__main__":
    main()
