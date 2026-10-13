#!/usr/bin/env python3
"""Build a public release of the permitted pretraining corpus for Hugging Face.

    python3 build_release.py --data $TASK_DATA_DIR --out RELEASE_DIR
    python3 build_release.py --data ... --out ... --verify-only

This is much simpler than pde-surrogate's equivalent, for a structural reason: there the
held-out trajectories live inside the same HDF5 files as the training ones, so files had to
be truncated and the result proved clean by content. Here the corpus and the evaluation data
are *separate repositories*. A corpus release carries no test data at all, needs no
truncation, and no file in it describes the held-out windows - checked, not assumed: see
docs/validation/republication_readiness.json.

What the release carries is the 95-directory allow-list, not GiftEvalPretrain's full 152.
Fifty-one climate shards are excluded for size and six because they were measured to hold
GIFT-Eval evaluation series up to an affine transform (`m4_hourly` is `pedestrian_counts`
plus a constant, r = 1.000000 across the whole series including its test horizon). Upstream
advertises the corpus as non-leaking and ships them anyway; republishing them here would
propagate a leak under this task's name, so the builder refuses to.

GIFT-Eval itself is never republished. It is the evaluation data.
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = "gifteval_pretrain"
FORMAT = "timeseries-forecasting-release/v1"
BLOCK = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(BLOCK), b""):
            h.update(b)
    return h.hexdigest()


def permitted(lock: dict):
    """(allow-listed directory names, every file under them) from the pinned lock."""
    src = next(s for s in lock["sources"] if s["id"] == SOURCE)
    allowed = set(src["directories"])
    leaking = set(src.get("excluded_for_leakage", {}))
    if allowed & leaking:
        raise SystemExit(f"allow-list and the leaking set overlap: {sorted(allowed & leaking)}")
    files = [f for f in src["files"] if f["path"].split("/")[0] in allowed]
    return src, allowed, leaking, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="a materialization made by materialize.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--release-id", default="timeseries-forecasting-corpus-v1")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N files (for tests)")
    a = ap.parse_args()

    lock = json.loads((HERE / "sources.lock.json").read_text())
    src, allowed, leaking, files = permitted(lock)
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.limit:
        files = files[:a.limit]

    inventory, missing = [], []
    for f in files:
        local = data / SOURCE / f["path"]
        if not local.exists():
            missing.append(f["path"])
            continue
        rec = {"path": f["path"], "size": local.stat().st_size, "sha256": sha256_file(local),
               "read_from": str(local)}
        if f.get("sha256") and rec["sha256"] != f["sha256"]:
            raise SystemExit(f"{f['path']}: local bytes do not match the pinned lock digest")
        inventory.append(rec)
    if missing and not a.verify_only:
        raise SystemExit(f"{len(missing)} allow-listed files are not materialized, first: {missing[:3]}")

    inv = out / "payload-files.jsonl"
    inv.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in inventory))
    by_dir = {}
    for r in inventory:
        d = r["path"].split("/")[0]
        e = by_dir.setdefault(d, {"files": 0, "bytes": 0})
        e["files"] += 1
        e["bytes"] += r["size"]
    manifest = {
        "format": FORMAT, "release_id": a.release_id,
        "upstream": {"repo": src["repo"], "revision": src["revision"]},
        "payload_files": len(inventory), "payload_bytes": sum(r["size"] for r in inventory),
        "directories": len(by_dir), "per_directory": by_dir,
        "withholds": {
          "excluded_for_leakage": sorted(leaking),
          "why": "measured to contain GIFT-Eval evaluation series up to an affine transform; see "
                 "docs/validation/corpus_leakage_verified.json",
          "excluded_for_size": src.get("excluded_bytes", 0),
          "gifteval": "the evaluation data is a separate repository and is never republished",
        },
        "licence": {
          "release": "CC BY-NC 4.0",
          "why": "LargeST is 43.1% of the allow-list and is CC BY-NC 4.0. CC BY-NC permits redistribution with "
                 "attribution and the notice carried through, so the release is possible, but it cannot be labelled "
                 "Apache-2.0 as the upstream card is. See UPSTREAM_LICENSES.md and "
                 "docs/validation/republication_readiness.json.",
          "unresolved": ["m5", "favorita_sales", "favorita_transactions", "godaddy"],
        },
        "inventory": {"path": "payload-files.jsonl", "size": inv.stat().st_size, "sha256": sha256_file(inv)},
    }
    (out / "release-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(inventory)} files across {len(by_dir)} directories, "
          f"{manifest['payload_bytes'] / 1e9:.2f} GB")
    print(f"withheld for leakage: {sorted(leaking)}")
    if missing:
        print(f"WARNING: {len(missing)} allow-listed files are not materialized")
    print(f"manifest sha256: {sha256_file(out / 'release-manifest.json')}")


if __name__ == "__main__":
    main()
