#!/usr/bin/env python3
"""Content checksums of a materialized release (public/, verifier/, private/).

An LMDB is identified by the sha256 over its (key, value) pairs in key order plus its entry count, so the digest
does not depend on page layout; every other file by its sha256.
usage: checksums.py write ROOT CHECKSUMS.json | checksums.py verify ROOT CHECKSUMS.json [--procs N]
"""
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

VIEWS = ("public", "verifier", "private")


def digest(path: Path) -> dict:
    h = hashlib.sha256()
    if path.suffix == ".lmdb":
        import lmdb
        env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=True, max_readers=1)
        n = 0
        with env.begin() as txn:
            for k, v in txn.cursor():
                h.update(len(k).to_bytes(4, "big") + k + len(v).to_bytes(8, "big") + v); n += 1
        env.close()
        return {"lmdb_entries": n, "content_sha256": h.hexdigest()}
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for v in VIEWS for p in (root / v).rglob("*")
                  if p.is_file() and not p.name.endswith("-lock"))


def main() -> None:
    mode, root, table = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 16
    names = files(root)
    with Pool(procs) as pool:
        got = dict(zip(names, pool.map(digest, [root / n for n in names], chunksize=1)))
    if mode == "write":
        table.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
        print(f"wrote {len(got)} checksums to {table}")
        return
    want = json.loads(table.read_text())
    bad = sorted(n for n in set(want) | set(got) if want.get(n) != got.get(n))
    for n in bad:
        print(f"MISMATCH {n}: expected {want.get(n)} got {got.get(n)}")
    print(f"{len(got)} files checked, {len(bad)} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
