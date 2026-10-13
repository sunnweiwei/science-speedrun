#!/usr/bin/env python3
"""Materialize the pinned timeseries-forecasting sources into an operator-selected directory.

Every file is downloaded individually at the revision recorded in
sources.lock.json and verified against the digest recorded there (LFS SHA-256,
or the git blob SHA-1 for small non-LFS files). A receipt is written next to
the data. Nothing here runs inside Harbor; it is an explicit preparation step.

    python3 materialize.py DEST                                   # gifteval + full pretraining allowlist (275 GB)
    python3 materialize.py DEST --sources gifteval                # evaluation side only (1.6 GB; enough for the verifier)
    python3 materialize.py DEST --tier fast                       # gifteval + the 1.25 GB fast tier of the corpus (smoke runs)
    python3 materialize.py DEST --sources leaderboard             # leaderboard CSV snapshot (reference, 3 MB)
    python3 materialize.py DEST --verify-only                     # hash what is already there
    python3 materialize.py DEST --pretrained timesfm-2.5,chronos-2 --sources -   # maintainers: external-target weights

Layout under DEST:  gifteval/<repo path>, gifteval_pretrain/<repo path>, leaderboard/<repo path>, pretrained/<id>/<repo path>
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

LOCK = Path(__file__).with_name("sources.lock.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    h = hashlib.sha1()
    h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_file(local: Path, f: dict, entry: dict):
    entry["bytes"] = local.stat().st_size
    if entry["bytes"] != f["bytes"]:
        return f"size {entry['bytes']} != {f['bytes']}"
    if f.get("sha256"):
        entry["sha256"] = sha256(local)
        if entry["sha256"] != f["sha256"]:
            return "sha256 mismatch"
    elif f.get("git_blob_sha1"):
        entry["git_blob_sha1"] = git_blob_sha1(local)
        if entry["git_blob_sha1"] != f["git_blob_sha1"]:
            return "git blob sha1 mismatch"
    return None


def fetch(repo, repo_type, revision, filename, root):
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo_id=repo, repo_type=repo_type, revision=revision, filename=filename, local_dir=str(root))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--sources", default="gifteval,gifteval_pretrain", help="comma-separated source ids from the lock ('-' for none)")
    ap.add_argument("--tier", default="full", choices=["full", "fast"], help="gifteval_pretrain: all 95 directories or the small ones only")
    ap.add_argument("--dirs", default="", help="gifteval_pretrain: comma-separated directory names to restrict to")
    ap.add_argument("--verify-only", action="store_true"); ap.add_argument("--plan", action="store_true")
    ap.add_argument("--lock", default=str(LOCK))
    ap.add_argument("--pretrained", default="", help="comma-separated pretrained ids to stage under DEST/pretrained (maintainers only)")
    args = ap.parse_args()
    lock = json.loads(Path(args.lock).read_text())
    dest = Path(args.dest)
    receipt = {"schema": "timeseries-forecasting-materialize-receipt/v1", "lock_schema": lock["schema"],
               "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sources": {}}
    failures = []

    def process(sid, repo, repo_type, revision, files, root):
        total = sum(f["bytes"] for f in files)
        print(f"[{sid}] {repo} @ {revision[:12]}: {len(files)} files, {total/1e9:.3f} GB", file=sys.stderr)
        if args.plan:
            for f in files:
                print(f"    {f['path']}  {f['bytes']/1e6:.1f} MB")
            return
        root.mkdir(parents=True, exist_ok=True)
        entries, t0 = [], time.time()
        for f in files:
            local = root / f["path"]
            if not args.verify_only and not local.exists():
                fetch(repo, repo_type, revision, f["path"], root)
            entry = {"path": f["path"], "bytes_expected": f["bytes"]}
            if not local.exists():
                entry["status"] = "missing"; failures.append(f"{sid}:{f['path']} missing")
            else:
                problem = check_file(local, f, entry)
                if problem and not args.verify_only:  # a truncated or corrupted file is re-fetched once
                    print(f"    re-fetching {f['path']}: {problem}", file=sys.stderr)
                    local.unlink()
                    fetch(repo, repo_type, revision, f["path"], root)
                    entry = {"path": f["path"], "bytes_expected": f["bytes"], "refetched": True}
                    problem = check_file(local, f, entry) if local.exists() else "missing after re-fetch"
                entry["status"] = "ok" if problem is None else problem
                if problem:
                    failures.append(f"{sid}:{f['path']} {problem}")
            entries.append(entry)
        receipt["sources"][sid] = {"repo": repo, "revision": revision, "files": entries, "bytes": total,
                                   "ok": sum(e["status"] == "ok" for e in entries), "total": len(entries), "seconds": round(time.time() - t0, 1)}
        print(f"    {receipt['sources'][sid]['ok']}/{len(entries)} ok in {receipt['sources'][sid]['seconds']} s", file=sys.stderr)

    by_id = {s["id"]: s for s in lock["sources"]}
    for sid in [s for s in args.sources.split(",") if s and s != "-"]:
        src = by_id[sid]
        files = src["files"]
        if sid == "gifteval_pretrain":
            dirs = src["directories"]
            wanted = set(args.dirs.split(",")) if args.dirs else {d for d, v in dirs.items() if args.tier == "full" or v["tier"] == "fast"}
            files = [f for f in files if "/" not in f["path"] or f["path"].split("/")[0] in wanted]
            receipt.setdefault("pretrain_directories", sorted(wanted))
        process(sid, src["repo"], src["repo_type"], src["revision"], files, dest / sid)
    for pid in [p for p in args.pretrained.split(",") if p]:
        p = next(x for x in lock["pretrained"] if x["id"] == pid)
        process(f"pretrained/{pid}", p["repo"], "model", p["revision"], p["files"], dest / "pretrained" / pid)
        if not args.plan:
            (dest / "pretrained" / pid).mkdir(parents=True, exist_ok=True)
            (dest / "pretrained" / pid / "PIN.json").write_text(json.dumps({k: p[k] for k in ("repo", "revision", "license")}, indent=1) + "\n")
    if args.plan:
        return
    receipt["ok"] = not failures
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "materialize-receipt.json"
    if out.exists():  # merge with earlier runs on the same DEST
        old = json.loads(out.read_text())
        merged = {**old.get("sources", {}), **receipt["sources"]}
        receipt["sources"] = merged
        receipt["ok"] = receipt["ok"] and all(v["ok"] == v["total"] for v in merged.values())
    out.write_text(json.dumps(receipt, indent=1) + "\n")
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures[:50]), file=sys.stderr)
        sys.exit(1)
    if not receipt["ok"]:
        print("FAILED: an earlier run in this receipt did not verify; rerun it with --verify-only", file=sys.stderr)
        sys.exit(1)
    print(f"receipt written to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
