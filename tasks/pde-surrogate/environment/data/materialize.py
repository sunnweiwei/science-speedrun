#!/usr/bin/env python3
"""Materialize pinned PDE surrogate sources into an operator-selected directory.

Every file is downloaded individually at the pinned revision recorded in
sources.lock.json, its SHA-256 is checked against the upstream LFS digest, and
its trajectory count is read from the file and compared with the count the
lock inferred from the byte size. A receipt is written next to the data.

    python3 materialize.py DEST --sources ape2d,well-turbulent_radiative_layer_2D
    python3 materialize.py DEST --sources ape2d --verify-only
    python3 materialize.py DEST --sources ape2d-full --files burgers.hdf5,decay_turb_test.hdf5

Layout under DEST:  <source id>/<path inside the HF repository>
This is an explicit preparation step; Harbor does not run it.
"""
import argparse
import hashlib
import json
import os
import shutil
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


def read_trajectory_count(path: Path, family: str):
    import h5py  # local import so --plan works without h5py
    with h5py.File(path, "r") as f:
        if family == "ape2d":
            sims = f["sims"]
            n = len(sims)
            shape = tuple(sims["sim0"].shape)
            return n, shape
        if family == "pdearena":
            g = f[next(iter(f.keys()))]
            return int(g["u"].shape[0]), (int(g["u"].shape[1]), 3, int(g["u"].shape[2]), int(g["u"].shape[3]))
        n = int(f.attrs["n_trajectories"])
        t = int(f["dimensions"]["time"].shape[0])
        first = next(iter(f["t0_fields"].values()), None) or next(iter(f["t1_fields"].values()))
        x, y = int(first.shape[2]), int(first.shape[3])
        return n, (t, None, x, y)


PRETRAINED = {
    "pde-transformer": {"repo": "thuerey-group/pde-transformer", "type": "model",
                        "revision": "d5889aab16b1fe82e82940b612d54a8f0773e542", "license": "MIT",
                        "files": [f"{v}/{n}" for v in ("mc-s", "mc-b", "mc-l", "sc-s", "sc-b", "sc-l")
                                  for n in ("config.json", "diffusion_pytorch_model.safetensors")]},
    "dpot": {"repo": "hzk17/DPOT", "type": "model", "revision": "2adec1cf9a55942f1456aa7463cd7ade908398d0",
             "license": "Apache-2.0", "files": ["model_Ti.pth", "model_S.pth", "model_M.pth"]},
}


RELEASE_SOURCE = Path(__file__).with_name("download-source.json")


def merge_members(target: Path, parts, src, verify_only: bool, plan: bool) -> bool:
    """Reassemble a file the release split across subtrees into the whole upstream file.

    `train/` holds `sims/sim0..sim{keep-1}` and `eval/` the tail from `sim{keep}` on, both under
    their original names, so the merge is a copy of every group with no renumbering. The result
    has as many trajectories as the upstream file, which is what the verifier needs and what every
    reader in this task assumes.
    """
    import h5py
    from huggingface_hub import hf_hub_download

    want = sum(r["size"] for r in parts)
    if plan:
        print(f"  merge  {target.name:56s} {want / 1e9:7.3f} GB from {len(parts)} members")
        return True
    if target.exists():
        print(f"  already present (not re-merged): {target}")
        return False
    if verify_only:
        print(f"  MISSING (needs merge): {target}", file=sys.stderr)
        return False
    # Root datasets (the norm_* statistics) must come from the eval tail, not the train head:
    # build_release.py recomputes them over the kept trajectories when it writes the head, so the
    # head's are deliberately not upstream's, while write_tail copies upstream's through unchanged.
    # A merged file is meant to be the upstream file, so the tail's are the right ones.
    parts = sorted(parts, key=lambda r: (not r["path"].startswith("eval/"), r["path"]))
    local = []
    for r in parts:
        got = hf_hub_download(src["repo_id"], r["path"], revision=src["revision"], repo_type="dataset")
        if sha256(Path(got)) != r["sha256"]:
            raise SystemExit(f"{r['path']}: downloaded bytes do not match the manifest")
        local.append(Path(got))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".partial")
    with h5py.File(tmp, "w") as out:
        g = out.create_group("sims", track_order=True)
        seen = set()
        for i, path in enumerate(local):
            with h5py.File(path, "r") as fin:
                for name in fin["sims"]:
                    if name in seen:
                        raise SystemExit(f"{target.name}: {name} appears in more than one member")
                    seen.add(name)
                    d = g.create_dataset(name, data=fin["sims"][name][()])
                    for k, v in fin["sims"][name].attrs.items():
                        d.attrs[k] = v
                if i == 0:                       # the eval tail, per the note above
                    for k, v in fin["sims"].attrs.items():   # PDE, Fields, Fields Scheme, Dt, ...
                        g.attrs[k] = v
                    for name in fin:
                        if name != "sims":
                            out.create_dataset(name, data=fin[name][()])
                    for k, v in fin.attrs.items():
                        if k != "pde_surrogate_public_release":
                            out.attrs[k] = v
    tmp.rename(target)
    check_corpus_readable(target)
    print(f"  merged  {target.name:56s} {len(seen):4d} trajectories from {len(parts)} members")
    return True



def release_head(path: Path):
    """The release note a truncated training head carries, or None for an upstream file."""
    import h5py
    try:
        with h5py.File(path, "r") as f:
            note = f.attrs.get("pde_surrogate_public_release")
    except OSError:
        return None
    if note is None:
        return None
    note = note.decode() if isinstance(note, bytes) else str(note)
    return note if note.startswith("training-only") else None


def check_corpus_readable(path: Path):
    """Refuse a corpus file the task's readers cannot open (missing `sims` attributes).

    The first published release shipped every member without them; the SHA-256 checks passed
    because the bytes were exactly what the manifest described. Bytes matching is not enough."""
    import h5py
    with h5py.File(path, "r") as f:
        if "sims" not in f:
            return
        missing = [k for k in ("PDE", "Fields", "Fields Scheme", "Dt") if k not in f["sims"].attrs]
    if missing:
        raise SystemExit(f"{path}: `sims` lacks the attributes {missing}; the release member is unreadable by "
                         f"environment/evaluation/pde_io.py. Report it; do not use this release.")


def from_release(dest: Path, verify_only: bool, plan: bool, subtree: str = ""):
    """Fetch the pre-built public training release instead of the four upstream repositories.

    The release carries the same layout materialize.py produces, minus everything a held-out
    case touches (see build_release.py). Each file is verified against the manifest's SHA-256,
    an existing correct file is left alone, and an existing *wrong* file is reported and never
    overwritten - a mismatch means the directory is not what the manifest describes, which the
    operator should resolve rather than have a download paper over.
    """
    if not RELEASE_SOURCE.exists():
        raise SystemExit("download-source.json is not present: no release has been published yet")
    src = json.loads(RELEASE_SOURCE.read_text())
    from huggingface_hub import hf_hub_download

    man_path = hf_hub_download(src["repo_id"], "release-manifest.json", revision=src["revision"],
                               repo_type="dataset")
    if sha256(Path(man_path)) != src["manifest_sha256"]:
        raise SystemExit("release-manifest.json does not match the pinned manifest_sha256")
    manifest = json.loads(Path(man_path).read_text())
    inv_path = hf_hub_download(src["repo_id"], manifest["inventory"]["path"], revision=src["revision"],
                               repo_type="dataset")
    if sha256(Path(inv_path)) != manifest["inventory"]["sha256"]:
        raise SystemExit("payload-files.jsonl does not match the manifest")
    records = [json.loads(l) for l in Path(inv_path).read_text().splitlines() if l.strip()]
    if subtree:
        records = [r for r in records if r.get("subtree") == subtree]
        if not records:
            raise SystemExit(f"the release has no '{subtree}' subtree")

    label = f" [{subtree} only]" if subtree else ""
    print(f"release {manifest['release_id']}{label}: {len(records)} files, "
          f"{sum(r['size'] for r in records) / 1e9:.2f} GB")
    if plan:
        for r in records:
            print(f"  {r['path']:62s} {r['size'] / 1e9:7.3f} GB")
        return

    # Members are stored as "<subtree>/<source>/<path>", but every reader in this task expects the
    # materialization layout, "<dest>/<source>/<path>" (make_cases.py, baseline/starter/data.py, the direct
    # path of this script). Strip the subtree, and where a train head and an eval tail share a
    # destination, merge them back into the whole upstream file so the result is indistinguishable
    # from a direct materialization. An agent host asks for --subtree train and gets the head alone.
    root = dest
    by_dest = {}
    for r in records:
        by_dest.setdefault(r["path"].split("/", 1)[1], []).append(r)
    ok = fetched = conflict = merged = 0
    for rel, parts in sorted(by_dest.items()):
        if len(parts) > 1:
            if merge_members(root / rel, parts, src, verify_only, plan):
                merged += 1
            continue
        r = parts[0]
        target = root / rel
        if target.exists() and target.stat().st_size == r["size"] and sha256(target) == r["sha256"]:
            if target.suffix == ".hdf5":
                check_corpus_readable(target)
            ok += 1
            continue
        if target.exists():
            print(f"  CONFLICT (left alone): {r['path']}", file=sys.stderr)
            conflict += 1
            continue
        if verify_only:
            print(f"  MISSING: {r['path']}", file=sys.stderr)
            continue
        got = hf_hub_download(src["repo_id"], r["path"], revision=src["revision"], repo_type="dataset")
        if sha256(Path(got)) != r["sha256"]:
            raise SystemExit(f"{r['path']}: downloaded bytes do not match the manifest")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(got, target)
        if target.suffix == ".hdf5":
            check_corpus_readable(target)
        fetched += 1
        print(f"  fetched {r['path']:62s} {r['size'] / 1e9:7.3f} GB")
    print(f"already correct {ok}, fetched {fetched}, merged {merged}, conflicting {conflict}")
    receipt = {"release_id": manifest["release_id"], "repo_id": src["repo_id"], "revision": src["revision"],
               "manifest_sha256": src["manifest_sha256"], "subtree": subtree or "all",
               "files_verified": ok + fetched, "files_merged": merged, "files_conflicting": conflict}
    (dest / "release-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    if conflict:
        raise SystemExit(f"{conflict} existing files do not match the release and were not overwritten")


def stage_pretrained(root: Path, names, verify_only: bool):
    """Download the permitted pretrained assets at their pinned revisions (see instruction.md)."""
    from huggingface_hub import hf_hub_download
    for name in names:
        spec = PRETRAINED[name]
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        for f in spec["files"]:
            if not verify_only and not (out / f).exists():
                hf_hub_download(repo_id=spec["repo"], repo_type=spec["type"], revision=spec["revision"],
                                filename=f, local_dir=str(out))
            print(f"    {'ok ' if (out / f).exists() else 'MISSING'} {name}/{f}", file=sys.stderr)
        (out / "PIN.json").write_text(json.dumps({k: v for k, v in spec.items() if k != "files"}, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--sources", default="ape2d", help="comma-separated source ids from the lock")
    ap.add_argument("--files", default="", help="comma-separated basenames to restrict to (optional)")
    ap.add_argument("--splits", default="", help="comma-separated splits to restrict to (optional)")
    ap.add_argument("--verify-only", action="store_true", help="hash and inspect existing files, no download")
    ap.add_argument("--plan", action="store_true", help="print what would be downloaded and exit")
    ap.add_argument("--lock", default=str(LOCK))
    ap.add_argument("--manifest", default="", help="split manifest: restrict every source to the files its held-out cases reference (verifier materialization)")
    ap.add_argument("--pools", default="", help="split manifest: restrict every source to the files its train/valid pools reference (agent host); "
                                                 "a truncated head already fetched with --from-release --subtree train is accepted as is")
    ap.add_argument("--from-release", action="store_true",
                    help="download the published release instead of the upstream repositories")
    ap.add_argument("--subtree", default="", choices=["", "train", "eval"],
                    help="with --from-release: fetch only one subtree. An agent host wants 'train'; "
                         "the verifier needs both, which is the default.")
    ap.add_argument("--pretrained", default="", help="comma-separated pinned assets to stage under DEST/pretrained: pde-transformer,dpot")
    args = ap.parse_args()
    if args.from_release:
        from_release(Path(args.dest), args.verify_only, args.plan, args.subtree)
        return
    if args.pretrained:
        stage_pretrained(Path(args.dest) / "pretrained", [a for a in args.pretrained.split(",") if a], args.verify_only)
        if not args.sources or args.sources == "-":
            return

    lock = json.loads(Path(args.lock).read_text())
    by_id = {s["id"]: s for s in lock["sources"]}
    wanted_files = {f for f in args.files.split(",") if f}
    case_files = {}
    if args.manifest:
        man = json.loads(Path(args.manifest).read_text())
        for c in man["cases"]:
            case_files.setdefault(c["source"], set()).add(c["file"])
        if args.sources in ("", "-"):
            args.sources = ",".join(sorted(case_files))
    if args.pools:
        man = json.loads(Path(args.pools).read_text())
        for split in ("train", "valid"):
            for p in man["pools"][split]:
                case_files.setdefault(p["source"], set()).add(p["file"])
        if args.sources in ("", "-"):
            args.sources = ",".join(sorted(case_files))
    wanted_splits = {s for s in args.splits.split(",") if s}
    dest = Path(args.dest)
    receipt = {"schema": "pde-surrogate-materialize-receipt/v1", "lock_schema": lock["schema"],
               "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sources": {}}
    failures = []

    for sid in [s for s in args.sources.split(",") if s]:
        src = by_id[sid]
        # The Well's trainer and package.py need its stats.yaml and <name>.yaml; other YAML (PDEArena's
        # generator seed configs, including the test shards') stays off the host
        well_meta = ("stats.yaml", f"{sid[len('well-'):]}.yaml") if sid.startswith("well-") else ()
        files = [f for f in src["files"] if f["kind"] == "hdf5" or f["path"].endswith((".json", "README.md")) or f["path"] in well_meta]
        files = [f for f in files if not wanted_files or Path(f["path"]).name in wanted_files]
        if case_files:
            files = [f for f in files if f["path"] in case_files.get(sid, set()) or f["path"].endswith("meta_all.json") or f["path"] in well_meta]
        files = [f for f in files if not wanted_splits or f.get("split") in wanted_splits or f["kind"] != "hdf5"]
        total = sum(f["bytes"] for f in files)
        print(f"[{sid}] {src['repo']} @ {src['revision'][:12]}: {len(files)} files, {total/1e9:.2f} GB", file=sys.stderr)
        if args.plan:
            for f in files:
                print(f"    {f['path']}  {f['bytes']/1e9:.3f} GB")
            continue
        root = dest / sid
        root.mkdir(parents=True, exist_ok=True)
        entries = []
        for f in files:
            local = root / f["path"]
            if not args.verify_only and not local.exists():
                from huggingface_hub import hf_hub_download
                hf_hub_download(repo_id=src["repo"], repo_type="dataset", revision=src["revision"],
                                filename=f["path"], local_dir=str(root))
            entry = {"path": f["path"], "bytes_expected": f["bytes"], "sha256_expected": f["sha256"]}
            if not local.exists():
                entry["status"] = "missing"; failures.append(f"{sid}:{f['path']} missing")
                entries.append(entry); continue
            entry["bytes"] = local.stat().st_size
            head = release_head(local) if f["kind"] == "hdf5" else None
            if head is not None:
                # a truncated training head from the published release (materialize.py --from-release --subtree
                # train): deliberately not the upstream file, so no upstream size or digest can match it. It is
                # accepted when the split's pools ask for it, which is what an agent host does.
                if args.pools:
                    check_corpus_readable(local)
                    entry.update({"status": "release-head", "release": head})
                else:
                    entry["status"] = "release-head-not-upstream"
                    failures.append(f"{sid}:{f['path']} is a truncated release head ({head}); a verifier host needs the upstream file")
                entries.append(entry); continue
            if entry["bytes"] != f["bytes"]:      # metadata files carry no LFS digest; their size is still checked
                entry["status"] = "size-mismatch"; failures.append(f"{sid}:{f['path']} lock says {f['bytes']} bytes, file has {entry['bytes']}")
                entries.append(entry); continue
            if f["sha256"]:
                entry["sha256"] = sha256(local)
                if entry["sha256"] != f["sha256"]:
                    entry["status"] = "sha256-mismatch"; failures.append(f"{sid}:{f['path']} sha256 mismatch")
                    entries.append(entry); continue
            if f["kind"] == "hdf5":
                n, shape = read_trajectory_count(local, src["family"])
                entry["n_trajectories"] = n
                entry["trajectory_shape"] = list(shape)
                if f.get("n_trajectories_inferred") not in (None, n):
                    entry["status"] = "count-mismatch"
                    failures.append(f"{sid}:{f['path']} lock inferred {f.get('n_trajectories_inferred')} trajectories, file has {n}")
                    entries.append(entry); continue
            entry["status"] = "ok"
            entries.append(entry)
            print(f"    ok  {f['path']}  n={entry.get('n_trajectories','-')}", file=sys.stderr)
        receipt["sources"][sid] = {"repo": src["repo"], "revision": src["revision"], "files": entries,
                                   "ok": sum(e["status"] in ("ok", "release-head") for e in entries), "total": len(entries)}
    if args.plan:
        return
    receipt["ok"] = not failures
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "materialize-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
        sys.exit(1)
    print(f"receipt written to {dest / 'materialize-receipt.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
