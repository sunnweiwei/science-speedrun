#!/usr/bin/env python3
"""Build a public release of the task's four sources for Hugging Face, split agent side from verifier side.

    python3 build_release.py --data $TASK_PRIVATE_DIR --out RELEASE_DIR
    python3 build_release.py --data ... --out ... --verify-only   # re-hash, rebuild nothing

The release has two subtrees, and which one a host fetches is what separates an agent
environment from a verifier environment:

  train/   what an agent may see: the training and validation pools, and nothing else
  eval/    what the verifier needs: the held-out files and the held-out trajectory tails

Publishing both is consistent with what the task already claims - "All upstream data are
public, so this is a benchmark-visible evaluation, not a secret test set" (instruction.md).
Anyone can download the MIT-licensed corpus and read trajectory 500 of burgers.hdf5. What
must never happen is an *agent* seeing eval, and that is now enforced by which subtree the
host materializes rather than by trusting a flag.

`train/` is built as follows:

  * files referenced only by held-out cases go to `eval/`, not `train/`;
  * the nine corpus files that carry training *and* held-out trajectories are written
    truncated, keeping `sims/sim0 .. sim{keep-1}` where `keep` is the end of the
    validation range. The held-out trajectories are always a contiguous tail, so this is
    exact rather than a filter;
  * the `norm_*` statistics inside a truncated file are recomputed over the kept
    trajectories, because upstream's were computed over all of them and would otherwise
    carry aggregate information about the held-out tail into a public file.

`eval/` carries the test-only files whole, plus, for each truncated corpus file, a companion
holding only the held-out tail - so the two subtrees together reconstruct the upstream file
and neither alone gives an agent the answers.

Nothing under `pretrained/` is ever included: those checkpoints are pinned downloads with
their own licences, and two of the audited baseline families carry no licence at all.

Unlike foldbench's release this one is **not** tarred. That release had 186k small feature
files and needed sharding to avoid a flat folder on the Hub; this one is 18 derived members
(the truncated head and the held-out tail of each of the nine cut corpus files) of 0.4-4.4 GB,
where a tar layer would only defeat per-file resumption and range requests.
The manifest carries a SHA-256 per file, which is what `materialize.py --from-release`
checks, and is the same guarantee under a simpler layout.
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_SOURCES = ("ape2d-full", "pdearena-ns", "well-active_matter", "well-turbulent_radiative_layer_2D")
FORMAT = "pde-surrogate-release/v1"
BLOCK = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(BLOCK), b""):
            h.update(b)
    return h.hexdigest()


# "test" as a whole token anywhere in the path: `pde_test_seed_*.yaml`, `data/test/...`.
# Matching a bare substring would also catch "latest", so the boundaries matter.
HELD_OUT_RE = re.compile(r"(?:^|[/_\-])test(?:[/_\-.]|$)", re.I)


def is_held_out_metadata(path: str) -> bool:
    """True for a file that describes the held-out set rather than the training pools."""
    return HELD_OUT_RE.search(path) is not None


def public_payload(lock: dict, split: dict):
    """(source, path) -> (subtree, keep).

    subtree is "train" or "eval"; keep is the trajectory count for a file that must be cut,
    or None for one that ships whole. Grouping mirrors foldbench's release-manifest groups.
    """
    pools = defaultdict(list)
    for name in ("train", "valid"):
        for e in split["pools"][name]:
            pools[(e["source"], e["file"])].append(tuple(e["trajectories"]))
    touched = {(c["source"], c["file"]) for c in split["cases"] if "file" in c}
    info = {(s["id"], f["path"]): f for s in lock["sources"] for f in s.get("files", [])}

    out = {}
    for key in sorted(pools):
        if key[0] not in TASK_SOURCES:
            continue
        if key not in info:
            raise SystemExit(f"{key} is in the split but not in the lock")
        keep = max(hi for _, hi in pools[key])
        out[key] = ("train", keep if key in touched else None)
    for key in sorted(touched):                      # held-out files the pools never reference
        if key[0] in TASK_SOURCES and key not in out and key in info:
            out[key] = ("eval", None)
    # Metadata siblings (README, per-family .json, generator configs) travel with their source, but
    # only the ones that do not describe the held-out set. PDEArena ships `pde_test_seed_*.yaml`,
    # the generator seeds for the test trajectories, and instruction.md forbids regenerating held-out
    # cases from them - publishing those files would hand an agent exactly that.
    for s in lock["sources"]:
        if s["id"] not in TASK_SOURCES:
            continue
        for f in s.get("files", []):
            if f.get("kind") != "metadata":
                continue
            # metadata that describes the held-out set belongs with eval, never with train:
            # pdearena's pde_test_seed_*.yaml are the generator seeds for the test trajectories
            subtree = "eval" if is_held_out_metadata(f["path"]) else "train"
            out.setdefault((s["id"], f["path"]), (subtree, None))
    return out, info


def write_truncated(src: Path, dst: Path, keep: int) -> int:
    """Copy sims/sim0..sim{keep-1}, recompute norm_* over them. Returns the source count."""
    import h5py
    import numpy as np

    dst.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        sims_in = fin["sims"]
        n_src = max(int(n[3:]) for n in sims_in if n.startswith("sim")) + 1
        if n_src < keep:
            raise SystemExit(f"{src.name}: asked to keep {keep} trajectories, file has {n_src}")
        g = fout.create_group("sims", track_order=True)
        for k, v in sims_in.attrs.items():       # PDE, Fields, Fields Scheme, Dt, ...: pde_io reads these
            g.attrs[k] = v
        acc_n, acc_sum, acc_sqs, vmin, vmax = 0, None, None, None, None
        for i in range(keep):
            a = sims_in[f"sim{i}"][()]
            d = g.create_dataset(f"sim{i}", data=a)
            for k, v in sims_in[f"sim{i}"].attrs.items():
                d.attrs[k] = v
            flat = a.astype(np.float64).reshape(a.shape[0], a.shape[1], -1)
            s, q = flat.sum(axis=(0, 2)), (flat ** 2).sum(axis=(0, 2))
            acc_sum = s if acc_sum is None else acc_sum + s
            acc_sqs = q if acc_sqs is None else acc_sqs + q
            acc_n += flat.shape[0] * flat.shape[2]
            lo, hi = flat.min(axis=(0, 2)), flat.max(axis=(0, 2))
            vmin = lo if vmin is None else np.minimum(vmin, lo)
            vmax = hi if vmax is None else np.maximum(vmax, hi)
        mean = acc_sum / acc_n
        std = np.sqrt(np.maximum(acc_sqs / acc_n - mean ** 2, 0.0))
        sh = lambda v: v.reshape(-1, 1, 1)
        for name, val, dt in (("norm_fields_sca_mean", sh(mean), "float64"),
                              ("norm_fields_sca_std", sh(std), "float64"),
                              ("norm_fields_sca_min", sh(vmin), "float64"),
                              ("norm_fields_sca_max", sh(vmax), "float64"),
                              ("norm_fields_std", sh(std), "float32")):
            if name in fin:
                fout.create_dataset(name, data=val.astype(dt))
        for name in fin:                      # constants and anything else at root, copied as-is
            if name != "sims" and name not in fout:
                fout.create_dataset(name, data=fin[name][()])
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        fout.attrs["pde_surrogate_public_release"] = (
            f"training-only: trajectories 0..{keep-1} of {n_src}; norm_* recomputed over those")
    return n_src


def write_tail(src: Path, dst: Path, keep: int) -> int:
    """The other half of a truncated file: sims/sim{keep}.. under their original names.

    Original numbering is kept so a reader that asks for trajectory 500 finds sim500, and so
    train/ and eval/ together reconstruct the upstream file exactly.
    """
    import h5py

    dst.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        sims_in = fin["sims"]
        n_src = max(int(n[3:]) for n in sims_in if n.startswith("sim")) + 1
        g = fout.create_group("sims", track_order=True)
        for k, v in sims_in.attrs.items():
            g.attrs[k] = v
        for i in range(keep, n_src):
            d = g.create_dataset(f"sim{i}", data=sims_in[f"sim{i}"][()])
            for k, v in sims_in[f"sim{i}"].attrs.items():
                d.attrs[k] = v
        for name in fin:
            if name != "sims":
                fout.create_dataset(name, data=fin[name][()])
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        fout.attrs["pde_surrogate_public_release"] = (
            f"held-out tail: trajectories {keep}..{n_src - 1} of {n_src}, original numbering")
    return n_src - keep


def validate_member(path: Path) -> dict:
    """Open a derived member with the task's own reader (environment/evaluation/pde_io.py).

    The first release shipped every derived file without the `sims` group attributes
    (PDE, Fields, Fields Scheme, Dt) because the writers copied datasets but not attributes,
    and no test read a member back with the reader the verifier and the starter use. This
    does; a member that `read_window` cannot open is a build failure, not a warning.
    """
    import h5py
    sys.path.insert(0, str(HERE.parent / "evaluation"))
    from pde_io import read_window  # noqa: E402

    with h5py.File(path, "r") as f:
        sims = sorted((n for n in f["sims"] if n.startswith("sim")), key=lambda n: int(n[3:]))
        missing = [k for k in ("PDE", "Fields", "Fields Scheme", "Dt") if k not in f["sims"].attrs]
        if missing:
            raise SystemExit(f"{path}: `sims` lacks the attributes {missing}; the task's readers cannot open it")
        first = int(sims[0][3:])
    frames, meta = read_window("ape2d", path, first, 0, 1, 0)
    if frames.ndim != 4 or not meta.get("fields"):
        raise SystemExit(f"{path}: read_window returned an unusable window")
    return {"validated_with": "pde_io.read_window", "first_trajectory": first, "n_trajectories": len(sims),
            "pde": meta["pde"], "fields": meta["fields"]}


def resolve(data: Path, source: str, name: str):
    """Find a source file on disk.

    materialize.py writes `<dest>/<source id>/<path>`, but a materialization made before that
    convention keeps The Well under `hf/well/<dataset>/`, so `well-active_matter` lives at
    `hf/well/active_matter`. Try the documented layouts first and that split last.
    """
    head, _, tail = source.partition("-")
    for candidate in (data / "hf" / source / name,
                      data / source / name,
                      data / "hf" / head / tail / name if tail else None):
        if candidate is not None and candidate.exists():
            return candidate
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="a materialization made by materialize.py")
    ap.add_argument("--out", required=True, help="release directory to write")
    ap.add_argument("--release-id", default="pde-surrogate-split-v2",
                    help="v1 (5bcba793) shipped members without the sims attributes and carried the duplicated hyp family")
    ap.add_argument("--verify-only", action="store_true", help="re-hash an existing release, write nothing new")
    ap.add_argument("--limit", type=int, default=0, help="only the first N payload files (for tests)")
    ap.add_argument("--min-free-gb", type=float, default=20.0,
                    help="stop before a write that would leave less than this much free")
    ap.add_argument("--mirror", action="store_true",
                    help="also carry the members that are byte-identical to an upstream file. Off by default: "
                         "everything the split does not rewrite already exists at pinned, public, MIT/CC-BY "
                         "upstream repositories and materialize.py fetches it from there with the same SHA-256 "
                         "check. A release therefore carries only what this task's split creates - the truncated "
                         "train heads and the eval tails - about 39 GB rather than the 185 GB of the four sources.")
    ap.add_argument("--assemble", action="store_true",
                    help="also copy the files that ship whole into --out. Off by default: those bytes already "
                         "exist in the materialization, and duplicating 50 GB to upload it once is wasteful. "
                         "The manifest records where each one lives so the uploader can read it in place.")
    a = ap.parse_args()

    lock = json.loads((HERE / "sources.lock.json").read_text())
    split = json.loads((HERE / "split.json").read_text())
    payload, info = public_payload(lock, split)
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    metadata_keys = {k for k, v in payload.items()
                     if v[1] is None and info.get(k, {}).get("kind") == "metadata"}
    items = sorted(payload.items())
    if not a.mirror:   # keep only members this task's split creates; the rest come from upstream
        items = [(k, v) for k, v in items if v[1] is not None]
    if a.limit:
        items = items[:a.limit]

    inventory = []
    for (source, name), (subtree, keep) in items:
        src = resolve(data, source, name)
        if src is None:
            if keep is None and (source, name) in metadata_keys:
                print(f"  skipped (not materialized): {source}/{name}")
                continue
            raise SystemExit(f"missing from the materialization: {source}/{name}")

        # a truncated corpus file yields two members: the head under train/, the tail under eval/
        plan = [(subtree, keep)] if keep is None else [("train", keep), ("eval", -keep)]
        for tree, cut in plan:
            dst = out / tree / source / name
            stage = cut is not None or a.assemble
            if stage and not a.verify_only and not dst.exists():
                free = shutil.disk_usage(out).free
                need = src.stat().st_size
                if free - need < a.min_free_gb * 1e9:
                    print(f"stopping before {tree}/{source}/{name}: {free/1e9:.1f} GB free, need "
                          f"{need/1e9:.1f} GB and a {a.min_free_gb:.0f} GB floor. Re-run to continue "
                          f"after freeing space.", file=sys.stderr)
                    raise SystemExit(2)
            if stage and not a.verify_only:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if cut is None:
                    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                        shutil.copy2(src, dst)
                elif not dst.exists():
                    if cut > 0:
                        write_truncated(src, dst, cut)
                    else:
                        write_tail(src, dst, -cut)
            if stage and not dst.exists():
                raise SystemExit(f"release is missing {tree}/{source}/{name}; run without --verify-only first")
            member = dst if stage else src
            rec = {"path": f"{tree}/{source}/{name}", "subtree": tree,
                   "size": member.stat().st_size, "sha256": sha256_file(member)}
            if cut is not None:
                rec["trajectories"] = ("0..%d" % (cut - 1)) if cut > 0 else ("%d.." % -cut)
                rec["upstream_trajectories"] = info[(source, name)].get("n_trajectories_inferred")
                checked = validate_member(dst)
                if checked["n_trajectories"] != (cut if cut > 0 else rec["upstream_trajectories"] + cut):
                    raise SystemExit(f"{dst}: holds {checked['n_trajectories']} trajectories, expected "
                                     f"{cut if cut > 0 else rec['upstream_trajectories'] + cut}")
                rec["readable"] = checked
            rec["upstream_sha256"] = info[(source, name)].get("sha256")
            if not stage:
                rec["read_from"] = str(src)
            inventory.append(rec)
            mark = f"  [{rec['trajectories']}]" if cut is not None else ""
            print(f"  {rec['path']:68s} {rec['size']/1e9:7.3f} GB{mark}")

    inv = out / "payload-files.jsonl"
    inv.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in inventory))
    manifest = {
        "format": FORMAT, "release_id": a.release_id,
        "kind": "mirror" if a.mirror else "derived-only",
        "upstream_note": ("this release carries only the members this task's split creates. Every other file is "
                          "byte-identical to a pinned public upstream file and materialize.py fetches it there, "
                          "verified against the same SHA-256 recorded in sources.lock.json."
                          if not a.mirror else "this release mirrors every member, upstream-identical ones included"),
        "staged_locally": sum(1 for r in inventory if "read_from" not in r),
        "read_from_materialization": sum(1 for r in inventory if "read_from" in r),
        "contract": split["contract"], "split_schema": split["schema"],
        "payload_files": len(inventory), "payload_bytes": sum(r["size"] for r in inventory),
        "groups": {tree: {"files": sum(1 for r in inventory if r["subtree"] == tree),
                          "bytes": sum(r["size"] for r in inventory if r["subtree"] == tree),
                          "meaning": ("what an agent environment may mount" if tree == "train"
                                      else "what the verifier needs; never mounted for an agent")}
                   for tree in ("train", "eval")},
        "per_source": {s: {"files": sum(1 for r in inventory if r["path"].split("/")[1] == s),
                           "bytes": sum(r["size"] for r in inventory if r["path"].split("/")[1] == s)}
                       for s in TASK_SOURCES},
        "split_files": sum(1 for r in inventory if "trajectories" in r) // 2,
        "separation": "train/ holds the training and validation pools only; eval/ holds the held-out files and the "
                      "held-out trajectory tails. Together they reconstruct the upstream files; neither alone gives "
                      "an agent the answers. Nothing under pretrained/ is included: those checkpoints are pinned "
                      "downloads with their own licences, two of them unlicensed.",
        "inventory": {"path": "payload-files.jsonl", "size": inv.stat().st_size, "sha256": sha256_file(inv)},
    }
    (out / "release-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"\n{len(inventory)} members, {manifest['payload_bytes']/1e9:.2f} GB")
    for tree, g in manifest["groups"].items():
        print(f"  {tree:6s} {g['files']:4d} files  {g['bytes']/1e9:7.2f} GB")
    print(f"manifest sha256: {sha256_file(out / 'release-manifest.json')}")


if __name__ == "__main__":
    main()
