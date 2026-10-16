"""Assemble, validate, publish and install the v2 data release (controller side, not agent-visible).

    python release.py assemble --pool POOL --physical-val DIR --physical-test DIR \
        --extra-dev DIR --extra-private DIR --model UMA.pt --mp-standard DIR --baselines DIR --output RELEASE
    python release.py validate --root RELEASE
    python release.py publish  --root RELEASE --repo-id OWNER/NAME --staging DIR     (HF_TOKEN)
    python release.py install  --root RELEASE --repo-id OWNER/NAME --revision SHA --manifest-sha256 SHA

Materialized layout (one directory serves as TASK_DATA_DIR and TASK_PRIVATE_DIR):

    pool/<source>/*.aselmdb, metadata.npz; pool/manifest.json, blocked.json
    val/<suite>/inputs/, val/<suite>/references/            agent-visible
    test/<suite>/inputs/                                     verifier main (label-free)
    test/<suite>/references/                                 grader only
    models/uma-s-1p1.pt, models/<baseline>/model.pt          starting checkpoint, baseline weights
    baselines/mp-standard/                                   original 500k MP sample, v2-filtered

Suites: physical (discovery, phonons, elasticity), kappa, mof, amorphous, cleavage, neb.
On the Hub, every pool shard is one file; the other trees are deterministic tars. release-manifest.json
(uploaded last) pins the size and SHA-256 of every artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

SCHEMA = "atomistic-materials-hf/v2"
SUITES = ("physical", "kappa", "mof", "amorphous", "cleavage", "neb")
EXTRA = SUITES[1:]


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(64 << 20):
            d.update(chunk)
    return d.hexdigest()


def place(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, copy_function=_link)
    else:
        _link(src, dst)


def _link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# ----------------------------------------------------------------------------------- assemble
def assemble(a):
    out = a.output
    if out.exists():
        raise FileExistsError(out)
    files = json.loads((a.pool / "manifest.json").read_text())["files"]
    for rel in sorted(files) + ["manifest.json", "blocked.json"]:
        place(a.pool / rel, out / "pool" / rel)
    for split, src in (("val", a.physical_val), ("test", a.physical_test)):
        place(src / "inputs", out / split / "physical" / "inputs")
        place(src / "references", out / split / "physical" / "references")
    for split, root in (("val", a.extra_dev), ("test", a.extra_private)):
        for suite in EXTRA:
            src = root / suite / split
            for f in src.rglob("*"):
                if f.is_dir():
                    continue
                kind = "references" if f.name.startswith("references.") else "inputs"
                place(f, out / split / suite / kind / f.relative_to(src))
    place(a.model, out / "models" / "uma-s-1p1.pt")
    place(a.mp_standard, out / "baselines" / "mp-standard")
    if a.baselines:
        for d in sorted(p for p in a.baselines.iterdir() if p.is_dir()):
            for f in d.iterdir():
                place(f, out / "models" / d.name / f.name)
    return validate(out)


def validate(root: Path) -> dict:
    report = {"status": "FAIL", "checks": {}}
    pool = json.loads((root / "pool" / "manifest.json").read_text())
    bad = [k for k, v in pool["files"].items() if (root / "pool" / k).stat().st_size != v["bytes"]]
    report["checks"]["pool_files"] = {"files": len(pool["files"]), "size_mismatch": bad[:5]}
    for split in ("val", "test"):
        for suite in SUITES:
            inputs, refs = root / split / suite / "inputs", root / split / suite / "references"
            ok = inputs.is_dir() and refs.is_dir() and any(refs.iterdir())
            leaked = [p.name for p in inputs.rglob("*") if p.name.startswith(("references", "targets"))]
            report["checks"][f"{split}/{suite}"] = {"ok": ok, "leaked": leaked}
    good = not bad and all(c.get("ok", True) and not c.get("leaked") for c in report["checks"].values()
                           if isinstance(c, dict) and "ok" in c)
    report["status"] = "PASS" if good and (root / "models" / "uma-s-1p1.pt").is_file() else "FAIL"
    report["pool_frames"] = sum(s["frames"] for s in pool["sources"].values())
    report["pool_bytes"] = pool["bytes"]
    (root / "validation_report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


# ----------------------------------------------------------------------------------- publish
def deterministic_tar(output: Path, root: Path, members: list[str]):
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "--sort=name", "--mtime=@0", "--owner=0", "--group=0", "--numeric-owner",
                    "--mode=u+rw,go+r,a+X", "--format=gnu", "-cf", str(output), "-C", str(root), *members], check=True)


def plan(root: Path):
    items = []
    for f in sorted((root / "pool").rglob("*")):
        if f.is_file():
            rel = str(f.relative_to(root))
            items.append(("pool", rel, {"file": rel}))
    items.append(("val", "archives/val.tar", {"members": ["val"]}))
    items.append(("test-inputs", "archives/test-inputs.tar",
                  {"members": [f"test/{s}/inputs" for s in SUITES]}))
    items.append(("test-references", "archives/test-references.tar",
                  {"members": [f"test/{s}/references" for s in SUITES]}))
    items.append(("models", "archives/models.tar", {"members": ["models"]}))
    items.append(("baselines", "archives/baselines.tar", {"members": ["baselines"]}))
    return items


def publish(a):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    report = json.loads((a.root / "validation_report.json").read_text())
    if report["status"] != "PASS":
        raise SystemExit("validate the release first")
    api.create_repo(a.repo_id, repo_type="dataset", private=a.private, exist_ok=True)
    a.staging.mkdir(parents=True, exist_ok=True)
    state_path = a.staging / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"artifacts": {}}
    items = plan(a.root)
    groups = set(a.groups.split(",")) if a.groups else None
    pending = [i for i in items if i[1] not in state["artifacts"] and (not groups or i[0] in groups)]
    for start in range(0, len(pending), a.batch):
        batch = pending[start:start + a.batch]
        folder = a.staging / "batch"
        shutil.rmtree(folder, ignore_errors=True)
        entries = []
        for group, remote, how in batch:
            local = folder / remote
            if "members" in how:
                deterministic_tar(local, a.root, how["members"])
                entry = {"group": group, "path": remote, "kind": "tar"}
            else:
                local.parent.mkdir(parents=True, exist_ok=True)
                _link(a.root / how["file"], local)
                entry = {"group": group, "path": remote, "kind": "file", "install_path": how["file"]}
            entry.update(size=local.stat().st_size, sha256=sha256(local))
            entries.append(entry)
        api.upload_folder(repo_id=a.repo_id, repo_type="dataset", folder_path=str(folder),
                          commit_message=f"{batch[0][1]} .. {batch[-1][1]}")
        for e in entries:
            state["artifacts"][e["path"]] = e
        state_path.write_text(json.dumps(state, indent=1) + "\n")
        shutil.rmtree(folder)
        print(json.dumps({"event": "uploaded", "done": len(state["artifacts"]), "total": len(items)}), flush=True)
    if groups:
        print(json.dumps({"event": "partial", "groups": sorted(groups)}))
        return
    manifest = {"schema": SCHEMA, "repo_id": a.repo_id, "validation": report,
                "artifacts": [state["artifacts"][r] for _, r, _ in items]}
    (a.staging / "release-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    api.upload_file(path_or_fileobj=str(Path(__file__).with_name("HF_DATASET_CARD.md")), path_in_repo="README.md",
                    repo_id=a.repo_id, repo_type="dataset", commit_message="dataset card")
    info = api.upload_file(path_or_fileobj=str(a.staging / "release-manifest.json"),
                           path_in_repo="release-manifest.json", repo_id=a.repo_id, repo_type="dataset",
                           commit_message="release manifest")
    print(json.dumps({"event": "published", "repo_id": a.repo_id, "revision": getattr(info, "oid", str(info)),
                      "manifest_sha256": sha256(a.staging / "release-manifest.json")}))


# ----------------------------------------------------------------------------------- install
def install(a):
    from huggingface_hub import hf_hub_download
    root = a.root
    tmp = root.with_name(root.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    cache = tmp / ".download"
    mpath = Path(hf_hub_download(a.repo_id, "release-manifest.json", repo_type="dataset", revision=a.revision,
                                 local_dir=cache))
    if sha256(mpath) != a.manifest_sha256:
        raise SystemExit("release-manifest.json does not match the pinned SHA-256")
    manifest = json.loads(mpath.read_text())
    groups = set(a.groups.split(",")) if a.groups else None
    for art in manifest["artifacts"]:
        if groups and art["group"] not in groups:
            continue
        marker = tmp / ".done" / (art["path"].replace("/", "__") + ".ok")
        if marker.exists():
            continue
        local = Path(hf_hub_download(a.repo_id, art["path"], repo_type="dataset", revision=a.revision,
                                     local_dir=cache))
        if local.stat().st_size != art["size"] or sha256(local) != art["sha256"]:
            raise SystemExit(f"checksum mismatch: {art['path']}")
        if art["kind"] == "tar":
            with tarfile.open(local) as t:
                t.extractall(tmp, filter="data")
            local.unlink()
        else:
            dst = tmp / art["install_path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(local, dst)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(art["sha256"])
        print(json.dumps({"event": "installed", "path": art["path"]}), flush=True)
    shutil.rmtree(cache, ignore_errors=True)
    shutil.rmtree(tmp / ".done", ignore_errors=True)
    if not groups:
        report = validate(tmp)
        if report["status"] != "PASS":
            raise SystemExit("installed release failed validation")
    os.replace(tmp, root)
    print(json.dumps({"event": "complete", "root": str(root)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("assemble")
    for k in ("pool", "physical-val", "physical-test", "extra-dev", "extra-private", "model", "mp-standard", "output"):
        s.add_argument("--" + k, type=Path, required=True)
    s.add_argument("--baselines", type=Path)
    s = sub.add_parser("validate"); s.add_argument("--root", type=Path, required=True)
    s = sub.add_parser("publish")
    s.add_argument("--root", type=Path, required=True); s.add_argument("--repo-id", required=True)
    s.add_argument("--staging", type=Path, required=True); s.add_argument("--batch", type=int, default=40)
    s.add_argument("--private", action="store_true")
    s.add_argument("--groups", help="upload only these groups now (no manifest is published)")
    s = sub.add_parser("install")
    s.add_argument("--root", type=Path, required=True); s.add_argument("--repo-id", required=True)
    s.add_argument("--revision", required=True); s.add_argument("--manifest-sha256", required=True)
    s.add_argument("--groups", help="comma list of pool,val,test-inputs,test-references,models,baselines (default all)")
    a = ap.parse_args()
    if a.cmd == "assemble":
        print(json.dumps(assemble(a), indent=1))
    elif a.cmd == "validate":
        print(json.dumps(validate(a.root), indent=1))
    elif a.cmd == "publish":
        publish(a)
    else:
        install(a)


if __name__ == "__main__":
    sys.exit(main())
