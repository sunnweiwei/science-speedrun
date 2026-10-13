#!/usr/bin/env python3
"""Pack a validated release into checksummed archives and upload it to a Hugging Face dataset.

    python publish_hf_release.py --data-root DATA_ROOT --repo-id OWNER/NAME --staging DIR

Layout on the Hub (the format of `agbench.hf_release`):
  archives/support.tar               everything except targets/ (manifests, labels, scoring queries,
                                     metadata, reference genome, variant evaluations)
  archives/train/part-NNNN-of-0250.tar   one deterministic tar per 8-bit Zarr train shard
  archives/{valid,test}/part-*.zarr.zip  full-precision target ZipStores, uploaded unchanged
  release-manifest.json              size and SHA-256 of every archive; uploaded last, so a partial
                                     upload is never a release

Archives are staged and uploaded in batches, then deleted, so staging needs about one batch of
space. A rerun resumes from `STAGING/state.json`. The token is read from HF_TOKEN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SCHEMA = "genome-sequence-to-function-hf/v4"
EXCLUDE = {"targets", "HF_MATERIALIZATION.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(64 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_tar(output: Path, root: Path, members: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "--sort=name", "--mtime=@0", "--owner=0", "--group=0", "--numeric-owner",
                    "--format=gnu", "-cf", str(output), "-C", str(root), *members], check=True)


def payload(root: Path, members: list[str]) -> tuple[int, int]:
    files = size = 0
    for member in members:
        path = root / member
        for item in ([path] if path.is_file() else path.rglob("*")):
            if item.is_file():
                files += 1
                size += item.stat().st_size
    return files, size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=12, help="archives per upload commit")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    root = args.data_root.resolve()
    report = json.loads((root / "validation_report.json").read_text())
    if report.get("status") != "PASS":
        raise SystemExit("validate the release first (validation_report.json must be PASS)")
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    args.staging.mkdir(parents=True, exist_ok=True)
    state_path = args.staging / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"artifacts": {}}

    # (group, archive path on the Hub, how to build it)
    plan: list[tuple[str, str, dict]] = []
    support = sorted(p.name for p in root.iterdir() if p.name not in EXCLUDE)
    plan.append(("support", "archives/support.tar", {"members": support}))
    for split in ("train", "valid", "test"):
        for part in sorted((root / "targets" / split).iterdir()):
            relative = str(part.relative_to(root))
            if part.is_dir():
                plan.append((split, f"archives/{split}/{part.name.removesuffix('.zarr')}.tar",
                             {"members": [relative]}))
            else:
                plan.append((split, f"archives/{split}/{part.name}", {"file": relative}))

    pending = [item for item in plan if item[1] not in state["artifacts"]]
    print(json.dumps({"event": "plan", "archives": len(plan), "pending": len(pending)}), flush=True)
    for start in range(0, len(pending), args.batch):
        batch = pending[start:start + args.batch]
        folder = args.staging / "batch"
        subprocess.run(["rm", "-rf", str(folder)], check=True)
        entries = []
        for group, remote, how in batch:
            local = folder / remote
            if "members" in how:
                deterministic_tar(local, root, how["members"])
                files, size = payload(root, how["members"])
                entry = {"group": group, "path": remote, "kind": "archive", "files": files,
                         "payload_bytes": size}
            else:
                local.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(root / how["file"], local)
                except OSError:
                    subprocess.run(["cp", str(root / how["file"]), str(local)], check=True)
                entry = {"group": group, "path": remote, "kind": "file", "install_path": how["file"]}
            entry.update(size=local.stat().st_size, sha256=sha256(local))
            entries.append(entry)
        api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=str(folder),
                          commit_message=f"archives {batch[0][1]} .. {batch[-1][1]}")
        for entry in entries:
            state["artifacts"][entry["path"]] = entry
        state_path.write_text(json.dumps(state, indent=1) + "\n")
        subprocess.run(["rm", "-rf", str(folder)], check=True)
        print(json.dumps({"event": "uploaded", "done": len(state["artifacts"]), "total": len(plan)}),
              flush=True)

    artifacts = [state["artifacts"][remote] for _, remote, _ in plan]
    manifest = {
        "schema": SCHEMA,
        "repo_id": args.repo_id,
        "release_id": f"genome-sequence-to-function-v4-{report.get('target_bytes', 0)}",
        "validation_status": report["status"],
        "payload_bytes": sum(a.get("payload_bytes", a["size"]) for a in artifacts),
        "artifacts": artifacts,
    }
    (args.staging / "release-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    card = Path(__file__).with_name("HF_DATASET_CARD.md")
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=args.repo_id,
                    repo_type="dataset", commit_message="dataset card")
    info = api.upload_file(path_or_fileobj=str(args.staging / "release-manifest.json"),
                           path_in_repo="release-manifest.json", repo_id=args.repo_id,
                           repo_type="dataset", commit_message="release manifest")
    revision = info.oid if hasattr(info, "oid") else str(info).rsplit("/", 1)[-1]
    print(json.dumps({"event": "published", "repo_id": args.repo_id, "revision": revision,
                      "manifest_sha256": sha256(args.staging / "release-manifest.json")}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
