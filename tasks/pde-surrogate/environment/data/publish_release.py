#!/usr/bin/env python3
"""Upload a built release to the Hub and write the pin that materialize.py reads.

    python3 publish_release.py --release RELEASE_DIR --repo deqing/ai4sci-pde-surrogate-data
    python3 publish_release.py --release ... --repo ... --dry-run     # list, upload nothing

Maintainers only; Harbor never runs this. The upload is deliberately boring: create the repo
if it is missing, push the payload in batches (the Hub degrades past ~100 files per commit),
then push the manifest and inventory last so a half-finished upload never looks complete.

Afterwards it writes `download-source.json` - repo id, the revision the upload landed on, and
the manifest's SHA-256 - which is the whole download contract that lives in Git. A consumer
verifies the manifest against that hash, every file against the manifest, and nothing else is
trusted.
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BLOCK = 8 * 1024 * 1024
CARD = """---
license: other
license_name: mixed-upstream
pretty_name: PDE surrogate benchmark split
tags: [pde, surrogate, benchmark, scientific-ml]
---

# PDE surrogate: the benchmark split

Companion data for the `pde-surrogate` task in
[T0-RSI/ai4sci-tasks](https://github.com/T0-RSI/ai4sci-tasks). It carries **only the
files this benchmark's split creates**, not a mirror of the upstream datasets.

| subtree | what |
| --- | --- |
| `train/` | the nine corpus files that carry training *and* held-out trajectories, truncated at the end of the validation range, with `norm_*` recomputed over the kept trajectories |
| `eval/` | the held-out tail of each of those files, under its original trajectory numbering |

Every member keeps the `sims` group attributes (`PDE`, `Fields`, `Fields Scheme`, `Dt`, ...) and
was opened with the task's own reader (`environment/evaluation/pde_io.py`) when it was built.
Release `pde-surrogate-split-v2` replaces `pde-surrogate-split-v1` (revision `5bcba793`), whose
members were written without those attributes and which carried upstream's `hyp.hdf5`, a content
copy of `diff.hdf5`; the task's split no longer uses that file.

Everything else the task uses is byte-identical to a pinned, public upstream file and is
fetched from there: `thuerey-group/pde-transformer-ape2d-full` (MIT),
`polymathic-ai/turbulent_radiative_layer_2D` and `polymathic-ai/active_matter` (CC-BY-4.0),
and `pdearena/NavierStokes-2D` (MIT). Their licences and attribution apply unchanged; no
relicensing is claimed here, and this repository does not duplicate their bytes.

## Use

```bash
python3 environment/data/materialize.py DEST --from-release --subtree train   # agent host
python3 environment/data/materialize.py DEST --from-release                   # verifier host
```

`release-manifest.json` and `payload-files.jsonl` carry a SHA-256 per member; the task pins
this repository by revision and manifest hash in `environment/data/download-source.json`.

## Evaluation visibility

The task states: *"All upstream data are public, so this is a benchmark-visible evaluation,
not a secret test set."* `train/` and `eval/` are separated so that an agent environment
mounts only the former; together they reconstruct the upstream files exactly.
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(BLOCK), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--repo", required=True, help="e.g. deqing/ai4sci-pde-surrogate-data")
    ap.add_argument("--private", action="store_true", help="create the repo private (default public)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", type=int, default=40, help="files per commit")
    a = ap.parse_args()

    rel = Path(a.release)
    manifest = json.loads((rel / "release-manifest.json").read_text())
    records = [json.loads(l) for l in (rel / "payload-files.jsonl").read_text().splitlines() if l.strip()]
    members = [r for r in records if "read_from" not in r]
    if len(members) != len(records):
        print(f"note: {len(records) - len(members)} members are read from the materialization and are not uploaded")

    total = sum(r["size"] for r in members)
    print(f"release {manifest['release_id']} ({manifest.get('kind', '?')})")
    print(f"uploading {len(members)} files, {total / 1e9:.2f} GB, to {a.repo}")
    for r in members:
        print(f"  {r['path']:58s} {r['size'] / 1e9:7.3f} GB")
    if a.dry_run:
        print("\ndry run: nothing uploaded")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=a.private, exist_ok=True)
    (rel / "README.md").write_text(CARD)

    for i in range(0, len(members), a.batch):
        chunk = members[i:i + a.batch]
        api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=str(rel),
                          allow_patterns=[r["path"] for r in chunk],
                          commit_message=f"payload {i // a.batch + 1}: {len(chunk)} files")
        print(f"  pushed {i + len(chunk)}/{len(members)}")
    # manifest last, so a half-finished upload never looks complete
    api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=str(rel),
                      allow_patterns=["release-manifest.json", "payload-files.jsonl", "README.md"],
                      commit_message=f"release {manifest['release_id']}: manifest and inventory")

    revision = api.list_repo_commits(a.repo, repo_type="dataset")[0].commit_id
    pin = {"repo_id": a.repo, "revision": revision,
           "manifest_sha256": sha256_file(rel / "release-manifest.json")}
    (HERE / "download-source.json").write_text(json.dumps(pin, indent=1) + "\n")
    print(f"\npinned at {revision}")
    print(f"wrote {HERE / 'download-source.json'}")


if __name__ == "__main__":
    main()
