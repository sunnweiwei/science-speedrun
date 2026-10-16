"""Download pinned public releases and build the reduced train/full eval data."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent


def verify(path, info):
    if path.stat().st_size != info["size"]:
        raise ValueError(f"Wrong size: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != info["sha256"]:
        raise ValueError(f"Checksum mismatch: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--verify-downloads-only", action="store_true")
    p.add_argument(
        "--pretrain-source",
        choices=["subset_1k", "shard0000", "two-shards"],
        default="subset_1k",
    )
    a = p.parse_args()
    root = a.output.resolve()
    downloads = root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    sources = json.loads((HERE / "sources.json").read_text())
    archive_names = (
        ["subset_1k_h5py_data.tar"]
        if a.pretrain_source == "subset_1k"
        else ["h5py_data-0000.tar"]
    )
    if a.pretrain_source == "two-shards":
        archive_names.append("h5py_data-0001.tar")
    sources = {
        name: info
        for name, info in sources.items()
        if name in {*archive_names, "awf.tar", "lfmc.tar", "lfmc_annotations.geojson"}
    }
    for name, info in sources.items():
        target = downloads / name
        if not target.exists():
            if a.verify_downloads_only:
                raise FileNotFoundError(target)
            partial = target.with_suffix(target.suffix + ".partial")
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "3",
                    "--continue-at",
                    "-",
                    "--output",
                    str(partial),
                    info["url"],
                ],
                check=True,
            )
            verify(partial, info)
            partial.rename(target)
        else:
            verify(target, info)
        print("verified", name, flush=True)
    if a.verify_downloads_only:
        return
    if a.pretrain_source == "subset_1k":
        archive_name = archive_names[0]
        pretrain = root / "pretrain"
        pretrain.mkdir(exist_ok=True)
        marker = pretrain / "extracted.json"
        if not marker.exists():
            with tarfile.open(downloads / archive_name) as archive:
                archive.extractall(pretrain, filter="data")
            marker.write_text(json.dumps(sources[archive_name]))
        metadata = list(pretrain.rglob("sample_metadata.csv"))
        if len(metadata) != 1:
            raise ValueError("Ambiguous upstream HDF5 root")
        source_dir = metadata[0].parent
        filtered = root / "pretrain-filtered"
        expected_samples = 3996
    else:
        shard_dirs = []
        for archive_name in archive_names:
            shard_id = archive_name.removeprefix("h5py_data-").removesuffix(".tar")
            directory = root / f"pretrain-shard{shard_id}"
            marker = directory / "source.json"
            if not marker.exists():
                subprocess.run(
                    [
                        sys.executable,
                        str(HERE / "extract_h5_shard.py"),
                        "--archive",
                        str(downloads / archive_name),
                        "--sha256",
                        sources[archive_name]["sha256"],
                        "--output",
                        str(directory),
                    ],
                    check=True,
                )
            source_release = json.loads(marker.read_text())
            if (
                source_release["sha256"] != sources[archive_name]["sha256"]
                or source_release["samples"] != sources[archive_name]["source_samples"]
            ):
                raise ValueError("Extracted source marker does not match pinned shard")
            shard_dirs.append(directory)
        expected_samples = sum(
            sources[name]["source_samples"] for name in archive_names
        )
        source_dir = shard_dirs[0]
        filtered = root / "pretrain-shard0000-filtered"
        if a.pretrain_source == "two-shards":
            source_dir = root / "pretrain-shards0000-0001"
            if not source_dir.exists():
                subprocess.run(
                    [
                        sys.executable,
                        str(HERE / "combine_h5_shards.py"),
                        "--sources",
                        *map(str, shard_dirs),
                        "--output",
                        str(source_dir),
                    ],
                    check=True,
                )
            combined = json.loads((source_dir / "source.json").read_text())
            if (
                combined["samples"] != expected_samples
                or not combined["unique_source_ids"]
                or {s["sha256"] for s in combined["sources"]}
                != {sources[name]["sha256"] for name in archive_names}
            ):
                raise ValueError("Combined source does not match the pinned shards")
            filtered = root / "pretrain-two-shards-filtered"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "prepare_pretrain.py"),
            "--source",
            str(source_dir),
            "--expected-source-samples",
            str(expected_samples),
            "--lfmc-annotations",
            str(downloads / "lfmc_annotations.geojson"),
            "--output",
            str(filtered),
        ],
        check=True,
    )
    count = json.loads((filtered / "provenance.json").read_text())["kept"]
    link = root / str(count)
    if link.exists() and link.resolve() != filtered.resolve():
        raise ValueError("Numeric loader alias points to another source")
    if not link.exists():
        link.symlink_to(filtered.name)
    for task in ["awf", "lfmc"]:
        dest = root / "eval" / task
        if not (dest / "release.json").exists():
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "prepare_eval.py"),
                    "--tar",
                    str(downloads / f"{task}.tar"),
                    "--task",
                    task,
                    "--output",
                    str(dest),
                    "--workers",
                    str(a.workers),
                ],
                check=True,
            )
        if not (dest / "inputs.h5").is_file() or not (dest / "labels.json").is_file():
            raise ValueError(f"Incomplete {task}")
    (root / "release.json").write_text(
        json.dumps(
            {
                "sources": sources,
                "pretrain_samples": count,
                "pretrain_source": a.pretrain_source,
                "protocol": "docs/PROTOCOL.md",
            },
            indent=2,
        )
    )
    print("Prepared", root, flush=True)


if __name__ == "__main__":
    main()
