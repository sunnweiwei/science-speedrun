"""Prepare pinned HF data: selected pretraining, point probes, and native train/val."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def safe_path(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or ".." in p.parts or "\\" in name:
        raise ValueError("Unsafe release path")
    return p


def verify(path, entry):
    if path.stat().st_size != entry["size"] or digest(path) != entry["sha256"]:
        raise ValueError(f"Integrity failure: {entry['path']}")


def copy_stream(source, target):
    """Resume only when an existing member matches its verified archive bytes."""
    if target.is_symlink():
        raise ValueError("Refusing symlink output")
    if target.exists():
        with target.open("rb") as old:
            while True:
                chunk = source.read(8 * 1024 * 1024)
                if old.read(len(chunk)) != chunk:
                    raise ValueError("Existing prepared file differs")
                if not chunk:
                    if old.read(1):
                        raise ValueError("Existing prepared file is longer")
                    return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".partial")
    if tmp.is_symlink():
        raise ValueError("Refusing symlink temporary output")
    with tmp.open("wb") as out:
        shutil.copyfileobj(source, out, length=8 * 1024 * 1024)
    tmp.replace(target)


def prepare_pretraining(archives, selection, output):
    rows = selection["samples"]
    if len(rows) != selection["kept"] or [r["sample_index"] for r in rows] != list(
        range(len(rows))
    ):
        raise ValueError("Invalid selected sample indices")
    wanted = {r["source"]: r["sample_index"] for r in rows}
    if len(wanted) != len(rows) or set(wanted) & set(selection["excluded"]):
        raise ValueError("Invalid inclusion/exclusion mapping")
    if len(rows) + len(selection["excluded"]) != selection["source_samples"]:
        raise ValueError("Wrong source coverage")
    seen = set()
    for path in archives:
        with tarfile.open(path, "r:") as tar:
            for member in tar:
                tar.members.clear()
                name = safe_path(member.name).name
                if member.issym() or member.islnk():
                    raise ValueError("Archive links are not allowed")
                if not member.isfile() or name not in wanted:
                    continue
                if name in seen:
                    raise ValueError("Duplicate selected sample")
                seen.add(name)
                with tar.extractfile(member) as source:
                    copy_stream(source, output / f"sample_{wanted[name]}.h5")
    if seen != set(wanted):
        raise ValueError("Incomplete pretraining subset")
    return len(seen)


def prepare_windows(archive, output, expected):
    decisions = {}
    counts = {"train": 0, "val": 0, "test": 0}
    with tarfile.open(archive, "r:") as tar:
        for member in tar:
            tar.members.clear()
            parts = safe_path(member.name).parts
            if not member.isfile() or "windows" not in parts:
                continue
            i = parts.index("windows")
            if parts[i + 3 :] != ("metadata.json",):
                continue
            with tar.extractfile(member) as f:
                meta = json.load(f)
            key = "/".join(parts[i + 1 : i + 3])
            split = meta["options"]["split"]
            if key in decisions or split not in counts:
                raise ValueError("Duplicate window or invalid split")
            decisions[key] = split
            counts[split] += 1
    if counts != expected:
        raise ValueError(f"Wrong split counts: {counts}")
    files = 0
    size = 0
    with tarfile.open(archive, "r:") as tar:
        for member in tar:
            tar.members.clear()
            parts = safe_path(member.name).parts
            if member.issym() or member.islnk():
                raise ValueError("Archive links are not allowed")
            if not member.isfile():
                continue
            if "windows" in parts:
                i = parts.index("windows")
                key = "/".join(parts[i + 1 : i + 3])
                if key not in decisions:
                    raise ValueError("Payload without window metadata")
                if decisions[key] == "test":
                    continue
                relative = Path(*parts[i:])
            elif parts[-1] == "config.json" and len(parts) == 2:
                relative = Path("config.json")
            else:
                continue
            with tar.extractfile(member) as source:
                copy_stream(source, output / relative)
            files += 1
            size += member.size
    if not (output / "config.json").exists():
        raise ValueError("Missing native dataset config")
    result = dict(
        window_counts=counts,
        extracted_splits=["train", "val"],
        heldout_window_payloads_read=False,
        files=files,
        payload_bytes_written=size,
    )
    (output / "extraction.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="sanxing/olmoearth-starter-data")
    p.add_argument(
        "--revision",
        required=True,
        help="Release commit or tag; resolved once to an immutable commit",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument(
        "--components",
        nargs="+",
        choices=["pretrain", "probe", "downstream"],
        default=["pretrain", "probe", "downstream"],
    )
    p.add_argument("--download-only", action="store_true")
    a = p.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    revision = HfApi().dataset_info(a.repo, revision=a.revision).sha

    def fetch(name):
        safe_path(name)
        return Path(
            hf_hub_download(
                a.repo,
                name,
                repo_type="dataset",
                revision=revision,
                cache_dir=a.cache_dir,
            )
        )

    previous = a.output / "hf-prepared.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        if old["repo"] != a.repo or old["revision"] != revision:
            raise ValueError("Use a new output for a different release revision")
    manifest_path = fetch("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported release manifest")
    files = {}
    entries = {}
    for entry in manifest["files"]:
        name = entry["path"]
        safe_path(name)
        if name in entries:
            raise ValueError("Duplicate manifest path")
        entries[name] = entry
        if set(entry["components"]) & set(a.components):
            file = fetch(name)
            verify(file, entry)
            files[name] = file
            print("verified", name, flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    result = {
        "repo": a.repo,
        "revision": revision,
        "manifest_sha256": digest(manifest_path),
        "components": a.components,
        "download_only": a.download_only,
    }
    if not a.download_only:
        if "pretrain" in a.components:
            subset = json.loads(files["pretrain/selection.json"].read_text())
            target = a.output / str(subset["kept"])
            prepare_pretraining(
                [files[f"archives/h5py_data-{i:04d}.tar"] for i in range(2)],
                subset,
                target,
            )
            for name in [
                "sample_metadata.csv",
                "latlon_distribution.npy",
                "selection.json",
            ]:
                dest = "provenance.json" if name == "selection.json" else name
                with files[f"pretrain/{name}"].open("rb") as src:
                    copy_stream(src, target / dest)
            result["pretrain_samples"] = subset["kept"]
        if "probe" in a.components:
            for name, src in files.items():
                if name.startswith("eval/"):
                    with src.open("rb") as f:
                        copy_stream(f, a.output / name)
        if "downstream" in a.components:
            for task, expected in [
                ("awf", {"train": 1115, "val": 344, "test": 0}),
                ("lfmc", {"train": 31175, "val": 8262, "test": 4585}),
            ]:
                prepare_windows(
                    files[f"archives/{task}.tar"],
                    a.output / f"{task}-downstream-trainval",
                    expected,
                )
            # Native LFMC trainer consumes this exact indexed window layout.
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "downstream"))
            from lfmc_batch import build_index

            import tempfile

            index_path = a.output / "lfmc-downstream-index.json"
            with tempfile.TemporaryDirectory(dir=a.output) as tmp:
                checked = Path(tmp) / "index.json"
                build_index(a.output / "lfmc-downstream-trainval", checked)
                if index_path.exists():
                    if index_path.read_bytes() != checked.read_bytes():
                        raise ValueError(
                            "Existing LFMC index differs from prepared windows"
                        )
                else:
                    checked.replace(index_path)
    marker = a.output / ("hf-download.json" if a.download_only else "hf-prepared.json")
    marker.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
