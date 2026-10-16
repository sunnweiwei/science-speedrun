"""Plan, fetch and verify pinned HydroGym assets. No downloads by default."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


CATALOG = Path(__file__).with_name("catalog.json")


def file_digest(path, algorithm):
    digest = hashlib.new(algorithm)
    if algorithm == "sha1":
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(root, files):
    root = root.resolve()
    for item in files:
        relative = Path(item["rfilename"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"Escaping resource: {relative}")
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError(f"Missing/truncated resource: {relative}")
        lfs = item.get("lfs")
        expected = lfs["sha256"] if lfs else item["blobId"]
        if file_digest(path, "sha256" if lfs else "sha1") != expected:
            raise ValueError(f"Resource checksum mismatch: {relative}")


def download_source(catalog, release, names):
    """Separate the byte-identical delivery mirror from scientific provenance."""
    from public_release import digest
    upstream = {"repo_id": catalog["repo_id"], "revision": catalog["revision"],
                "path_prefix": ""}
    path = CATALOG.with_name("download-source.json")
    if not path.exists():
        return upstream
    mirror = json.loads(path.read_text())
    if (mirror.get("data_release_id") != release["release_id"] or
            not set(names).issubset(release["training"]["case_ids"])):
        # Historical releases/inventory diagnostics still use their full source.
        return upstream
    if mirror.get("data_release_sha256") != digest(release):
        raise ValueError("Download mirror is bound to another frozen release")
    if (not isinstance(mirror.get("repo_id"), str) or
            not re.fullmatch(r"[\w.-]+/[\w.-]+", mirror["repo_id"]) or
            not isinstance(mirror.get("revision"), str) or
            not re.fullmatch(r"[0-9a-f]{40}", mirror["revision"]) or
            mirror.get("path_prefix") != "environments"):
        raise ValueError("Download mirror requires a pinned Hub commit and environments prefix")
    return {key: mirror[key] for key in ("repo_id", "revision", "path_prefix")}


def main():
    from public_release import load_release, digest, validate_index

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--release", type=Path, help="Explicit versioned manifest; default is the task's current release")
    parser.add_argument("--inventory", action="store_true",
                        help="Prepare all source directories for diagnostics, not the frozen benchmark")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--from-local", type=Path,
                      help="Create a selected read-only-use data view with hard links, then verify; requires the same filesystem")
    args = parser.parse_args()
    catalog = json.loads(CATALOG.read_text())
    release = load_release(args.release) if args.release else load_release()
    index = json.loads(CATALOG.with_name("source-files.json").read_text())
    validate_index(index)
    if digest(index) != release["source_index_sha256"]:
        raise ValueError("Source file index differs from frozen release")
    known = {row["name"]: row for row in catalog["directories"]}
    if args.case and args.inventory:
        parser.error("--case and --inventory are mutually exclusive")
    names = sorted(set(args.case) or (known if args.inventory else release["training"]["case_ids"]))
    if set(names) - known.keys():
        parser.error(f"Unknown resource directories: {sorted(set(names) - known.keys())}")
    destination = args.destination.expanduser().resolve()
    delivery = download_source(catalog, release, names)
    prefix = delivery["path_prefix"]
    local_dir = destination if prefix else destination / "environments"
    command = ["hf", "download", delivery["repo_id"], "--type", "dataset",
               "--revision", delivery["revision"], "--local-dir", str(local_dir)]
    # This hf release accepts one --include string; repeated flags overwrite it.
    commands = [command + ["--include", f"{prefix + '/' if prefix else ''}{name}/*"] for name in names]
    whole_release = set(release["training"]["case_ids"]).issubset(names)
    print(json.dumps({"data_release_id": release["release_id"],
                      "complete_release_selection": whole_release,
                      "directories": names, "bytes": sum(known[n]["bytes"] for n in names),
                      "download_source": delivery,
                      "download_commands": commands, "mode": "download" if args.download else
                      "verify" if args.verify_only else "local-reuse" if args.from_local else "plan"}, indent=2))
    if not args.download and not args.verify_only and not args.from_local:
        return
    destination.mkdir(parents=True, exist_ok=True)
    receipt = destination / "materialized.json"
    # Never replace a different release in a shared data root.
    published = destination / "release.json"
    if published.exists() and json.loads(published.read_text()) != release:
        raise ValueError("Destination contains another release; use a separate data root")
    previous = json.loads(receipt.read_text()) if receipt.exists() else {}
    if args.download:
        # A downloader may replace selected files. Invalidate before mutation.
        receipt.unlink(missing_ok=True)
        # Local-dir downloads still use Xet's global cache unless redirected.
        # Keep it on the selected data disk, not a possibly root-owned home cache.
        download_env = dict(os.environ)
        download_env.setdefault("HF_XET_CACHE", str(destination / ".cache/huggingface/xet"))
        for command in commands:
            subprocess.run(command, check=True, env=download_env)
    # The checked-in Hub file metadata makes --verify-only fully offline and
    # reproducible even if the remote repository changes or disappears.
    files = [f for f in index["files"] if f["rfilename"].split("/")[0] in names]
    if not files or any(not any(f["rfilename"].startswith(n + "/") for f in files) for n in names):
        raise ValueError("Incomplete upstream file metadata")
    if args.from_local:
        source_root = (args.from_local.expanduser().resolve() / "environments").resolve()
        target_root = (destination / "environments").resolve()
        for item in files:
            relative = Path(item["rfilename"])
            source = (source_root / relative).resolve()
            target = target_root / relative
            if not source.is_relative_to(source_root) or not target.resolve().is_relative_to(target_root):
                raise ValueError("Escaping local data resource")
            if not source.is_file() or source.stat().st_size != item["size"]:
                raise ValueError(f"Missing/truncated local source: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                # No implicit terabyte copy. Inputs are immutable and mounted
                # read-only during research; verify their bytes below.
                os.link(source, target)
            elif not target.samefile(source):
                raise ValueError(f"Refusing to replace an existing local resource: {target}")
    try:
        verify_files(destination / "environments", files)
    except ValueError:
        # A read-only recheck leaves a prior receipt available while it runs,
        # but a detected file mismatch must invalidate that receipt.
        receipt.unlink(missing_ok=True)
        raise
    verified = set(names)
    if previous.get("revision") == catalog["revision"]:
        verified.update(previous.get("verified_directories", []))
    value = {"repo_id": catalog["repo_id"], "revision": catalog["revision"],
             "verified_directories": sorted(verified),
             "verified_files": sum(f["rfilename"].split("/")[0] in verified for f in index["files"]),
             "benchmark_validated": False}
    # The receipt's original source identity remains compatible with the frozen
    # evaluator. Record the actual delivery repository separately, never relabel
    # unchanged physics/data as a new scientific release.
    if args.download:
        value["download_source"] = delivery
    elif previous.get("download_source"):
        value["download_source"] = previous["download_source"]
    if whole_release:
        if not published.exists():
            pending_release = published.with_suffix(".pending.json")
            pending_release.write_text(json.dumps(release, indent=2) + "\n")
            os.replace(pending_release, published)
        value.update(data_release_id=release["release_id"], data_release_sha256=digest(release),
                     source_index_sha256=digest(index))
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, receipt)
    print("Pinned resource hashes verified offline. Data selection is frozen; native full-suite execution is a separate acceptance check.")


if __name__ == "__main__":
    main()
