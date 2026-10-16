"""Materialize released rslearn windows without extracting heldout window files."""

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def extract(archive, output, expected=None):
    if output.exists():
        raise ValueError("Use a new dataset output; never overwrite partial runs")
    output.mkdir(parents=True)
    decisions = {}
    counts = {"train": 0, "val": 0, "test": 0}
    files = 0
    bytes_written = 0
    # Tar member ordering is not a metadata-before-payload contract. First index
    # only split metadata; seek past every imagery/label payload in this pass.
    with tarfile.open(archive, "r:") as tar:
        for member in tar:
            tar.members.clear()
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or "windows" not in parts:
                continue
            index = parts.index("windows")
            if parts[index + 3 :] != ("metadata.json",):
                continue
            metadata = json.load(tar.extractfile(member))
            key = "/".join(parts[index + 1 : index + 3])
            split = metadata["options"]["split"]
            if split not in counts or key in decisions:
                raise ValueError("Invalid split or duplicate metadata")
            decisions[key] = split
            counts[split] += 1
    if expected is not None and counts != expected:
        raise ValueError(f"Wrong window coverage: {counts}")
    print({"indexed_windows": counts}, flush=True)
    with tarfile.open(archive, "r:") as tar:
        for member in tar:
            tar.members.clear()
            parts = PurePosixPath(member.name).parts
            if (
                not parts
                or ".." in parts
                or member.name.startswith("/")
                or member.issym()
                or member.islnk()
            ):
                raise ValueError("Unsafe archive member")
            if not member.isfile():
                continue
            relative = Path(*parts[1:])
            if "windows" in parts:
                index = parts.index("windows")
                if len(parts) < index + 4:
                    raise ValueError("Unexpected window layout")
                key = "/".join(parts[index + 1 : index + 3])
                if key not in decisions:
                    raise ValueError("Window payload has no split metadata")
                if decisions[key] == "test":
                    continue
            elif relative.as_posix() != "config.json":
                continue
            dest = output / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise ValueError("Duplicate output member")
            with tar.extractfile(member) as src, dest.open("wb") as target:
                shutil.copyfileobj(src, target)
            files += 1
            bytes_written += member.size
            if files % 100000 == 0:
                print({"files": files, "windows": counts}, flush=True)
            tar.members.clear()
    if expected is not None and counts != expected:
        raise ValueError(f"Wrong window coverage: {counts}")
    if not (output / "config.json").exists():
        raise ValueError("Missing rslearn config")
    report = {
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "window_counts": counts,
        "extracted_splits": ["train", "val"],
        "heldout_window_payloads_read": False,
        "files": files,
        "payload_bytes_written": bytes_written,
    }
    (output / "extraction.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(
        json.dumps(
            extract(a.archive, a.output, {"train": 31175, "val": 8262, "test": 4585}),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
