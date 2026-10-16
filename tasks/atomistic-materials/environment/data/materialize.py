"""Download explicit components of the immutable HF release on the operator host."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

import requests

PIN = Path(__file__).with_name("release.json")
GROUPS = (
    "training",
    "protocol",
    "records",
    "initial-model",
    "trained-models",
    "eval-val",
    "eval-test",
    "eval-regression",
    "licenses",
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def within(root, name):
    parts = PurePosixPath(name)
    if parts.is_absolute() or ".." in parts.parts or not parts.parts:
        raise ValueError(f"Unsafe release path: {name}")
    result = root.joinpath(*parts.parts).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes output directory: {name}")
    return result


def checked(path, size, digest):
    if path.stat().st_size != size or sha256(path) != digest:
        raise ValueError(f"Release checksum/size mismatch: {path}")


def fetch(url, target, expected_sha, expected_bytes=None, token=None):
    if target.exists():
        checked(
            target,
            expected_bytes if expected_bytes is not None else target.stat().st_size,
            expected_sha,
        )
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    temporary = None
    try:
        with requests.get(
            url, headers=headers, stream=True, timeout=(30, 180)
        ) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                size = 0
                for block in response.iter_content(1 << 20):
                    size += len(block)
                    if size > (
                        expected_bytes if expected_bytes is not None else 1 << 20
                    ):
                        raise ValueError("Release download exceeds declared size")
                    stream.write(block)
        checked(
            temporary,
            expected_bytes if expected_bytes is not None else size,
            expected_sha,
        )
        temporary.replace(target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def unpack(source, destination, item):
    """Verify before publishing; never leave a partially written dataset file."""
    if destination.exists():
        checked(destination, item["unpacked_bytes"], item["unpacked_sha256"])
        return
    checked(source, item["bytes"], item["sha256"])
    if item["encoding"] not in ("gzip", "identity"):
        raise ValueError("Unknown release encoding")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        opener = gzip.open if item["encoding"] == "gzip" else open
        with (
            opener(source, "rb") as reader,
            tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as writer,
        ):
            temporary = Path(writer.name)
            size = 0
            while block := reader.read(1 << 20):
                size += len(block)
                if size > item["unpacked_bytes"]:
                    raise ValueError("Unpacked data exceed declared size")
                writer.write(block)
        checked(temporary, item["unpacked_bytes"], item["unpacked_sha256"])
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def materialize(root, groups, token=None):
    pin = json.loads(PIN.read_text())
    if not re.fullmatch(r"[0-9a-f]{40}", pin["revision"]):
        raise ValueError("HF revision must be an immutable commit")
    if not set(groups) <= set(GROUPS):
        raise ValueError("Unknown data component")
    base = f"https://huggingface.co/datasets/{pin['repo_id']}/resolve/{pin['revision']}"
    cache = root / ".hf-cache" / pin["revision"]
    manifest = fetch(
        base + "/manifest.json",
        cache / "manifest.json",
        pin["manifest_sha256"],
        token=token,
    )
    release = json.loads(manifest.read_text())
    required = release.get("required_groups", [])
    if not set(required) <= {"licenses"}:
        raise ValueError("Only license notices may be implicit components")
    groups = list(dict.fromkeys([*groups, *required]))
    files = [item for item in release["files"] if item["group"] in groups]
    if {item["group"] for item in files} != set(groups):
        raise ValueError("Release is missing a requested component")
    destinations = [within(root, item["destination"]) for item in files]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Duplicate release destination")
    for item, target in zip(files, destinations):
        if target.exists():
            checked(target, item["unpacked_bytes"], item["unpacked_sha256"])
        else:
            cached = within(cache, item["path"])
            fetch(
                base + "/" + item["path"], cached, item["sha256"], item["bytes"], token
            )
            unpack(cached, target, item)
        print(f"Verified {target}", flush=True)
    return {
        "repo_id": pin["repo_id"],
        "revision": pin["revision"],
        "components": groups,
        "files": len(files),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".data"))
    parser.add_argument(
        "--components", nargs="+", choices=GROUPS, default=["training", "protocol"]
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        help="Operator credential; never mount into model runtime",
    )
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    credential = args.token_file or Path(
        os.environ.get(
            "HF_TOKEN_PATH",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
            / "token",
        )
    )
    if credential.is_file():
        token = credential.read_text().strip()
    print(json.dumps(materialize(args.root, args.components, token), indent=2))
