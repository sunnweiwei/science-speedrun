#!/usr/bin/env python3
"""Trusted CrystalGen grader; never imports or executes submission code."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from main_verifier import validate_final_suite
from scoring.evaluate import evaluate, write_results


EXPECTED_ASSETS = {
    "lemat_bulk": "0dc17eea904b860ad7288141e9870f67f8e6bb2c",
    "mlip_hull": "70d505bb294c16658a9551e2b81ace69a28d9790",
    "uma": "f611b917d9c68566bbbeccbb0aa0f7cad1696cb2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "READY.json"):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(sha256(path).encode() + b"\n")
        count += 1
        size += path.stat().st_size
    return digest.hexdigest(), count, size


def validate_snapshot(sealed: Path, input_root: Path) -> None:
    ready = sealed.parent / "READY"
    if ready.read_text().strip() != sha256(sealed / "snapshot.json"):
        raise ValueError("snapshot READY hash mismatch")
    snapshot = json.loads((sealed / "snapshot.json").read_text())
    if snapshot.get("schema") != "crystalgen-output-snapshot/v0":
        raise ValueError("invalid output snapshot schema")
    if snapshot.get("requests_sha256") != sha256(input_root / "requests.jsonl"):
        raise ValueError("snapshot was produced for different requests")
    expected_files = set(snapshot.get("files", {})) | {"snapshot.json"}
    actual_files = {
        path.relative_to(sealed).as_posix()
        for path in sealed.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("sealed snapshot file set changed")
    for relative, expected_hash in snapshot["files"].items():
        path = sealed / relative
        if path.is_symlink() or not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"sealed output hash mismatch: {relative}")


def validate_assets(asset_root: Path) -> None:
    ready = json.loads((asset_root / "READY.json").read_text())
    if (ready.get("schema") != "crystalgen-scorer-assets/v0"
            or not ready.get("hull_license_review_acknowledged")
            or not ready.get("mlips_warmed")):
        raise ValueError("scorer assets are absent or not approved/warmed")
    for name, revision in EXPECTED_ASSETS.items():
        if ready.get("assets", {}).get(name, {}).get("revision") != revision:
            raise ValueError(f"wrong scorer asset revision: {name}")
    observed = tree_digest(asset_root)
    declared = (ready.get("tree_sha256"), ready.get("file_count"), ready.get("byte_count"))
    if observed != declared:
        raise ValueError("scorer asset tree does not match its prepared digest")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--input", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--assets", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--output", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--scorer-root", type=lambda value: Path(value).resolve(), default=Path("/opt/lemat"))
    parser.add_argument("--scorer-python", type=lambda value: Path(value).resolve(), default=Path("/opt/lemat/.venv/bin/python"))
    parser.add_argument("--fixture-results", type=lambda value: Path(value).resolve())
    parser.add_argument("--allow-small-suite", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=172800)
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait_seconds
    while not (args.sealed.parent / "READY").is_file():
        if (args.sealed.parent / "INVALID").is_file():
            print("main verifier rejected the submission; nothing to score")
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("sealed output was not published")
        time.sleep(1)
    validate_final_suite(args.input, args.allow_small_suite)
    validate_snapshot(args.sealed, args.input)
    if args.fixture_results is None:
        validate_assets(args.assets)
    rewards, report = evaluate(
        args.input,
        args.sealed,
        args.scorer_root,
        args.scorer_python,
        fixture_results=args.fixture_results,
        artifact_root=args.output,
    )
    report["evaluation_mode"] = "sealed_replay_uma200_then_exact_lemat"
    write_results(args.output, rewards, report)


if __name__ == "__main__":
    main()
