"""CLI for the single 1-Mb, 4,400-example benchmark protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .constants import SPLIT_TO_UPSTREAM
from .errors import ContractError
from .hf_release import HuggingFaceUnavailable, install_huggingface_release
from .reference import fetch_reference
from .source_index import index_source
from .materialize import materialize_smoke, materialize_split
from .pipeline import bootstrap, inspect
from .release import export_release, migrate_layout
from .validate import validate_release


def _splits(value: str) -> list[str]:
    result = value.split(",")
    unknown = set(result) - set(SPLIT_TO_UPSTREAM)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown splits: {sorted(unknown)}")
    return result


def _json(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agbench")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("bootstrap", "fetch-reference", "inspect", "smoke", "index", "materialize", "migrate-release", "validate", "install-hf"):
        item = sub.add_parser(name)
        item.add_argument("--data-dir", type=Path, required=True)
        if name in {"inspect", "index", "validate"}:
            item.add_argument("--splits", type=_splits, default=list(SPLIT_TO_UPSTREAM))
        if name == "migrate-release":
            item.add_argument("--splits", type=_splits, default=list(SPLIT_TO_UPSTREAM))
    sub.choices["inspect"].add_argument("--probe", action="store_true")
    sub.choices["smoke"].add_argument("--split", choices=tuple(SPLIT_TO_UPSTREAM), default="valid")
    sub.choices["materialize"].add_argument("--split", choices=tuple(SPLIT_TO_UPSTREAM), required=True)
    sub.choices["materialize"].add_argument("--workers", type=int, default=1)
    sub.choices["validate"].add_argument("--allow-incomplete", action="store_true")
    sub.choices["install-hf"].add_argument("--repo-id", required=True)
    sub.choices["install-hf"].add_argument("--revision", required=True)
    sub.choices["install-hf"].add_argument("--manifest-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "bootstrap":
            _json(bootstrap(args.data_dir))
        elif args.command == "fetch-reference":
            fasta, fai = fetch_reference(args.data_dir)
            _json({"status": "PASS", "fasta": str(fasta), "fai": str(fai)})
        elif args.command == "inspect":
            report = inspect(args.data_dir, args.splits, probe=args.probe)
            _json({
                "status": report["status"],
                "schema": report["schema"],
                "source_bytes": report["inventory"]["source_bytes"],
                "object_count": len(report["inventory"]["objects"]),
                "record_probes": report["record_probes"],
                "full_report": str(args.data_dir / "metadata" / "upstream_inspection.json"),
            })
        elif args.command == "smoke":
            _json(materialize_smoke(args.data_dir, split=args.split))
        elif args.command == "index":
            _json(index_source(args.data_dir, args.splits))
        elif args.command == "materialize":
            _json(materialize_split(
                args.data_dir, split=args.split, max_shards=None, workers=args.workers
            ))
        elif args.command == "migrate-release":
            migration = migrate_layout(args.data_dir)
            release = export_release(args.data_dir, args.splits, resume=True)
            _json({"migration": migration, "release": release})
        elif args.command == "validate":
            _json(validate_release(args.data_dir, splits=args.splits, require_complete=not args.allow_incomplete))
        elif args.command == "install-hf":
            _json(install_huggingface_release(
                args.data_dir, repo_id=args.repo_id, revision=args.revision,
                manifest_sha256=args.manifest_sha256
            ))
        return 0
    except HuggingFaceUnavailable as exc:
        print(f"HF_UNAVAILABLE: {exc}", file=sys.stderr)
        return 3
    except (ContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
