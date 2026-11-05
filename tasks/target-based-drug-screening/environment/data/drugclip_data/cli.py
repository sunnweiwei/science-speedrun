from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .biolip import (download_biolip2, filter_biolip_contamination, make_split,
                     prepare_biolip2_structures)
from .processed import build_processed_data
from .profsa import prepare_profsadb
from .report import build_report
from .smoke import create_smoke_inputs
from .testdata import build_test_index, download_test_data
from .validate import validate_dataset


def validate_materialization_root(root: Path) -> None:
    """Reject broad host paths while permitting the container's bind mount."""
    container_data_root = Path("/data")
    if (root == Path("/") or root == Path.home() or
            (len(root.parts) < 3 and root != container_data_root)):
        raise SystemExit(f"refusing unsafe materialization target: {root}")


def clear_materialization_root(root: Path) -> None:
    """Remove a materialization's contents without deleting its mount point."""
    if not root.exists():
        return
    for child in root.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def run_all(root: Path, smoke: bool, overwrite: bool, biolip_lock: Path | None,
            profsa_source_fasta: Path | None) -> None:
    # `/data` is the dedicated bind mount in the supplied container. On the
    # host, continue rejecting broad/shallow targets such as `/tmp` or `/var`.
    validate_materialization_root(root)
    if overwrite and root.exists():
        # The Docker launcher bind-mounts the destination at `/data`. A mount
        # point cannot itself be removed, so preserve the root and clear only
        # its children.
        clear_materialization_root(root)
    root.mkdir(parents=True, exist_ok=True)
    def stage(message: str) -> None:
        print(f"[target-screening-data] {message}", flush=True)
    if smoke:
        stage("creating synthetic smoke inputs (NOT TRAINING DATA)")
        create_smoke_inputs(root)
    else:
        stage("downloading and freezing author-released DUD-E/LIT-PCBA")
        download_test_data(root)
    stage("building test protein and ligand indices")
    build_test_index(root)
    if not smoke:
        stage("downloading pinned BioLiP2/Q-BioLiP snapshot")
        download_biolip2(root, biolip_lock)
    stage("preparing canonical BioLiP2 structures, pockets, and conformers")
    prepare_biolip2_structures(root)
    stage("filtering BioLiP2 test contamination")
    filter_biolip_contamination(root)
    stage("clustering and freezing BioLiP2 train/dev splits")
    make_split(root)
    stage("filtering and selecting fixed ProFSADB subset")
    prepare_profsadb(root, profsa_source_fasta, download=not smoke,
                     smoke_count=2 if smoke else None)
    stage("materializing derived LMDB views")
    build_processed_data(root)
    stage("validating all dataset invariants")
    receipt = validate_dataset(root, smoke=smoke)
    stage("writing dataset report")
    build_report(root, smoke=smoke)
    (root / ".complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
    stage("complete")


def main(default_command: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Target-based drug-screening data materializer")
    parser.add_argument("command", nargs="?", default=default_command,
                        choices=["release", "all", "download-test", "index-test", "download-biolip2",
                                             "prepare-biolip2", "filter", "split", "prepare-profsa",
                                             "processed", "validate", "report"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--biolip-lock", type=Path)
    parser.add_argument("--profsa-source-fasta", type=Path)
    args = parser.parse_args()
    root = args.data_root.resolve()
    actions = {
        "download-test": lambda: download_test_data(root),
        "index-test": lambda: build_test_index(root),
        "download-biolip2": lambda: download_biolip2(root, args.biolip_lock),
        "prepare-biolip2": lambda: prepare_biolip2_structures(root),
        "filter": lambda: filter_biolip_contamination(root),
        "split": lambda: make_split(root),
        "prepare-profsa": lambda: prepare_profsadb(root, args.profsa_source_fasta,
                                                    download=not args.smoke,
                                                    smoke_count=2 if args.smoke else None),
        "processed": lambda: build_processed_data(root),
        "validate": lambda: validate_dataset(root, args.smoke),
        "report": lambda: build_report(root, args.smoke),
    }
    if args.command == "release":
        if args.smoke:
            run_all(root, True, args.overwrite, args.biolip_lock,
                    args.profsa_source_fasta)
            return
        from .hf_release import HubUnavailable, data_source, materialize_huggingface
        if data_source() == "huggingface":
            try:
                materialize_huggingface(root)
                return
            except HubUnavailable as error:
                print(f"HF dataset unavailable ({error}); using original sources",
                      flush=True)
        if (root / ".complete.json").is_file():
            validate_dataset(root)
        else:
            run_all(root, False, args.overwrite, args.biolip_lock,
                    args.profsa_source_fasta)
    elif args.command == "all":
        run_all(root, args.smoke, args.overwrite, args.biolip_lock, args.profsa_source_fasta)
    else:
        actions[args.command]()


if __name__ == "__main__":
    main()
