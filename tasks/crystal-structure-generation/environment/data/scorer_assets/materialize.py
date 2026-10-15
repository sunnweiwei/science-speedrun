#!/usr/bin/env python3
"""Download pinned evaluator snapshots and optionally warm MLIP caches."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


ASSETS = {
    "lemat_bulk": ("dataset", "LeMaterial/LeMat-Bulk", "0dc17eea904b860ad7288141e9870f67f8e6bb2c"),
    "mlip_hull": ("dataset", "LeMaterial/LeMat-Bulk-MLIP-Hull", "70d505bb294c16658a9551e2b81ace69a28d9790"),
    "uma": ("model", "facebook/UMA", "f611b917d9c68566bbbeccbb0aa0f7cad1696cb2"),
}


def tree_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "READY.json"):
        relative = path.relative_to(root).as_posix().encode()
        item_digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                item_digest.update(block)
        file_hash = item_digest.hexdigest().encode()
        digest.update(relative + b"\0" + file_hash + b"\n")
        count += 1
        size += path.stat().st_size
    return digest.hexdigest(), count, size


def warm_mlips() -> None:
    from lemat_genbench.models.mace.calculator import create_mace_calculator
    from lemat_genbench.models.orb.calculator import create_orb_calculator
    from lemat_genbench.models.uma.calculator import create_uma_calculator

    create_orb_calculator(model_type="orb_v3_conservative_inf_omat", device="cpu")
    create_mace_calculator(model_type="mp", device="cpu")
    create_uma_calculator(model_name="uma-s-1", task="omat", device="cpu")


def pin_main_ref(snapshot: Path, revision: str) -> None:
    """Make default-branch calls resolve to the audited commit while offline."""
    repository_cache = snapshot.parent.parent
    refs = repository_cache / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(revision)


def validate_datasets() -> None:
    """Force the exact configs used by the scorer through the datasets loader."""
    from datasets import load_dataset

    load_dataset(
        "LeMaterial/LeMat-Bulk", "compatible_pbe",
        revision=ASSETS["lemat_bulk"][2],
    )
    load_dataset(
        "LeMaterial/LeMat-Bulk-MLIP-Hull",
        revision=ASSETS["mlip_hull"][2],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--acknowledge-hull-license-review", action="store_true")
    parser.add_argument("--warm-mlips", action="store_true")
    parser.add_argument("--token-file", type=lambda value: Path(value).resolve())
    args = parser.parse_args()
    if not args.acknowledge_hull_license_review:
        parser.error("the unlicensed hull snapshot requires explicit operator review")

    args.output_root.mkdir(parents=True, exist_ok=True)
    ready = args.output_root / "READY.json"
    if ready.exists():
        raise FileExistsError("asset directory already has READY.json; validate or use a new directory")
    hf_home = args.output_root / "huggingface"
    hf_hub = hf_home / "hub"
    scorer_home = args.output_root / "home"
    home_cache = scorer_home / ".cache"
    xdg_cache = args.output_root / "xdg-cache"
    torch_cache = args.output_root / "torch-cache"
    for path in (hf_hub, home_cache, xdg_cache, torch_cache):
        path.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    if args.token_file is not None:
        token = args.token_file.read_text().strip()
    if not token:
        original_hf_home = Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
        )
        token_path = Path(os.environ.get("HF_TOKEN_PATH", original_hf_home / "token"))
        if token_path.is_file():
            token = token_path.read_text().strip()
    if not token:
        raise RuntimeError(
            "no Hugging Face token found; run `hf auth login` or pass --token-file"
        )

    os.environ.update({
        # MACE and ORB use hard-coded paths below ~/.cache. Keep HOME inside
        # the sealed asset tree so warm-up artifacts survive materialization.
        "HOME": str(scorer_home),
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_hub),
        "XDG_CACHE_HOME": str(xdg_cache),
        "TORCH_HOME": str(torch_cache),
        "HF_HUB_DISABLE_XET": "1",
        "HF_TOKEN": token,
    })
    fairchem_cache = home_cache / "fairchem"
    if fairchem_cache.exists() or fairchem_cache.is_symlink():
        if fairchem_cache.is_dir() and not fairchem_cache.is_symlink():
            shutil.rmtree(fairchem_cache)
        else:
            fairchem_cache.unlink()
    fairchem_cache.symlink_to(Path("../../huggingface/hub"), target_is_directory=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required for asset preparation") from error

    resolved = {}
    for name, (repo_type, repo_id, revision) in ASSETS.items():
        snapshot = Path(snapshot_download(
            repo_id=repo_id,
            repo_type=None if repo_type == "model" else repo_type,
            revision=revision,
            cache_dir=hf_hub,
            token=token,
        ))
        pin_main_ref(snapshot, revision)
        resolved[name] = str(snapshot)
    validate_datasets()
    if args.warm_mlips:
        # FairChem loads UMA by name from the default branch. Online, that re-points refs/main at
        # the live Hub head (facebook/UMA moved past the audited commit), so warm offline: the
        # pinned refs/main then resolves to the audited snapshot. ORB and MACE do not use the Hub.
        # The Hub client caches its HTTP session, so the online session from the downloads above
        # must be dropped for offline mode to apply.
        import huggingface_hub.constants
        from huggingface_hub.utils import reset_sessions
        os.environ["HF_HUB_OFFLINE"] = "1"
        huggingface_hub.constants.HF_HUB_OFFLINE = True
        reset_sessions()
        warm_mlips()

    digest, file_count, byte_count = tree_digest(args.output_root)
    document = {
        "schema": "crystalgen-scorer-assets/v0",
        "assets": {
            name: {"repo_type": kind, "repo_id": repo, "revision": revision}
            for name, (kind, repo, revision) in ASSETS.items()
        },
        "hull_license_review_acknowledged": True,
        "mlips_warmed": args.warm_mlips,
        "tree_sha256": digest,
        "file_count": file_count,
        "byte_count": byte_count,
    }
    ready.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
