#!/usr/bin/env python3
"""Train and publish the deliberately weak CrystalGen smoke generator."""

import argparse
import ast
import csv
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile


MP20_TRAIN_SHA256 = "133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d"


def swap_into(staged: Path, final: Path) -> None:
    """Replace ``final`` by ``staged`` with one rename, so a reader never sees a partial package."""
    old = final.with_name(f".{final.name}.old-{os.getpid()}")
    if final.exists():
        try:
            os.replace(final, old)
        except OSError as error:
            # A directory from a lower image layer cannot be renamed on overlayfs (EXDEV): move its
            # contents aside instead; the swap below is still a single rename.
            if error.errno != errno.EXDEV:
                raise
            shutil.copytree(final, old, symlinks=True)
            shutil.rmtree(final)
    os.replace(staged, final)
    shutil.rmtree(old, ignore_errors=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train(path: Path) -> dict:
    element_counts = {}
    atom_counts = {}
    densities = []
    rows = 0
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"elements", "pretty_formula", "density"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"training CSV is missing {sorted(required)}")
        try:
            from pymatgen.core import Composition
        except ImportError as error:
            raise RuntimeError("pymatgen is required to train the smoke generator") from error
        for row in reader:
            rows += 1
            elements = ast.literal_eval(row["elements"])
            if not isinstance(elements, list) or not elements:
                raise ValueError("invalid elements column")
            for element in elements:
                element_counts[element] = element_counts.get(element, 0) + 1
            count = max(1, int(round(float(Composition(row["pretty_formula"]).num_atoms))))
            atom_counts[str(min(count, 20))] = atom_counts.get(str(min(count, 20)), 0) + 1
            density = float(row["density"])
            if math.isfinite(density) and density > 0:
                densities.append(math.log(density))
    if rows == 0 or not densities:
        raise ValueError("empty training data")
    mean = sum(densities) / len(densities)
    variance = sum((value - mean) ** 2 for value in densities) / len(densities)
    return {
        "schema": "crystalgen-smoke-model/v0",
        "rows": rows,
        "element_counts": element_counts,
        "atom_counts": atom_counts,
        "log_density_mean": mean,
        "log_density_std": math.sqrt(variance),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--submission-dir", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--starter-root", required=True, type=lambda value: Path(value).resolve())
    args = parser.parse_args()
    args.submission_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".crystalgen-submission-", dir=args.submission_dir.parent))
    try:
        model = train(args.train_csv)
        model_path = temporary / "model.json"
        model_path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
        for name in ("generate.sh", "generate.py"):
            shutil.copyfile(args.starter_root / name, temporary / name)
        (temporary / "README.md").write_text(
            "# CrystalGen smoke submission\n\n"
            "A deliberately weak aggregate-statistics generator used for interface testing.\n"
        )
        parameter_count = len(model["element_counts"]) + len(model["atom_counts"]) + 2
        manifest = {
            "schema": "crystalgen-model/v0",
            "checkpoint": {"path": "model.json", "sha256": sha256(model_path), "format": "json"},
            "parameter_count": parameter_count,
            "one_parameter_set": True,
            "trained_from_scratch": True,
            "training_data": {"name": "MP-20", "split": "train", "source_sha256": MP20_TRAIN_SHA256},
            "model_config": {"architecture": "aggregate-statistics-smoke-generator"},
        }
        (temporary / "model_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.chmod(temporary / "generate.sh", 0o755)
        swap_into(temporary, args.submission_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
