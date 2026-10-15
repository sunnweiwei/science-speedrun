#!/usr/bin/env python3
"""Generate random periodic structures from the smoke model."""

import argparse
import csv
import json
import math
from pathlib import Path
import random
import re


AMU_PER_ANGSTROM3_TO_G_PER_CM3 = 1.66053906660


def weighted_choice(rng: random.Random, counts: dict[str, int]):
    names = sorted(counts)
    weights = [counts[name] for name in names]
    return rng.choices(names, weights=weights, k=1)[0]


def load_requests(path: Path) -> list[dict]:
    requests = []
    seen = set()
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            request = json.loads(line)
            required = {"schema_version", "request_id", "mode", "condition", "n_samples", "seed"}
            if set(request) != required or request.get("schema_version") != "crystalgen-request/v0":
                raise ValueError(f"invalid request schema at line {line_number}")
            request_id = request["request_id"]
            if (not isinstance(request_id, str)
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", request_id)
                    or request_id in seen):
                raise ValueError(f"bad or duplicate request_id at line {line_number}")
            if request.get("mode") != "unconditional" or request.get("condition") is not None:
                raise ValueError("only unconditional requests with condition=null are supported")
            if type(request["n_samples"]) is not int or not 1 <= request["n_samples"] <= 10_000:
                raise ValueError(f"bad n_samples at line {line_number}")
            if type(request["seed"]) is not int or request["seed"] < 0:
                raise ValueError(f"bad seed at line {line_number}")
            seen.add(request_id)
            requests.append(request)
    if not requests:
        raise ValueError("empty request file")
    return requests


def make_structure(model: dict, request: dict, sample_index: int):
    try:
        from pymatgen.core import Element, Lattice, Structure
    except ImportError as error:
        raise RuntimeError("pymatgen is required for generation") from error
    rng = random.Random((request["seed"] << 32) ^ sample_index)
    n_species = max(1, min(20, int(weighted_choice(rng, model["atom_counts"]))))
    element_count = min(n_species, rng.randint(1, min(4, n_species)))
    chosen = []
    while len(chosen) < element_count:
        element = weighted_choice(rng, model["element_counts"])
        if element not in chosen:
            chosen.append(element)
    species = [chosen[index % len(chosen)] for index in range(n_species)]
    density = math.exp(rng.gauss(model["log_density_mean"], model["log_density_std"]))
    density = min(20.0, max(0.2, density))
    total_mass = sum(float(Element(name).atomic_mass) for name in species)
    volume = total_mass * AMU_PER_ANGSTROM3_TO_G_PER_CM3 / density
    lattice = Lattice.cubic(max(2.5, volume ** (1.0 / 3.0)))
    coords = [[rng.random(), rng.random(), rng.random()] for _ in species]
    return Structure(lattice, species, coords, coords_are_cartesian=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--input-root", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--output-root", required=True, type=lambda value: Path(value).resolve())
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("output directory must start empty")
    args.output_root.mkdir(parents=True, exist_ok=True)
    structure_root = args.output_root / "structures"
    structure_root.mkdir()
    model = json.loads(args.model.read_text())
    requests = load_requests(args.input_root / "requests.jsonl")
    with (args.output_root / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["request_id", "sample_index", "structure_path"])
        writer.writeheader()
        for request in requests:
            for sample_index in range(request["n_samples"]):
                relative = Path("structures") / f"{request['request_id']}-{sample_index:04d}.cif"
                structure = make_structure(model, request, sample_index)
                structure.to(filename=args.output_root / relative, fmt="cif")
                writer.writerow({
                    "request_id": request["request_id"],
                    "sample_index": sample_index,
                    "structure_path": relative.as_posix(),
                })


if __name__ == "__main__":
    main()
