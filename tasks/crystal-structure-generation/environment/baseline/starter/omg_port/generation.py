#!/usr/bin/env python3
"""Offline, multi-GPU request replay for the unconditional OMatG starter."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import re
import shutil
import subprocess
import sys


def load_requests(input_root: Path) -> list[dict]:
    path = input_root / "requests.jsonl"
    requests = []
    seen = set()
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            request = json.loads(line)
            request_id = request.get("request_id")
            mode = request.get("mode")
            required = {"schema_version", "request_id", "mode", "condition", "n_samples", "seed"}
            if set(request) != required or request.get("schema_version") != "crystalgen-request/v0":
                raise ValueError(f"bad request schema at line {line_number}")
            if (not isinstance(request_id, str)
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", request_id)
                    or request_id in seen):
                raise ValueError(f"bad or duplicate request_id at line {line_number}")
            if mode != "unconditional":
                raise ValueError(f"only unconditional requests are supported (line {line_number})")
            if type(request.get("n_samples")) is not int or not 1 <= request["n_samples"] <= 10_000:
                raise ValueError(f"bad n_samples at line {line_number}")
            if type(request.get("seed")) is not int or request["seed"] < 0:
                raise ValueError(f"bad seed at line {line_number}")
            if request.get("condition") is not None:
                raise ValueError("unconditional request must have condition=null")
            seen.add(request_id)
            requests.append(request)
    if not requests:
        raise ValueError("empty request suite")
    return requests


def jobs(requests: list[dict]) -> list[dict]:
    return [
        {"request": request, "sample_index": sample_index}
        for request in requests
        for sample_index in range(request["n_samples"])
    ]


def device_tokens() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible not in {"-1", "none", "None"}:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if tokens:
            return tokens
    try:
        import torch

        return [str(index) for index in range(torch.cuda.device_count())]
    except ImportError:
        return []


def choose_atom_count(stats: dict, request: dict, sample_index: int) -> int:
    counts = stats["counts"]
    population = sorted(int(value) for value in counts)
    weights = [int(counts[str(value)]) for value in population]
    rng = random.Random((request["seed"] << 32) ^ sample_index)
    n_atoms = rng.choices(population, weights=weights, k=1)[0]
    return min(20, n_atoms)


def write_prediction_lmdb(path: Path, assigned: list[dict], stats: dict) -> None:
    import lmdb
    import torch

    environment = lmdb.Environment(str(path), subdir=False, map_size=1 << 30, lock=False)
    try:
        with environment.begin(write=True) as transaction:
            for index, job in enumerate(assigned):
                request = job["request"]
                n_atoms = choose_atom_count(stats, request, job["sample_index"])
                record = {
                    "pos": torch.zeros((n_atoms, 3), dtype=torch.float32),
                    "cell": torch.eye(3, dtype=torch.float32) * 5.0,
                    "atomic_numbers": torch.ones(n_atoms, dtype=torch.int64),
                    "identifier": f"{request['request_id']}:{job['sample_index']}",
                }
                transaction.put(str(index).encode(), pickle.dumps(record))
    finally:
        environment.close()


def instantiate_and_predict(package_root: Path, dataset_root: Path, xyz_path: Path, seed: int):
    import torch
    from lightning.pytorch import seed_everything
    from omg.datamodule import OMGDataModule
    from omg.main import OMGCLI
    from omg.omg_lightning import OMGLightning
    from omg.omg_trainer import OMGTrainer
    from safetensors.torch import load_file

    seed_everything(seed, workers=True)
    device = "gpu" if torch.cuda.is_available() else "cpu"
    arguments = [
        f"--config={package_root / 'config.yaml'}",
        f"--data.train_dataset.init_args.file_path={dataset_root / 'train.lmdb'}",
        f"--data.val_dataset.init_args.file_path={dataset_root / 'val.lmdb'}",
        f"--data.pred_dataset.init_args.file_path={dataset_root / 'pred.lmdb'}",
        "--data.batch_size=64",
        "--data.num_workers=0",
        "--data.pin_memory=false",
        "--data.persistent_workers=false",
        f"--trainer.accelerator={device}",
        "--trainer.devices=1",
        "--trainer.strategy=auto",
        "--trainer.logger=false",
        f"--trainer.default_root_dir={dataset_root}",
        f"--model.generation_xyz_filename={xyz_path}",
    ]
    cli = OMGCLI(
        model_class=OMGLightning,
        datamodule_class=OMGDataModule,
        trainer_class=OMGTrainer,
        save_config_callback=None,
        args=arguments,
        run=False,
    )
    state = load_file(package_root / "model.safetensors", device="cpu")
    cli.model.load_state_dict(state, strict=True)
    cli.model.eval()
    return cli.trainer.predict(cli.model, datamodule=cli.datamodule, return_predictions=True)


def invalid_sentinel_structure():
    """Return a parseable low-density sentinel that fails scorer validity."""
    from pymatgen.core import Lattice, Structure

    return Structure(Lattice.cubic(10.0), ["H"], [[0.0, 0.0, 0.0]])


def serialize_predictions(predictions, assigned: list[dict], structure_root: Path) -> list[dict]:
    import numpy as np
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter

    generated = []
    for batch in predictions:
        batch = batch.to("cpu")
        for index in range(len(batch.n_atoms)):
            lower = int(batch.ptr[index])
            upper = int(batch.ptr[index + 1])
            generated.append((
                batch.cell[index].detach().numpy(),
                batch.species[lower:upper].detach().numpy(),
                batch.pos[lower:upper].detach().numpy(),
                bool(batch.pos_is_fractional[index]),
            ))
    if len(generated) != len(assigned):
        raise RuntimeError(f"OMatG returned {len(generated)} structures for {len(assigned)} jobs")

    rows = []
    for job, (cell, species, positions, fractional) in zip(assigned, generated):
        request = job["request"]
        sample_index = job["sample_index"]
        relative = Path("structures") / f"{request['request_id']}-{sample_index:04d}.cif"
        destination = structure_root.parent / relative
        try:
            if (
                not np.isfinite(cell).all()
                or not np.isfinite(positions).all()
                or abs(float(np.linalg.det(cell))) < 1.0e-6
                or np.any(species < 1)
                or np.any(species > 118)
            ):
                raise ValueError("nonphysical raw OMatG output")
            structure = Structure(
                Lattice(cell),
                [int(value) for value in species],
                positions,
                coords_are_cartesian=not fractional,
            )
        except Exception as error:
            print(
                f"warning: using serialization fallback for {request['request_id']}[{sample_index}]: {error}",
                file=sys.stderr,
            )
            structure = invalid_sentinel_structure()
        CifWriter(structure).write_file(destination)
        rows.append({
            "request_id": request["request_id"],
            "sample_index": sample_index,
            "structure_path": relative.as_posix(),
        })
    return rows


def worker(args, requests: list[dict], stats: dict) -> None:
    all_jobs = jobs(requests)
    assigned = [
        job for ordinal, job in enumerate(all_jobs)
        if ordinal % args.workers == args.worker_index
    ]
    worker_root = args.output_root / f".crystalgen-worker-{args.worker_index}"
    worker_root.mkdir()
    lmdb_path = worker_root / "pred.lmdb"
    write_prediction_lmdb(lmdb_path, assigned, stats)
    # OMatG's data module constructs all three split objects even for predict.
    # Keep their LMDB environments distinct within the worker process.
    shutil.copyfile(lmdb_path, worker_root / "train.lmdb")
    shutil.copyfile(lmdb_path, worker_root / "val.lmdb")
    seed_payload = "\n".join(
        f"{job['request']['request_id']}:{job['request']['seed']}:{job['sample_index']}"
        for job in assigned
    ).encode()
    seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:4], "big")
    predictions = instantiate_and_predict(
        args.package_root, worker_root, worker_root / "generated.xyz", seed
    )
    rows = serialize_predictions(predictions, assigned, args.output_root / "structures")
    (worker_root / "manifest.json").write_text(json.dumps(rows) + "\n")


def orchestrate(args, requests: list[dict], stats: dict) -> None:
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("output directory must start empty")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "structures").mkdir()
    tokens = device_tokens()
    worker_count = min(len(jobs(requests)), max(1, len(tokens)))
    processes = []
    for worker_index in range(worker_count):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--package-root", str(args.package_root),
            "--input-root", str(args.input_root),
            "--output-root", str(args.output_root),
            "--worker-index", str(worker_index),
            "--workers", str(worker_count),
        ]
        environment = os.environ.copy()
        if tokens:
            environment["CUDA_VISIBLE_DEVICES"] = tokens[worker_index]
        processes.append(subprocess.Popen(command, env=environment))
    failures = [process.wait() for process in processes]
    if any(code != 0 for code in failures):
        raise RuntimeError(f"generation workers failed: {failures}")

    expected = {
        (job["request"]["request_id"], job["sample_index"])
        for job in jobs(requests)
    }
    rows = []
    for worker_index in range(worker_count):
        worker_root = args.output_root / f".crystalgen-worker-{worker_index}"
        rows.extend(json.loads((worker_root / "manifest.json").read_text()))
        shutil.rmtree(worker_root)
    observed = {(row["request_id"], row["sample_index"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise RuntimeError("workers did not produce the exact requested sample set")
    rows.sort(key=lambda row: (row["request_id"], row["sample_index"]))
    with (args.output_root / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["request_id", "sample_index", "structure_path"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--input-root", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--output-root", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    requests = load_requests(args.input_root)
    stats = json.loads((args.package_root / "sampler_stats.json").read_text())
    if stats.get("schema") != "crystalgen-atom-count-prior/v0":
        raise ValueError("invalid atom-count prior")
    if args.worker_index is None:
        orchestrate(args, requests, stats)
    else:
        if args.workers is None or not 0 <= args.worker_index < args.workers:
            raise ValueError("invalid worker shard")
        worker(args, requests, stats)


if __name__ == "__main__":
    main()
