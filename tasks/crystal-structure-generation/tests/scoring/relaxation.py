"""Fixed pre-scoring relaxation shared by development and trusted final tests."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


RECIPE = {
    "model": "uma-s-1", "task": "omat",
    "optimizer": "BFGSLineSearch", "cell_filter": "FrechetCellFilter",
    "fmax_ev_per_angstrom": 0.02, "max_steps": 200,
    "sample_timeout_seconds": 300,
    "selection": "none; every requested sample retained",
    "sample_failure": "preserve original bytes and report failure",
    "unconverged": "retain finite last iterate and report nonconvergence",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def gpu_tokens(python: Path) -> list[str]:
    count = int(subprocess.check_output(
        [str(python), "-c", "import torch; print(torch.cuda.device_count())"], text=True
    ).strip())
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if count < 1 or (visible and len(visible) != count):
        raise RuntimeError("relaxation requires working CUDA GPUs with an unambiguous device mask")
    return visible or [str(index) for index in range(count)]


def workers_per_gpu(gpus: int) -> int:
    """CRYSTALGEN_RELAX_WORKERS_PER_GPU, else as many two-thread workers as the CPUs allow (at most 4)."""
    value = os.environ.get("CRYSTALGEN_RELAX_WORKERS_PER_GPU")
    if value:
        if int(value) < 1:
            raise ValueError("CRYSTALGEN_RELAX_WORKERS_PER_GPU must be positive")
        return int(value)
    return max(1, min(4, len(os.sched_getaffinity(0)) // (2 * max(gpus, 1))))


def run_workers(inventory: Path, output: Path, python: Path, samples: int) -> None:
    tokens = gpu_tokens(python)
    limit = int(os.environ.get("CRYSTALGEN_LEMAT_GPU_WORKERS", "8"))
    if limit < 1:
        raise ValueError("CRYSTALGEN_LEMAT_GPU_WORKERS must be positive")
    tokens = tokens[:min(limit, samples)]
    # Each worker is CPU-bound (ASE line search, two threads) and keeps its GPU mostly idle, so several
    # workers share a GPU. They claim samples one at a time, so no worker is left with a long tail.
    slots = [token for _ in range(workers_per_gpu(len(tokens))) for token in tokens][:samples]
    claims = output / f"claims-{os.getpid()}-{time.time_ns()}"
    claims.mkdir()
    processes, logs = [], []
    try:
        for index, token in enumerate(slots):
            log = (output / f"worker-{index}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen(
                [str(python), str(Path(__file__).with_name("relax_worker.py")),
                 "--inventory", str(inventory), "--output", str(output),
                 "--worker", str(index), "--workers", str(len(slots)), "--claims", str(claims)],
                stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": token, "OMP_NUM_THREADS": "2"},
            ))
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError(f"relaxation worker failed; inspect {output}/worker-*.log")
            time.sleep(0.2)
        if any(process.returncode for process in processes):
            raise RuntimeError(f"relaxation worker failed; inspect {output}/worker-*.log")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


def relax_records(records: dict, expected: int, output: Path, python: Path) -> tuple[dict, dict]:
    """Never overwrite inputs or use submission-supplied relaxation metadata."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    inventory = [
        {"request_id": request_id, "sample_index": index,
         "source_path": str(source.resolve()), "source_sha256": sha256(source),
         "structure_path": f"structures/{request_id}--{index:06d}.cif"}
        for (request_id, index), source in sorted(records.items())
    ]
    inventory_path = output / "inventory.json"
    write_json(inventory_path, inventory)
    (output / "generated").mkdir()
    if inventory:
        run_workers(inventory_path, output, python, len(inventory))
    samples, relaxed = [], {}
    for ordinal, item in enumerate(inventory):
        record = json.loads((output / "samples" / f"{ordinal:06d}.json").read_text())
        destination = output / "generated" / item["structure_path"]
        if (record["input"] != item or record["recipe"] != RECIPE
                or not destination.is_file() or sha256(destination) != record["output_sha256"]
                or record["status"] not in {"relaxed", "fallback_original"}):
            raise RuntimeError("relaxation output failed integrity checks")
        samples.append(record)
        relaxed[(item["request_id"], item["sample_index"])] = destination
    with (output / "generated" / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["request_id", "sample_index", "structure_path"])
        writer.writeheader()
        writer.writerows({key: item[key] for key in writer.fieldnames} for item in inventory)
    summary = {
        "schema": "crystalgen-relaxation/v1", "recipe": RECIPE,
        "requested": expected, "present": len(samples), "missing_or_invalid_manifest": expected - len(samples),
        "relaxed": sum(r["status"] == "relaxed" for r in samples),
        "converged": sum(r.get("converged", False) for r in samples),
        "unconverged": sum(r["status"] == "relaxed" and not r["converged"] for r in samples),
        "fallback_original": sum(r["status"] == "fallback_original" for r in samples),
        "duration_seconds": time.monotonic() - started,
        "summed_sample_seconds": sum(r["duration_seconds"] for r in samples),
    }
    write_json(output / "summary.json", summary)
    return relaxed, summary
