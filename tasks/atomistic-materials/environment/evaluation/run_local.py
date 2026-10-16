"""Run the physical suites (discovery, phonons, elasticity) on local GPUs.

One worker process per GPU pulls chunks from distributed.chunk_plan, writes each chunk with
predict.run, and distributed.merge assembles and checks the complete run:

    python run_local.py --inputs CASES.jsonl --artifact MODEL.pt [--energy-audit AUDIT.json] \
        --output OUT_DIR [--gpus 0,1,2,3] [--factory uma|module:function]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _worker(gpu, queue, args, chunk_root):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import sys
    sys.path.insert(0, str(HERE))
    from common import write_rows
    import predict
    calculator = predict.load_calculator(args.factory, args.artifact, "cuda")
    while True:
        item = queue.get()
        if item is None:
            return
        chunk, rows = item
        directory = chunk_root / chunk["chunk_id"]
        if (directory / "worker.json").exists():
            continue
        staging = chunk_root / f".{chunk['chunk_id']}.tmp"
        staging.mkdir(parents=True, exist_ok=True)
        write_rows(staging / "inputs.jsonl", rows)
        started = time.monotonic()
        predict.run(staging / "inputs.jsonl", staging / "out", args.factory, args.artifact, "cuda",
                    args.energy_audit, calculator=calculator, verbose=False)
        for name in ("run.json", "predictions.jsonl"):
            (staging / "out" / name).rename(staging / name)
        (staging / "out").rmdir()
        (staging / "worker.json").write_text(json.dumps({
            "worker_id": f"{socket.gethostname()}-gpu{gpu}", "chunk_id": chunk["chunk_id"],
            "chunk_input_sha256": chunk["input_sha256"], "seconds": time.monotonic() - started}))
        staging.rename(directory)
        print(json.dumps({"chunk": chunk["chunk_id"], "gpu": gpu}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--energy-audit", type=Path)
    parser.add_argument("--factory", default="uma")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(HERE))
    from common import digest, indexed, read_rows
    from distributed import chunk_plan, merge
    cases = read_rows(args.inputs)
    by_id = indexed(cases)
    plan = chunk_plan(cases)
    chunk_root = args.output.parent / f"{args.output.name}.chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    for chunk in plan:
        queue.put((chunk, [by_id[key] for key in chunk["case_ids"]]))
    gpus = [g for g in args.gpus.split(",") if g != ""]
    for _ in gpus:
        queue.put(None)
    procs = [ctx.Process(target=_worker, args=(g, queue, args, chunk_root)) for g in gpus]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    # Chunks left by a crashed worker must not block interpreter exit on the queue's feeder thread.
    queue.cancel_join_thread()
    failed = [p.exitcode for p in procs if p.exitcode]
    if failed:
        raise SystemExit(f"worker failed with exit code {failed[0]}")
    merge(args.inputs, chunk_root, plan, args.output, digest(args.artifact))


if __name__ == "__main__":
    main()
