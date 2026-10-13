"""SE-600M embeddings for the shipped training cells (dev/train/<ctx>/se_embeddings.npy, fp16,
row-aligned with cells.h5ad). Cells are embedded in row chunks so that several GPUs share one
context; each chunk is a `state emb transform` call (arc-state, dev image)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import anndata as ad
import numpy as np

CHUNK = 120_000


def _run_chunk(job: tuple, gpu_queue: Queue, se_dir: Path, batch_size: int, tmp: Path) -> tuple:
    ctx_dir, start, stop = job
    out = tmp / f"{ctx_dir.name}_{start}_{stop}.npy"
    if out.exists():  # resumable: a chunk that finished in an earlier attempt is reused
        return ctx_dir, start, stop, out
    gpu = gpu_queue.get()
    try:
        src = ad.read_h5ad(ctx_dir / "cells.h5ad", backed="r")
        piece = src[start:stop].to_memory()
        piece_path = tmp / f"{ctx_dir.name}_{start}_{stop}.h5ad"
        piece.write_h5ad(piece_path)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        cmd = [sys.executable, "-m", "state", "emb", "transform", "--model-folder", str(se_dir),
               "--checkpoint", str(se_dir / "se600m_epoch16.ckpt"), "--input", str(piece_path),
               "--output", str(out), "--batch-size", str(batch_size)]
        with (ctx_dir / "se_embeddings.log").open("a") as fh:
            fh.write(f"### chunk {start}-{stop} on GPU {gpu}\n")
            subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, check=True)
        piece_path.unlink()
        return ctx_dir, start, stop, out
    finally:
        gpu_queue.put(gpu)


def stage_embed(root: Path, se_dir: Path, gpus: str, batch_size: int = 256) -> None:
    dirs = sorted(p for p in (root / "dev" / "train").iterdir() if (p / "cells.h5ad").exists())
    gpu_list = gpus.split(",")
    jobs = []
    sizes = {}
    for d in dirs:
        if (d / "se_embeddings.npy").exists():
            print(f"[embed] {d.name}: exists"); continue
        n = ad.read_h5ad(d / "cells.h5ad", backed="r").n_obs
        sizes[d] = n
        jobs += [(d, s, min(s + CHUNK, n)) for s in range(0, n, CHUNK)]
    jobs.sort(key=lambda j: -(j[2] - j[1]))
    q: Queue = Queue()
    for g in gpu_list:
        q.put(g)
    tmp = root / ".embed-chunks"  # stable location so that a re-run resumes
    tmp.mkdir(exist_ok=True)
    results = {}
    with ThreadPoolExecutor(max_workers=len(gpu_list)) as pool:
        for ctx_dir, start, stop, out in pool.map(lambda j: _run_chunk(j, q, se_dir, batch_size, tmp), jobs):
            results.setdefault(ctx_dir, []).append((start, stop, out))
            print(f"[embed] {ctx_dir.name} {start}-{stop} done", flush=True)
    for d, parts in results.items():
        parts.sort()
        emb = np.concatenate([np.load(p[2]) for p in parts]).astype(np.float16)
        assert emb.shape[0] == sizes[d], (d, emb.shape, sizes[d])
        np.save(d / "se_embeddings.npy", emb)
        for p in parts:
            p[2].unlink()
        print(f"[embed] {d.name}: {emb.shape}")
    if not any(tmp.iterdir()):
        tmp.rmdir()
    meta = {"model": "arcinstitute/SE-600M @ 5a9a80f (se600m_epoch16.ckpt)", "dtype": "float16",
            "alignment": "row i of se_embeddings.npy is cell i of cells.h5ad", "contexts": [d.name for d in dirs]}
    (root / "dev" / "train" / "se_embeddings.json").write_text(json.dumps(meta, indent=2) + "\n")
