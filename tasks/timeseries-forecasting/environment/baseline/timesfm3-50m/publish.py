#!/usr/bin/env python3
"""Assemble a contract package from a training checkpoint directory (config.json + model.safetensors).

    python3 publish.py --checkpoint RUN/best --out /workspace/submission --sources-from /workspace/research/packed
"""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sources-from", default=""); ap.add_argument("--name", default="timesfm3-arch-from-scratch")
    a = ap.parse_args()
    final = Path(a.out).resolve(); final.parent.mkdir(parents=True, exist_ok=True)
    out = final.parent / f".{final.name}.staging-{os.getpid()}"; shutil.rmtree(out, ignore_errors=True); out.mkdir()
    ck = Path(a.checkpoint)
    for f in ("config.json", "model.safetensors", "training.json"):
        shutil.copy(ck / f, out / f)
    for f in ("predict.py", "infer.sh"):
        shutil.copy(HERE / f, out / f)
    shutil.copytree(HERE / "vendor", out / "vendor")
    cands = [HERE / "ts_io.py", Path("/workspace/evaluation/ts_io.py")] + [q / "evaluation" / "ts_io.py" for q in HERE.parents]
    ts_io = next(p for p in cands if p.exists())
    shutil.copy(ts_io, out / "ts_io.py")
    sources = []
    if a.sources_from:
        sources = [f"gifteval_pretrain/{d}" for d in json.loads((Path(a.sources_from) / "meta.json").read_text())["dirs"]]
    sources.append("synthetic:kernelsynth")
    cfg = json.loads((ck / "config.json").read_text())
    manifest = {"name": a.name, "contract": "timeseries-forecast/v1", "pretrained_assets": [], "sources_used": sources,
                "architecture": "TimesFM-3 (google-research/timesfm 3.0.2 code, random init)", "config": cfg,
                "training": json.loads((ck / "training.json").read_text()),
                "weights_sha256": hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest()}
    (out / "model_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if final.exists():
        old = final.parent / f".{final.name}.old-{int(time.time())}"; os.replace(final, old); shutil.rmtree(old, ignore_errors=True)
    os.replace(out, final)
    print(final)


if __name__ == "__main__":
    main()
