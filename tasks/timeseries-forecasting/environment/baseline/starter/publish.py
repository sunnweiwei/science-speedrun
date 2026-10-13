#!/usr/bin/env python3
"""Assemble a self-contained package: python3 publish.py --weights RUN/best.pt --out /workspace/submission --sources gifteval_pretrain/... [--synthetic kernelsynth]

The package is assembled in a staging directory next to --out and swapped in with one rename, so a reader of
--out never sees a half-written package."""
import argparse
import errno
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import PatchForecaster, count_parameters  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="starter-patch-transformer"); ap.add_argument("--sources", default="")
    ap.add_argument("--sources-from", default="", help="a packed corpus directory: take sources_used from its meta.json")
    ap.add_argument("--synthetic", default=""); ap.add_argument("--notes", default="")
    args = ap.parse_args()
    final = Path(args.out).resolve(); final.parent.mkdir(parents=True, exist_ok=True)
    out = final.parent / f".{final.name}.staging-{os.getpid()}"; shutil.rmtree(out, ignore_errors=True)
    (out / "weights").mkdir(parents=True)
    ck = torch.load(args.weights, map_location="cpu")
    slim = {"model": ck["model"], "config": ck["config"], "step": ck.get("step")}
    torch.save(slim, out / "weights" / "best.pt")
    for f in ("model.py", "predict.py", "infer.sh"):
        shutil.copy(HERE / f, out / f)
    ts_io = HERE / "ts_io.py" if (HERE / "ts_io.py").exists() else HERE.parents[1] / "evaluation" / "ts_io.py"
    shutil.copy(ts_io, out / "ts_io.py")
    model = PatchForecaster(**ck["config"])
    sources = [s for s in args.sources.split(",") if s]
    if args.sources_from:
        meta = json.loads((Path(args.sources_from) / "meta.json").read_text())
        sources += [f"gifteval_pretrain/{d}" for d in meta["dirs"]]
    if args.synthetic:
        sources.append(f"synthetic:{args.synthetic}")
    manifest = {"name": args.name, "contract": "timeseries-forecast/v1", "parameters": count_parameters(model), "pretrained_assets": [],
                "sources_used": sources, "config": ck["config"], "training_step": ck.get("step"),
                "weights_sha256": hashlib.sha256((out / "weights" / "best.pt").read_bytes()).hexdigest(), "notes": args.notes}
    (out / "model_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (out / "README.md").write_text(f"# {args.name}\n\nPackage built by environment/baseline/starter/publish.py; see model_manifest.json.\n")
    old = final.parent / f".{final.name}.old-{int(time.time())}"
    if final.exists():
        try:
            os.replace(final, old)
        except OSError as error:        # a directory from a lower image layer cannot be renamed on overlayfs
            if error.errno != errno.EXDEV:
                raise
            shutil.copytree(final, old, symlinks=True); shutil.rmtree(final)
    os.replace(out, final); shutil.rmtree(old, ignore_errors=True)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
