#!/usr/bin/env python3
"""Wrap a checkpoint trained by The Well's trainer as a pde-surrogate-rollout/v1 package.

    python3 package.py --run WELL_RUN_DIR --dataset-dir WELL_BASE/<dataset> --out SUBMISSION_DIR [--dataset-name well/turbulent_radiative_layer_2D]
WELL_RUN_DIR holds the trainer's checkpoint (best.pt or the latest *.pt) with the model config used;
the dataset directory supplies stats.yaml (z-score statistics) and <name>.yaml (field order).
The package reproduces the trainer's formatting: z-scored frames stacked time-major then channel,
the model predicts the next z-scored frame, rolled out by shifting the window.
"""
import argparse
import errno
import json
import os
import shutil
import time
from pathlib import Path

import torch
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--dataset-dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--dataset-name", required=True, help="task dataset name this per-dataset model serves, e.g. well/active_matter")
ap.add_argument("--init-features", type=int, default=42); ap.add_argument("--blocks-per-stage", type=int, default=2)
ap.add_argument("--checkpoint", default="", help="checkpoint inside --run to wrap (default: best.pt when present, else the newest *.pt)")
a = ap.parse_args()
# assemble in a staging directory next to --out and swap it in with one rename at the end, so a reader of
# --out (for example /workspace/submission) never sees a half-written package
run, final = Path(a.run), Path(a.out).resolve(); final.parent.mkdir(parents=True, exist_ok=True)
out = final.parent / f".{final.name}.staging-{os.getpid()}"; shutil.rmtree(out, ignore_errors=True); out.mkdir()
ckpts = sorted(run.rglob("*.pt"), key=lambda p: p.stat().st_mtime)
if a.checkpoint:                                    # an explicit choice wins
    ckpts = [Path(a.checkpoint) if Path(a.checkpoint).is_absolute() else run / a.checkpoint]
elif any(p.name == "best.pt" for p in ckpts):       # the best-validation checkpoint, never the newest file by accident
    ckpts = [p for p in ckpts if p.name == "best.pt"]
ck = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck.get("state_dict", ck)
state = {k.replace("module.", "", 1): v for k, v in state.items()}
stats = yaml.safe_load(open(Path(a.dataset_dir) / "stats.yaml"))
meta = yaml.safe_load(open(next(Path(a.dataset_dir).glob("*.yaml"))))  # <name>.yaml first alphabetically? ensure not stats
if "field_names" not in meta:
    meta = yaml.safe_load(open([p for p in Path(a.dataset_dir).glob("*.yaml") if p.name != "stats.yaml"][0]))
fields = []
for order in (0, 1, 2):
    for f in meta["field_names"].get(order, []) or []:
        fields.append(f)
def flat(kind):
    vals = []
    for order in (0, 1, 2):
        for f in meta["field_names"].get(order, []) or []:
            base = f.rsplit("_", 1)[0] if order > 0 else f
            v = stats[kind][base]
            if isinstance(v, list):
                comp = f.rsplit("_", 1)[1]; idx = {"x": 0, "y": 1}.get(comp, 0) if order == 1 else 0
                v = v[idx] if not isinstance(v[0], list) else v[idx // 2][idx % 2]
            vals.append(float(v))
    return vals
mean, std = flat("mean"), flat("std")
cfg = {"contract": "pde-surrogate-rollout/v1", "dataset": a.dataset_name, "fields": fields, "mean": mean, "std": std,
       "n_steps_input": 4, "init_features": a.init_features, "blocks_per_stage": a.blocks_per_stage,
       "spatial_resolution": meta["spatial_resolution"], "checkpoint": ckpts[-1].name}
json.dump(cfg, open(out / "config.json", "w"), indent=1)
torch.save(state, out / "weights.pt")
(out / "predict.py").write_text('''#!/usr/bin/env python3
"""pde-surrogate-rollout/v1 inference for a CNextU-net trained with The Well's recipe (one dataset)."""
import json, sys
from pathlib import Path
import h5py, numpy as np, torch
from the_well.benchmark.models import UNetConvNext
HERE = Path(__file__).resolve().parent
cfg = json.loads((HERE / "config.json").read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"
C = len(cfg["fields"]); T = cfg["n_steps_input"]
model = UNetConvNext(dim_in=T * C, dim_out=C, n_spatial_dims=2, spatial_resolution=tuple(cfg["spatial_resolution"]),
                     init_features=cfg["init_features"], blocks_per_stage=cfg["blocks_per_stage"]).to(device).eval()
model.load_state_dict(torch.load(HERE / "weights.pt", map_location=device))
m = np.array(cfg["mean"], np.float32)[:, None, None]; s = np.array(cfg["std"], np.float32)[:, None, None]
inp, out = Path(sys.argv[1]), Path(sys.argv[2]); (out / "cases").mkdir(parents=True, exist_ok=True)
for case in json.loads((inp / "index.json").read_text())["cases"]:
    with h5py.File(inp / "cases" / f"{case['case_id']}.h5") as f: x = f["input"][()].astype(np.float32)
    K = int(case["n_steps_out"])
    if x.shape[1] != C:                       # not this model's dataset: predict persistence (scored honestly)
        pred = np.repeat(x[-1:], K, 0)
    else:
        z = torch.from_numpy((x - m) / s).to(device)                # (T, C, X, Y) z-scored
        frames = list(z[-T:]); preds = []
        with torch.no_grad():
            for _ in range(K):
                xin = torch.stack(frames[-T:], 0).reshape(1, T * C, *z.shape[-2:])   # time-major then channel
                nxt = model(xin)[0]; preds.append(nxt); frames.append(nxt)
        pred = torch.stack(preds).cpu().numpy() * s + m
    with h5py.File(out / "cases" / f"{case['case_id']}.h5", "w") as f: f.create_dataset("prediction", data=pred.astype(np.float32))
print(f"wrote predictions to {out}")
''')
(out / "infer.sh").write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec python3 "$(dirname "$(readlink -f "$0")")/predict.py" "$1" "$2"\n')
(out / "infer.sh").chmod(0o755)
json.dump({"contract": "pde-surrogate-rollout/v1", "name": f"cnextunet-well-recipe-{a.dataset_name.split('/')[-1]}",
           "description": "CNextU-net trained from scratch with The Well's published recipe on one transfer dataset (per-dataset reference)",
           "parameters": sum(v.numel() for v in state.values()), "context_frames": 4, "sources_used": [a.dataset_name],
           "pretrained_assets": []}, open(out / "model_manifest.json", "w"), indent=1)
(out / "README.md").write_text("# CNextU-net reference package\n\nSee environment/baseline/cnext_unet/README.md.\n")
old = final.parent / f".{final.name}.old-{int(time.time())}"
if final.exists():
    try:
        os.replace(final, old)
    except OSError as error:        # a directory from a lower image layer cannot be renamed on overlayfs
        if error.errno != errno.EXDEV:
            raise
        shutil.copytree(final, old, symlinks=True); shutil.rmtree(final)
os.replace(out, final); shutil.rmtree(old, ignore_errors=True)
print(f"packaged {ckpts[-1]} -> {final} ({cfg['fields']})")
