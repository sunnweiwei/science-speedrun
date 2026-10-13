#!/usr/bin/env python3
"""Assemble a portable submission from a training run.

    usage: python3 publish.py RUN_DIR SUBMISSION_DIR

RUN_DIR is what train.py wrote; SUBMISSION_DIR receives the package the contract expects
(infer.sh, predict.py, model.py, the readers and model_manifest.json). The package is assembled in a
staging directory next to SUBMISSION_DIR and swapped in with one rename, so a reader of SUBMISSION_DIR
never sees a half-written package.
"""
import errno
import json
import os
import shutil
import sys
import time
from pathlib import Path

if sys.argv[1:2] in (["-h"], ["--help"]):        # --help is a question, not an error
    print(__doc__)
    raise SystemExit(0)
if len(sys.argv) != 3:
    raise SystemExit(f"usage: python3 publish.py RUN_DIR SUBMISSION_DIR\n{__doc__}")
run, final = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
final.parent.mkdir(parents=True, exist_ok=True)
sub = final.parent / f".{final.name}.staging-{os.getpid()}"
shutil.rmtree(sub, ignore_errors=True); sub.mkdir()
here = Path(__file__).resolve().parent
for name in ("model.py", "predict.py", "infer.sh", "pde_io.py"):
    src = here / name if (here / name).exists() else here.parents[1] / "evaluation" / name
    shutil.copy(src, sub / name)
for name in ("weights.pt", "normalization.json", "config.json"):
    shutil.copy(run / name, sub / name)
cfg = json.loads((run / "config.json").read_text())
trained_on = sorted(json.loads((run / "normalization.json").read_text()))   # the datasets the run indexed
import torch  # noqa: E402
n_params = sum(v.numel() for v in torch.load(run / "weights.pt", map_location="cpu").values())
(sub / "model_manifest.json").write_text(json.dumps({
    "contract": "pde-surrogate-rollout/v1", "name": "channel-shared-unet-starter",
    "description": "Channel-shared residual U-Net, 4 context frames -> next frame, autoregressive rollout",
    "parameters": n_params, "context_frames": cfg["context"],
    "sources_used": [f"training pools of {cfg.get('manifest', 'the split manifest')}: {', '.join(trained_on)}"]
                    + ([f"pools absent at training time: {', '.join(cfg['missing_pools'])}"] if cfg.get("missing_pools") else []),
    "selection": "last checkpoint; the starter has no validation loop and selects nothing",
    "pretrained_assets": []}, indent=1))
(sub / "README.md").write_text("# Starter submission\n\nChannel-shared U-Net trained by `environment/baseline/starter/train.py` "
                               f"on {len(trained_on)} datasets ({', '.join(trained_on)}); see config.json for the run "
                               "configuration. Model selection: none, the last checkpoint is shipped. "
                               "`bash infer.sh INPUT_DIR OUTPUT_DIR`.\n")
old = final.parent / f".{final.name}.old-{int(time.time())}"
if final.exists():
    try:
        os.replace(final, old)
    except OSError as error:
        # A directory from a lower image layer cannot be renamed on overlayfs (EXDEV): move its
        # contents aside instead; the swap below is still a single rename.
        if error.errno != errno.EXDEV:
            raise
        shutil.copytree(final, old, symlinks=True); shutil.rmtree(final)
os.replace(sub, final)
shutil.rmtree(old, ignore_errors=True)
print(f"published {final} ({n_params} parameters)")
