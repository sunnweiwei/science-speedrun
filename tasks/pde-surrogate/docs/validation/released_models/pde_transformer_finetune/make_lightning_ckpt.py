#!/usr/bin/env python3
"""Wrap a released PDE-Transformer safetensors file as a Lightning checkpoint upstream's `ckpt_path` can load.

    python3 make_lightning_ckpt.py --hf /path/to/pde-transformer --variant sc-s --out sc-s_init.ckpt
The separate-channel module `Supervised` holds the network under `model.`; the 550 released keys
match the network's state dict exactly (checked 2026-09-16).
"""
import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True); ap.add_argument("--variant", default="sc-s"); ap.add_argument("--out", required=True)
a = ap.parse_args()
sd = load_file(str(Path(a.hf) / a.variant / "diffusion_pytorch_model.safetensors"))
torch.save({"state_dict": {f"model.{k}": v for k, v in sd.items()}, "pde_surrogate_source": f"thuerey-group/pde-transformer/{a.variant}"}, a.out)
print(f"{len(sd)} tensors -> {a.out}")
