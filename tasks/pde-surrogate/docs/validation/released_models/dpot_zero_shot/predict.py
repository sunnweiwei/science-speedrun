#!/usr/bin/env python3
"""Zero-shot replay of a released DPOT checkpoint under pde-surrogate-rollout/v1 (reference use only).

    DPOT_CODE_DIR=/path/to/DPOT DPOT_CKPT=/path/to/model_Ti.pth python3 predict.py INPUT_DIR OUTPUT_DIR

Upstream conventions reproduced: input (B, X, Y, T_in=10, C=4) in physical units
(the checkpoints were trained with normalize=0), channels one-padded to 4 in
upstream's order (PDEArena: u, vx, vy), bilinear resize to 128x128, output one
frame, rollout by shifting the window. The task supplies four context frames;
DPOT needs ten, so the first frame is repeated (DPOT_CONTEXT=pad, default) or,
for DPOT_CONTEXT=native, the case is skipped unless it carries ten frames.
The DPOT code repositories carry no license; they are used here locally and
are not part of the task package.
"""
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.environ["DPOT_CODE_DIR"])
from models.dpot import DPOTNet  # noqa: E402

T_IN, RES, NCH = 10, 128, 4


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = vars(ck["args"])
    model = DPOTNet(img_size=a["res"], patch_size=a["patch_size"], in_channels=NCH, in_timesteps=a["T_in"],
                    out_timesteps=a["T_bundle"], out_channels=NCH, normalize=bool(a["normalize"]), embed_dim=a["width"],
                    modes=a["modes"], depth=a["n_layers"], n_blocks=a["n_blocks"], mlp_ratio=a["mlp_ratio"],
                    out_layer_dim=a.get("out_layer_dim", 32), act=a["act"], n_cls=a.get("n_class", 12))
    model.load_state_dict(ck["model"], strict=True)
    return model.to(device).eval(), a


def main(input_dir, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, a = load_model(os.environ["DPOT_CKPT"], device)
    mode = os.environ.get("DPOT_CONTEXT", "pad")
    index = json.loads((Path(input_dir) / "index.json").read_text())
    (Path(output_dir) / "cases").mkdir(parents=True, exist_ok=True)
    for case in index["cases"]:
        with h5py.File(Path(input_dir) / "cases" / f"{case['case_id']}.h5", "r") as f:
            x = f["input"][()].astype(np.float32)               # (T, C, X, Y)
        K, C, (X, Y) = int(case["n_steps_out"]), x.shape[1], x.shape[2:]
        if x.shape[0] < T_IN:
            if mode == "native":
                continue
            x = np.concatenate([np.repeat(x[:1], T_IN - x.shape[0], axis=0), x], axis=0)
        x = x[-T_IN:]
        xt = torch.from_numpy(x).to(device)                     # (T, C, X, Y)
        xt = F.interpolate(xt, size=(RES, RES), mode="bilinear", align_corners=False)
        # DPOT takes exactly four channels; datasets with more are processed in independent blocks of four
        blocks = [list(range(i, min(i + NCH, C))) for i in range(0, C, NCH)]
        pred = torch.zeros(K, C, RES, RES, device=device)
        with torch.no_grad():
            for chans in blocks:
                xx = torch.ones(1, RES, RES, T_IN, NCH, device=device)
                xx[0, :, :, :, :len(chans)] = xt[:, chans].permute(2, 3, 0, 1)   # (X, Y, T, c)
                for k in range(K):
                    im, _ = model(xx)                                           # (1, X, Y, 1, 4)
                    pred[k, chans] = im[0, :, :, 0, :len(chans)].permute(2, 0, 1)
                    xx = torch.cat((xx[..., 1:, :], im), dim=-2)
        pred = F.interpolate(pred, size=(X, Y), mode="bilinear", align_corners=False).cpu().numpy()
        with h5py.File(Path(output_dir) / "cases" / f"{case['case_id']}.h5", "w") as f:
            f.create_dataset("prediction", data=pred.astype(np.float32))
    print(f"wrote predictions to {output_dir} (mode={mode})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
