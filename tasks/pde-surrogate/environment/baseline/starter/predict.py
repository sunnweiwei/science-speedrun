#!/usr/bin/env python3
"""Inference entry for the pde-surrogate-rollout/v1 contract: python3 predict.py INPUT_DIR OUTPUT_DIR"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import UNet  # noqa: E402


def main(input_dir, output_dir):
    cfg = json.loads((HERE / "config.json").read_text())
    stats = json.loads((HERE / "normalization.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(context=cfg["context"], width=cfg["width"]).to(device).eval()
    model.load_state_dict(torch.load(HERE / "weights.pt", map_location=device))
    torch.manual_seed(0)
    index = json.loads((Path(input_dir) / "index.json").read_text())
    (Path(output_dir) / "cases").mkdir(parents=True, exist_ok=True)
    for case in index["cases"]:
        with h5py.File(Path(input_dir) / "cases" / f"{case['case_id']}.h5", "r") as f:
            x = f["input"][()].astype(np.float32)             # (T_in, C, X, Y)
        ds = case["dataset"]
        if ds in stats:                                        # known dataset: training statistics
            m, s = (np.array(stats[ds][k], dtype=np.float32)[:, None, None] for k in ("mean", "std"))
        else:                                                  # unseen dataset: statistics of the context frames
            m, s = x.mean(axis=(0, 2, 3), keepdims=False)[:, None, None], x.std(axis=(0, 2, 3))[:, None, None] + 1e-6
        xn = torch.from_numpy((x - m) / s).permute(1, 0, 2, 3).to(device)   # (C, T_in, X, Y)
        if xn.shape[1] < cfg["context"]:                       # pad context by repeating the first frame
            xn = torch.cat([xn[:, :1].expand(-1, cfg["context"] - xn.shape[1], -1, -1), xn], dim=1)
        pred = model.rollout(xn, int(case["n_steps_out"])).cpu().numpy() * s + m   # (K, C, X, Y)
        with h5py.File(Path(output_dir) / "cases" / f"{case['case_id']}.h5", "w") as f:
            f.create_dataset("prediction", data=pred.astype(np.float32))
    print(f"wrote {len(index['cases'])} predictions to {output_dir}")


if __name__ == "__main__":
    if sys.argv[1:2] in (["-h"], ["--help"]) or len(sys.argv) != 3:
        print(__doc__, file=sys.stdout if sys.argv[1:2] in (["-h"], ["--help"]) else sys.stderr)
        raise SystemExit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
    main(sys.argv[1], sys.argv[2])
