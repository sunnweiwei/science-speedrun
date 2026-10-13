#!/usr/bin/env python3
"""timeseries-forecast/v1 inference for the starter: python3 predict.py INPUT_DIR OUTPUT_DIR [--weights best.pt]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ts_io  # noqa: E402
from model import PatchForecaster  # noqa: E402


def load_model(weights: Path, device):
    ck = torch.load(weights, map_location=device)
    model = PatchForecaster(**ck["config"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return model


def run(model, device, input_dir: Path, output_dir: Path, configs=None, batch: int = 512):
    index = json.loads((Path(input_dir) / "index.json").read_text())
    torch.manual_seed(0)
    for cfg in index["configs"]:
        if configs and cfg["config_id"] not in configs:
            continue
        case = ts_io.CaseFile(Path(input_dir) / "cases" / f"{cfg['config_id']}.h5")
        n, d, h = case.n, case.n_variates, case.prediction_length
        mean = np.zeros((n, d, h), np.float32); quant = np.zeros((n, d, 9, h), np.float32)
        # flatten (instance, variate) into univariate rows; sort by length for tight padding
        rows = [(i, v) for i in range(n) for v in range(d)]
        lengths = np.array([min(int(case.context_length[i]), model.max_context) for i, _ in rows])
        order = np.argsort(lengths, kind="stable")
        for s in range(0, len(rows), batch):
            sel = order[s:s + batch]
            L = int(lengths[sel].max())
            ctx = np.full((len(sel), L), np.nan, np.float32)
            for r, k in enumerate(sel):
                i, v = rows[k]
                c = case.context(i)[v][-L:]
                ctx[r, L - len(c):] = c
            with torch.no_grad():
                m, q = model.predict(torch.from_numpy(ctx).to(device), h)
            m, q = m.float().cpu().numpy(), q.float().cpu().numpy()
            for r, k in enumerate(sel):
                i, v = rows[k]
                mean[i, v] = m[r]; quant[i, v] = q[r]
        ts_io.write_prediction_file(Path(output_dir) / "cases" / f"{cfg['config_id']}.h5", mean, quant, {"model": "starter"})
        case.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--weights", default=str(HERE / "weights" / "best.pt")); ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = load_model(Path(args.weights), device)
    run(model, device, Path(args.input), Path(args.output), batch=args.batch)
    manifest = HERE / "model_manifest.json"
    if manifest.exists():
        (Path(args.output) / "model_manifest.json").write_text(manifest.read_text())


if __name__ == "__main__":
    main()
