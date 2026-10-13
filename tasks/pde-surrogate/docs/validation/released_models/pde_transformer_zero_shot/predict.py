#!/usr/bin/env python3
"""Zero-shot replay of a released PDE-Transformer mixed-channel checkpoint under pde-surrogate-rollout/v1.

    PDET_CKPT_DIR=/path/to/thuerey-group/pde-transformer PDET_VARIANT=mc-s python3 predict.py INPUT_DIR OUTPUT_DIR

Faithful to upstream inference: inputs normalized with the training file's
norm_fields_sca_{mean,std} buffers (bundled in normalization.json), channels
zero-padded to 2, class label = index of the PDE name in upstream's all_pdes,
one-step supervised model rolled out autoregressively from the LAST context
frame, padded output channels zeroed. Datasets absent from normalization.json
(transfer datasets the checkpoints never saw) are normalized from their own
context frames and labelled by name when upstream knows the name, else
'unknown'; those numbers are reported but are not a reference.
"""
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
from pdetransformer.core.mixed_channels import PDETransformer  # noqa: E402
from pdetransformer.data.metadata_remapping import all_pdes  # noqa: E402


def main(input_dir, output_dir):
    ckpt = os.environ.get("PDET_CKPT_DIR", str(HERE / "checkpoint"))
    variant = os.environ.get("PDET_VARIANT", "mc-s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PDETransformer.from_pretrained(ckpt, subfolder=variant).to(device).eval()
    stats = json.loads((HERE / "normalization.json").read_text())
    index = json.loads((Path(input_dir) / "index.json").read_text())
    (Path(output_dir) / "cases").mkdir(parents=True, exist_ok=True)
    for case in index["cases"]:
        with h5py.File(Path(input_dir) / "cases" / f"{case['case_id']}.h5", "r") as f:
            x = f["input"][()].astype(np.float32); pde = str(f.attrs["pde"])
        ds, K = case["dataset"], int(case["n_steps_out"])
        C = x.shape[1]
        if ds in stats:
            m, s = (np.array(stats[ds][k], dtype=np.float32)[:C] for k in ("mean", "std"))
        else:
            m, s = x.mean(axis=(0, 2, 3)), x.std(axis=(0, 2, 3)) + 1e-6
        label = all_pdes.index(pde.lower()) if pde.lower() in all_pdes else 0
        last = (x[-1] - m[:, None, None]) / s[:, None, None]                    # (C, X, Y)
        labels = torch.tensor([label], device=device)
        # the mixed-channel model has two channels; datasets with more are processed in independent blocks of two
        pred = np.zeros((K, C, *last.shape[1:]), np.float32)
        with torch.no_grad():
            for a in range(0, C, 2):
                chans = list(range(a, min(a + 2, C)))
                cur = np.zeros((1, 2, *last.shape[1:]), np.float32); cur[0, :len(chans)] = last[chans]
                cur = torch.from_numpy(cur).to(device)
                for k in range(K):
                    cur = model(cur, class_labels=labels).sample
                    cur[:, len(chans):] = 0.0                                   # zero padded channel
                    pred[k, chans] = cur[0, :len(chans)].float().cpu().numpy()
        pred = pred * s[:, None, None] + m[:, None, None]
        with h5py.File(Path(output_dir) / "cases" / f"{case['case_id']}.h5", "w") as f:
            f.create_dataset("prediction", data=pred.astype(np.float32))
    print(f"wrote {len(index['cases'])} predictions ({variant}) to {output_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
