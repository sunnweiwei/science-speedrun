#!/usr/bin/env python3
"""Inference for a trained starter package. Copied into the submission as
predict.py by train.py's export step, so it must stay self-contained: it may
import only esm2.py, which is copied alongside it."""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from esm2 import Model, encode, pad_batch  # noqa: E402

TASKS = ("fluorescence", "stability", "beta_lactamase", "solubility",
         "subcellular_localization", "binary_localization", "fold",
         "secondary_structure", "yeast_ppi", "human_ppi", "ppi_affinity")
MAX_LEN = 1024


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: predict.py INPUT_ROOT OUTPUT_ROOT")
    source, destination = Path(sys.argv[1]), Path(sys.argv[2]) / "predictions"
    destination.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = json.loads((HERE / "config.json").read_text())
    model = Model(layers=config["layers"], dim=config["dim"],
                  heads=config["heads"]).to(device)
    model.load_state_dict(torch.load(HERE / "model.pt", map_location=device))
    model.eval()
    for task in TASKS:
        rows = [json.loads(line) for line in (source / f"{task}.jsonl").open()]
        with (destination / f"{task}.jsonl").open("w") as sink, torch.no_grad():
            for start in range(0, len(rows), 32):
                chunk = rows[start:start + 32]
                features = [r["features"] for r in chunk]
                if task == "secondary_structure":
                    tokens = pad_batch(
                        [encode(f["sequence"], MAX_LEN) for f in features], device)
                    hidden, _ = model.encoder(tokens)
                    logits = model.heads[task](hidden).float().softmax(-1).cpu()
                    for row, record in enumerate(chunk):
                        length = len(record["features"]["sequence"])
                        used = min(length, MAX_LEN - 2)
                        scores = [[float(x) for x in logits[row, 1 + i]]
                                  for i in range(used)]
                        scores += [[1.0, 1.0, 1.0]] * (length - used)
                        sink.write(json.dumps(
                            {"example_id": record["example_id"],
                             "prediction": {"ss3_scores": scores}},
                            separators=(",", ":")) + "\n")
                    continue
                representation = model.represent(task, features, device, MAX_LEN)
                output = model.heads[task](representation).float().cpu()
                for row, record in enumerate(chunk):
                    if task in Model.CLASSES:
                        probabilities = output[row].softmax(-1).clamp(min=1e-9)
                        prediction = {"scores": [float(x) for x in probabilities]}
                    else:
                        prediction = {"value": float(output[row, 0])}
                    sink.write(json.dumps(
                        {"example_id": record["example_id"], "prediction": prediction},
                        separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
