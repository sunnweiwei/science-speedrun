"""Round-trip smoke test for the scaffold using a test-only tiny network."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from data import HORIZONS


TRAIN_WRAPPER = r"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, sys.argv.pop(1))
import data, model, train

class Tiny(torch.nn.Module):
    def __init__(self, config, metadata):
        super().__init__()
        self.linear = torch.nn.Linear(1, len(metadata["scored_diseases"]) + 1)
        self.diseases = len(metadata["scored_diseases"])
    def forward(self, batch):
        risk = torch.sigmoid(self.linear(batch["age"]))
        disease = risk[:, :self.diseases, None].expand(-1, -1, 13)
        death = risk[:, self.diseases:, None].expand(-1, -1, 13).squeeze(1)
        return {"disease_risk": disease, "death_risk": death}

def tiny_model(config, metadata):
    return Tiny(config, metadata)

def tiny_loader(input_dir, config):
    batch = {"age": torch.tensor([[0.25], [0.75]]),
             "target": torch.tensor([0.0, 1.0])}
    return [batch], {"age_scale": 100.0}

def tiny_loss(outputs, batch, config):
    return torch.nn.functional.binary_cross_entropy(
        outputs["disease_risk"][:, 0, -1], batch["target"])

model.build_model = tiny_model
train.build_model = tiny_model
data.build_training_loader = tiny_loader
train.build_training_loader = tiny_loader
train.compute_loss = tiny_loss
train.main()
"""


PREDICT_WRAPPER = r"""
import sys
import torch
sys.path.insert(0, sys.argv.pop(1))
import data, model

class Tiny(torch.nn.Module):
    def __init__(self, config, metadata):
        super().__init__()
        self.linear = torch.nn.Linear(1, len(metadata["scored_diseases"]) + 1)
        self.diseases = len(metadata["scored_diseases"])
    def forward(self, batch):
        risk = torch.sigmoid(self.linear(batch["age"]))
        disease = risk[:, :self.diseases, None].expand(-1, -1, 13)
        death = risk[:, self.diseases:, None].expand(-1, -1, 13).squeeze(1)
        return {"disease_risk": disease, "death_risk": death}

class TinyPreprocessor:
    def __init__(self, config, state=None):
        self.scale = state["age_scale"]
    def transform(self, history):
        return {"age": torch.tensor([history["approx_age"] / self.scale])}

model.build_model = lambda config, metadata: Tiny(config, metadata)
data.build_preprocessor = lambda config, state=None: TinyPreprocessor(config, state)
import predict
predict.build_model = model.build_model
predict.build_preprocessor = data.build_preprocessor
predict.main()
"""


def _write_fixture(root: Path) -> None:
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({
        "horizons_days": list(HORIZONS),
        "scored_diseases": ["d0", "d1"],
    }), encoding="utf-8")
    pq.write_table(pa.Table.from_pylist([
        {"sample_id": "s0", "gender": "F", "approx_age": 25.0},
        {"sample_id": "s1", "gender": "M", "approx_age": 75.0},
    ]), root / "samples.parquet")
    pq.write_table(pa.Table.from_pylist([
        {"sample_id": "s0", "event_type": "diagnosis", "code": "A",
         "value": None, "unit": None, "available_time": "2020-01-01T00:00:00",
         "event_time": "2020-01-01T00:00:00", "seconds_before_cutoff": 1.0},
        {"sample_id": "s1", "event_type": "diagnosis", "code": "B",
         "value": None, "unit": None, "available_time": "2020-01-01T00:00:00",
         "event_time": "2020-01-01T00:00:00", "seconds_before_cutoff": 1.0},
    ]), root / "events.parquet")


def test_one_step_train_then_independent_inference(tmp_path: Path) -> None:
    starter = Path(__file__).resolve().parent
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "tiny.pt"
    _write_fixture(input_dir)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([
        sys.executable, "-c", TRAIN_WRAPPER, str(starter),
        "--input", str(input_dir), "--config", str(starter / "config.json"),
        "--checkpoint", str(checkpoint), "--device", "cpu", "--max-steps", "1",
    ], check=True, env=env)
    subprocess.run([
        sys.executable, "-c", PREDICT_WRAPPER, str(starter),
        str(input_dir), str(output_dir), "--checkpoint", str(checkpoint),
        "--device", "cpu", "--batch-size", "1",
    ], check=True, env=env)
    predictions = pq.read_table(output_dir / "predictions.parquet")
    assert predictions.num_rows == 2
    assert predictions.column_names == ["sample_id", "disease_risk", "death_risk"]
