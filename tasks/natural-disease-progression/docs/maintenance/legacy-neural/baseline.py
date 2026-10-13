"""Baselines for directly predicted first-recorded disease and mortality risk."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

from adapter import fit_adapter, history_to_ids, iter_histories
HORIZONS = tuple(range(30, 361, 30)) + (365,)


class RiskNet(nn.Module):
    """One shared encoder and interval-hazard head for every target disease/death."""

    def __init__(self, vocab_size: int, n_diseases: int, hidden: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, hidden, padding_idx=0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, (n_diseases + 1) * len(HORIZONS)))
        self.n_diseases = n_diseases

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(tokens)
        present = (tokens != 0).unsqueeze(-1)
        pooled = (embedded * present).sum(1) / present.sum(1).clamp(min=1)
        hazards = torch.sigmoid(self.head(pooled).reshape(
            -1, self.n_diseases + 1, len(HORIZONS)))
        cumulative = 1 - torch.cumprod(1 - hazards, dim=-1)
        return cumulative[:, :self.n_diseases], cumulative[:, self.n_diseases]


def pad_features(rows: list[list[int]], device: torch.device) -> torch.Tensor:
    length = max(map(len, rows))
    array = np.zeros((len(rows), length), dtype=np.int64)
    for i, row in enumerate(rows):
        array[i, :len(row)] = row
    return torch.as_tensor(array, device=device)


def iter_target_rows(label_file: Path, diseases: list[str]):
    """Stream sorted direct labels in bounded Arrow batches."""
    label_names = [f"disease_label_{h}" for h in HORIZONS]
    valid_names = [f"disease_valid_{h}" for h in HORIZONS]
    death_names = [f"death_label_{h}" for h in HORIZONS]
    death_valid_names = [f"death_valid_{h}" for h in HORIZONS]
    columns = ["sample_id", "disease_code", *label_names, *valid_names,
               *death_names, *death_valid_names]
    carry = None
    previous_sid = None

    def make_target(values: dict[str, np.ndarray], start: int, end: int):
        nonlocal previous_sid
        sid = values["sample_id"][start]
        if end - start != len(diseases):
            raise ValueError(f"Training labels for {sid} do not cover every disease")
        if values["disease_code"][start:end].tolist() != diseases:
            raise ValueError(f"Training label disease order differs from manifest for {sid}")
        if previous_sid is not None and sid <= previous_sid:
            raise ValueError("Training label file is not strictly ordered by sample_id")
        previous_sid = sid
        y = np.stack([values[name][start:end] for name in label_names], axis=1).astype(np.float32)
        mask = np.stack([values[name][start:end] for name in valid_names], axis=1).astype(np.float32)
        death_y = np.asarray([values[name][start] for name in death_names], dtype=np.float32)
        death_mask = np.asarray([values[name][start] for name in death_valid_names], dtype=np.float32)
        return sid, y, mask, death_y, death_mask

    for batch in pq.ParquetFile(label_file).iter_batches(batch_size=65536, columns=columns):
        values = {name: np.asarray(batch.column(name).to_pylist()) for name in columns}
        if carry is not None:
            values = {name: np.concatenate((carry[name], values[name])) for name in columns}
        ids = values["sample_id"]
        boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        start = 0
        for end in boundaries:
            yield make_target(values, start, int(end))
            start = int(end)
        carry = {name: values[name][start:].copy() for name in columns}
    if carry is not None and len(carry["sample_id"]):
        yield make_target(carry, 0, len(carry["sample_id"]))


def train(train_dir: Path, manifest: Path, output: Path, *, modalities: str = "all",
          epochs: int = 2, batch_size: int = 128, max_samples: int | None = None,
          device_name: str = "cpu") -> dict:
    if modalities not in {"all", "diagnosis"}:
        raise ValueError("modalities must be all or diagnosis")
    start = time.monotonic()
    manifest_data = json.loads(manifest.read_text())
    disease_codes = manifest_data["scored_diseases"]
    if manifest_data["horizons_days"] != list(HORIZONS):
        raise ValueError("Baseline requires the direct 13-point v1 time grid")
    adapter = fit_adapter(train_dir)
    device = torch.device(device_name)
    torch.manual_seed(20260924)
    model = RiskNet(len(adapter["vocab"]), len(disease_codes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rows_seen = 0
    losses = []
    for epoch in range(epochs):
        model.train()
        target_stream = iter(iter_target_rows(train_dir / "label_windows.parquet", disease_codes))
        target = next(target_stream, None)
        pending = []

        def step(batch):
            features, y, mask, dy, dm = zip(*batch)
            x = pad_features(features, device)
            y = torch.as_tensor(np.stack(y), device=device)
            mask = torch.as_tensor(np.stack(mask), device=device)
            dy = torch.as_tensor(np.stack(dy), device=device)
            dm = torch.as_tensor(np.stack(dm), device=device)
            pred, dpred = model(x)
            disease_loss = nn.functional.binary_cross_entropy(pred.clamp(1e-6,1-1e-6), y, reduction="none")
            death_loss = nn.functional.binary_cross_entropy(dpred.clamp(1e-6,1-1e-6), dy, reduction="none")
            loss = (disease_loss * mask).sum() / mask.sum().clamp(min=1) + (death_loss * dm).sum() / dm.sum().clamp(min=1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return float(loss.detach().cpu())

        for sid, history in iter_histories(train_dir, adapter["max_events"]):
            if max_samples is not None and rows_seen >= max_samples * (epoch + 1):
                break
            while target is not None and target[0] < sid:
                target = next(target_stream, None)
            if target is None or target[0] != sid:
                raise ValueError(f"Missing training label for {sid}")
            pending.append((history_to_ids(history, adapter, modalities), *target[1:]))
            target = next(target_stream, None)
            rows_seen += 1
            if len(pending) == batch_size:
                losses.append(step(pending))
                pending = []
        if pending:
            losses.append(step(pending))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "adapter": adapter,
                "diseases": disease_codes, "horizons": HORIZONS,
                "kind": "neural", "modalities": modalities}, output)
    report = {"samples_seen": rows_seen, "epochs": epochs, "batch_size": batch_size,
              "modalities": modalities, "duration_seconds": round(time.monotonic()-start,2),
              "final_loss": losses[-1] if losses else None, "device": device_name}
    output.with_suffix(".training.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def train_rate_baseline(train_dir: Path, manifest: Path, output: Path,
                        kind: str) -> dict:
    """Fit train-only prevalence or age/sex-stratified cumulative risks."""
    if kind not in {"prevalence", "demographic"}:
        raise ValueError("Rate baseline kind must be prevalence or demographic")
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    manifest_data = json.loads(manifest.read_text())
    diseases = manifest_data["scored_diseases"]
    if manifest_data["horizons_days"] != list(HORIZONS):
        raise ValueError("Rate baseline requires the direct 13-point v1 time grid")
    db = duckdb.connect()
    labels = (train_dir / "label_windows.parquet").as_posix()
    samples = (train_dir / "samples.parquet").as_posix()
    groups = [("ALL", -1)] if kind == "prevalence" else [
        ("ALL", -1),
        *((sex, age_bin) for sex in ("F", "M") for age_bin in range(4)),
    ]
    disease_rates, death_rates = {}, {}
    for sex, age_bin in groups:
        condition = "TRUE" if sex == "ALL" else (
            f"s.gender='{sex}' AND CASE WHEN s.approx_age<45 THEN 0 "
            "WHEN s.approx_age<65 THEN 1 WHEN s.approx_age<80 THEN 2 ELSE 3 END="
            f"{age_bin}")
        aggregate_columns = ",".join(
            f"count(*) FILTER(WHERE l.disease_label_{h}) AS p_{h},"
            f"count(*) FILTER(WHERE l.disease_valid_{h}) AS n_{h}"
            for h in HORIZONS)
        rows = db.execute(f"""SELECT l.disease_code,{aggregate_columns}
          FROM read_parquet(?) l JOIN read_parquet(?) s USING(sample_id)
          WHERE {condition}
            AND l.diagnosis_observation_end>=l.cutoff+INTERVAL 365 DAY
            AND l.death_observation_end>=l.cutoff+INTERVAL 365 DAY
          GROUP BY l.disease_code""", [labels, samples]).fetchall()
        by_code = {row[0]: row[1:] for row in rows}
        group_key = f"{sex}:{age_bin}"
        disease_rates[group_key] = {}
        for code in diseases:
            counts = by_code.get(code, (0, 0) * len(HORIZONS))
            values = [(counts[2*k] + 1) / (counts[2*k+1] + 2)
                      for k in range(len(HORIZONS))]
            disease_rates[group_key][code] = np.maximum.accumulate(values).tolist()
        first_code = diseases[0]
        death_columns = ",".join(
            f"count(*) FILTER(WHERE l.death_label_{h}) AS p_{h},"
            f"count(*) FILTER(WHERE l.death_valid_{h}) AS n_{h}"
            for h in HORIZONS)
        counts = db.execute(f"""SELECT {death_columns}
          FROM read_parquet(?) l JOIN read_parquet(?) s USING(sample_id)
          WHERE l.disease_code=? AND {condition}
            AND l.diagnosis_observation_end>=l.cutoff+INTERVAL 365 DAY
            AND l.death_observation_end>=l.cutoff+INTERVAL 365 DAY""",
          [labels, samples, first_code]).fetchone()
        values = [(counts[2*k] + 1) / (counts[2*k+1] + 2)
                  for k in range(len(HORIZONS))]
        death_rates[group_key] = np.maximum.accumulate(values).tolist()
    db.close()
    if kind == "demographic":
        disease_rates["fallback"] = disease_rates.pop("ALL:-1")
        death_rates["fallback"] = death_rates.pop("ALL:-1")
    torch.save({"kind": kind, "diseases": diseases, "horizons": HORIZONS,
                "disease_rates": disease_rates, "death_rates": death_rates,
                "adapter": {"max_events": 512}}, output)
    report = {"kind": kind, "duration_seconds": round(time.monotonic()-start, 2),
              "training_source": "complete-year train labels only"}
    output.with_suffix(".training.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


class Predictor:
    def __init__(self, checkpoint: Path, device_name: str = "cpu"):
        self.device = torch.device(device_name)
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.adapter = state["adapter"]
        self.diseases = state["diseases"]
        self.horizons = tuple(state["horizons"])
        self.kind = state["kind"]
        if self.horizons != HORIZONS:
            raise ValueError("Checkpoint does not directly predict the v1 time grid")
        if self.kind == "neural":
            self.modalities = state["modalities"]
            self.model = RiskNet(len(self.adapter["vocab"]), len(self.diseases)).to(self.device)
            self.model.load_state_dict(state["state_dict"])
            self.model.eval()
        else:
            self.disease_rates = state["disease_rates"]
            self.death_rates = state["death_rates"]

    def predict_risk(self, history: list[dict], horizons: list[int] | tuple[int, ...]):
        if tuple(horizons) != self.horizons:
            raise ValueError("Risks must be requested on the fixed direct-prediction grid")
        if self.kind == "neural":
            features = [history_to_ids(h, self.adapter, self.modalities) for h in history]
            with torch.no_grad():
                d, death = self.model(pad_features(features, self.device))
            return {"disease_risk": d.cpu().numpy(), "death_risk": death.cpu().numpy()}
        disease, death = [], []
        for row in history:
            if self.kind == "prevalence":
                key = "ALL:-1"
            else:
                age = row.get("approx_age") or 0
                age_bin = 0 if age < 45 else 1 if age < 65 else 2 if age < 80 else 3
                key = f"{row.get('gender', 'unknown')}:{age_bin}"
                if key not in self.disease_rates:
                    key = "fallback"
            disease.append([self.disease_rates[key][code] for code in self.diseases])
            death.append(self.death_rates[key])
        return {"disease_risk": np.asarray(disease, dtype=np.float32),
                "death_risk": np.asarray(death, dtype=np.float32)}


def load_predictor(submission_dir: Path, manifest: dict, device: str = "cpu") -> Predictor:
    """Compatibility hook used only by the verifier's synthetic smoke fixture."""
    predictor = Predictor(submission_dir / "model.pt", device)
    if predictor.diseases != manifest["scored_diseases"]:
        raise ValueError("Checkpoint disease list does not match input manifest")
    return predictor
