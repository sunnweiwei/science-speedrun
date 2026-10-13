"""Architecture-neutral readers, preprocessing hooks, and output helpers.

These utilities intentionally impose no vocabulary, truncation, batching, or model
architecture. ``sample_id`` is retained only for alignment.
"""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

HORIZONS = tuple(range(30, 361, 30)) + (365,)


class HistoryDataset(Dataset):
    """Small adapter for already-indexed examples.

    Agents may replace this with a streaming or memory-mapped dataset. Keeping the
    adapter here makes the same example structure available to training and
    inference without prescribing how a complete training corpus is cached.
    """

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


class HistoryPreprocessor:
    """Replaceable preprocessing contract shared by ``train.py`` and ``predict.py``."""

    def __init__(self, config: dict, state: dict | None = None):
        self.config = config
        self.state = state

    def fit(self, histories) -> "HistoryPreprocessor":
        raise NotImplementedError(
            "Implement training-only vocabulary/statistic fitting in HistoryPreprocessor.fit"
        )

    def transform(self, history: dict) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "Implement one-history feature encoding in HistoryPreprocessor.transform"
        )

    def state_dict(self) -> dict:
        if self.state is None:
            raise NotImplementedError(
                "Return JSON-like fitted preprocessing state from HistoryPreprocessor.state_dict"
            )
        return self.state


def build_preprocessor(config: dict, state: dict | None = None) -> HistoryPreprocessor:
    """Single construction hook used for both fitted and restored preprocessing."""
    return HistoryPreprocessor(config, state)


def collate_histories(histories: list[dict], preprocessor: HistoryPreprocessor) -> dict:
    """Encode and batch histories.

    The default only handles fixed-shape tensors. Sequence models should replace
    this function with their padding/mask/time-gap policy and keep that policy
    common to training and inference.
    """
    if not histories:
        raise ValueError("Cannot collate an empty history batch")
    encoded = [preprocessor.transform(history) for history in histories]
    keys = tuple(encoded[0])
    if any(tuple(row) != keys for row in encoded):
        raise ValueError("Preprocessor returned inconsistent feature keys")
    try:
        return {key: torch.stack([row[key] for row in encoded]) for key in keys}
    except RuntimeError as error:
        raise ValueError(
            "Encoded tensors have variable shapes; implement explicit padding and masks "
            "in collate_histories"
        ) from error


def build_training_loader(input_dir: Path, config: dict):
    """Return ``(loader, preprocessing_state)`` for optimization.

    This is deliberately the feature/target integration seam: fit preprocessing
    only on training records, attach label and at-risk tensors to each batch, and
    return ordinary serializable preprocessing state for the checkpoint.
    """
    raise NotImplementedError(
        "Implement build_training_loader with training labels, masks, and a fitted "
        "HistoryPreprocessor; return (loader, preprocessing_state)"
    )


def read_manifest(input_dir: Path) -> dict:
    manifest = json.loads((input_dir / "manifest.json").read_text())
    if tuple(manifest.get("horizons_days", ())) != HORIZONS:
        raise ValueError("Input horizon grid differs from the frozen task grid")
    if not manifest.get("scored_diseases"):
        raise ValueError("Input manifest has no scored disease vocabulary")
    return manifest


def load_histories(input_dir: Path) -> tuple[list[str], list[dict]]:
    """Load complete cutoff-bounded histories in the input sample order."""
    samples = pq.read_table(input_dir / "samples.parquet").to_pylist()
    ids = [row["sample_id"] for row in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Input sample_id values must be unique")
    by_id = {row["sample_id"]: {
        "gender": row["gender"], "approx_age": row["approx_age"], "events": []
    } for row in samples}
    for batch in pq.ParquetFile(input_dir / "events.parquet").iter_batches(batch_size=65536):
        for event in batch.to_pylist():
            if event["sample_id"] not in by_id:
                raise ValueError("Event sample_id does not exist in samples.parquet")
            by_id[event["sample_id"]]["events"].append(event)
    for history in by_id.values():
        history["events"].sort(
            key=lambda e: (e["available_time"], e["event_time"], e["event_type"], e["code"]))
    return ids, [by_id[sid] for sid in ids]


def iter_histories(input_dir: Path, max_events: int | None = None):
    """Stream histories in sorted sample order; return every event by default.

    ``max_events`` is an explicit caller policy. When set, the most recent events
    in availability/event-time order are retained; it is not a benchmark limit.
    """
    samples = sorted(pq.read_table(input_dir / "samples.parquet").to_pylist(),
                     key=lambda row: row["sample_id"])
    ids = [row["sample_id"] for row in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Input sample_id values must be unique")
    db = duckdb.connect()
    event_path = str(input_dir / "events.parquet")
    columns = {row[0] for row in db.execute("DESCRIBE SELECT * FROM read_parquet(?)",
                                           [event_path]).fetchall()}
    stay = "stay_key" if "stay_key" in columns else (
        "hadm_id" if "hadm_id" in columns else "NULL")
    group = "simultaneous_group" if "simultaneous_group" in columns else (
        "same_time_group" if "same_time_group" in columns else "NULL")
    cursor = db.execute(f"""SELECT sample_id,event_type,code,"value",unit,
      available_time,event_time,seconds_before_cutoff,{stay} AS stay_key,
      {group} AS simultaneous_group FROM read_parquet(?)
      ORDER BY sample_id,available_time,event_time,event_type,code""", [event_path])

    def rows():
        while True:
            batch = cursor.fetchmany(65536)
            if not batch:
                break
            yield from batch

    names = ("sample_id", "event_type", "code", "value", "unit",
             "available_time", "event_time", "seconds_before_cutoff", "stay_key",
             "simultaneous_group")
    stream = iter(rows())
    row = next(stream, None)
    for sample in samples:
        sid = sample["sample_id"]
        events = deque(maxlen=max_events) if max_events is not None else []
        while row is not None and row[0] == sid:
            events.append(dict(zip(names, row)))
            row = next(stream, None)
        yield sid, {"gender": sample["gender"], "approx_age": sample["approx_age"],
                    "cutoff": sample.get("cutoff"), "events": list(events)}
    if row is not None:
        raise ValueError("events.parquet contains a sample_id absent from samples.parquet")
    db.close()


def write_predictions(output: Path, sample_ids: list[str], disease_risk: np.ndarray,
                      death_risk: np.ndarray, disease_count: int) -> Path:
    """Validate and write the frozen prediction schema."""
    disease_risk = np.asarray(disease_risk, dtype=np.float32)
    death_risk = np.asarray(death_risk, dtype=np.float32)
    expected_disease = (len(sample_ids), disease_count, len(HORIZONS))
    expected_death = (len(sample_ids), len(HORIZONS))
    if disease_risk.shape != expected_disease or death_risk.shape != expected_death:
        raise ValueError(f"Prediction shapes must be {expected_disease} and {expected_death}")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Prediction sample_id values must be unique")
    for name, values in (("disease", disease_risk), ("death", death_risk)):
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{name} risks must be finite probabilities")
        if np.any(np.diff(values, axis=-1) < -1e-7):
            raise ValueError(f"{name} risks must be nondecreasing over time")
    schema = pa.schema([
        pa.field("sample_id", pa.string()),
        pa.field("disease_risk", pa.list_(pa.float32(), disease_count * len(HORIZONS))),
        pa.field("death_risk", pa.list_(pa.float32(), len(HORIZONS))),
    ])
    rows = [{"sample_id": sid,
             "disease_risk": disease_risk[i].reshape(-1).tolist(),
             "death_risk": death_risk[i].tolist()}
            for i, sid in enumerate(sample_ids)]
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "predictions.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), destination,
                   compression="zstd")
    return destination
