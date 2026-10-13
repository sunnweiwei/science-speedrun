"""Trainable longitudinal-v2 starter and its predeclared ablations.

The cache retains every event for the unordered summary and a deterministic,
history-spanning sequence for the temporal encoder.  Continuous lab values are
standardized by train-only code/unit statistics; time is measured backwards
from the cutoff.  sample_id is used only for alignment and data splitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import roc_auc_score

HORIZONS = tuple(range(30, 361, 30)) + (365,)
MODALITIES = {"diagnosis": 1, "lab": 2, "prescription": 3, "administration": 4}


def _token(event_type: str, code: str, unit: str | None) -> str:
    return f"{event_type}:{code}:{unit or ''}" if event_type == "lab" else f"{event_type}:{code}"


def fit_spec(train: Path, max_codes: int = 16000, min_code_count: int = 20) -> dict:
    db = duckdb.connect()
    events = str(train / "events.parquet")
    rows = db.execute("""SELECT event_type,code,unit,count(*) n,
      avg(try_cast(value AS DOUBLE)) mean_value,
      stddev_pop(try_cast(value AS DOUBLE)) std_value,
      count(try_cast(value AS DOUBLE)) numeric_n
      FROM read_parquet(?) GROUP BY event_type,code,unit
      HAVING count(*)>=? ORDER BY n DESC,event_type,code,unit LIMIT ?""",
      [events, min_code_count, max_codes]).fetchall()
    db.close()
    vocab, numeric = {}, {}
    for event_type, code, unit, _, mean, std, numeric_n in rows:
        key = _token(event_type, code, unit)
        vocab[key] = len(vocab) + 2  # 0 padding, 1 unknown
        if numeric_n >= 50 and mean is not None and std is not None and std > 1e-8:
            numeric[key] = [float(mean), float(std)]
    return {"schema": "ndp-longitudinal-adapter/v3", "vocab": vocab,
            "numeric": numeric, "max_sequence_events": 512,
            "numeric_imputation": "train_mean_with_explicit_missing_indicator",
            "selection": "all_if_short_else_256_uniform_older_plus_256_most_recent"}


def _select_sequence(n: int, limit: int) -> np.ndarray:
    if n <= limit:
        return np.arange(n, dtype=np.int64)
    recent = limit // 2
    older = np.linspace(0, n - recent - 1, limit - recent, dtype=np.int64)
    return np.concatenate((older, np.arange(n - recent, n, dtype=np.int64)))


def _numeric_value(kind: str, key: str, raw, stats: dict) -> tuple[float, float]:
    parsed = None
    try:
        parsed = float(raw)
        if not math.isfinite(parsed):
            parsed = None
    except (TypeError, ValueError):
        pass
    numeric_modality = kind in {"lab", "prescription", "administration"}
    observed = parsed is not None and key in stats
    value = (float(np.clip((parsed - stats[key][0]) / stats[key][1], -8, 8))
             if observed else 0.0)
    return value, float(numeric_modality and not observed)


def _sequence_delta(backwards_time: np.ndarray) -> np.ndarray:
    """Gaps for rows ordered by event_time (oldest to newest)."""
    return np.maximum(0.0, np.r_[0.0, backwards_time[:-1] - backwards_time[1:]]).astype(np.float32)


def _load_labels_aligned(labels: Path, ids: list[str], diseases: list[str]) -> dict:
    """Load a keyed label grid after proving key uniqueness and exact coverage."""
    db = duckdb.connect()
    db.register("wanted_samples", pa.table({"sample_id": ids,
        "sample_order": np.arange(len(ids), dtype=np.int32)}))
    db.register("wanted_diseases", pa.table({"disease_code": diseases,
        "disease_order": np.arange(len(diseases), dtype=np.int16)}))
    label_path = str(labels)
    total, unique = db.execute("""SELECT count(*),count(DISTINCT (sample_id,disease_code))
        FROM read_parquet(?)""", [label_path]).fetchone()
    expected = len(ids) * len(diseases)
    if total != unique or total != expected:
        db.close()
        raise ValueError("Label keys are duplicated or do not match the requested sample/disease grid")
    joined = db.execute("""SELECT count(*) FROM wanted_samples s CROSS JOIN wanted_diseases d
        ANTI JOIN read_parquet(?) l USING(sample_id,disease_code)""", [label_path]).fetchone()[0]
    if joined:
        db.close()
        raise ValueError("Label grid is missing requested sample/disease keys")
    shape = (len(ids), len(diseases), len(HORIZONS))
    y, mask = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    dy = np.zeros((len(ids), len(HORIZONS)), np.uint8)
    dm = np.zeros_like(dy)
    fields = [f"l.{name}_{horizon}" for horizon in HORIZONS for name in
              ("disease_label", "disease_valid", "death_label", "death_valid")]
    aligned = db.execute(f"""SELECT {','.join(fields)}
      FROM wanted_samples s CROSS JOIN wanted_diseases d
      JOIN read_parquet(?) l USING(sample_id,disease_code)
      ORDER BY s.sample_order,d.disease_order""", [label_path]).fetchnumpy()
    for k, horizon in enumerate(HORIZONS):
        y[:, :, k] = np.asarray(aligned[f"disease_label_{horizon}"], dtype=np.uint8).reshape(
            len(ids), len(diseases))
        mask[:, :, k] = np.asarray(aligned[f"disease_valid_{horizon}"], dtype=np.uint8).reshape(
            len(ids), len(diseases))
        death_grid = np.asarray(aligned[f"death_label_{horizon}"], dtype=np.uint8).reshape(
            len(ids), len(diseases))
        death_mask_grid = np.asarray(aligned[f"death_valid_{horizon}"], dtype=np.uint8).reshape(
            len(ids), len(diseases))
        if not ((death_grid == death_grid[:, :1]).all()
                and (death_mask_grid == death_mask_grid[:, :1]).all()):
            db.close()
            raise ValueError("Death labels disagree across disease rows for a sample")
        dy[:, k], dm[:, k] = death_grid[:, 0], death_mask_grid[:, 0]
    primary_rows = db.execute("""SELECT s.sample_order,
      bool_and(l.cutoff + INTERVAL 365 DAY <= l.diagnosis_observation_end
               AND l.cutoff + INTERVAL 365 DAY <= l.death_observation_end) AS full_year
      FROM wanted_samples s JOIN read_parquet(?) l USING(sample_id)
      GROUP BY s.sample_order ORDER BY s.sample_order""", [label_path]).fetchall()
    db.close()
    if len(primary_rows) != len(ids):
        raise ValueError("Label grid cannot determine complete-year primary cohort")
    primary = np.asarray([row[1] for row in primary_rows], dtype=np.uint8)
    return {"y": y, "mask": mask, "death_y": dy, "death_mask": dm,
            "primary": primary}


def build_cache(input_dir: Path, cache: Path, spec: dict,
                labels: Path | None = None) -> dict:
    start = time.monotonic()
    samples = sorted(pq.read_table(input_dir / "samples.parquet").to_pylist(),
                     key=lambda x: x["sample_id"])
    ids = [x["sample_id"] for x in samples]
    vocab, stats = spec["vocab"], spec["numeric"]
    all_code, all_mod, all_value, all_missing, all_time, offsets = [], [], [], [], [], [0]
    seq_code, seq_mod, seq_value, seq_missing, seq_time, seq_delta, seq_len = [], [], [], [], [], [], []
    db = duckdb.connect()
    cursor = db.execute("""SELECT sample_id,event_type,code,unit,value,
      greatest(seconds_before_cutoff,0) seconds_before_cutoff
      FROM read_parquet(?) ORDER BY sample_id,event_time,available_time,
      event_type,code,event_group_key""", [str(input_dir / "events.parquet")])
    stream = iter(lambda: cursor.fetchmany(65536), [])
    flat_stream = (row for batch in stream for row in batch)
    row = next(flat_stream, None)
    for sid in ids:
        codes, mods, values, missing, times = [], [], [], [], []
        while row is not None and row[0] == sid:
            _, kind, code, unit, raw, seconds = row
            key = _token(kind, code, unit)
            codes.append(vocab.get(key, 1)); mods.append(MODALITIES.get(kind, 0))
            # Zero is the standardized training mean, never an asserted raw zero.
            value, is_missing = _numeric_value(kind, key, raw, stats)
            values.append(value)
            missing.append(is_missing)
            times.append(float(np.log1p(float(seconds) / 86400.0)))
            row = next(flat_stream, None)
        code_a = np.asarray(codes, dtype=np.int32)
        mod_a = np.asarray(mods, dtype=np.uint8)
        value_a = np.asarray(values, dtype=np.float32)
        missing_a = np.asarray(missing, dtype=np.float32)
        time_a = np.asarray(times, dtype=np.float32)
        all_code.append(code_a); all_mod.append(mod_a); all_value.append(value_a)
        all_missing.append(missing_a); all_time.append(time_a)
        offsets.append(offsets[-1] + len(code_a))
        choose = _select_sequence(len(code_a), int(spec["max_sequence_events"]))
        sc, sm, sv, sx, st = (code_a[choose], mod_a[choose], value_a[choose],
                              missing_a[choose], time_a[choose])
        # Rows are chronological, so backwards-from-cutoff time decreases.
        delta = _sequence_delta(st)
        pad = int(spec["max_sequence_events"]) - len(choose)
        seq_code.append(np.pad(sc, (0, pad))); seq_mod.append(np.pad(sm, (0, pad)))
        seq_value.append(np.pad(sv, (0, pad))); seq_time.append(np.pad(st, (0, pad)))
        seq_missing.append(np.pad(sx, (0, pad)))
        seq_delta.append(np.pad(delta, (0, pad))); seq_len.append(len(choose))
    db.close()
    payload = {
        "ids": ids,
        "age": torch.tensor([x["approx_age"] for x in samples], dtype=torch.float32),
        "sex": torch.tensor([1.0 if x["gender"] == "F" else 0.0 for x in samples]),
        "all_code": torch.from_numpy(np.concatenate(all_code) if all_code else np.empty(0, np.int32)),
        "all_mod": torch.from_numpy(np.concatenate(all_mod) if all_mod else np.empty(0, np.uint8)),
        "all_value": torch.from_numpy(np.concatenate(all_value) if all_value else np.empty(0, np.float32)),
        "all_missing": torch.from_numpy(np.concatenate(all_missing) if all_missing else np.empty(0, np.float32)),
        "all_time": torch.from_numpy(np.concatenate(all_time) if all_time else np.empty(0, np.float32)),
        "offsets": torch.tensor(offsets, dtype=torch.int64),
        "seq_code": torch.from_numpy(np.stack(seq_code)).to(torch.int32),
        "seq_mod": torch.from_numpy(np.stack(seq_mod)).to(torch.uint8),
        "seq_value": torch.from_numpy(np.stack(seq_value)),
        "seq_missing": torch.from_numpy(np.stack(seq_missing)),
        "seq_time": torch.from_numpy(np.stack(seq_time)),
        "seq_delta": torch.from_numpy(np.stack(seq_delta)),
        "seq_len": torch.tensor(seq_len, dtype=torch.int16),
    }
    if labels is not None:
        manifest = json.loads((input_dir.parent / "manifest.json").read_text())
        diseases = manifest["scored_diseases"]
        aligned = _load_labels_aligned(labels, ids, diseases)
        payload.update({key: torch.from_numpy(value) for key, value in aligned.items()})
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache)
    return {"samples": len(ids), "events": int(payload["all_code"].numel()),
            "cache": str(cache), "duration_seconds": round(time.monotonic()-start, 2)}


class CacheDataset(Dataset):
    def __init__(self, path: Path):
        self.x = torch.load(path, map_location="cpu", weights_only=False)
    def __len__(self): return len(self.x["ids"])
    def __getitem__(self, i):
        a, b = int(self.x["offsets"][i]), int(self.x["offsets"][i+1])
        out = {"index": i, "id": self.x["ids"][i], "age": self.x["age"][i],
               "sex": self.x["sex"][i], "all_code": self.x["all_code"][a:b],
               "all_mod": self.x["all_mod"][a:b], "all_value": self.x["all_value"][a:b],
               "all_missing": self.x["all_missing"][a:b]}
        for key in ("seq_code", "seq_mod", "seq_value", "seq_missing", "seq_time",
                    "seq_delta", "seq_len", "y", "mask", "death_y", "death_mask", "primary"):
            if key in self.x: out[key] = self.x[key][i]
        return out


def collate(rows: list[dict], diagnosis_only: bool = False) -> dict:
    result = {k: torch.stack([r[k] for r in rows]) for k in
              ("age", "sex", "seq_code", "seq_mod", "seq_value", "seq_missing",
               "seq_time", "seq_delta", "seq_len")}
    result["ids"] = [r["id"] for r in rows]
    codes, mods, values, missing, groups = [], [], [], [], []
    for i, r in enumerate(rows):
        use = r["all_mod"] == 1 if diagnosis_only else torch.ones_like(r["all_mod"], dtype=torch.bool)
        codes.append(r["all_code"][use]); mods.append(r["all_mod"][use])
        values.append(r["all_value"][use]); missing.append(r["all_missing"][use])
        groups.append(torch.full((int(use.sum()),), i, dtype=torch.int64))
    result["all_code"] = torch.cat(codes).long() if codes else torch.empty(0, dtype=torch.long)
    result["all_mod"] = torch.cat(mods).long() if mods else torch.empty(0, dtype=torch.long)
    result["all_value"] = torch.cat(values) if values else torch.empty(0)
    result["all_missing"] = torch.cat(missing) if missing else torch.empty(0)
    result["all_group"] = torch.cat(groups) if groups else torch.empty(0, dtype=torch.long)
    if diagnosis_only:
        # Snapshot the original modality mask before mutating any tensor.  The
        # previous loop derived later masks from the already-compacted seq_mod.
        original = {key: result[key].clone() for key in
                    ("seq_code", "seq_mod", "seq_value", "seq_missing", "seq_time")}
        keep = original["seq_mod"] == 1
        for key in original:
            compact = torch.zeros_like(result[key])
            for i in range(len(rows)):
                use = keep[i]
                compact[i, :int(use.sum())] = original[key][i, use]
            result[key] = compact
        result["seq_len"] = keep.sum(1).clamp(min=1)
        result["seq_delta"].zero_()
        for i, length in enumerate(result["seq_len"].tolist()):
            if length > 1:
                result["seq_delta"][i, 1:length] = torch.clamp(
                    result["seq_time"][i, :length-1] - result["seq_time"][i, 1:length], min=0)
    for key in ("y", "mask", "death_y", "death_mask", "primary"):
        if key in rows[0]: result[key] = torch.stack([r[key] for r in rows]).float()
    return result


class LongitudinalRiskModel(nn.Module):
    def __init__(self, vocab: int, diseases: int, kind: str, hidden: int = 128):
        super().__init__(); self.kind = kind; self.diseases = diseases
        self.code = nn.Embedding(vocab + 2, 64, padding_idx=0)
        self.modality = nn.Embedding(5, 64, padding_idx=0)
        self.value = nn.Linear(1, 64, bias=False)
        self.missing = nn.Linear(1, 64, bias=False)
        self.time = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 64))
        if kind != "unordered": self.gru = nn.GRU(64, hidden, batch_first=True)
        width = 64 + 16 + (hidden if kind != "unordered" else 0)
        self.demo = nn.Sequential(nn.Linear(2, 16), nn.SiLU())
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 256), nn.SiLU(),
                                  nn.Dropout(.15), nn.Linear(256, (diseases + 1) * len(HORIZONS)))
        nn.init.constant_(self.head[-1].bias, -7.0)

    def forward(self, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
        n = b["age"].shape[0]
        if b["all_code"].numel():
            e = (self.code(b["all_code"]) + self.modality(b["all_mod"])
                 + self.value(b["all_value"][:,None])
                 + self.missing(b["all_missing"][:,None]))
            summary = torch.zeros((n, 64), device=e.device).index_add_(0, b["all_group"], e)
            count = torch.zeros(n, device=e.device).index_add_(0, b["all_group"], torch.ones(len(e), device=e.device))
            summary = summary / count.clamp(min=1)[:,None]
        else: summary = torch.zeros((n,64), device=b["age"].device)
        demo = self.demo(torch.stack(((b["age"]-65)/20, b["sex"]), dim=1))
        parts = [summary, demo]
        if self.kind != "unordered":
            seq = self.code(b["seq_code"].long()) + self.modality(b["seq_mod"].long())
            seq = seq + self.value(b["seq_value"][:,:,None])
            seq = seq + self.missing(b["seq_missing"][:,:,None])
            seq = seq + self.time(torch.stack((b["seq_time"], b["seq_delta"]), dim=2))
            seq = seq * (b["seq_code"] != 0)[:,:,None]
            encoded, _ = self.gru(seq)
            last_index = (b["seq_len"].long()-1).clamp(min=0)
            parts.append(encoded[torch.arange(n,device=encoded.device),last_index])
        increments = nn.functional.softplus(self.head(torch.cat(parts,1)).reshape(
            -1,self.diseases+1,len(HORIZONS)))
        risk = 1-torch.exp(-torch.cumsum(increments,dim=2))
        return risk[:,:self.diseases], risk[:,self.diseases]


def move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}


def loss_fn(model: nn.Module, batch: dict, positive_weight: torch.Tensor) -> torch.Tensor:
    p, dp = model(batch)
    p = p.clamp(1e-6, 1 - 1e-6)
    element = -(positive_weight[None] * batch["y"] * torch.log(p)
                + (1 - batch["y"]) * torch.log1p(-p)) * batch["mask"]
    # Each disease/time cell has equal optimization weight. Positive weights
    # are train-fit and capped so rare outcomes create ranking signal without
    # allowing a handful of events to dominate gradients.
    cell = element.sum(0) / batch["mask"].sum(0).clamp(min=1)
    disease = cell.mean()
    death = (((dp-batch["death_y"])**2*batch["death_mask"]).sum()
             / batch["death_mask"].sum().clamp(min=1))
    return disease + .1*death


def macro_time_auroc(labels: np.ndarray, prediction: np.ndarray,
                     valid: np.ndarray) -> float | None:
    """The official disease-first primary metric, for checkpoint selection."""
    by_disease = []
    for j in range(labels.shape[1]):
        by_time = []
        for k in range(labels.shape[2]):
            use = valid[:, j, k]
            y = labels[use, j, k]
            if len(y) and y.min() != y.max():
                by_time.append(float(roc_auc_score(y, prediction[use, j, k])))
        if by_time:
            by_disease.append(float(np.mean(by_time)))
    return float(np.mean(by_disease)) if by_disease else None


def train_model(materialized: Path, cache: Path, output: Path, kind: str,
                epochs: int, batch_size: int, seed: int, device_name: str) -> dict:
    if kind not in {"temporal", "unordered", "diagnosis"}: raise ValueError(kind)
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device(device_name)
    manifest = json.loads((materialized/"manifest.json").read_text())
    spec = json.loads(cache.with_suffix(".adapter.json").read_text())
    data = CacheDataset(cache)
    folds = np.asarray([int(hashlib.sha256(x.encode()).hexdigest()[:8],16)%10 for x in data.x["ids"]])
    fit_idx, val_idx = np.flatnonzero(folds<8), np.flatnonzero(folds==8)
    diagnosis_only = kind == "diagnosis"
    make_collate = lambda rows: collate(rows, diagnosis_only)
    train_loader = DataLoader(Subset(data,fit_idx),batch_size=batch_size,shuffle=True,
        num_workers=0,collate_fn=make_collate,pin_memory=True)
    val_loader = DataLoader(Subset(data,val_idx),batch_size=batch_size,shuffle=False,
        num_workers=0,collate_fn=make_collate,pin_memory=True)
    model = LongitudinalRiskModel(len(spec["vocab"]),len(manifest["scored_diseases"]),
                                  "unordered" if kind=="unordered" else "temporal").to(device)
    fit_y = data.x["y"][fit_idx].float()
    fit_mask = data.x["mask"][fit_idx].float()
    positive = (fit_y * fit_mask).sum(0)
    negative = ((1 - fit_y) * fit_mask).sum(0)
    positive_weight = torch.clamp(negative / positive.clamp(min=1), min=1, max=100).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=.5,patience=1)
    best, best_state, patience, history = -float("inf"), None, 0, []
    started=time.monotonic()
    for epoch in range(epochs):
        model.train(); train_loss=[]
        for batch in train_loader:
            batch=move(batch,device); loss=loss_fn(model,batch,positive_weight)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); train_loss.append(float(loss))
        model.eval(); val_loss=[]; val_prediction=[]; val_y=[]; val_mask=[]; val_primary=[]
        with torch.no_grad():
            for batch in val_loader:
                batch=move(batch,device); val_loss.append(float(loss_fn(model,batch,positive_weight)))
                predicted, _ = model(batch)
                val_prediction.append(predicted.cpu().numpy())
                val_y.append(batch["y"].cpu().numpy())
                val_mask.append(batch["mask"].cpu().numpy().astype(bool))
                val_primary.append(batch["primary"].cpu().numpy().astype(bool))
        yp=np.concatenate(val_y); pp=np.concatenate(val_prediction); vm=np.concatenate(val_mask)
        primary=np.concatenate(val_primary)
        val_auc=macro_time_auroc(yp,pp,vm & primary[:,None,None])
        if val_auc is None:
            raise ValueError("Internal validation fold has no scorable disease/time AUROC cell")
        record={"epoch":epoch+1,"train_loss":float(np.mean(train_loss)),
                "internal_val_loss":float(np.mean(val_loss)),
                "internal_val_macro_time_averaged_auroc":val_auc,
                "lr":opt.param_groups[0]["lr"]}
        history.append(record); print(json.dumps(record),flush=True); scheduler.step(val_auc)
        if val_auc > best+1e-5:
            best=val_auc; patience=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            patience+=1
            if patience>=3: break
    model.load_state_dict(best_state)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.cpu().state_dict(),"spec":spec,"diseases":manifest["scored_diseases"],
                "kind":kind,"model_kind":"unordered" if kind=="unordered" else "temporal",
                "seed":seed,"horizons":HORIZONS},output)
    report={"kind":kind,"seed":seed,"epochs_completed":len(history),
            "best_internal_val_macro_time_averaged_auroc":best,
            "parameters":sum(p.numel() for p in model.parameters()),"duration_seconds":time.monotonic()-started,
            "device":device_name,"history":history,
            "checkpoint_rule":"maximum full internal validation macro time-averaged AUROC",
            "objective":"equal-cell capped train-prevalence-balanced binary cross entropy plus 0.1 mortality Brier",
            "split":"sha256(sample_id) mod 10: 0-7 fit, 8 early-stop, 9 reserved calibration"}
    output.with_suffix(".training.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def predict(cache: Path, checkpoint: Path, output: Path, batch_size: int, device_name: str) -> dict:
    state=torch.load(checkpoint,map_location="cpu",weights_only=False); device=torch.device(device_name)
    data=CacheDataset(cache); diagnosis_only=state["kind"]=="diagnosis"
    loader=DataLoader(data,batch_size=batch_size,shuffle=False,num_workers=0,
        collate_fn=lambda rows: collate(rows,diagnosis_only),pin_memory=True)
    model=LongitudinalRiskModel(len(state["spec"]["vocab"]),len(state["diseases"]),state["model_kind"]).to(device)
    model.load_state_dict(state["state_dict"]); model.eval(); rows=[]
    with torch.no_grad():
        for batch in loader:
            ids=batch["ids"]; batch=move(batch,device); p,d=model(batch)
            p=p.cpu().numpy(); d=d.cpu().numpy()
            if "calibration" in state:
                cal=state["calibration"]
                p=cal["disease_alpha"]*p+(1-cal["disease_alpha"])*cal["disease_rate"][None]
                d=cal["death_alpha"]*d+(1-cal["death_alpha"])*cal["death_rate"][None]
                p=np.maximum.accumulate(p,axis=2); d=np.maximum.accumulate(d,axis=1)
            rows.extend({"sample_id":sid,"disease_risk":p[i].reshape(-1).tolist(),"death_risk":d[i].tolist()} for i,sid in enumerate(ids))
    schema=pa.schema([pa.field("sample_id",pa.string()),pa.field("disease_risk",pa.list_(pa.float32(),len(state["diseases"])*len(HORIZONS))),pa.field("death_risk",pa.list_(pa.float32(),len(HORIZONS)))])
    output.parent.mkdir(parents=True,exist_ok=True); pq.write_table(pa.Table.from_pylist(rows,schema=schema),output,compression="zstd")
    return {"samples":len(rows),"output":str(output)}


def calibrate(cache: Path, checkpoint: Path, output: Path, batch_size: int,
              device_name: str) -> dict:
    """Fit conservative prevalence shrinkage on the reserved train fold only."""
    state=torch.load(checkpoint,map_location="cpu",weights_only=False); device=torch.device(device_name)
    data=CacheDataset(cache); x=data.x
    folds=np.asarray([int(hashlib.sha256(s.encode()).hexdigest()[:8],16)%10 for s in x["ids"]])
    fit=torch.tensor(folds<9); cal_idx=np.flatnonzero(folds==9)
    y=x["y"].float(); mask=x["mask"].float(); dy=x["death_y"].float(); dm=x["death_mask"].float()
    disease_rate=((y[fit]*mask[fit]).sum(0)+1)/(mask[fit].sum(0)+2)
    death_rate=((dy[fit]*dm[fit]).sum(0)+1)/(dm[fit].sum(0)+2)
    disease_rate=torch.cummax(disease_rate,1).values.numpy()
    death_rate=torch.cummax(death_rate,0).values.numpy()
    diagnosis_only=state["kind"]=="diagnosis"
    loader=DataLoader(Subset(data,cal_idx),batch_size=batch_size,shuffle=False,num_workers=0,
        collate_fn=lambda rows: collate(rows,diagnosis_only),pin_memory=True)
    model=LongitudinalRiskModel(len(state["spec"]["vocab"]),len(state["diseases"]),state["model_kind"]).to(device)
    model.load_state_dict(state["state_dict"]); model.eval(); pp=[]; dd=[]
    with torch.no_grad():
        for batch in loader:
            batch=move(batch,device); p,d=model(batch); pp.append(p.cpu().numpy()); dd.append(d.cpu().numpy())
    pp=np.concatenate(pp); dd=np.concatenate(dd)
    cy=y[cal_idx].numpy(); cm=mask[cal_idx].numpy(); cdy=dy[cal_idx].numpy(); cdm=dm[cal_idx].numpy()
    # One global shrinkage coefficient avoids 3,406-cell calibration overfit.
    grid=np.linspace(0,1,51)
    disease_loss=[]; death_loss=[]
    for a in grid:
        q=a*pp+(1-a)*disease_rate[None]
        disease_loss.append(float((((q-cy)**2)*cm).sum()/cm.sum()))
        qd=a*dd+(1-a)*death_rate[None]
        death_loss.append(float((((qd-cdy)**2)*cdm).sum()/cdm.sum()))
    da=float(grid[int(np.argmin(disease_loss))]); ma=float(grid[int(np.argmin(death_loss))])
    state["calibration"]={"method":"global_convex_shrinkage_to_fit_fold_prevalence",
        "fit_folds":"sha256(sample_id) mod 10 < 9","calibration_fold":"sha256(sample_id) mod 10 == 9",
        "disease_alpha":da,"death_alpha":ma,"disease_rate":disease_rate,"death_rate":death_rate}
    torch.save(state,output)
    report={"disease_alpha":da,"death_alpha":ma,"calibration_samples":len(cal_idx),
            "disease_brier_before":disease_loss[-1],"disease_brier_after":min(disease_loss),
            "death_brier_before":death_loss[-1],"death_brier_after":min(death_loss)}
    output.with_suffix(".calibration.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main() -> None:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("cache"); p.add_argument("--input",type=Path,required=True); p.add_argument("--cache",type=Path,required=True); p.add_argument("--adapter",type=Path); p.add_argument("--labels",type=Path)
    p=sub.add_parser("train"); p.add_argument("--materialized",type=Path,required=True); p.add_argument("--cache",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--kind",choices=["temporal","unordered","diagnosis"],required=True); p.add_argument("--epochs",type=int,default=15); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--seed",type=int,default=20260929); p.add_argument("--device",default="cuda")
    p=sub.add_parser("predict"); p.add_argument("--cache",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--device",default="cuda")
    p=sub.add_parser("calibrate"); p.add_argument("--cache",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--device",default="cuda")
    args=parser.parse_args()
    if args.command=="cache":
        if args.adapter:
            spec=json.loads(args.adapter.read_text())
        else:
            spec=fit_spec(args.input)
        args.cache.with_suffix(".adapter.json").write_text(json.dumps(spec,indent=2)+"\n")
        print(json.dumps(build_cache(args.input,args.cache,spec,args.labels),indent=2))
    elif args.command=="train": print(json.dumps(train_model(args.materialized,args.cache,args.output,args.kind,args.epochs,args.batch_size,args.seed,args.device),indent=2))
    elif args.command=="predict": print(json.dumps(predict(args.cache,args.checkpoint,args.output,args.batch_size,args.device),indent=2))
    else: print(json.dumps(calibrate(args.cache,args.checkpoint,args.output,args.batch_size,args.device),indent=2))


if __name__=="__main__": main()
