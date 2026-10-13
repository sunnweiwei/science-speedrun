"""Shared I/O for the timeseries-forecasting task (contract timeseries-forecast/v1).

Three things live here because the case builder, the grader, the starter and
the reference packages must agree on them exactly:

1. GIFT-Eval protocol constants and the frequency -> horizon / seasonality
   tables, transcribed from gift-eval `src/gift_eval/data.py` at commit
   db743860 and gluonts 0.15.1 `time_feature.seasonality`. They are tables,
   not runtime derivations, so a pandas alias rename cannot silently change a
   horizon or a seasonality.
2. Readers for the Arrow `save_to_disk` layout of Salesforce/GiftEval
   (`<dataset>/[<freq>/]data-00000-of-00001.arrow`), implemented with pyarrow
   only, and the rolling-window instance generator that reproduces
   `Dataset.test_data` / `Dataset.validation_dataset` of gift-eval.
3. The case-file format: one HDF5 file per (dataset, frequency, term) config
   holding every forecast instance's context, and one HDF5 prediction file per
   config holding the mean and the nine decile forecasts.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path

import numpy as np

CONTRACT = "timeseries-forecast/v1"
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
CONTEXT_CAP = 16384  # observations of history given per instance; the only deviation from GIFT-Eval (see contract.md)
TEST_SPLIT = 0.1
MAX_WINDOW = 20
TERM_MULTIPLIER = {"short": 1, "medium": 10, "long": 15}
PRED_LENGTH_MAP = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
M4_PRED_LENGTH_MAP = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
# gluonts DEFAULT_SEASONALITIES for the base unit; a multiple n gives base // n when it divides, else 1
BASE_SEASONALITY = {"S": 3600, "T": 1440, "H": 24, "D": 1, "W": 1, "M": 12, "B": 5, "Q": 4, "A": 1}
_UNIT_ALIASES = {"min": "T", "h": "H", "s": "S", "Y": "A", "YE": "A", "QE": "Q", "ME": "M", "MS": "M", "AS": "A", "YS": "A", "QS": "Q"}
_FREQ_RE = re.compile(r"^(\d*)([A-Za-z]+)(?:-([A-Za-z]+))?$")


def parse_freq(freq: str) -> tuple[int, str]:
    """'15T' -> (15, 'T'); 'W-SUN' -> (1, 'W'); 'A-DEC' -> (1, 'A'); 'h' -> (1, 'H')."""
    m = _FREQ_RE.match(freq.strip())
    if not m:
        raise ValueError(f"unrecognised frequency {freq!r}")
    n = int(m.group(1) or 1)
    unit = _UNIT_ALIASES.get(m.group(2), m.group(2))
    if unit not in BASE_SEASONALITY:
        raise ValueError(f"unrecognised frequency unit in {freq!r}")
    return n, unit


def prediction_length(dataset: str, freq: str, term: str) -> int:
    _, unit = parse_freq(freq)
    table = M4_PRED_LENGTH_MAP if "m4" in dataset else PRED_LENGTH_MAP
    return TERM_MULTIPLIER[term] * table[unit]


def seasonality(freq: str) -> int:
    n, unit = parse_freq(freq)
    base = BASE_SEASONALITY[unit]
    q, r = divmod(base, n)
    return q if r == 0 else 1


def n_windows(dataset: str, min_series_length: int, pred_len: int) -> int:
    if "m4" in dataset:
        return 1
    w = math.ceil(TEST_SPLIT * min_series_length / pred_len)
    return min(max(1, w), MAX_WINDOW)


def config_id(dataset: str, freq_unit: str, term: str) -> str:
    """Leaderboard key 'electricity/H/short' -> file stem 'electricity__H__short' (the freq subfolder is dropped: keys are unique)."""
    return f"{dataset.split('/')[0]}__{freq_unit}__{term}"


# gift-eval notebooks: results keys use the frequency sub-folder (or dataset_properties.json for flat datasets) and pretty names
LEADERBOARD_FREQ = {"m4_yearly": "A", "m4_quarterly": "Q", "m4_monthly": "M", "m4_weekly": "W", "m4_daily": "D", "m4_hourly": "H",
                    "hospital": "M", "covid_deaths": "D", "temperature_rain_with_missing": "D", "car_parts_with_missing": "M",
                    "restaurant": "D", "bizitobs_application": "10S", "bizitobs_service": "10S"}
LEADERBOARD_NAME = {"saugeenday": "saugeen", "temperature_rain_with_missing": "temperature_rain",
                    "kdd_cup_2018_with_missing": "kdd_cup_2018", "car_parts_with_missing": "car_parts"}


def leaderboard_key(dataset: str, term: str) -> str:
    """Key used in the GIFT-Eval results CSVs, e.g. 'electricity/15T/short', 'kdd_cup_2018/H/short', 'loop_seattle/5T/short' (lower-cased)."""
    base, _, sub = dataset.partition("/")
    freq = sub if sub else LEADERBOARD_FREQ[base]
    base = base.lower()
    return f"{LEADERBOARD_NAME.get(base, base)}/{freq}/{term}"


VALID_MIN_CONTEXT = 8   # a validation window is staged only if at least this many observations precede it


# ----------------------------------------------------------------------------- Arrow readers


def arrow_path(root: Path, dataset: str) -> Path:
    return Path(root) / dataset / "data-00000-of-00001.arrow"


def read_arrow_dataset(root: Path, dataset: str):
    """Return (item_ids, starts (datetime64[s]), freq, series) for one GIFT-Eval dataset.

    `series` is a list of float32 arrays of shape (D, T); univariate datasets
    give D = 1. Missing values are NaN as stored.
    """
    import pyarrow as pa

    with open(arrow_path(root, dataset), "rb") as f:
        table = pa.ipc.open_stream(f).read_all()
    item_ids = table.column("item_id").to_pylist()
    starts = table.column("start").to_pandas().to_numpy().astype("datetime64[s]")
    freq = table.column("freq")[0].as_py()
    target = table.column("target").combine_chunks()
    series = []
    if pa.types.is_list(target.type.value_type) or pa.types.is_fixed_size_list(target.type.value_type):
        # multivariate: fixed_size_list<list<float>> (or list<list<float>>) -> (D, T) per item
        if pa.types.is_fixed_size_list(target.type):
            k = target.type.list_size
            outer_off = (np.arange(len(target) + 1, dtype=np.int64) + target.offset) * k
        else:
            outer_off = target.offsets.to_numpy()
        inner = target.values
        inner_off = inner.offsets.to_numpy()
        flat = inner.values.to_numpy(zero_copy_only=False).astype(np.float32)
        for i in range(len(target)):
            d0, d1 = outer_off[i], outer_off[i + 1]
            rows = [flat[inner_off[j]:inner_off[j + 1]] for j in range(d0, d1)]
            if len({len(r) for r in rows}) != 1:
                raise ValueError(f"{dataset} item {item_ids[i]}: variates of unequal length")
            series.append(np.stack(rows))
    else:
        off = target.offsets.to_numpy()
        flat = target.values.to_numpy(zero_copy_only=False).astype(np.float32)
        for i in range(len(target)):
            series.append(flat[off[i]:off[i + 1]][None, :])
    return item_ids, starts, freq, series


def read_arrow_index(root: Path, dataset: str):
    """Series metadata without building one array per series: (item_ids, starts, freq, lengths, fingerprints).

    The planning passes (cutoff cohorts, validation boundaries) need only lengths and identities, and a dataset can
    hold 48,000 series, so they avoid the per-series slicing that `read_arrow_dataset` does.
    """
    import pyarrow as pa

    with open(arrow_path(root, dataset), "rb") as f:
        table = pa.ipc.open_stream(f).read_all()
    item_ids = table.column("item_id").to_pylist()
    starts = table.column("start").to_pandas().to_numpy().astype("datetime64[s]")
    freq = table.column("freq")[0].as_py()
    target = table.column("target").combine_chunks()
    if pa.types.is_list(target.type.value_type) or pa.types.is_fixed_size_list(target.type.value_type):
        inner = target.values
        off = inner.offsets.to_numpy()
        flat = inner.values.to_numpy(zero_copy_only=False)
        width = target.type.list_size if pa.types.is_fixed_size_list(target.type) else None
        if width is None:
            outer = target.offsets.to_numpy()
            first = outer[:-1]
        else:
            first = (np.arange(len(target)) + target.offset) * width
    else:
        off = target.offsets.to_numpy()
        flat = target.values.to_numpy(zero_copy_only=False)
        first = np.arange(len(target))
    lengths = (off[first + 1] - off[first]).astype(np.int64)
    fingerprints = []
    for i, start in enumerate(first):
        a, b = off[start], off[start + 1]
        head, tail = flat[a:min(a + 32, b)], flat[max(a, b - 32):b]
        fingerprints.append((int(lengths[i]), hashlib.sha256(head.tobytes() + tail.tobytes()).hexdigest()[:16]))
    return item_ids, starts, freq, lengths, fingerprints


def config_table(root: Path, dataset: str, term: str) -> dict:
    """Protocol quantities of one (dataset, term) config, read from the Arrow file."""
    item_ids, starts, freq, series = read_arrow_dataset(root, dataset)
    n, unit = parse_freq(freq)
    lengths = np.array([s.shape[1] for s in series])
    pred_len = prediction_length(dataset, freq, term)
    windows = n_windows(dataset, int(lengths.min()), pred_len)
    return {
        "dataset": dataset, "freq": freq, "freq_unit": unit, "freq_multiple": n, "term": term,
        "config_id": config_id(dataset, unit, term), "leaderboard_key": leaderboard_key(dataset, term),
        "prediction_length": pred_len, "seasonality": seasonality(freq), "windows": windows,
        "n_series": len(series), "n_variates": int(series[0].shape[0]),
        "min_length": int(lengths.min()), "max_length": int(lengths.max()),
        "n_test_instances": len(series) * windows, "n_valid_instances": len(series),
    }


def test_cutoffs(length: int, pred_len: int, windows: int) -> list[int]:
    """gluonts split(offset=-L*W).generate_instances(L, windows=W, distance=L): cutoff k = T - L*W + k*L."""
    return [length - pred_len * windows + k * pred_len for k in range(windows)]


def valid_cutoff(length: int, pred_len: int, windows: int) -> int:
    """This config's own validation window: the last horizon of gift-eval's `validation_dataset` prefix, which
    `training_dataset` (offset -L*(W+1)) excludes.

    A series usually appears in several configs (terms, and aggregated frequencies of the same meter), whose test
    regions start at different times; the earliest of them is the only safe boundary for the agent-visible validation
    data. `make_cases.py` therefore stages validation against that boundary and uses this function only as the
    per-config upper bound.
    """
    return length - pred_len * (windows + 1)


def steps_between(start: np.datetime64, freq: str, timestamp: str) -> int:
    """How many steps of `freq` separate `start` from `timestamp` (the index of that timestamp in the series grid)."""
    import pandas as pd

    multiple, _ = parse_freq(freq)
    f = _pandas_freq(freq)
    # a Period difference is in base units ('15T' counts minutes), so divide by the multiple to get steps
    delta = (pd.Period(pd.Timestamp(timestamp), freq=f) - pd.Period(pd.Timestamp(start), freq=f)).n
    return int(delta) // multiple


@functools.lru_cache(maxsize=4096)
def _base_period(start: str, freq: str):
    import pandas as pd

    return pd.Period(pd.Timestamp(start), freq=_pandas_freq(freq))


def timestamp_after(start: np.datetime64, freq: str, steps: int) -> str:
    """ISO timestamp of the observation `steps` after `start` under pandas frequency `freq`."""
    return (_base_period(str(start), freq) + int(steps)).to_timestamp().isoformat()


def _pandas_freq(freq: str) -> str:
    n, unit = parse_freq(freq)
    anchor = freq.split("-")[1] if "-" in freq else ""
    new = {"T": "min", "H": "h", "S": "s", "A": "Y", "M": "M", "Q": "Q", "W": "W", "D": "D", "B": "B"}[unit]
    return f"{n if n != 1 else ''}{new}{'-' + anchor if anchor else ''}"


# ----------------------------------------------------------------------------- case files


def write_case_file(path: Path, meta: dict, contexts: list[np.ndarray], item_ids, windows, forecast_starts, history_lengths, cohorts=None):
    """contexts: list of (D, T_i) float32 arrays (already capped); history_lengths: uncapped history length per instance;
    cohorts: verifier invocation group per instance (see make_cases.py), optional."""
    import h5py

    d = int(meta["n_variates"])
    lengths = np.array([c.shape[1] for c in contexts], dtype=np.int64)
    offsets = np.zeros(len(contexts) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths * d)
    values = np.concatenate([c.reshape(-1) for c in contexts]).astype(np.float32) if contexts else np.zeros(0, np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("values", data=values)
        f.create_dataset("offsets", data=offsets)
        f.create_dataset("context_length", data=lengths)
        f.create_dataset("history_length", data=np.asarray(history_lengths, dtype=np.int64))
        f.create_dataset("item_id", data=np.array(item_ids, dtype=h5py.string_dtype()))
        f.create_dataset("window", data=np.asarray(windows, dtype=np.int64))
        f.create_dataset("forecast_start", data=np.array(forecast_starts, dtype=h5py.string_dtype()))
        if cohorts is not None:
            f.create_dataset("cohort", data=np.asarray(cohorts, dtype=np.int64))
        for k, v in meta.items():
            f.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
        f.attrs["contract"] = CONTRACT
        f.attrs["quantile_levels"] = np.array(QUANTILE_LEVELS)


def write_case_subset(src: "CaseFile", indices, dst: Path):
    """Write the instances `indices` of an open CaseFile to a new case file (same attrs, contexts copied)."""
    import h5py

    indices = np.asarray(indices, dtype=np.int64)
    contexts = [src.context(int(i)) for i in indices]
    meta = {k: v for k, v in src.meta.items() if k not in ("contract", "quantile_levels")}
    write_case_file(dst, meta, contexts, [src.item_id[i] for i in indices], src.window[indices],
                    [src.forecast_start[i] for i in indices], src.history_length[indices])
    with h5py.File(dst, "r+") as f:
        f.create_dataset("cohort", data=src.cohort[indices])
        f.create_dataset("source_index", data=indices)


class CaseFile:
    """Read-only view of one staged config: `meta` attrs and `context(i)` -> (D, T_i) float32."""

    def __init__(self, path: Path):
        import h5py

        self.path = Path(path)
        self._f = h5py.File(self.path, "r")
        self.meta = {k: (v.item() if hasattr(v, "item") and np.ndim(v) == 0 else v) for k, v in self._f.attrs.items()}
        self.n_variates = int(self.meta["n_variates"])
        self.prediction_length = int(self.meta["prediction_length"])
        self.offsets = self._f["offsets"][()]
        self.n = len(self.offsets) - 1
        self.item_id = [s.decode() if isinstance(s, bytes) else s for s in self._f["item_id"][()]]
        self.window = self._f["window"][()]
        self.context_length = self._f["context_length"][()]
        self.forecast_start = [s.decode() if isinstance(s, bytes) else s for s in self._f["forecast_start"][()]]
        self.history_length = self._f["history_length"][()]
        self.cohort = self._f["cohort"][()] if "cohort" in self._f else np.zeros(self.n, dtype=np.int64)
        self.source_index = self._f["source_index"][()] if "source_index" in self._f else np.arange(self.n)

    def context(self, i: int) -> np.ndarray:
        v = self._f["values"][self.offsets[i]:self.offsets[i + 1]]
        return v.reshape(self.n_variates, -1)

    def contexts(self):
        values = self._f["values"][()]
        for i in range(self.n):
            yield values[self.offsets[i]:self.offsets[i + 1]].reshape(self.n_variates, -1)

    def close(self):
        self._f.close()


def write_prediction_file(path: Path, mean: np.ndarray, quantiles: np.ndarray, meta: dict | None = None):
    """mean (N, D, H) float32; quantiles (N, D, 9, H) float32 ordered by QUANTILE_LEVELS."""
    import h5py

    mean = np.asarray(mean, dtype=np.float32)
    quantiles = np.asarray(quantiles, dtype=np.float32)
    if mean.ndim != 3 or quantiles.ndim != 4 or quantiles.shape[:2] != mean.shape[:2] or quantiles.shape[3] != mean.shape[2] or quantiles.shape[2] != len(QUANTILE_LEVELS):
        raise ValueError(f"bad prediction shapes mean={mean.shape} quantiles={quantiles.shape}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("mean", data=mean)
        f.create_dataset("quantiles", data=quantiles)
        f.attrs["quantile_levels"] = np.array(QUANTILE_LEVELS)
        f.attrs["contract"] = CONTRACT
        for k, v in (meta or {}).items():
            f.attrs[k] = v


PREDICTION_DATASETS = ("mean", "quantiles")


def open_package_h5(path, names=PREDICTION_DATASETS):
    """Open an HDF5 file a package wrote, refusing anything but a regular file whose `names` are plain datasets.

    The verifier reads package output with its own privileges, outside every isolation mode, so it must not follow
    anything the package put in the file: an external or soft link, a virtual dataset or external raw storage can
    point at the held-out labels and would be resolved here. A FIFO or device in place of the file would block."""
    import h5py

    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode):
        raise ValueError("not a regular file (symlinks, FIFOs and devices are refused)")
    f = h5py.File(path, "r")
    try:
        for name in names:
            link = f.get(name, getlink=True)
            if link is None:
                raise KeyError(f"no {name!r} dataset")
            if not isinstance(link, h5py.HardLink):
                raise ValueError(f"{name!r} is a {type(link).__name__}; only plain datasets are read")
            d = f[name]
            if not isinstance(d, h5py.Dataset) or d.is_virtual or d.external:
                raise ValueError(f"{name!r} is not a plain dataset (virtual or external storage is refused)")
    except Exception:
        f.close()
        raise
    return f


def prediction_shapes(path: Path):
    """(mean shape, quantiles shape, attrs) without reading the arrays, so a mis-declared file cannot exhaust memory."""
    with open_package_h5(path) as f:
        attrs = {k: (v.item() if hasattr(v, "item") and np.ndim(v) == 0 else v) for k, v in f.attrs.items()}
        return tuple(f["mean"].shape), tuple(f["quantiles"].shape), attrs


def read_prediction_file(path: Path, expect: tuple | None = None):
    """Read mean and quantiles. `expect` is the required mean shape (N, D, H); the file is rejected before it is read."""
    with open_package_h5(path) as f:
        got_m, got_q = tuple(f["mean"].shape), tuple(f["quantiles"].shape)
        want_q = None if expect is None else (expect[0], expect[1], len(QUANTILE_LEVELS), expect[2])
        if expect is not None and (got_m != tuple(expect) or got_q != want_q):
            raise ValueError(f"bad shapes mean={got_m} quantiles={got_q} expected {tuple(expect)} / {want_q}")
        return f["mean"][()], f["quantiles"][()]


def write_truth_file(path: Path, labels: np.ndarray, seasonal_error: np.ndarray, meta: dict):
    """labels (N, D, H) float32 with NaN where the benchmark has missing targets; seasonal_error (N, D) float64."""
    import h5py

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("labels", data=np.asarray(labels, dtype=np.float32))
        f.create_dataset("seasonal_error", data=np.asarray(seasonal_error, dtype=np.float64))
        for k, v in meta.items():
            f.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v


def read_truth_file(path: Path):
    import h5py

    with h5py.File(path, "r") as f:
        meta = {k: (v.item() if hasattr(v, "item") and np.ndim(v) == 0 else v) for k, v in f.attrs.items()}
        return f["labels"][()], f["seasonal_error"][()], meta


def seasonal_error(history: np.ndarray, m: int) -> np.ndarray:
    """gluonts 0.15.1 seasonal error on the full (uncapped) history, per variate.

    Reproduces `gluonts.model.evaluation._get_data_batch` with
    `mask_invalid_label=True`: the history is a masked array, so the mean of
    |x[t] - x[t-m]| runs over the pairs where both observations are finite.
    If m > T the lag is 1 (gluonts.ev.ts_stats.seasonal_error). A variate with
    no valid pair gets NaN.
    """
    t = history.shape[-1]
    lag = m if m <= t else 1
    if t <= lag:
        return np.full(history.shape[0], np.nan)
    x = history.astype(np.float64)
    diff = np.abs(x[:, lag:] - x[:, :-lag])
    valid = np.isfinite(diff)
    count = valid.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(valid, diff, 0.0).sum(axis=-1) / count
    out[count == 0] = np.nan
    return out
