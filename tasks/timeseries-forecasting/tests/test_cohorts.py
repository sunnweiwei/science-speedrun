"""Cohort rules of make_cases (the second integration review round): related directories share a timeline, and
placeholder-time (M4) series whose values hold another series' test label, or copies of one series, never share an
infer.sh call; such a series gets no validation window, and a copy's validation window stops before every copy's
test region."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environment" / "evaluation"))
import make_cases  # noqa: E402
import ts_io  # noqa: E402


def test_related_directories_share_one_timeline():
    times = {"bizitobs_application": {"app": {"2020-01-01T00:00:00", "2020-01-01T00:00:20"}},
             "bizitobs_service": {"svc": {"2020-01-01T00:00:10", "2020-01-01T00:00:30"}},
             "electricity": {"m": {"2020-01-01T00:00:10"}}}
    table = make_cases.rank_cohorts(times)
    assert table["bizitobs_application"]["app"] == {"2020-01-01T00:00:00": 0, "2020-01-01T00:00:20": 2}
    assert table["bizitobs_service"]["svc"] == {"2020-01-01T00:00:10": 1, "2020-01-01T00:00:30": 3}
    assert table["electricity"]["m"] == {"2020-01-01T00:00:10": 0}      # its own timeline


def _write_arrow(path: Path, freq: str, items: dict):
    import pyarrow as pa
    path.mkdir(parents=True)
    table = pa.table({"item_id": list(items), "start": pa.array([np.datetime64("1750-01-01T00:00:00", "s")] * len(items), type=pa.timestamp("s")),
                      "freq": [freq] * len(items), "target": pa.array([v.tolist() for v in items.values()], type=pa.list_(pa.float32()))})
    with pa.OSFile(str(path / "data-00000-of-00001.arrow"), "wb") as f, pa.ipc.new_stream(f, table.schema) as w:
        w.write_table(table)


def _cfg(dataset, freq, unit, L, n):
    return {"dataset": dataset, "freq": freq, "freq_unit": unit, "freq_multiple": 1, "term": "short",
            "config_id": f"{dataset}__{unit}__short", "leaderboard_key": f"{dataset}/{unit}/short", "prediction_length": L,
            "seasonality": 1, "windows": 1, "n_series": n, "n_variates": 1, "min_length": 60, "max_length": 200,
            "domain": "Econ/Fin", "gifteval_subdir": "gifteval"}


def test_m4_content_copies_are_split_across_calls_and_kept_out_of_validation(tmp_path):
    rng = np.random.default_rng(0)
    a = np.round(rng.normal(100, 10, 80), 1).astype(np.float32)          # its label: the last 6 values
    b = np.concatenate([rng.normal(50, 5, 40), a[-6:], rng.normal(50, 5, 40)]).astype(np.float32)   # holds a's label
    c = np.round(rng.normal(0, 1, 90), 1).astype(np.float32)             # unrelated
    _write_arrow(tmp_path / "gifteval" / "m4_syn", "D", {"a": a, "b": b, "c": c})
    _write_arrow(tmp_path / "gifteval" / "m4_syn2", "W-SUN", {"a_copy": a.copy()})   # the same series elsewhere
    cfgs = [_cfg("m4_syn", "D", "D", 6, 3), _cfg("m4_syn2", "W-SUN", "W", 4, 1)]
    edges, containers = make_cases.content_conflicts(tmp_path, cfgs)
    A, B, C, A2 = ("m4_syn", "a"), ("m4_syn", "b"), ("m4_syn", "c"), ("m4_syn2", "a_copy")
    assert frozenset((A, B)) in edges and frozenset((A, A2)) in edges
    assert B in containers and C not in containers and A not in containers
    table = make_cases.cohort_table(tmp_path, cfgs, "test")
    call = {k: min(table[k[0]][k[1]].values()) for k in (A, B, C, A2)}
    assert call[A] != call[B] and call[A] != call[A2], call
    man = tmp_path / "split.json"; man.write_text(json.dumps({"configs": cfgs}))
    subprocess.run([sys.executable, str(ROOT / "environment" / "evaluation" / "make_cases.py"), "--data", str(tmp_path), "--manifest",
                    str(man), "--split", "valid", "--input", str(tmp_path / "v" / "input"), "--truth", str(tmp_path / "v" / "truth")],
                   check=True, capture_output=True)
    case = ts_io.CaseFile(tmp_path / "v" / "input" / "cases" / "m4_syn__D__short.h5")
    assert "b" not in case.item_id and {"a", "c"} <= set(case.item_id)
    case.close()
    # the weekly copy's validation window ends before the daily copy's test region (index 80 - 6 = 74)
    copy = ts_io.CaseFile(tmp_path / "v" / "input" / "cases" / "m4_syn2__W__short.h5")
    assert int(copy.history_length[0]) + 4 <= 74, (copy.history_length, "validation label must end by index 74")
    copy.close()
