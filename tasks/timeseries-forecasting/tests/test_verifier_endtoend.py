"""The verifier end to end on a synthetic GIFT-Eval directory: staging, cohort calls, merging, the determinism
rerun and scoring, with packages that try what the second integration review found.

The fixture is one hourly directory with three series of different lengths, so their test cutoffs fall on five
distinct timestamps: absolute-time cohorts give five calls (per-series ranking gave two). Tests that need uid mode
(the integrator's: verifier as root, package as uid 10001) are skipped unless run as root.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "environment" / "evaluation"
sys.path.insert(0, str(EVAL))
import ts_io  # noqa: E402

CID = "syn_hourly__H__short"
LENGTHS = (200, 212, 224)
ROOT_ONLY = pytest.mark.skipif(os.geteuid() != 0, reason="uid mode needs root (the integrator's verifier container)")


def _traversable(path: Path):
    for p in [Path(path), *Path(path).parents]:
        if str(p) in ("/", "/tmp"):
            break
        os.chmod(p, os.stat(p).st_mode | 0o011)


def make_gifteval(root: Path):
    """data/gifteval/syn/H as GIFT-Eval stores it (an Arrow IPC stream), a one-config manifest and a baseline."""
    import pyarrow as pa
    d = root / "data" / "gifteval" / "syn" / "H"; d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    series = [(10 * np.sin(np.arange(n) * 2 * np.pi / 24) + 50 + rng.normal(size=n)).astype(np.float32) for n in LENGTHS]
    table = pa.table({"item_id": ["a", "b", "c"],
                      "start": pa.array([np.datetime64("2020-01-01T00:00:00", "s")] * 3, type=pa.timestamp("s")),
                      "freq": ["H"] * 3, "target": pa.array([s.tolist() for s in series], type=pa.list_(pa.float32()))})
    with pa.OSFile(str(d / "data-00000-of-00001.arrow"), "wb") as f, pa.ipc.new_stream(f, table.schema) as w:
        w.write_table(table)
    cfg = {"dataset": "syn/H", "freq": "H", "freq_unit": "H", "freq_multiple": 1, "term": "short", "config_id": CID,
           "leaderboard_key": "syn/H/short", "prediction_length": 24, "seasonality": 24, "windows": 2,
           "n_series": 3, "n_variates": 1, "min_length": min(LENGTHS), "max_length": max(LENGTHS), "domain": "Energy",
           "gifteval_subdir": "gifteval"}
    man = root / "split.json"
    man.write_text(json.dumps({"contract": "timeseries-forecast/v1", "context_cap": 16384, "configs": [cfg]}))
    base = root / "baseline.json"
    base.write_text(json.dumps({"configs": {CID: {"mase": 1.0, "wql": 0.1}}}))
    (root / "data").chmod(0o700)   # root-only, as /private is in the verifier image
    return root / "data", man, base


PRELUDE = r'''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
python3 - "$1" "$2" <<'PY'
import json, os, sys
from pathlib import Path
import numpy as np
import ts_io
inp, out = Path(sys.argv[1]), Path(sys.argv[2]); (out / "cases").mkdir(parents=True, exist_ok=True)
index = json.loads((inp / "index.json").read_text())
def naive(case):
    last = np.array([[np.nan_to_num(case.context(i)[v, -1]) for v in range(case.n_variates)] for i in range(case.n)], np.float32)
    mean = np.repeat(last[:, :, None], case.prediction_length, axis=2)
    return mean, np.repeat(mean[:, :, None, :], 9, axis=2)
'''
NAIVE = PRELUDE + r'''for cfg in index["configs"]:
    case = ts_io.CaseFile(inp / "cases" / f"{cfg['config_id']}.h5")
    ts_io.write_prediction_file(out / "cases" / f"{cfg['config_id']}.h5", *naive(case)); case.close()
PY
'''


def make_package(root: Path, infer_sh: str, name="submission", manifest=None):
    pkg = root / name; pkg.mkdir()
    shutil.copy(EVAL / "ts_io.py", pkg / "ts_io.py")
    (pkg / "infer.sh").write_text(infer_sh); os.chmod(pkg / "infer.sh", 0o755)
    (pkg / "model_manifest.json").write_text(manifest if manifest is not None else
                                             json.dumps({"contract": "timeseries-forecast/v1", "name": "test", "pretrained_assets": []}))
    return pkg


def run_verifier(root: Path, pkg: Path, data: Path, man: Path, base: Path, env_extra=None, timeout=300):
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    env.update(env_extra or {})
    sys.path.insert(0, str(ROOT / "tests"))
    import isolation
    if os.geteuid() == 0:
        _traversable(root); _traversable(pkg)
    elif isolation.select_mode() == "subprocess" or env.get("TIMESERIES_NO_BWRAP"):
        env.setdefault("TIMESERIES_ALLOW_NO_ISOLATION", "1")
    return subprocess.run([sys.executable, str(ROOT / "tests" / "main_verifier.py"), "--submission", str(pkg), "--data", str(data),
                           "--manifest", str(man), "--baseline", str(base), "--output", str(root / "out"), "--work", str(root / "work"),
                           "--timeout", "120"], capture_output=True, env=env, timeout=timeout)


def _status(root):
    return json.loads((root / "out" / "status.json").read_text())


def test_cohorts_are_absolute_cutoffs_and_a_plain_package_scores(tmp_path):
    data, man, base = make_gifteval(tmp_path); pkg = make_package(tmp_path, NAIVE)
    r = run_verifier(tmp_path, pkg, data, man, base)
    assert r.returncode == 0, r.stderr[-1500:]
    st = _status(tmp_path)
    assert st["inference"]["n_cohorts"] == 5 and st["inference"]["invocations"] == 5, st["inference"]
    assert st["n_failed"] == 0 and st["determinism"]["deterministic"] is True, st
    case = ts_io.CaseFile(tmp_path / "work" / "input" / "cases" / f"{CID}.h5")
    starts = {}
    for i in range(case.n):
        starts.setdefault(int(case.cohort[i]), set()).add(case.forecast_start[i])
    case.close()
    assert all(len(s) == 1 for s in starts.values()), starts     # one cutoff per call, across items
    assert (os.stat(tmp_path / "out" / "infer.log").st_mode & 0o077) == 0


VIRTUAL = PRELUDE.replace("import ts_io\n", "import ts_io, h5py\n") + r'''truth = Path("%s") / "truth" / "cases"
for cfg in index["configs"]:
    cid = cfg["config_id"]; case = ts_io.CaseFile(inp / "cases" / f"{cid}.h5")
    n, D, H = case.n, case.n_variates, case.prediction_length
    src = h5py.VirtualSource(str(truth / f"{cid}.h5"), "labels", shape=(6, D, H))   # 3 series x 2 windows
    mean, quant = h5py.VirtualLayout((n, D, H), "f4"), h5py.VirtualLayout((n, D, 9, H), "f4")
    for j, s in enumerate(case.source_index):
        mean[j] = src[int(s)]
        for q in range(9):
            quant[j, :, q, :] = src[int(s)]
    with h5py.File(out / "cases" / f"{cid}.h5", "w", libver="latest") as f:
        f.create_virtual_dataset("mean", mean); f.create_virtual_dataset("quantiles", quant)
    case.close()
PY
'''


def test_a_virtual_prediction_mapped_onto_the_labels_scores_nothing(tmp_path):
    """The review's reproduction (reward 0.999 before the guard): VDS rows mapped by source_index onto the truth."""
    data, man, base = make_gifteval(tmp_path)
    pkg = make_package(tmp_path, VIRTUAL % (tmp_path / "work").resolve())
    r = run_verifier(tmp_path, pkg, data, man, base)
    assert r.returncode == 0, r.stderr[-1500:]
    st = _status(tmp_path)
    assert st["n_failed"] == 1 and st["reward"] == 0.0, {k: st.get(k) for k in ("n_failed", "reward", "failure_reasons")}


HOME_PROBE = PRELUDE + r'''report = Path("%s")
seen = {"home": (Path(os.environ["HOME"]) / "cache").exists(), "tmp": Path("/tmp/ts-probe-%s").exists()}
with open(report, "a") as f: f.write(json.dumps(seen) + "\n")
(Path(os.environ["HOME"]) / "cache").write_text("x"); Path("/tmp/ts-probe-%s").write_text("x")
for cfg in index["configs"]:
    case = ts_io.CaseFile(inp / "cases" / f"{cfg['config_id']}.h5")
    ts_io.write_prediction_file(out / "cases" / f"{cfg['config_id']}.h5", *naive(case)); case.close()
PY
'''


def test_no_cohort_call_sees_state_an_earlier_call_left_in_home(tmp_path):
    """Forced out of Bubblewrap (plain subprocess here, uid mode as root): HOME is fresh for every call, and as root
    the /tmp file is swept too."""
    data, man, base = make_gifteval(tmp_path)
    report = tmp_path / "report.jsonl"; report.write_text(""); os.chmod(report, 0o666)
    tag = str(os.getpid())
    pkg = make_package(tmp_path, HOME_PROBE % (report, tag, tag))
    try:
        r = run_verifier(tmp_path, pkg, data, man, base, env_extra={"TIMESERIES_NO_BWRAP": "1"})
        assert r.returncode == 0, r.stderr[-1500:]
        seen = [json.loads(l) for l in report.read_text().splitlines()]
        assert len(seen) == 6, seen                      # 5 cohort calls and the determinism rerun
        assert not any(s["home"] for s in seen), seen
        if os.geteuid() == 0:
            assert not any(s["tmp"] for s in seen), seen
    finally:
        Path(f"/tmp/ts-probe-{tag}").unlink(missing_ok=True)


NOISY = PRELUDE + r'''sys.stdout.buffer.write(b"\xff\xfe not utf-8 \x80\n" * 1000); sys.stdout.flush()
for cfg in index["configs"]:
    case = ts_io.CaseFile(inp / "cases" / f"{cfg['config_id']}.h5")
    ts_io.write_prediction_file(out / "cases" / f"{cfg['config_id']}.h5", *naive(case)); case.close()
PY
'''


def test_invalid_utf8_on_stdout_is_logged_not_fatal(tmp_path):
    data, man, base = make_gifteval(tmp_path); pkg = make_package(tmp_path, NOISY)
    r = run_verifier(tmp_path, pkg, data, man, base)
    assert r.returncode == 0, r.stderr[-1500:]
    assert _status(tmp_path)["n_failed"] == 0
    assert b"\xff\xfe not utf-8" in (tmp_path / "out" / "infer.log").read_bytes()


FIFO = NAIVE.replace("PY\n", "PY\nrm -f \"$2/cases/%s.h5\"; mkfifo \"$2/cases/%s.h5\"\n" % (CID, CID), 1)


def test_a_fifo_in_place_of_a_prediction_is_a_gap_not_a_hang(tmp_path):
    data, man, base = make_gifteval(tmp_path); pkg = make_package(tmp_path, FIFO)
    r = run_verifier(tmp_path, pkg, data, man, base, timeout=240)
    assert r.returncode == 0, r.stderr[-1500:]
    assert _status(tmp_path)["n_failed"] == 1


@pytest.mark.parametrize("manifest", ["[]", "null", '"timeseries-forecast/v1"'])
def test_a_manifest_that_is_not_an_object_scores_zero(tmp_path, manifest):
    data, man, base = make_gifteval(tmp_path); pkg = make_package(tmp_path, NAIVE, manifest=manifest)
    r = run_verifier(tmp_path, pkg, data, man, base)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "JSON object" in _status(tmp_path)["rule_violation"]


@ROOT_ONLY
def test_uid_mode_refuses_data_the_runner_can_reach(tmp_path):
    data, man, base = make_gifteval(tmp_path); pkg = make_package(tmp_path, NAIVE)
    os.chmod(data, 0o711)                       # traversable but not listable still exposes the public layout
    r = run_verifier(tmp_path, pkg, data, man, base, env_extra={"TIMESERIES_NO_BWRAP": "1"})
    assert r.returncode == 2 and not (tmp_path / "out" / "reward.json").exists()
    assert "runner uid, which can read" in _status(tmp_path)["error"]
