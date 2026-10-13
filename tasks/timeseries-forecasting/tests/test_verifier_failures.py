"""The verifier must score model failures (never crash) and distinguish them from setup errors."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environment" / "evaluation"))
sys.path.insert(0, str(ROOT / "environment" / "evaluation"))
import ts_io  # noqa: E402
from score import REL_MAX, score  # noqa: E402
from make_example import make  # noqa: E402


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    out = tmp_path_factory.mktemp("example")
    make(out, seed=1)
    return out


BASE = {c: {"mase": 1.0, "wql": 0.1} for c in ("example_hourly__H__short", "example_daily__D__short", "example_multi__T__medium")}


def test_missing_config_scores_max(example, tmp_path):
    out = tmp_path / "partial"
    (out / "cases").mkdir(parents=True)
    src = example / "output" / "cases" / "example_daily__D__short.h5"
    (out / "cases" / src.name).write_bytes(src.read_bytes())
    rep = score(example / "truth", out, BASE)
    assert rep["n_failed"] == 2 and all(f["reason"] == "missing prediction file" for f in rep["failures"])
    failed = [c for c in rep["configs"] if c["status"] == "failed"]
    assert all(c["relative"] == {"mase": REL_MAX, "wql": REL_MAX} for c in failed)
    assert rep["reward"] == 0.0


def test_bad_shape_and_nan_are_model_failures(example, tmp_path):
    out = tmp_path / "bad"
    for cid, kind in (("example_hourly__H__short", "shape"), ("example_daily__D__short", "nan"), ("example_multi__T__medium", "garbage")):
        p = out / "cases" / f"{cid}.h5"
        p.parent.mkdir(parents=True, exist_ok=True)
        labels, _, _ = ts_io.read_truth_file(example / "truth" / "cases" / f"{cid}.h5")
        n, d, h = labels.shape
        if kind == "shape":
            ts_io.write_prediction_file(p, np.zeros((n, d, h - 1)), np.zeros((n, d, 9, h - 1)))
        elif kind == "nan":
            q = np.zeros((n, d, 9, h)); q[0, 0, 0, 0] = np.nan
            ts_io.write_prediction_file(p, np.zeros((n, d, h)), q)
        else:
            p.write_bytes(b"not an hdf5 file")
    rep = score(example / "truth", out, BASE)
    reasons = {f["config_id"]: f["reason"] for f in rep["failures"]}
    assert reasons["example_hourly__H__short"].startswith("bad shapes")
    assert reasons["example_daily__D__short"] == "non-finite values in prediction"
    assert reasons["example_multi__T__medium"].startswith("unreadable prediction file")


def test_missing_baseline_is_a_setup_error(example):
    with pytest.raises(RuntimeError):
        score(example / "truth", example / "output", {"example_daily__D__short": {"mase": 1, "wql": 1}})


def _open_to_runner(path):
    """When the suite runs as root (the verifier image), the runner uid must traverse pytest's 0700 tmp tree."""
    import os
    if os.geteuid() != 0:
        return
    p = Path(path).resolve()
    while str(p) not in ("/", "/tmp"):
        try:
            os.chmod(p, os.stat(p).st_mode | 0o011)
        except OSError:
            pass
        p = p.parent


def _run_verifier(tmp_path, sub):
    _open_to_runner(tmp_path)
    sys.path.insert(0, str(ROOT / "tests"))
    import isolation  # noqa: E402
    env = dict(os.environ)
    if isolation.select_mode() == "subprocess":
        # neither a namespace nor root here (e.g. a non-root container): these tests are about the package rules,
        # which are decided before inference; the isolation boundary has its own tests in test_isolation.py
        env.setdefault("TIMESERIES_ALLOW_NO_ISOLATION", "1")
    return subprocess.run([sys.executable, str(ROOT / "tests" / "main_verifier.py"), "--submission", str(sub), "--data", str(tmp_path),
                           "--manifest", str(ROOT / "environment" / "data" / "split_smoke.json"), "--output", str(tmp_path / "out"),
                           "--work", str(tmp_path / "work")], capture_output=True, text=True, env=env)


@pytest.mark.parametrize("broken", ["wrong contract", "no infer.sh", "unreadable manifest", "no manifest", "pretrained assets"])
def test_package_defects_score_zero_rather_than_erroring(tmp_path, broken):
    """A malformed package is the package's fault: reward 0 and exit 0, never an infrastructure error."""
    sub = tmp_path / "sub"; sub.mkdir()
    if broken != "no infer.sh":
        (sub / "infer.sh").write_text("#!/bin/bash\nexit 0\n")
    if broken == "unreadable manifest":
        (sub / "model_manifest.json").write_text("{not json")
    elif broken != "no manifest":
        contract = "something-else/v9" if broken == "wrong contract" else "timeseries-forecast/v1"
        manifest = {"contract": contract}
        if broken == "pretrained assets":
            manifest["pretrained_assets"] = ["amazon/chronos-2"]
        (sub / "model_manifest.json").write_text(json.dumps(manifest))
    proc = _run_verifier(tmp_path, sub)
    assert proc.returncode == 0, proc.stderr[-400:]
    assert json.loads((tmp_path / "out" / "reward.json").read_text())["reward"] == 0.0
    assert (tmp_path / "out" / "reward.txt").read_text().strip() == "0.0"
    assert json.loads((tmp_path / "out" / "status.json").read_text())["rule_violation"]


def test_mis_shaped_prediction_is_a_model_failure_not_a_crash(example, tmp_path):
    """A prediction file whose declared shape is wrong must be scored, not raised (and never materialized)."""
    out = tmp_path / "bad"
    cid = "example_daily__D__short"
    labels, _, _ = ts_io.read_truth_file(example / "truth" / "cases" / f"{cid}.h5")
    n, d, h = labels.shape
    (out / "cases").mkdir(parents=True)
    import h5py
    with h5py.File(out / "cases" / f"{cid}.h5", "w") as f:   # 1-D mean with the right N: passes a shape[0]-only check
        f.create_dataset("mean", data=np.zeros((n,), np.float32))
        f.create_dataset("quantiles", data=np.zeros((n, d, 9, h), np.float32))
    for other in BASE:
        if other != cid:
            src = example / "output" / "cases" / f"{other}.h5"
            (out / "cases" / src.name).write_bytes(src.read_bytes())
    rep = score(example / "truth", out, BASE)
    reasons = {f["config_id"]: f["reason"] for f in rep["failures"]}
    assert reasons[cid].startswith("bad shapes") and rep["configs"][1]["relative"] == {"mase": REL_MAX, "wql": REL_MAX}


def test_a_huge_declared_shape_is_rejected_before_it_is_read(tmp_path):
    """ts_io must refuse a mis-declared prediction dataset without allocating it."""
    import h5py
    p = tmp_path / "huge.h5"
    with h5py.File(p, "w") as f:   # 400M float32 declared, a few KB on disk
        f.create_dataset("mean", shape=(400_000_000,), dtype="float32", chunks=(1024,), compression="gzip")
        f.create_dataset("quantiles", shape=(4, 1, 9, 3), dtype="float32")
    assert ts_io.prediction_shapes(p)[0] == (400_000_000,)
    with pytest.raises(ValueError, match="bad shapes"):
        ts_io.read_prediction_file(p, expect=(4, 1, 3))


def test_partial_coverage_is_scored_and_penalized(example, tmp_path):
    """A run cut short mid-way scores the instances it covered, blended toward the failure value, not all-or-nothing."""
    full = tmp_path / "full"
    coverage = {}
    for cid in BASE:
        labels, _, _ = ts_io.read_truth_file(example / "truth" / "cases" / f"{cid}.h5")
        clean = np.nan_to_num(labels); quant = np.repeat(clean[:, :, None], 9, axis=2)
        ts_io.write_prediction_file(full / "cases" / f"{cid}.h5", clean, quant)
        covered = np.ones(len(clean), bool); covered[len(clean) // 2:] = False
        coverage[cid] = covered.tolist()
    good = score(example / "truth", full, BASE)
    half = score(example / "truth", full, BASE, coverage=coverage)
    assert good["n_partial"] == 0 and half["n_partial"] == len(BASE) and half["n_failed"] == 0
    assert 0.0 < half["reward"] < good["reward"]
    assert half["n_instances_scored"] < good["n_instances_scored"]


def test_coverage_is_not_read_from_the_package_output(example, tmp_path):
    """A package cannot exclude its worst instances by writing a coverage mask into its own prediction file."""
    import h5py

    out = tmp_path / "selfreport"
    for cid in BASE:
        labels, _, _ = ts_io.read_truth_file(example / "truth" / "cases" / f"{cid}.h5")
        bad = np.zeros_like(np.nan_to_num(labels))
        ts_io.write_prediction_file(out / "cases" / f"{cid}.h5", bad, np.repeat(bad[:, :, None], 9, axis=2))
        keep = np.zeros(len(bad), bool); keep[0] = True
        with h5py.File(out / "cases" / f"{cid}.h5", "r+") as f:
            f.create_dataset("covered", data=keep)
    rep = score(example / "truth", out, BASE)
    assert rep["n_partial"] == 0
    assert all(c["coverage"]["fraction"] == 1.0 for c in rep["configs"])


@pytest.mark.parametrize("kind", ["escaping symlink", "broken symlink"])
def test_symlink_out_of_the_package_is_a_rule_violation(tmp_path, kind):
    """A package must not reach outside itself: copying it must never dereference a link to the labels or the data."""
    secret = tmp_path / "truth-like"; secret.mkdir(); (secret / "labels.h5").write_text("the answers")
    sub = tmp_path / "sub"; sub.mkdir()
    (sub / "infer.sh").write_text("#!/bin/bash\nexit 0\n")
    (sub / "model_manifest.json").write_text(json.dumps({"contract": "timeseries-forecast/v1", "pretrained_assets": []}))
    (sub / "assets").symlink_to(secret if kind == "escaping symlink" else tmp_path / "nowhere")
    proc = _run_verifier(tmp_path, sub)
    assert proc.returncode == 0, proc.stderr[-400:]
    status = json.loads((tmp_path / "out" / "status.json").read_text())
    assert "symlink" in status["rule_violation"]
    assert json.loads((tmp_path / "out" / "reward.json").read_text())["reward"] == 0.0
    assert not (tmp_path / "work" / "sealed" / "assets").exists()


def test_symlink_inside_the_package_is_dereferenced(tmp_path):
    """The legitimate case the seal must keep working: a package that symlinks its own shared module."""
    sys.path.insert(0, str(ROOT / "tests"))
    from main_verifier import seal_package

    sub = tmp_path / "sub"; (sub / "lib").mkdir(parents=True)
    (sub / "lib" / "ts_io.py").write_text("# shared reader\n")
    (sub / "ts_io.py").symlink_to(sub / "lib" / "ts_io.py")
    sealed = tmp_path / "sealed"
    assert seal_package(sub, sealed) is None
    assert (sealed / "ts_io.py").is_file() and not (sealed / "ts_io.py").is_symlink()
