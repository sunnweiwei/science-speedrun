"""Data-free checks of the grader, the contract and the split manifest.  Run: python3 -m pytest tests/ -q"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
sys.path.insert(0, str(TASK / "environment" / "evaluation"))
import score as S  # noqa: E402


def _case(dirpath, cid, arr, truth=False):
    (dirpath / "cases").mkdir(parents=True, exist_ok=True)
    with h5py.File(dirpath / "cases" / f"{cid}.h5", "w") as f:
        f.create_dataset("target" if truth else "prediction", data=arr)
        if truth:
            f.attrs["fields"] = ["a", "b"]; f.attrs["field_scheme"] = "ab"; f.attrs["dataset"] = "syn/x"


def _setup(tmp):
    rng = np.random.default_rng(0)
    inp, truth, out = tmp / "in", tmp / "truth", tmp / "out"
    inp.mkdir(); truth.mkdir(); out.mkdir()
    cases, targets = [], {}
    for i in range(3):
        t = rng.normal(size=(5, 2, 8, 8)).astype(np.float32) * (i + 1)
        targets[f"c{i}"] = t
        _case(truth, f"c{i}", t, truth=True)
        cases.append({"case_id": f"c{i}", "dataset": "syn/x", "n_steps_out": 5})
    (inp / "index.json").write_text(json.dumps({"contract": S.CONTRACT, "cases": cases}))
    return inp, truth, out, targets


def test_spatial_mean_predictor_scores_one():
    with tempfile.TemporaryDirectory() as d:
        inp, truth, out, targets = _setup(Path(d))
        for cid, t in targets.items():
            _case(out, cid, np.broadcast_to(t.mean(axis=(2, 3), keepdims=True), t.shape).astype(np.float32))
        r = S.score(inp, truth, out)
        assert r["n_failed"] == 0
        assert abs(r["mean_vrmse"] - 1.0) < 1e-2, r["mean_vrmse"]  # sqrt(var/(var+eps)) with ddof=1 vs population mean


def test_exact_prediction_scores_zero_and_full_reward():
    with tempfile.TemporaryDirectory() as d:
        inp, truth, out, targets = _setup(Path(d))
        for cid, t in targets.items():
            _case(out, cid, t)
        r = S.score(inp, truth, out)
        assert r["mean_vrmse"] < 1e-3 and r["reward"] == 1.0


def test_failures_stay_in_denominator_with_reasons():
    with tempfile.TemporaryDirectory() as d:
        inp, truth, out, targets = _setup(Path(d))
        _case(out, "c0", targets["c0"])                                    # good
        _case(out, "c1", np.zeros((5, 2, 8, 7), np.float32))                # wrong shape
        bad = targets["c2"].copy(); bad[0, 0, 0, 0] = np.nan; _case(out, "c2", bad)  # non-finite
        r = S.score(inp, truth, out)
        assert r["n_cases"] == 3 and r["n_failed"] == 2
        reasons = {f["case_id"]: f["reason"] for f in r["failures"]}
        assert reasons["c1"].startswith("wrong shape") and reasons["c2"] == "non-finite values"
        assert abs(r["datasets"]["syn/x"]["vrmse"] - (0 + 10 + 10) / 3) < 1e-6
        (out / "cases" / "c1.h5").unlink()
        r = S.score(inp, truth, out)
        assert {f["case_id"]: f["reason"] for f in r["failures"]}["c1"] == "missing"


def test_reward_anchors():
    assert S.reward_from_vrmse(0.01) == 1.0 and S.reward_from_vrmse(10) == 0.0 and S.reward_from_vrmse(50) == 0.0
    assert abs(S.reward_from_vrmse(1.0) - 1 / 3) < 1e-9 and abs(S.reward_from_vrmse(0.1) - 2 / 3) < 1e-9


def test_vrmse_matches_reference_definition():
    rng = np.random.default_rng(1)
    t = rng.normal(size=(4, 3, 6, 5)); p = t + 0.1 * rng.normal(size=t.shape)
    v = S.vrmse(p, t)
    ref = np.sqrt(((p - t) ** 2).mean(axis=(2, 3)) / (t.var(axis=(2, 3), ddof=1) + 1e-7))
    assert np.allclose(v, ref)


def _manifest(name):
    return json.loads((TASK / "environment" / "data" / name).read_text())


def test_manifests_are_consistent_with_lock():
    lock = json.loads((TASK / "environment" / "data" / "sources.lock.json").read_text())
    files = {(s["id"], f["path"]): f for s in lock["sources"] for f in s["files"] if f["kind"] == "hdf5"}
    for name in ("split.json", "split_smoke.json"):
        m = _manifest(name)
        assert m["contract"] == S.CONTRACT and m["context_frames"] == 4
        seen = set()
        for c in m["cases"]:
            f = files[(c["source"], c["file"])]
            assert 0 <= c["trajectory"] < f["n_trajectories_inferred"], c
            assert c["start"] + m["context_frames"] + c["n_steps_out"] <= f["trajectory_shape_tchw"][0], c
            key = (c["source"], c["file"], c["trajectory"])
            assert c["split"] == "test_long" or key not in seen, ("duplicate held-out case", c)
            seen.add(key)
        held = {(c["source"], c["file"], c["trajectory"]) for c in m["cases"]}
        for split in ("train", "valid"):
            for p in m["pools"][split]:
                f = files[(p["source"], p["file"])]
                a, b = p["trajectories"]
                assert 0 <= a < b <= f["n_trajectories_inferred"], p
                for t in range(a, b):
                    assert (p["source"], p["file"], t) not in held, ("held-out trajectory leaks into", split, p, t)
        train = {(p["source"], p["file"], t) for p in m["pools"]["train"] for t in range(*p["trajectories"])}
        valid = {(p["source"], p["file"], t) for p in m["pools"]["valid"] for t in range(*p["trajectories"])}
        assert not (train & valid)


def test_starter_infer_sh_is_a_bash_script():
    assert subprocess.run(["bash", "-n", str(TASK / "environment" / "baseline" / "starter" / "infer.sh")]).returncode == 0
    assert subprocess.run(["bash", "-n", str(HERE / "test.sh")]).returncode == 0


def test_per_simulation_domain_extent_takes_precedence():
    sys.path.insert(0, str(TASK / "environment" / "evaluation"))
    from pde_io import read_window
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ks.hdf5"
        with h5py.File(path, "w") as f:
            g = f.create_group("sims")
            g.attrs["PDE"] = "Kuramoto-Sivashinsky"; g.attrs["Fields"] = ["Density"]; g.attrs["Fields Scheme"] = "d"
            g.attrs["Dt"] = 0.5; g.attrs["Constants"] = ["Domain Extent"]; g.attrs["Boundary Conditions"] = ["periodic"] * 4
            ds = g.create_dataset("sim0", data=np.zeros((6, 1, 8, 8), np.float32)); ds.attrs["Domain Extent"] = 63.5
        frames, meta = read_window("ape2d", path, 0, 0, 4, 1)
        assert meta["domain_extent"] == [63.5, 63.5] and meta["constants"] == {"Domain Extent": 63.5}


def test_split_excludes_the_upstream_duplicate_family():
    """ape2d-full/hyp.hdf5 is a content copy of diff.hdf5 upstream (same PDE attribute, constants,
    arrays and sidecar); a split that used it would score diffusion twice."""
    for name in ("split.json", "split_smoke.json"):
        m = _manifest(name)
        files = {p["file"] for s in m["pools"].values() for p in s} | {c["file"] for c in m["cases"] if "file" in c}
        assert not any(Path(f).name.startswith("hyp") for f in files), name
    m = _manifest("split.json")
    corpus = {c["dataset"] for c in m["cases"] if c["source"] == "ape2d-full" and c["split"] == "test"}
    assert len(corpus) == 16
    assert sum(1 for c in m["cases"] if c["split"] == "test") == 1085


def test_starter_pools_refuse_a_missing_pool_file(tmp_path):
    sys.path.insert(0, str(TASK / "environment" / "baseline" / "starter"))
    from data import Pools
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"pools": {"train": [{"source": "ape2d-full", "dataset": "ape2d/burgers",
                                                            "file": "absent.hdf5", "trajectories": [0, 2]}]}, "cases": []}))
    with pytest.raises(FileNotFoundError):
        Pools(tmp_path, manifest, "train")
    pools = Pools(tmp_path, manifest, "train", allow_missing=True)
    assert pools.missing == ["ape2d-full/absent.hdf5"] and pools.datasets == []


def test_agent_mount_check_accepts_heads_and_refuses_full_files(tmp_path):
    """environment/data/check_agent_mount.py (the dev entrypoint's guard): a truncated head passes, an upstream file with
    the held-out tail is refused. The first version of the guard raised on every start."""
    import subprocess
    split = {"pools": {"train": [{"source": "s", "file": "a.hdf5", "trajectories": [0, 4]}],
                       "valid": [{"source": "s", "file": "a.hdf5", "trajectories": [4, 5]}]},
             "cases": [{"source": "s", "file": "a.hdf5", "trajectory": 5}]}
    (tmp_path / "split.json").write_text(json.dumps(split))
    data = tmp_path / "data" / "s"; data.mkdir(parents=True)
    script = TASK / "environment" / "data" / "check_agent_mount.py"
    def write(n):
        with h5py.File(data / "a.hdf5", "w") as f:
            g = f.create_group("sims")
            for i in range(n):
                g.create_dataset(f"sim{i}", data=np.zeros((2, 1, 2, 2), dtype="float32"))
    write(5)      # the agent-side head: trajectories 0..4
    ok = subprocess.run([sys.executable, str(script), str(tmp_path / "data"), str(tmp_path / "split.json")], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    write(6)      # the upstream file with the held-out trajectory 5
    bad = subprocess.run([sys.executable, str(script), str(tmp_path / "data"), str(tmp_path / "split.json")], capture_output=True, text=True)
    assert bad.returncode == 1 and "held-out" in bad.stderr
