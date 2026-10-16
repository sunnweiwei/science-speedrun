"""v2 suites: scoring of failures and caps, run binding, submission factory, verifier boundaries."""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

import score_extra
from common import digest


def write(path: Path, value, gz=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if gz:
        with gzip.open(path, "wt") as f:
            json.dump(value, f)
    else:
        path.write_text(json.dumps(value))


def run_dir(tmp_path, suite, inputs, items):
    run = tmp_path / "run"
    (run / "items").mkdir(parents=True)
    for i, item in enumerate(items):
        (run / "items" / f"{i}.json").write_text(json.dumps(item))
    write(run / "run.json", {"suite": suite, "complete": True,
                             "inputs_manifest_sha256": digest(inputs / "manifest.json")})
    return run


def test_neb_errors_are_capped_and_failures_score_the_cap(tmp_path):
    refs = tmp_path / "refs"
    write(refs / "references.json", [{"id": "a", "barrier_eV": 0.5}, {"id": "b", "barrier_eV": 0.3},
                                     {"id": "c", "barrier_eV": 0.4}])
    pred = {"a": {"status": "ok", "barrier_eV": 0.6, "converged": True},
            "b": {"status": "ok", "barrier_eV": 40.0, "converged": False},  # diverged NEB
            "c": {"status": "failed"}}
    out = score_extra.score_neb(pred, refs)
    assert out["barrier_MAE_eV"] == pytest.approx((0.1 + 2.0 + 2.0) / 3)
    assert out["barrier_MAE_converged_eV"] == pytest.approx(0.1)
    assert out["failed"] == 1 and out["converged_frac"] == pytest.approx(1 / 3)


def test_failing_never_beats_a_bad_prediction(tmp_path):
    refs = tmp_path / "refs"
    write(refs / "references.json", [{"id": "m", "B0_GPa": 10.0}])
    worst = score_extra.score_mof({"m": {"status": "ok", "B_GPa": 1000.0}}, refs)["bulk_modulus_MAPE_pct"]
    failed = score_extra.score_mof({}, refs)["bulk_modulus_MAPE_pct"]
    assert failed >= worst == 200.0
    write(refs / "references.json", [{"id": "s", "cleavage_energy": 0.1}])
    bad = score_extra.score_cleavage({"s": {"status": "ok", "cleavage_energy": 5.0}}, refs)
    gone = score_extra.score_cleavage({}, refs)
    assert gone["cleavage_relMAE_pct"] >= bad["cleavage_relMAE_pct"] == pytest.approx(200.0)


def test_kappa_failures_and_broken_symmetry_score_two():
    ref = {"kappa_tot_avg": [10.0], "mode_kappa_tot_avg": np.ones((2, 3)).tolist(), "init_spg_num": 225, "weights": [1, 1]}
    assert score_extra.srme_one(None, ref)[0] == 2.0
    broken = {"status": "ok", "init_spg_num": 225, "final_spg_num": 221, "kappa_tot_rta": [10, 10, 10, 0, 0, 0]}
    assert score_extra.srme_one(broken, ref)[0] == 2.0
    exact = {"status": "ok", "init_spg_num": 225, "final_spg_num": 225,
             "kappa_tot_rta": [10, 10, 10, 0, 0, 0], "mode_weights": [1, 1],
             "mode_kappa_tot_rta": np.tile([1.0, 1.0, 1.0, 0, 0, 0], (2, 3, 1)).tolist()}
    assert score_extra.srme_one(exact, ref) == (0.0, 0.0, False)


def test_kappa_normalization_and_initial_symmetry_are_reference_owned():
    ref = {"kappa_tot_avg": [10.0], "mode_kappa_tot_avg": np.ones((1, 2, 3)).tolist(),
           "init_spg_num": 225, "weights": [1, 1]}
    pred = {"status": "ok", "init_spg_num": 225, "final_spg_num": 225,
            "kappa_tot_rta": [10, 10, 10, 0, 0, 0], "mode_weights": [1, 1],
            "mode_kappa_tot_rta": np.tile([1.5, 1.5, 1.5, 0, 0, 0], (1, 2, 3, 1)).tolist()}
    score = score_extra.srme_one(pred, ref)
    assert score == pytest.approx((0.15, 0, False))
    pred["mode_weights"] = [1e12, 1e12]
    assert score_extra.srme_one(pred, ref) == score
    pred.update(init_spg_num=221, final_spg_num=221)
    assert score_extra.srme_one(pred, ref)[0] == 2


def test_pool_split_groups_supercells_and_keeps_parents_disjoint(tmp_path, monkeypatch):
    import importlib.util
    from ase import Atoms
    from ase.db import connect

    path = Path(__file__).resolve().parents[1] / "environment/baseline/uma-mixed-replay/prepare_mixed.py"
    spec = importlib.util.spec_from_file_location("mixed_selection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.composition_key(Atoms("SiO2").symbols) == module.composition_key(Atoms("Si14O28").symbols)
    shard = tmp_path / "source/frames.aselmdb"
    shard.parent.mkdir()
    with connect(shard) as db:
        for i, formula in enumerate(("SiO2", "Si14O28")):
            db.write(Atoms(formula), data={"sid": str(i), "parent": "same-parent"})
    rows = module.scan(str(shard))
    assert len({row[-1] for row in rows}) == 1

    class SerialPool:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def map(self, func, items, **kwargs): return map(func, items)

    monkeypatch.setattr(module, "ProcessPoolExecutor", SerialPool)
    train, dev = module.select(tmp_path, ["source"], 10, 10, 1)
    assert sorted(len(x) for x in (train, dev)) == [0, 2]


def test_pretraining_workspace_does_not_mount_historical_weights():
    import yaml

    root = Path(__file__).resolve().parents[1] / "environment"
    config = yaml.safe_load((root / "docker-compose.yaml").read_text())
    targets = {v["target"] for v in config["services"]["main"]["volumes"]}
    assert targets == {"/workspace/data/pool", "/workspace/data/val",
                       "/workspace/data/baselines/mp-standard", "/workspace/research"}


def test_invalid_package_publishes_zero_versioned_score(tmp_path):
    import subprocess
    import sys

    script = Path(__file__).with_name("main_verifier.py")
    output = tmp_path / "output"
    subprocess.run([sys.executable, str(script), "--submission", str(tmp_path / "missing"),
                    "--input", str(tmp_path / "inputs"), "--sealed", str(tmp_path / "sealed"),
                    "--output", str(output)], check=True)
    report = json.loads((output / "report.json").read_text())
    reward = json.loads((output / "reward.json").read_text())
    assert report["aggregate"]["version"] == "crystal-potential-score-v2"
    assert reward["reward"] == reward["crystal_potential_score"] == 0
    assert reward["invalid_submission"] is True
    # Harbor rejects unavailable metrics serialized as null.
    assert all(isinstance(value, (int, float)) and np.isfinite(value) for value in reward.values())


def test_amorphous_relative_rmse_and_missing_structures(tmp_path):
    refs = tmp_path / "refs"
    rows = [{"id": f"x{i}", "system": "si", "energy_per_atom": float(i), "forces": [[1.0, 0, 0], [-1.0, 0, 0]]}
            for i in range(3)]
    write(refs / "references.json.gz", rows, gz=True)
    perfect = {r["id"]: {"status": "ok", "energy_per_atom": r["energy_per_atom"] + 5, "forces": r["forces"]}
               for r in rows}
    out = score_extra.score_amorphous(perfect, refs)
    assert out["si_energy_RelRMSE_pct"] == pytest.approx(0) and out["si_force_RelRMSE_pct"] == pytest.approx(0)
    perfect.pop("x0")
    assert score_extra.score_amorphous(perfect, refs)["failed"] == 1
    nothing = score_extra.score_amorphous({}, refs)
    assert nothing["si_energy_RelRMSE_pct"] > 150 and nothing["si_force_RelRMSE_pct"] > 150


def test_run_bound_to_its_inputs(tmp_path):
    inputs = tmp_path / "inputs"
    write(inputs / "manifest.json", {"suite": "neb", "ids": ["a"]})
    refs = tmp_path / "refs"
    write(refs / "references.json", [{"id": "a", "barrier_eV": 0.5}])
    run = run_dir(tmp_path, "neb", inputs, [{"id": "a", "status": "ok", "barrier_eV": 0.5, "converged": True}])
    assert score_extra.score("neb", run, inputs, refs)["barrier_MAE_eV"] == 0
    write(inputs / "manifest.json", {"suite": "neb", "ids": ["a", "b"]})
    with pytest.raises(SystemExit):
        score_extra.score("neb", run, inputs, refs)


def test_submission_factory_and_directory_digest(tmp_path):
    from predict import load_calculator
    sub = tmp_path / "submission"
    sub.mkdir()
    (sub / "calculator.py").write_text(
        "from ase.calculators.emt import EMT\n"
        "def make_calculator(submission_dir, device):\n"
        "    assert (submission_dir / 'calculator.py').is_file()\n"
        "    return EMT()\n")
    calc = load_calculator("submission", sub, "cpu")
    assert type(calc).__name__ == "EMT"
    first = digest(sub)
    (sub / "weights.bin").write_bytes(b"x")
    assert digest(sub) != first


def test_verifier_rejects_references_in_untrusted_inputs(tmp_path):
    import main_verifier
    for suite in main_verifier.SUITES:
        (tmp_path / suite / "inputs").mkdir(parents=True)
    main_verifier.validate_input(tmp_path)
    (tmp_path / "kappa" / "inputs" / "references.json.gz").write_bytes(b"")
    with pytest.raises(ValueError):
        main_verifier.validate_input(tmp_path)
    with pytest.raises(FileNotFoundError):
        main_verifier.validate_submission(tmp_path)


def test_reward_keys_are_flat_raw_metrics():
    import grader
    report = {"physical": {"metrics": {"discovery": {"F1": 0.8}, "elasticity": {"bulk_modulus_vrh": {"MAE": 5.0}}}},
              "kappa": {"kappa_SRME": 0.3}, "neb": {"valid": False}}
    reward = grader.flatten(report)
    assert reward["discovery_F1"] == 0.8 and reward["bulk_modulus_MAE_GPa"] == 5.0
    assert reward["kappa_SRME"] == 0.3 and reward["neb_barrier_MAE_eV"] is None
    assert len(reward) == 15
