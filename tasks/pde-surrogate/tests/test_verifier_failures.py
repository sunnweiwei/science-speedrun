"""Verifier-level tests on a synthetic source: a good package, a crashing package, a hanging
package, a package with the wrong contract, and the input/truth isolation boundary.
Run: python3 -m pytest pde-surrogate/tests -q"""
import json
import os
import shutil
import subprocess
import sys

import pytest
import tempfile
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
sys.path.insert(0, str(HERE))
import isolation  # noqa: E402
VERIFIER = HERE / "main_verifier.py"
PY = sys.executable


def make_source(root: Path):
    """Synthetic PBDL-format file: 3 simulations, 12 frames, 1 channel, 16x16."""
    src = root / "data" / "ape2d-synthetic"
    src.mkdir(parents=True)
    rng = np.random.default_rng(0)
    with h5py.File(src / "syn.hdf5", "w") as f:
        g = f.create_group("sims")
        g.attrs["PDE"] = "Burgers"; g.attrs["Fields"] = ["Velocity X"]; g.attrs["Fields Scheme"] = "V"
        g.attrs["Dt"] = 0.1; g.attrs["Constants"] = ["Viscosity"]; g.attrs["Domain Extent"] = 1.0
        g.attrs["Boundary Conditions"] = ["periodic"] * 4
        for i in range(3):
            base = rng.normal(size=(1, 16, 16)).astype(np.float32)
            frames = np.stack([base * (1 + 0.05 * t) for t in range(12)])  # slowly scaling field
            d = g.create_dataset(f"sim{i}", data=frames); d.attrs["Viscosity"] = 1e-3
    cases = [{"case_id": f"syn-{i}", "dataset": "ape2d/syn", "source": "ape2d-synthetic", "file": "syn.hdf5",
              "trajectory": i, "start": 0, "n_steps_out": 5, "split": "test"} for i in range(3)]
    man = root / "split.json"
    man.write_text(json.dumps({"contract": "pde-surrogate-rollout/v1", "cases": cases}))
    (root / "data").chmod(0o700)   # root-only, as /private is in the verifier image
    return root / "data", man


PERSIST = r'''#!/usr/bin/env bash
set -euo pipefail
python3 - "$1" "$2" <<'PY'
import json, sys, h5py, numpy as np
from pathlib import Path
inp, out = Path(sys.argv[1]), Path(sys.argv[2]); (out / "cases").mkdir(parents=True, exist_ok=True)
for c in json.loads((inp / "index.json").read_text())["cases"]:
    with h5py.File(inp / "cases" / f"{c['case_id']}.h5") as f: x = f["input"][()]
    with h5py.File(out / "cases" / f"{c['case_id']}.h5", "w") as f: f.create_dataset("prediction", data=np.repeat(x[-1:], c["n_steps_out"], 0))
PY
'''


def make_package(root: Path, infer_sh: str, contract="pde-surrogate-rollout/v1"):
    pkg = root / "submission"; pkg.mkdir()
    (pkg / "infer.sh").write_text(infer_sh); os.chmod(pkg / "infer.sh", 0o755)
    (pkg / "model_manifest.json").write_text(json.dumps({"contract": contract, "name": "test", "context_frames": 4}))
    (pkg / "README.md").write_text("test package\n")
    return pkg


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


def run_verifier(pkg, data, man, out, work, timeout=60):
    _open_to_runner(work); _open_to_runner(pkg)
    env = dict(os.environ, PATH=str(Path(PY).parent) + os.pathsep + os.environ.get("PATH", ""))
    if isolation.select_mode() == "subprocess":
        # neither a namespace nor root here (e.g. a non-root container): these tests are about scoring, and the
        # isolation boundary has its own tests in test_isolation.py and the root-only one below
        env.setdefault("PDE_SURROGATE_ALLOW_NO_ISOLATION", "1")
    return subprocess.run([PY, str(VERIFIER), "--submission", str(pkg), "--data", str(data), "--manifest", str(man),
                           "--output", str(out), "--work", str(work), "--timeout", str(timeout)],
                          capture_output=True, text=True, env=env)


def test_good_package_yields_reward_and_report():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); data, man = make_source(root); pkg = make_package(root, PERSIST)
        r = run_verifier(pkg, data, man, root / "out", root / "work")
        assert r.returncode == 0, r.stderr[-800:]
        rew = json.loads((root / "out" / "reward.json").read_text())["reward"]
        st = json.loads((root / "out" / "status.json").read_text())
        assert 0 < rew < 1 and st["n_failed"] == 0 and st["inference"]["exit_code"] == 0 and st["stage"] == "done"


def test_determinism_check_is_recorded():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); data, man = make_source(root); pkg = make_package(root, PERSIST)
        run_verifier(pkg, data, man, root / "out", root / "work")
        st = json.loads((root / "out" / "status.json").read_text())
        assert st["determinism"]["checked"] == 3 and st["determinism"]["deterministic"] is True
        assert st["determinism"]["max_abs_difference"] == 0.0


def test_crashing_package_is_scored_not_fatal():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); data, man = make_source(root)
        pkg = make_package(root, "#!/usr/bin/env bash\necho boom >&2\nexit 3\n")
        r = run_verifier(pkg, data, man, root / "out", root / "work")
        assert r.returncode == 0
        st = json.loads((root / "out" / "status.json").read_text())
        assert st["inference"]["exit_code"] == 3 and st["n_failed"] == 3 and st["reward"] == 0.0
        assert "missing" in st["failure_reasons"]


def test_hanging_package_times_out_and_is_scored():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); data, man = make_source(root)
        pkg = make_package(root, "#!/usr/bin/env bash\nsleep 30\n")
        r = run_verifier(pkg, data, man, root / "out", root / "work", timeout=3)
        assert r.returncode == 0
        st = json.loads((root / "out" / "status.json").read_text())
        assert st["inference"]["timed_out"] is True and st["n_failed"] == 3 and st["reward"] == 0.0


def test_wrong_contract_scores_zero_with_the_violation_named():
    """A malformed package is the package's fault: reward 0 and a reason, exit 0, never a setup error."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); data, man = make_source(root)
        pkg = make_package(root, PERSIST, contract="something-else/v9")
        r = run_verifier(pkg, data, man, root / "out", root / "work")
        assert r.returncode == 0, r.stderr[-800:]
        rew = json.loads((root / "out" / "reward.json").read_text())
        st = json.loads((root / "out" / "status.json").read_text())
        assert rew["reward"] == 0.0 and rew["rule_violation"] == 1.0 and "contract" in st["rule_violation"]


def test_declared_pretrained_assets_score_zero():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); data, man = make_source(root)
        pkg = make_package(root, PERSIST)
        m = json.loads((pkg / "model_manifest.json").read_text()); m["pretrained_assets"] = ["some/checkpoint"]
        (pkg / "model_manifest.json").write_text(json.dumps(m))
        r = run_verifier(pkg, data, man, root / "out", root / "work")
        assert r.returncode == 0, r.stderr[-800:]
        assert "pretrained" in json.loads((root / "out" / "status.json").read_text())["rule_violation"]


def test_a_symlink_escaping_the_package_is_a_violation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); data, man = make_source(root)
        pkg = make_package(root, PERSIST)
        (pkg / "peek").symlink_to(root)          # points outside the package
        r = run_verifier(pkg, data, man, root / "out", root / "work")
        assert r.returncode == 0, r.stderr[-800:]
        assert "symlink" in json.loads((root / "out" / "status.json").read_text())["rule_violation"]


def test_non_finite_truth_is_a_grader_error_not_a_score():
    """A NaN in the staged truth must abort scoring (exit 2), not become a reward of 0."""
    import h5py
    import numpy as np
    sys.path.insert(0, str(TASK / "environment" / "evaluation"))
    from score import score
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for d in ("input/cases", "truth/cases", "output/cases"):
            (root / d).mkdir(parents=True)
        (root / "input" / "index.json").write_text(json.dumps({"contract": "pde-surrogate-rollout/v1",
                                                               "cases": [{"case_id": "c0", "dataset": "ape2d/burgers"}]}))
        t = np.zeros((3, 1, 4, 4), dtype="float32"); t[1, 0, 2, 2] = np.nan
        with h5py.File(root / "truth" / "cases" / "c0.h5", "w") as f:
            f.create_dataset("target", data=t); f.attrs["fields"] = ["u"]
        with h5py.File(root / "output" / "cases" / "c0.h5", "w") as f:
            f.create_dataset("prediction", data=np.zeros_like(t))
        with pytest.raises(ValueError):
            score(root / "input", root / "truth", root / "output")


def test_reward_json_names_the_isolation_mode_and_the_bounds_resolve_in_the_image_layout():
    """`reward.json` carries the isolation mode; the bounds file is found next to the grader modules,
    which is where the verifier image bakes it (/tests/evaluation), not two directories up."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); data, man = make_source(root)
        pkg = make_package(root, PERSIST)
        r = run_verifier(pkg, data, man, root / "out", root / "work")
        assert r.returncode == 0, r.stderr[-800:]
        rew = json.loads((root / "out" / "reward.json").read_text())
        assert all(isinstance(v, (int, float)) for v in rew.values()) and "normalised_score" in rew
        st = json.loads((root / "out" / "status.json").read_text())
        assert st["inference"]["isolation"] in ("bwrap", "uid", "subprocess")
        assert st["determinism"].get("isolation") == st["inference"]["isolation"], "the rerun must use the same isolation"
        # the image layout: main_verifier.py beside an evaluation/ directory, no ../environment
        image = root / "tests"; (image / "evaluation").mkdir(parents=True)
        for name in ("main_verifier.py", "isolation.py"):
            shutil.copy(HERE / name, image / name)
        (image / "evaluation").mkdir(exist_ok=True)
        for name in ("score.py", "make_cases.py", "pde_io.py", "score_bounds.json"):
            shutil.copy(TASK / "environment" / "evaluation" / name, image / "evaluation" / name)
        probe = subprocess.run([sys.executable, "-c", "import main_verifier as m; print(m.normalised(0.3)['normalised_score'])"],
                               cwd=image, capture_output=True, text=True)
        assert probe.returncode == 0 and probe.stdout.strip(), probe.stderr[-500:]


def test_input_namespace_holds_no_targets():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); data, man = make_source(root); pkg = make_package(root, PERSIST)
        run_verifier(pkg, data, man, root / "out", root / "work")
        inp = root / "work" / "input"
        for f in (inp / "cases").glob("*.h5"):
            with h5py.File(f) as h:
                assert set(h.keys()) == {"input"} and h["input"].shape[0] == 4
        assert not (inp / "truth").exists() and (root / "work" / "truth").exists()


@pytest.mark.skipif(os.geteuid() != 0, reason="the uid drop needs root")
def test_uid_mode_refuses_data_the_runner_can_read(monkeypatch):
    """A package running as the runner uid must not reach the held-out files the truth was staged from."""
    monkeypatch.setenv("PDE_SURROGATE_NO_BWRAP", "1")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); data, man = make_source(root); pkg = make_package(root, PERSIST)
        data.chmod(0o755)
        r = run_verifier(pkg, data, man, root / "out", root / "work")
        assert r.returncode != 0 and not (root / "out" / "reward.json").exists()
        assert "runner uid, which can read" in (root / "out" / "status.json").read_text()


CACHING = r'''#!/usr/bin/env bash
set -euo pipefail
python3 - "$1" "$2" <<'PY'
import json, os, sys, h5py, numpy as np
from pathlib import Path
inp, out = Path(sys.argv[1]), Path(sys.argv[2]); (out / "cases").mkdir(parents=True, exist_ok=True)
cache = Path(os.environ["HOME"]) / "cache"; cache.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng()
for c in json.loads((inp / "index.json").read_text())["cases"]:
    with h5py.File(inp / "cases" / f"{c['case_id']}.h5") as f: x = f["input"][()]
    hit = cache / f"{c['case_id']}.npy"
    pred = np.load(hit) if hit.exists() else np.repeat(x[-1:], c["n_steps_out"], 0) + rng.normal(size=1).astype(x.dtype)
    np.save(hit, pred)
    with h5py.File(out / "cases" / f"{c['case_id']}.h5", "w") as f: f.create_dataset("prediction", data=pred)
PY
'''


@pytest.mark.parametrize("namespace", [True, False], ids=["as-selected", "no-namespace"])
def test_determinism_rerun_cannot_read_the_first_calls_home(monkeypatch, namespace):
    """A package that caches its first-call predictions in HOME and replays them must not pass the rerun:
    every call gets an empty HOME/TMPDIR (main_verifier.run_call). Bubblewrap gives each call a fresh
    /home/runner anyway; the uid and plain-subprocess modes share the verifier's filesystem, so the
    second case forces them (uid as root, else the opt-in untrusted mode)."""
    if not namespace:
        monkeypatch.setenv("PDE_SURROGATE_NO_BWRAP", "1")
        if os.geteuid() != 0:
            monkeypatch.setenv("PDE_SURROGATE_ALLOW_NO_ISOLATION", "1")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); data, man = make_source(root); pkg = make_package(root, CACHING)
        r = run_verifier(pkg, data, man, root / "out", root / "work")
        assert r.returncode == 0, r.stderr[-800:]
        st = json.loads((root / "out" / "status.json").read_text())
        assert st["determinism"]["checked"] == 3 and st["determinism"]["deterministic"] is False, st["determinism"]
        assert not list((root / "work").glob("home-*")), "per-call homes must be removed"
