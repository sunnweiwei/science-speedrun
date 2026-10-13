"""What a package can write, leave behind or declare must never raise its score or crash the verifier.

These cover the second integration review round: package output is read by the grader with the verifier's
privileges, so HDF5 links, virtual or external datasets and non-regular files are refused; a malformed manifest
scores 0; the sealed copy is not writable by the runner; a deep tree left in HOME is removed without crashing; and,
as root (the integrator's uid mode), no runner process, file, IPC object or output survives a call.
"""
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "environment" / "evaluation"))
import isolation  # noqa: E402
from score import open_package_h5  # noqa: E402
from test_verifier_failures import PERSIST, make_package, make_source, run_verifier  # noqa: E402

ROOT_ONLY = pytest.mark.skipif(os.geteuid() != 0, reason="uid mode needs root (the integrator's verifier container)")


def _traversable(path: Path):
    """pytest's root temp tree (/tmp/pytest-of-root) is 0700: open every parent so the runner uid can reach `path`."""
    for p in [Path(path), *Path(path).parents]:
        if str(p) in ("/", "/tmp"):
            break
        os.chmod(p, os.stat(p).st_mode | 0o011)


def _link_package(kind: str, work: Path) -> str:
    """infer.sh that writes, for every case, a 'prediction' pointing at the staged truth instead of a forecast."""
    return r'''#!/usr/bin/env bash
set -euo pipefail
python3 - "$1" "$2" <<'PY'
import json, sys, h5py
from pathlib import Path
inp, out = Path(sys.argv[1]), Path(sys.argv[2]); (out / "cases").mkdir(parents=True, exist_ok=True)
truth = Path("%s") / "truth" / "cases"
for c in json.loads((inp / "index.json").read_text())["cases"]:
    with h5py.File(inp / "cases" / f"{c['case_id']}.h5") as f: C, X, Y = f["input"].shape[1:]
    t = str(truth / f"{c['case_id']}.h5")
    with h5py.File(out / "cases" / f"{c['case_id']}.h5", "w", libver="latest") as f:
        if "%s" == "external":
            f["prediction"] = h5py.ExternalLink(t, "target")
        else:
            lay = h5py.VirtualLayout(shape=(c["n_steps_out"], C, X, Y), dtype="f4")
            lay[:] = h5py.VirtualSource(t, "target", shape=(c["n_steps_out"], C, X, Y))
            f.create_virtual_dataset("prediction", lay)
PY
''' % (work, kind)


@pytest.mark.parametrize("kind", ["external", "virtual"])
def test_prediction_pointing_at_the_truth_is_a_failed_case(kind):
    """Before this guard the grader followed the link as the verifier and scored the truth against itself."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); data, man = make_source(root)
        pkg = make_package(root, _link_package(kind, (root / "work").resolve()))
        r = run_verifier(pkg, data, man, root / "out", root / "work")
        assert r.returncode == 0, r.stderr[-800:]
        st = json.loads((root / "out" / "status.json").read_text())
        assert st["n_failed"] == 3, st
        assert json.loads((root / "out" / "reward.json").read_text())["reward"] < 0.05


def test_open_package_h5_accepts_only_plain_datasets(tmp_path):
    plain, soft, ext, fifo = tmp_path / "plain.h5", tmp_path / "soft.h5", tmp_path / "ext.h5", tmp_path / "fifo.h5"
    with h5py.File(plain, "w") as f:
        f.create_dataset("prediction", data=np.zeros((2, 2), np.float32))
    with h5py.File(soft, "w") as f:
        f.create_dataset("real", data=np.zeros((2, 2), np.float32)); f["prediction"] = h5py.SoftLink("/real")
    raw = tmp_path / "raw.bin"; raw.write_bytes(np.zeros(4, np.float32).tobytes())
    with h5py.File(ext, "w") as f:
        f.create_dataset("prediction", shape=(2, 2), dtype="f4", external=[(str(raw), 0, 16)])
    os.mkfifo(fifo)
    with open_package_h5(plain, ["prediction"]) as f:
        assert f["prediction"].shape == (2, 2)
    for bad in (soft, ext, fifo):
        with pytest.raises((ValueError, KeyError)):
            open_package_h5(bad, ["prediction"])
    (tmp_path / "link.h5").symlink_to(plain)
    with pytest.raises(ValueError):
        open_package_h5(tmp_path / "link.h5", ["prediction"])


FIFO = PERSIST.replace("PY\n", "PY\nrm -f \"$2/cases/syn-1.h5\"; mkfifo \"$2/cases/syn-1.h5\"\n", 1)


def test_a_fifo_in_place_of_a_prediction_fails_that_case_without_blocking():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); data, man = make_source(root); pkg = make_package(root, FIFO)
        r = run_verifier(pkg, data, man, root / "out", root / "work", timeout=60)
        assert r.returncode == 0, r.stderr[-800:]
        st = json.loads((root / "out" / "status.json").read_text())
        assert st["n_failed"] == 1 and st["stage"] == "done", st


@pytest.mark.parametrize("manifest", ["[]", "null", '"pde-surrogate-rollout/v1"', "42"])
def test_a_manifest_that_is_not_an_object_scores_zero(manifest):
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); data, man = make_source(root); pkg = make_package(root, PERSIST)
        (pkg / "model_manifest.json").write_text(manifest)
        r = run_verifier(pkg, data, man, root / "out", root / "work")
        assert r.returncode == 0, r.stderr[-800:]
        st = json.loads((root / "out" / "status.json").read_text())
        assert "JSON object" in st["rule_violation"]
        assert json.loads((root / "out" / "reward.json").read_text())["reward"] == 0.0


def test_a_deeply_nested_manifest_name_is_recorded_not_fatal():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); data, man = make_source(root); pkg = make_package(root, PERSIST)
        m = json.loads((pkg / "model_manifest.json").read_text())
        (pkg / "model_manifest.json").write_text(json.dumps(m)[:-1] + ', "name": ' + "[" * 900 + "]" * 900 + "}")
        r = run_verifier(pkg, data, man, root / "out", root / "work")
        assert r.returncode == 0, r.stderr[-800:]
        assert json.loads((root / "out" / "status.json").read_text())["stage"] == "done"


def test_the_sealed_copy_is_not_writable_by_others_and_has_no_set_id_bits(tmp_path):
    from main_verifier import seal_package
    sub = tmp_path / "sub"; (sub / "open").mkdir(parents=True)
    (sub / "infer.sh").write_text("exit 0\n"); os.chmod(sub / "infer.sh", 0o4777)
    (sub / "open" / "state.txt").write_text("x"); os.chmod(sub / "open" / "state.txt", 0o666); os.chmod(sub / "open", 0o1777)
    assert seal_package(sub, tmp_path / "sealed") is None
    for p in [tmp_path / "sealed", *(tmp_path / "sealed").rglob("*")]:
        mode = p.stat().st_mode
        assert not mode & (stat.S_IWGRP | stat.S_IWOTH | stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX), (p, oct(mode))


DEEP_HOME = PERSIST.replace("set -euo pipefail\n", "set -euo pipefail\n"
                            "python3 -c \"import os; os.chdir(os.environ['HOME']); [(os.mkdir('d'), os.chdir('d')) for _ in range(1500)]\"\n", 1)


def test_a_deep_tree_left_in_home_is_removed_without_crashing_the_verifier(monkeypatch):
    """rmtree recursed on the Python stack before 3.12.4: a 1,500-deep HOME made the verifier exit 2 with no reward."""
    monkeypatch.setenv("PDE_SURROGATE_NO_BWRAP", "1")
    if os.geteuid() != 0:
        monkeypatch.setenv("PDE_SURROGATE_ALLOW_NO_ISOLATION", "1")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); data, man = make_source(root); pkg = make_package(root, DEEP_HOME)
        r = run_verifier(pkg, data, man, root / "out", root / "work")
        assert r.returncode == 0, r.stderr[-800:]
        assert (root / "out" / "reward.json").exists()
        assert not list((root / "work").glob("home-*"))


@ROOT_ONLY
def test_after_uid_call_leaves_no_runner_process_file_ipc_or_readable_output(tmp_path):
    """The integrator's mode: a call's detached child, its /tmp file, its SysV segment and its output are gone or
    root-only before the next call starts."""
    out, home = tmp_path / "out", tmp_path / "home"
    out.mkdir(); home.mkdir()
    _traversable(tmp_path)
    for p in (out, home):
        os.chown(p, isolation.RUNNER_UID, isolation.RUNNER_GID)
    marker = Path("/tmp") / f"runner-marker-{os.getpid()}"
    script = (f"import os, subprocess, sys; open('{out}/pred.h5', 'w').write('x'); open('{marker}', 'w').write('x');"
              "subprocess.run(['ipcmk', '-M', '4096'], capture_output=True);"
              "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'], start_new_session=True)")
    subprocess.run([sys.executable, "-c", script], preexec_fn=isolation._drop_privileges, check=True)
    assert isolation._runner_pids(), "the detached child should be running before cleanup"
    isolation.after_uid_call(out, home)
    assert not isolation._runner_pids()
    assert not marker.exists()
    assert out.stat().st_uid == 0 and stat.S_IMODE(out.stat().st_mode) == 0o700
    assert not isolation.runner_can_reach(out)
    shm = Path("/proc/sysvipc/shm").read_text().splitlines()
    assert not any(l.split()[7] == str(isolation.RUNNER_UID) for l in shm[1:] if len(l.split()) > 7)


@ROOT_ONLY
def test_runner_can_reach_counts_search_permission(tmp_path):
    """A directory the runner may traverse but not list still exposes files whose names are known (the data layout
    is public)."""
    _traversable(tmp_path)
    d = tmp_path / "data"; d.mkdir(); os.chmod(d, 0o711)
    assert isolation.runner_can_reach(d)
    os.chmod(d, 0o700)
    assert not isolation.runner_can_reach(d)
