"""The namespace must hide the held-out targets from the package, and still run a real model."""
import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isolation  # noqa: E402

bwrap_only = pytest.mark.skipif(not isolation.bwrap_available(), reason="bubblewrap unavailable")


def _probe_package(root: Path, marker: str):
    sealed = root / "sealed"
    sealed.mkdir(parents=True)
    (sealed / "infer.sh").write_text(
        '#!/usr/bin/env bash\nexec python3 "$(dirname "$(readlink -f "$0")")/probe.py" "$1" "$2"\n')
    (sealed / "infer.sh").chmod(0o755)
    (sealed / "probe.py").write_text(textwrap.dedent(f"""
        import json, os, sys
        from pathlib import Path
        found = []
        for start in ("/", "/tmp", "/home", {str(root)!r}):
            try:
                for dirpath, _, files in os.walk(start):
                    found += [os.path.join(dirpath, f) for f in files if {marker!r} in f]
                    if len(found) > 2:
                        break
            except Exception:
                pass
        Path(sys.argv[2], "probe.json").write_text(json.dumps(found))
    """))
    return sealed


@bwrap_only
def test_truth_is_unreachable_inside_the_namespace(tmp_path):
    marker = "SECRET_TARGETS"
    sealed = _probe_package(tmp_path, marker)
    inp, outp, truth = tmp_path / "input", tmp_path / "out", tmp_path / "truth"
    for d in (inp, outp, truth):
        d.mkdir()
    (truth / f"{marker}.h5").write_text("held-out answers")

    with open(tmp_path / "log", "w") as log:
        rc, _, mode = isolation.run_isolated(sealed, inp, outp, 120, log, use_bwrap=True)
    assert mode == "bwrap" and rc == 0
    assert json.loads((outp / "probe.json").read_text()) == [], "the package reached the held-out targets"


def test_without_the_namespace_the_same_probe_succeeds(tmp_path):
    """The control: this is the boundary the namespace exists to close."""
    marker = "SECRET_TARGETS"
    sealed = _probe_package(tmp_path, marker)
    inp, outp, truth = tmp_path / "input", tmp_path / "out", tmp_path / "truth"
    for d in (inp, outp, truth):
        d.mkdir()
    (truth / f"{marker}.h5").write_text("held-out answers")

    with open(tmp_path / "log", "w") as log:
        rc, _, mode = isolation.run_isolated(sealed, inp, outp, 120, log, use_bwrap=False,
                                             fallback_env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)})
    assert mode == "subprocess" and rc == 0
    assert json.loads((outp / "probe.json").read_text()), "probe should find the targets without isolation"


@bwrap_only
def test_output_is_writable_and_input_is_not(tmp_path):
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    (sealed / "infer.sh").write_text(
        '#!/usr/bin/env bash\n'
        'echo ok > "$2"/wrote.txt\n'
        'if echo bad > "$1"/tampered.txt 2>/dev/null; then echo INPUT_WRITABLE; fi\n')
    inp, outp = tmp_path / "input", tmp_path / "out"
    inp.mkdir(); outp.mkdir()
    with open(tmp_path / "log", "w") as log:
        rc, _, mode = isolation.run_isolated(sealed, inp, outp, 120, log, use_bwrap=True)
    assert rc == 0 and mode == "bwrap"
    assert (outp / "wrote.txt").exists()
    assert not (inp / "tampered.txt").exists()
    assert "INPUT_WRITABLE" not in (tmp_path / "log").read_text()


@bwrap_only
def test_cuda_pip_libraries_are_on_the_sandbox_library_path():
    """cuDNN dlopens its sublibraries at first convolution; --clearenv drops torch's rpath."""
    cmd = isolation._bwrap_command(Path("/tmp"), Path("/tmp"), Path("/tmp"))
    ld = cmd[cmd.index("LD_LIBRARY_PATH") + 1]
    nvidia = list(Path(sys.prefix).glob("lib/python*/site-packages/nvidia/*/lib"))
    if not nvidia:
        pytest.skip("no pip-installed CUDA libraries in this runtime")
    assert any(str(d) in ld for d in nvidia), f"nvidia lib dirs missing from LD_LIBRARY_PATH: {ld}"


def test_mode_selection_prefers_the_namespace_then_the_uid_drop(monkeypatch):
    """bwrap when the host allows user namespaces; otherwise a uid drop, which needs root; otherwise nothing."""
    monkeypatch.setattr(isolation, "bwrap_available", lambda: True)
    assert isolation.select_mode() == "bwrap"
    monkeypatch.setattr(isolation, "bwrap_available", lambda: False)
    monkeypatch.setattr(isolation.os, "geteuid", lambda: 0)
    assert isolation.select_mode() == "uid"
    monkeypatch.setattr(isolation.os, "geteuid", lambda: 1000)
    assert isolation.select_mode() == "subprocess"


@pytest.mark.skipif(os.geteuid() != 0, reason="the uid drop needs root")
def test_uid_drop_hides_a_root_owned_truth_directory(tmp_path):
    marker = "SECRET_TARGETS"
    sealed = _probe_package(tmp_path, marker)
    inp, outp, truth = tmp_path / "input", tmp_path / "out", tmp_path / "truth"
    for d in (inp, outp, truth):
        d.mkdir()
    (truth / f"{marker}.h5").write_text("held-out answers")
    truth.chmod(0o700)
    import os
    p = tmp_path.resolve()
    while str(p) not in ('/', '/tmp'):
        os.chmod(p, os.stat(p).st_mode | 0o011); p = p.parent
    with open(tmp_path / "log", "w") as log:
        rc, _, mode = isolation.run_isolated(sealed, inp, outp, 120, log, mode="uid")
    assert mode == "uid" and rc == 0, (tmp_path / "log").read_text()
    assert json.loads((outp / "probe.json").read_text()) == [], "the runner uid reached the root-owned targets"
