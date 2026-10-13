from __future__ import annotations

import os
from pathlib import Path
import pwd
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from main_verifier import _agent_processes, _copy_snapshot, _stop_untrusted  # noqa: E402


def test_cleanup_kills_detached_agent_descendants(tmp_path: Path):
    try:
        agent = pwd.getpwnam("agent")
    except KeyError:
        pytest.skip("the generic materializer test image has no untrusted agent user")
    pid_path = Path("/tmp") / f"agbench-detached-{os.getpid()}.pid"
    pid_path.unlink(missing_ok=True)
    code = (
        "import os,time; child=os.fork(); "
        "child and os._exit(0); os.setsid(); "
        f"open({str(pid_path)!r},'w').write(str(os.getpid())); time.sleep(60)"
    )

    def demote():
        os.setgid(agent.pw_gid)
        os.setuid(agent.pw_uid)

    parent = subprocess.Popen(
        [sys.executable, "-c", code], start_new_session=True, preexec_fn=demote
    )
    parent.wait(timeout=5)
    deadline = time.monotonic() + 5
    while not pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pid_path.exists()

    _stop_untrusted(agent.pw_uid, parent.pid)

    assert _agent_processes(agent.pw_uid) == []
    pid_path.unlink(missing_ok=True)


def test_snapshot_copy_rejects_special_files(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    os.mkfifo(source / "pipe")
    with pytest.raises(ValueError, match="non-regular"):
        _copy_snapshot(source, tmp_path / "snapshot")
