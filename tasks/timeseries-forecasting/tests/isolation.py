"""Run `bash infer.sh INPUT OUTPUT` so that the package cannot reach the held-out targets.

    from isolation import run_isolated
    rc, seconds, mode = run_isolated(sealed, inp, outp, timeout, log)

Three modes, tried in this order:

  "bwrap"       a Bubblewrap namespace holding only the runtime, the sealed package, the input
                (read-only) and the output (read-write): no network, no data directory, no truth,
                no workspace, no host /tmp. Needs user namespaces, which Docker's default seccomp
                profile denies; a compose overlay with `security_opt: [seccomp:unconfined,
                apparmor:unconfined]` restores them (tests/docker-compose.yaml).
  "uid"         the verifier runs as root (Harbor's default) and Bubblewrap cannot start: the
                package runs as an unprivileged uid (TIMESERIES_RUNNER_UID, default 10001) while
                the truth directory stays root-owned with mode 0700. This is the boundary
                protein-sequence-modeling's verifier uses, and the one the integrator has reproduced
                under Docker; it hides only what root-only permissions hide, so the verifier data
                must sit under a root-only directory too (tests/docker-compose.yaml mounts it at
                /private/data inside a 0700 /private) and main_verifier.py refuses to run the
                package when runner_can_reach() says the runner could read the data or the labels.
                It does not cut the network: the compose overlay sets network_mode: none.
  "subprocess"  neither is possible (not root, no namespaces). The package sees the verifier's
                work directory, which holds the labels, so main_verifier.py refuses to
                score unless the operator sets TIMESERIES_ALLOW_NO_ISOLATION=1, and then marks
                the result untrusted.

Shared with pde-surrogate/tests/isolation.py; both contracts are `bash infer.sh INPUT OUTPUT`. Adapted from probabilistic-global-weather/wn2/namespace_inference.py.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


UNSHARE_ALL = ["--unshare-all"]
# Docker containers usually cannot configure the loopback of a new network namespace ("bwrap: loopback: Failed
# RTM_NEWADDR"); then every namespace but the network one is still available, and the container's own
# network policy (task.toml no-network, compose network_mode: none) is what keeps the package offline.
UNSHARE_NO_NET = ["--unshare-user-try", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup-try"]
_UNSHARE = None


def unshare_flags():
    """The bwrap namespace flags this host accepts, probed once; None when none work."""
    global _UNSHARE
    if _UNSHARE is None:
        _UNSHARE = False
        for flags in (UNSHARE_ALL, UNSHARE_NO_NET):
            try:
                subprocess.run(["bwrap", *flags, "--ro-bind", "/", "/", "true"], check=True, capture_output=True, timeout=30)
                _UNSHARE = flags
                break
            except Exception:  # noqa: BLE001
                continue
    return _UNSHARE or None


def _bwrap_command(sealed: Path, inp: Path, outp: Path):
    args = ["bwrap", *(unshare_flags() or UNSHARE_ALL), "--die-with-parent", "--new-session", "--cap-drop", "ALL", "--clearenv"]
    roots = [Path(p) for p in ("/usr", "/bin", "/lib", "/lib64", "/etc/alternatives", "/etc/ld.so.cache")]
    # the verifier's python (the declared runtime): literal and resolved prefixes, plus every link on the interpreter's
    # symlink chain (uv-managed venvs point at a symlinked install directory)
    roots += [Path(sys.prefix), Path(sys.prefix).resolve(), Path(sys.base_prefix), Path(sys.base_prefix).resolve()]
    link = Path(sys.executable)
    while link.is_symlink():
        link = Path(os.readlink(link)) if os.path.isabs(os.readlink(link)) else link.parent / os.readlink(link)
        roots.append(link.parent.parent)
    mounted = []
    for path in roots:
        if path.exists() and not any(path == p or path.is_relative_to(p) for p in mounted) and str(path) not in ("/", "/home"):
            args += ["--ro-bind", str(path), str(path)]
            mounted.append(path)
    args += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/etc", "--dir", "/home/runner"]
    for device in sorted(Path("/dev").glob("nvidia*")):
        args += ["--dev-bind", str(device), str(device)]
    # host CUDA driver libraries (the CUDA runtime itself lives in the python prefix)
    try:
        cache = subprocess.check_output(["/sbin/ldconfig", "-p"], text=True)
    except Exception:  # noqa: BLE001
        cache = ""
    args += ["--dir", "/tmp/driver-libs"]
    seen = set()
    for line in cache.splitlines():
        words = line.split()
        if words and words[0].startswith(("libcuda.", "libnvidia-")) and "x86-64" in line:
            name, path = words[0], Path(words[-1])
            if name not in seen and path.is_file():
                args += ["--ro-bind", str(path.resolve()), f"/tmp/driver-libs/{name}"]
                seen.add(name)
    args += ["--ro-bind", str(sealed), "/submission", "--ro-bind", str(inp), "/input", "--bind", str(outp), "/output", "--chdir", "/submission"]
    # cuDNN and the other CUDA libraries shipped as pip packages dlopen their sublibraries at first use
    # (cuDNN 9 splits conv into libcudnn_engines_precompiled.so and friends). Outside the namespace the
    # loader finds them through torch's rpath; --clearenv drops that, and a convolution then dies with
    # CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED. Put every nvidia/*/lib on the path explicitly.
    nvidia_libs = sorted({str(d) for prefix in {sys.prefix, str(Path(sys.prefix).resolve())}
                          for d in Path(prefix).glob("lib/python*/site-packages/nvidia/*/lib") if d.is_dir()})
    ld_path = ":".join(["/tmp/driver-libs", *nvidia_libs, "/usr/lib/x86_64-linux-gnu", "/usr/lib64"])
    env = {"PATH": f"{sys.prefix}/bin:/usr/bin:/bin", "HOME": "/home/runner", "TMPDIR": "/tmp", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "PIP_NO_INDEX": "1", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8"),
           "LD_LIBRARY_PATH": ld_path,
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")}
    for k, v in env.items():
        args += ["--setenv", k, v]
    return args + ["bash", "infer.sh", "/input", "/output"]


def bwrap_available() -> bool:
    if os.environ.get("TIMESERIES_NO_BWRAP"):
        return False
    if not shutil.which("bwrap"):
        return False
    return unshare_flags() is not None   # user namespaces may be disabled (Docker's default seccomp profile, other runtimes)


def bwrap_detail() -> str:
    flags = unshare_flags()
    return "all namespaces" if flags == UNSHARE_ALL else ("no network namespace (container policy keeps the package offline)" if flags else "unavailable")


RUNNER_UID = int(os.environ.get("TIMESERIES_RUNNER_UID", "10001"))
RUNNER_GID = int(os.environ.get("TIMESERIES_RUNNER_GID", str(RUNNER_UID)))


def uid_drop_available() -> bool:
    """Only root can become another uid; that is how Harbor runs a verifier unless told otherwise."""
    return os.geteuid() == 0 and RUNNER_UID != 0 and not os.environ.get("TIMESERIES_NO_UID_DROP")


def select_mode() -> str:
    return "bwrap" if bwrap_available() else ("uid" if uid_drop_available() else "subprocess")


def _make_readable(path: Path):
    """The runner uid must read the sealed package and the staged input (they are root-owned). chmod -R walks any
    depth; the sealed tree's shape is the package's choice."""
    subprocess.run(["chmod", "-R", "a+rX", "--", str(path)], capture_output=True, timeout=600)


def _drop_privileges():
    os.setgroups([])
    os.setgid(RUNNER_GID)
    os.setuid(RUNNER_UID)


def _rm_rf(path) -> None:
    """Remove a tree whatever its depth (rm's fts walk does not recurse on the Python stack, and never raises here)."""
    subprocess.run(["rm", "-rf", "--one-file-system", "--", str(path)], capture_output=True, timeout=600)


_RUNNER_WRITABLE = None


def runner_writable_dirs() -> list[str]:
    """Every directory the runner uid can write outside the verifier's own, found once by a scan run as the
    runner before its first call (then the verifier's per-call directories are still root-owned): /tmp, /var/tmp,
    its passwd home, /run/lock and whatever else the image leaves writable. uid mode has no mount namespace, so
    these persist across calls and are what sweep_runner_files empties after each one."""
    global _RUNNER_WRITABLE
    if _RUNNER_WRITABLE is None:
        found = set()
        try:
            scan = subprocess.run(["find", "/", "(", "-path", "/proc", "-o", "-path", "/sys", "-o", "-path", "/dev", ")",
                                   "-prune", "-o", "-type", "d", "-writable", "-print", "-prune"],
                                  preexec_fn=_drop_privileges, capture_output=True, timeout=600)
            found.update(l for l in scan.stdout.decode(errors="replace").splitlines() if l.startswith("/"))
        except Exception:  # noqa: BLE001 - the defaults below still apply
            pass
        for d in ("/tmp", "/var/tmp", "/dev/shm", "/dev/mqueue", "/run/lock"):
            if os.path.isdir(d):
                found.add(d)
        _RUNNER_WRITABLE = sorted(found)
    return list(_RUNNER_WRITABLE)


def sweep_runner_files(roots=None, exclude=()) -> int:
    """Delete everything the runner uid owns under `roots` (default: runner_writable_dirs()), skipping paths under
    `exclude`. find and rm walk any depth. Returns the number of top-level entries removed; a no-op unless root."""
    if os.geteuid() != 0:
        return 0
    removed = 0
    for root in (runner_writable_dirs() if roots is None else roots):
        if not os.path.isdir(root):
            continue
        cmd = ["find", root, "-mindepth", "1"]
        for e in exclude:
            cmd += ["(", "-path", str(Path(e).resolve()), "-prune", ")", "-o"]
        cmd += ["-user", str(RUNNER_UID), "-prune", "-print0", "-exec", "rm", "-rf", "--one-file-system", "--", "{}", "+"]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=600)
            removed += r.stdout.count(b"\0")
        except Exception:  # noqa: BLE001 - a sweep failure must not take the verifier down
            pass
    return removed


def _runner_pids() -> list[int]:
    """Live processes that hold the runner uid (real, effective, saved or filesystem), zombies excluded."""
    pids = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/status") as f:
                fields = dict(line.split(":", 1) for line in f.read().splitlines() if ":" in line)
        except OSError:
            continue
        if fields.get("State", "").strip().startswith("Z"):
            continue
        if str(RUNNER_UID) in fields.get("Uid", "").split():
            pids.append(int(d))
    return pids


def kill_runner_processes(wait: float = 10.0) -> None:
    """SIGKILL every process of the runner uid. uid mode has no pid namespace, so anything a package leaves in the
    background (setsid, double fork) would otherwise outlive its call and carry state into the next one. `kill -1`
    run as the runner reaches all of them at once; root then kills stragglers. Raises if any survive."""
    import signal

    deadline = time.time() + wait
    while True:
        pids = _runner_pids()
        if not pids:
            return
        if time.time() > deadline:
            raise RuntimeError(f"runner-uid processes survived SIGKILL: {pids[:10]}")
        try:
            subprocess.run([sys.executable, "-c", "import os, signal; os.kill(-1, signal.SIGKILL)"],
                           preexec_fn=_drop_privileges, capture_output=True, timeout=30)
        except Exception:  # noqa: BLE001
            pass
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(0.05)


def remove_runner_ipc() -> int:
    """Remove SysV shared memory, message queues and semaphores the runner uid owns (not files, so no sweep sees
    them; uid mode has no IPC namespace). POSIX shm and mqueues live in /dev/shm and /dev/mqueue, which the sweep
    covers. Returns the number removed."""
    removed = 0
    for table, flag, key in (("shm", "-m", "shmid"), ("msg", "-q", "msqid"), ("sem", "-s", "semid")):
        try:
            lines = Path(f"/proc/sysvipc/{table}").read_text().splitlines()
        except OSError:
            continue
        if not lines:
            continue
        head = lines[0].split()
        for line in lines[1:]:
            cols = line.split()
            row = dict(zip(head, cols))
            if row.get("uid") == str(RUNNER_UID) or row.get("cuid") == str(RUNNER_UID):
                # IPC_RMID needs ownership or CAP_SYS_ADMIN, which Docker withholds from root: remove as the
                # runner (it created or owns the object) and fall back to root outside such containers
                rm = ["ipcrm", flag, row[key]]
                if (subprocess.run(rm, preexec_fn=_drop_privileges, capture_output=True).returncode == 0
                        or subprocess.run(rm, capture_output=True).returncode == 0):
                    removed += 1
    return removed


def reclaim(path: Path) -> None:
    """Give a directory the runner wrote back to root (recursively, links not followed) with mode 0700, so no later
    call as the runner can read or change what an earlier one produced."""
    path = Path(path)
    if not path.exists():
        return
    subprocess.run(["chown", "-R", "-h", "0:0", "--", str(path)], capture_output=True, timeout=600)
    os.chmod(path, 0o700)


def after_uid_call(outp: Path, home: Path, protect=()) -> dict:
    """Everything that must happen between two uid-mode calls, in order: no runner process left, its output and
    home back to root, its files in the shared writable directories and its IPC objects gone."""
    kill_runner_processes()
    reclaim(outp)
    reclaim(home)
    return {"swept": sweep_runner_files(exclude=protect), "ipc_removed": remove_runner_ipc()}


def runner_can_reach(path: Path) -> bool:
    """Whether the runner uid could get at `path`: for a directory, search permission is enough (file names under
    the data root are public, so a traversable directory is a readable one); for a file, read permission."""
    path = Path(path)
    if not path.exists():
        return False
    flag = "-x" if path.is_dir() else "-r"
    probe = subprocess.run(["test", flag, str(path)], preexec_fn=_drop_privileges, capture_output=True, timeout=30)
    return probe.returncode == 0


def run_isolated(sealed: Path, inp: Path, outp: Path, timeout: float, log, use_bwrap: bool | None = None,
                 fallback_env: dict | None = None, mode: str | None = None, protect=()):
    """Returns (exit code or None on timeout, seconds, mode). On timeout the whole process group is killed.

    `mode` forces "bwrap", "uid" or "subprocess"; `use_bwrap` is the older switch and is kept for callers. In uid
    mode every call ends with after_uid_call (no runner process, output and home root-owned, shared writable
    directories and IPC emptied); `protect` lists verifier directories the sweep must not enter."""
    import signal

    if mode is None:
        mode = "bwrap" if use_bwrap else ("subprocess" if use_bwrap is False else select_mode())
    preexec = None
    if mode == "bwrap":
        cmd, cwd, env = _bwrap_command(Path(sealed).resolve(), Path(inp).resolve(), Path(outp).resolve()), None, None
    else:
        cmd, cwd = ["bash", "infer.sh", str(inp), str(outp)], str(sealed)
        env = dict(fallback_env) if fallback_env else {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "HOME": str(Path(outp).parent / "home"),
            "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8"), "HF_HUB_OFFLINE": "1"}
        for k in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
            if k in os.environ and k not in env:
                env[k] = os.environ[k]
        env.setdefault("PIP_NO_INDEX", "1")
        home = Path(env.get("HOME", "/tmp"))
        home.mkdir(parents=True, exist_ok=True)
        if mode == "uid":
            if not uid_drop_available():
                raise RuntimeError("uid isolation needs root")
            runner_writable_dirs()    # scanned before anything of the verifier's is handed to the runner
            # the runner owns its home and the output directory; it may only read the package and the input
            for p in (home, Path(outp)):
                os.chown(p, RUNNER_UID, RUNNER_GID)
                p.chmod(0o755)
            _make_readable(Path(sealed))
            _make_readable(Path(inp))
            env["TMPDIR"] = str(home)
            preexec = _drop_privileges
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            preexec_fn=preexec)
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:    # the package may have forked workers; take the whole session down
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait()
        rc = None
    seconds = time.time() - t0
    if mode == "uid":
        after_uid_call(Path(outp), Path(env.get("HOME", "/nonexistent")), protect=protect)
    return rc, seconds, mode
