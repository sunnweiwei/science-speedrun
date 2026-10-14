"""Run WN2 in a fresh offline PID/user/mount namespace with its model runtime.

Only the declared runtime, model, public inputs, GPU devices and output mount
cross this boundary. Run this launcher on the host, with either its declared
Python environment or WEATHER_RUNTIME_ROOT pointing to an unpacked task image.
"""
import os
from pathlib import Path
import signal
import subprocess
import sys


def command(submission, public, output):
    args = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
            "--cap-drop", "ALL", "--clearenv"]
    runtime_root = os.environ.get("WEATHER_RUNTIME_ROOT")
    if runtime_root:
        root = Path(runtime_root).resolve()
        if root == Path("/") or not (root / "opt/weather-python/bin/python3").is_file():
            raise RuntimeError("WEATHER_RUNTIME_ROOT must be an unpacked task image")
        args += ["--ro-bind", str(root), "/", "--ro-bind", "/sys", "/sys"]
        # Existing image directories avoid creating mountpoints on a read-only root.
        model_mount, input_mount, output_mount = "/mnt", "/media", "/run"
        python_prefix = "/opt/weather-python"
    else:
        # Evaluation uses a separate NumPy/xarray environment. Explicitly bind
        # the declared model interpreter instead of inheriting the scorer's.
        model_python = Path(os.environ.get("WEATHER_MODEL_PYTHON", "/opt/weather-python/bin/python"))
        if not model_python.is_file():
            raise RuntimeError("Declared WN2 model Python is unavailable")
        model_prefix = model_python.parent.parent
        roots = [Path(p) for p in ("/usr", "/bin", "/lib", "/lib64", "/sys",
                                          "/etc/alternatives")]
        roots += [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(), model_prefix,
                  Path("/opt/weathernext"), Path("/opt/xarray-jax")]
        mounted = []
        for path in roots:
            if path.exists() and not any(path == p or path.is_relative_to(p) for p in mounted):
                args += ["--ro-bind", str(path), str(path)]
                mounted.append(path)
        model_mount, input_mount, output_mount = "/submission", "/input", "/output"
        python_prefix = str(model_prefix)
    args += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/etc"]
    if not runtime_root and Path("/etc/ld.so.cache").exists():
        args += ["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"]
    for device in sorted(Path("/dev").glob("nvidia*")):
        args += ["--dev-bind", str(device), str(device)]
    args += ["--ro-bind", str(submission), model_mount, "--ro-bind", str(public), input_mount,
             "--bind", str(output), output_mount, "--chdir", model_mount]
    environment = dict(PATH=f"{python_prefix}/bin:/usr/bin:/bin", HOME="/tmp", TMPDIR="/tmp",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PIP_NO_INDEX="1",
                       HF_HUB_OFFLINE="1", XLA_PYTHON_CLIENT_PREALLOCATE="false",
                       OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if not runtime_root:
        # Debian/Ubuntu's /lib is commonly a symlink to /usr/lib. Bubblewrap
        # materializes mount points as directories, so retain the canonical
        # multiarch loader path explicitly inside the empty namespace root.
        environment["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/usr/lib64"
    if environment["CUDA_VISIBLE_DEVICES"]:
        # JAX's CUDA libraries are in the image; only the host driver crosses
        # the boundary. Use SONAME paths from the host loader, not an entire
        # host library directory that would override the pinned image runtime.
        cache = subprocess.check_output(["/sbin/ldconfig", "-p"], text=True)
        args += ["--dir", "/tmp/driver-libs"]
        names = set()
        for line in cache.splitlines():
            words = line.split()
            if words and words[0].startswith(("libcuda.", "libnvidia-")) and "x86-64" in line:
                name, path = words[0], Path(words[-1])
                if name not in names and path.is_file():
                    args += ["--ro-bind", str(path.resolve()), "/tmp/driver-libs/" + name]
                    names.add(name)
        # Apptainer exposes its injected drivers here rather than registering
        # them in the container's ldconfig cache. Copy only named driver files
        # into the inner namespace, retaining the pinned CUDA toolkit runtime.
        injected = Path("/.singularity.d/libs")
        if injected.is_dir():
            for path in sorted(injected.iterdir()):
                if path.name.startswith(("libcuda.", "libnvidia-")) and path.is_file() and path.name not in names:
                    args += ["--ro-bind", str(path.resolve()), "/tmp/driver-libs/" + path.name]
                    names.add(path.name)
        if "libcuda.so.1" not in names:
            raise RuntimeError("allocated GPU driver libcuda.so.1 is unavailable")
        previous = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = "/tmp/driver-libs" + (":" + previous if previous else "")
    for key, value in environment.items():
        args += ["--setenv", key, value]
    return args + ["bash", model_mount + "/infer.sh", input_mount, output_mount]


def run(submission, public, work, timeout):
    output = work / "untrusted-output"
    output.mkdir()
    process = None
    try:
        with (work / "inference.log").open("wb") as log:
            process = subprocess.Popen(command(submission, public, output), stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, process.args)
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    # Exiting bwrap destroys the child PID namespace, including daemonized children.
    return output
