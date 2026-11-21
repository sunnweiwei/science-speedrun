"""Execute a submission once per request and collect its fixed output slots."""
import hashlib
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
from xml.etree import ElementTree

from policy import InvalidPolicy, SECONDS_PER_SLOT
from requests_io import write_monomer_request


def package_digest(package):
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.is_symlink():
            payload = b"link:" + os.fsencode(os.readlink(path))
        elif path.is_file():
            with path.open("rb") as f:
                payload = b"file:" + hashlib.file_digest(f, "sha256").digest()
        elif path.is_dir():
            continue
        else:
            raise InvalidPolicy("The package contains a special file")
        digest.update(os.fsencode(path.relative_to(package)) + b"\0" + hashlib.sha256(payload).digest())
    return digest.hexdigest()


def gpu_device():
    # The NVIDIA runtime may remap device minors inside a container.
    info = ElementTree.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"]))
    gpus = info.findall("gpu")
    mask = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    for index, gpu in enumerate(gpus):
        uuid = gpu.findtext("uuid")
        if mask in (str(index), uuid):
            os.environ["CUDA_VISIBLE_DEVICES"] = uuid
            return {"name": gpu.findtext("product_name"), "uuid": uuid, "driver": info.findtext("driver_version"),
                    "device_file": f"/dev/nvidia{int(gpu.findtext('minor_number'))}"}
    raise RuntimeError("Select one allocated GPU through CUDA_VISIBLE_DEVICES")


def private_namespace(package, request, output, assets):
    """The trusted verifier exposes only this request, the runtime and frozen assets."""
    command = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
               "--cap-drop", "ALL", "--clearenv"]
    for root in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/sys", "/etc/ld.so.cache", "/etc/alternatives",
                 "/opt/af3", "/opt/af3env"):
        if Path(root).exists():
            command += ["--ro-bind", root, root]
    command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    nodes = sorted(Path("/dev").glob("nvidia*"))
    selected = gpu_device()["device_file"] if any(d.name[6:].isdigit() for d in nodes) else None
    for dev in nodes:
        if not dev.name[6:].isdigit() or str(dev) == selected:
            command += ["--dev-bind", str(dev), str(dev)]
    command += ["--ro-bind", str(assets / "rfd4"), "/assets/rfd4",
                "--ro-bind", str(assets / "af3"), "/assets/af3",
                "--ro-bind", str(package), "/submission", "--ro-bind", str(request.parent), "/request",
                "--bind", str(output), "/output", "--chdir", "/submission"]
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "RFD4_MODEL": "/assets/rfd4", "DATA_PATH": "/assets",
           "HOME": "/tmp", "TMPDIR": "/tmp", "HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled",
           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
           "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false"}
    for key in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
        if key in os.environ:
            env[key] = os.environ[key]
    for key, value in env.items():
        command += ["--setenv", key, value]
    return command + ["--"]


def collect_outputs(output, slots, length, sealed):
    # Child processes have exited before the verifier takes ownership of these files.
    sealed.mkdir()
    paths = []
    for index in range(slots):
        name = f"slot-{index:06d}.cif"
        file = output / name
        try:
            info = file.lstat()
        except FileNotFoundError:
            paths.append(None)
            continue
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= max(2**20, length * 8192):
            paths.append(None)
            continue
        try:
            shutil.copyfile(file, sealed / name, follow_symlinks=False)
        except PermissionError:
            paths.append(None)
            continue
        paths.append(sealed / name)
    return paths


def invoke_program(package, request, output, assets, seconds, log_path, *, private):
    """Dispatch an opaque request. The task adapter owns its payload and output semantics."""
    if (package / "infer.py").is_symlink() or not (package / "infer.py").is_file():
        raise InvalidPolicy("Expected a regular infer.py entry point")
    if private:
        if not shutil.which("bwrap"):
            raise RuntimeError("Private program runtime requires bubblewrap")
        namespace = private_namespace(package, request, output, assets)
        # A broken namespace setup is an infrastructure failure, including on GPU hosts.
        subprocess.run(namespace + ["/bin/true"], stdin=subprocess.DEVNULL, check=True, timeout=30)
        command = namespace + ["/usr/local/bin/python", "/submission/infer.py", "--request",
                               f"/request/{request.name}", "--output", "/output"]
    else:
        command = [sys.executable, str(package / "infer.py"), "--request", str(request), "--output", str(output)]
    env = dict(os.environ, RFD4_MODEL=str(assets / "rfd4"), DATA_PATH=str(assets),
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
               HF_HUB_OFFLINE="1", WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1")
    start = time.monotonic()
    timed_out = False
    with log_path.open("w") as log:
        with subprocess.Popen(command, cwd=package, env=env, stdin=subprocess.DEVNULL, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True) as process:
            try:
                status = process.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                status = process.wait()
    elapsed = time.monotonic() - start
    return {"generation_seconds": elapsed, "timed_out": timed_out, "program_exit_code": status}


def run_program(package, assets, work, length, slots, seed, *, private, seconds_per_slot=SECONDS_PER_SLOT,
                descriptor=None):
    work.mkdir(parents=True, exist_ok=False)
    seconds = slots * seconds_per_slot
    request = write_monomer_request(work / "request", assets, length, slots, seed, seconds, descriptor)
    output = work / "output"
    output.mkdir()
    costs = invoke_program(package, request, output, assets, seconds, work / "inference.log", private=private)
    if costs["program_exit_code"] != 0 or costs["timed_out"]:
        return [], costs
    return collect_outputs(output, slots, length, work / "accepted"), costs
