"""Private suite setup for the verifier image. Development uses the shared scorer with public draws."""
import hashlib
import hmac
import os
from pathlib import Path

from program import gpu_device


def suite_seeds(path):
    data = path.read_bytes()
    if len(data) != 32:
        raise RuntimeError("Private seed must contain 32 bytes")
    seed = int.from_bytes(hashlib.sha256(data).digest()[:4], "big") % (2**31 - 1024)
    final_seed = int.from_bytes(hmac.digest(data, b"rfd4-program-final-af3", "sha256")[:4], "big") % 2**31
    return seed, final_seed, hashlib.sha256(data).hexdigest()


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
