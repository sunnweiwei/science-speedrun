"""Private suite setup for the verifier image. Development uses the shared scorer with public draws."""
import hashlib
import hmac
import os
from pathlib import Path

from program import gpu_device


# Quartile-balanced 12-target verification roster (target-disjoint from DEV_4TRACK_TARGETS on PPI, SM, and MCSA).
VERIFY_4TRACK_TARGETS = {
    "ppi_backbone": ("33_TrkA", "37_IL17A", "rfd3_il7ra"),
    "studio_figure_2c_unfixed": (
        "Zn_ideal",
        "s-adenosylmethionine_final_0_Fe2-S2_ideal_combined",
        "thiamine_diphosphate_final_0",
    ),
    "mcsa_fixed": ("M0179_1q3s", "M0315_1ey3", "M0732_1xs1"),
    "dna_unfixed": ("7m5w", "7n5u", "7rte"),
}


def suite_seeds(path):
    data = path.read_bytes()
    if len(data) != 32:
        raise RuntimeError("Private seed must contain 32 bytes")
    seed = int.from_bytes(hashlib.sha256(data).digest()[:4], "big") % (2**31 - 1024)
    final_seed = int.from_bytes(hmac.digest(data, b"rfd4-program-final-af3", "sha256")[:4], "big") % 2**31
    return seed, final_seed, hashlib.sha256(data).hexdigest()


def four_track_seeds(path):
    data = Path(path).read_bytes()
    if len(data) != 32:
        raise RuntimeError("Private seed must contain 32 bytes")
    gen_offset = int.from_bytes(hmac.digest(data, b"rfd4-four-track-gen-v1", "sha256")[:4], "big") % (2**29)
    af3_seed = int.from_bytes(hmac.digest(data, b"rfd4-four-track-af3-v1", "sha256")[:4], "big") % (2**31 - 1)
    return gen_offset, af3_seed, hashlib.sha256(data).hexdigest()


def private_namespace(package, request, output, assets, *, gpu_info=None):
    """The trusted verifier exposes only this request, the runtime and frozen assets."""
    command = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
               "--cap-drop", "ALL", "--clearenv"]
    for root in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/sys", "/etc/ld.so.cache", "/etc/alternatives",
                 "/opt/af3", "/opt/af3env"):
        if Path(root).exists():
            command += ["--ro-bind", root, root]
    command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    nodes = sorted(Path("/dev").glob("nvidia*"))
    active_gpu = gpu_info if gpu_info is not None else (gpu_device() if any(d.name[6:].isdigit() for d in nodes) else None)
    selected = active_gpu["device_file"] if active_gpu is not None else None
    for dev in nodes:
        if not dev.name[6:].isdigit() or str(dev) == selected:
            command += ["--dev-bind", str(dev), str(dev)]
    command += ["--ro-bind", str(assets / "rfd4"), "/assets/rfd4",
                "--ro-bind", str(assets / "af3"), "/assets/af3"]
    if (assets / "of3").is_dir():
        command += ["--ro-bind", str(assets / "of3"), "/assets/of3"]
    command += ["--ro-bind", str(package), "/submission", "--ro-bind", str(request.parent), "/request",
                "--bind", str(output), "/output", "--chdir", "/submission"]
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "RFD4_MODEL": "/assets/rfd4", "DATA_PATH": "/assets",
           "HOME": "/tmp", "TMPDIR": "/tmp", "HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled",
           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
           "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false"}
    for key in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
        if key in os.environ:
            env[key] = os.environ[key]
    if active_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = active_gpu["uuid"]
    for key, value in env.items():
        command += ["--setenv", key, value]
    return command + ["--"]
