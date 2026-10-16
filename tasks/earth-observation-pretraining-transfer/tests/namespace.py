"""Offline model replay with only runtime, package, unlabeled inputs and output."""

import os
from pathlib import Path
import sys
import sysconfig


def command(package, inputs, outputs):
    # The interpreter and installed model dependencies are explicitly declared.
    prefix = Path(sys.prefix).resolve()
    base = Path(sys.base_prefix).resolve()
    cmd = [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--clearenv",
    ]
    for name in ["/usr", "/bin", "/lib", "/lib64", "/sys"]:
        if Path(name).exists():
            cmd += ["--ro-bind", name, name]
    # Hide nonstandard /usr subtrees (e.g. shared storage) from offline inference.
    standard = {
        "bin",
        "sbin",
        "lib",
        "lib32",
        "lib64",
        "libexec",
        "local",
        "share",
        "include",
        "src",
        "games",
    }
    for child in Path("/usr").iterdir():
        # External symlink targets are absent unless explicitly mounted below.
        # Do not mount over links such as RHEL's /usr/tmp -> ../var/tmp: their
        # target may not exist inside the isolated filesystem.
        if child.is_dir() and not child.is_symlink() and child.name not in standard:
            cmd += ["--tmpfs", str(child)]
    cmd += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/etc"]
    for path in [base, prefix]:
        cmd += ["--ro-bind", str(path), str(path)]
    if Path("/etc/ld.so.cache").exists():
        cmd += ["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"]
    for path in sorted(Path("/dev").glob("nvidia*")):
        cmd += ["--dev-bind", str(path), str(path)]
    cmd += [
        "--ro-bind",
        str(package),
        "/model",
        "--ro-bind",
        str(inputs),
        "/inputs",
        "--bind",
        str(outputs),
        "/outputs",
        "--chdir",
        "/model",
    ]
    site = Path(sysconfig.get_paths()["purelib"])
    runtime_libraries = [
        site / "torch/lib",
        *sorted((site / "nvidia").glob("*/lib")),
        prefix / "lib",
    ]
    runtime_libraries = [str(p) for p in runtime_libraries if p.is_dir()]
    environment = dict(
        PATH=f"{prefix}/bin:/usr/bin:/bin",
        PYTHON=str(prefix / "bin/python"),
        HOME="/tmp",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        PIP_NO_INDEX="1",
        SLURM_JOB_PARTITION=os.environ.get("SLURM_JOB_PARTITION", ""),
        LD_LIBRARY_PATH=":".join(
            runtime_libraries + ["/usr/lib/x86_64-linux-gnu", "/usr/lib64"]
        ),
    )
    # Pass the GPU selection through only when the scheduler set one: an empty CUDA_VISIBLE_DEVICES
    # hides every GPU (Docker/Harbor expose the allocated GPUs without setting the variable).
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    hash_seed = os.environ.get("OLMO_INFERENCE_HASHSEED")
    if hash_seed is not None:
        if not hash_seed.isdigit() or int(hash_seed) >= 2**32:
            raise ValueError("OLMO_INFERENCE_HASHSEED must be a uint32")
        environment["PYTHONHASHSEED"] = hash_seed
    injected = Path("/.singularity.d/libs")
    if injected.is_dir():
        cmd += ["--dir", "/tmp/driver-libs"]
        for path in sorted(injected.iterdir()):
            if path.name.startswith(("libcuda.", "libnvidia-")) and path.is_file():
                cmd += [
                    "--ro-bind",
                    str(path.resolve()),
                    "/tmp/driver-libs/" + path.name,
                ]
        environment["LD_LIBRARY_PATH"] = (
            "/tmp/driver-libs:" + environment["LD_LIBRARY_PATH"]
        )
    for key, value in environment.items():
        cmd += ["--setenv", key, value]
    return cmd + [
        str(prefix / "bin/python"),
        "/model/infer.py",
        "--model",
        "/model",
        "--input",
        "/inputs/inputs.h5",
        "--output",
        "/outputs/features.npz",
    ]


def direct(package, inputs, outputs):
    """Development fallback without Bubblewrap: the same entry point, a clean environment and the
    real paths. It gives no isolation and is never used by the verifier (tests/test.sh)."""
    sandboxed = command(package, inputs, outputs)
    environment = {}
    for i, word in enumerate(sandboxed):
        if word == "--setenv":
            environment[sandboxed[i + 1]] = sandboxed[i + 2]
    environment.pop("LD_LIBRARY_PATH", None)  # the sandbox's library view; use the image's own
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    python = sandboxed[-8]
    argv = [python, str(Path(package) / "infer.py"), "--model", str(package),
            "--input", str(Path(inputs) / "inputs.h5"), "--output", str(Path(outputs) / "features.npz")]
    return argv, environment
