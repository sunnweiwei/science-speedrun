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
from requests_io import write_conditioned_request, write_monomer_request


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


def visible_gpus():
    """Return metadata for all GPUs visible to this process via CUDA_VISIBLE_DEVICES (or all host GPUs if unset)."""
    info = ElementTree.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"]))
    driver = info.findtext("driver_version")
    all_gpus = [
        {
            "index": index,
            "name": gpu.findtext("product_name"),
            "uuid": gpu.findtext("uuid"),
            "driver": driver,
            "device_file": f"/dev/nvidia{int(gpu.findtext('minor_number'))}",
        }
        for index, gpu in enumerate(info.findall("gpu"))
    ]
    raw_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_mask is None or not raw_mask.strip():
        return all_gpus
    tokens = [t.strip() for t in raw_mask.split(",") if t.strip()]
    selected = []
    for tok in tokens:
        matched = next((g for g in all_gpus if tok in (str(g["index"]), g["uuid"])), None)
        if matched is None:
            raise RuntimeError(f"Requested GPU {tok!r} in CUDA_VISIBLE_DEVICES was not found")
        if matched["uuid"] not in {g["uuid"] for g in selected}:
            selected.append(matched)
    return selected


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


def verifier_runtime():
    directory = Path(__file__).resolve().parents[2] / "tests"
    if not (directory / "rfd4_private.py").is_file():
        raise RuntimeError("Private replay requires the verifier environment")
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    import rfd4_private
    return rfd4_private


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
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= max(8 * 2**20, length * 8192):
            paths.append(None)
            continue
        try:
            shutil.copyfile(file, sealed / name, follow_symlinks=False)
        except PermissionError:
            paths.append(None)
            continue
        paths.append(sealed / name)
    return paths


def invoke_program(package, request, output, assets, seconds, log_path, *, private, gpu_info=None):
    """Dispatch an opaque request. The task adapter owns its payload and output semantics."""
    if (package / "infer.py").is_symlink() or not (package / "infer.py").is_file():
        raise InvalidPolicy("Expected a regular infer.py entry point")
    if private:
        if not shutil.which("bwrap"):
            raise RuntimeError("Private program runtime requires bubblewrap")
        namespace = verifier_runtime().private_namespace(package, request, output, assets, gpu_info=gpu_info)
        # A broken namespace setup is an infrastructure failure, including on GPU hosts.
        subprocess.run(namespace + ["/bin/true"], stdin=subprocess.DEVNULL, check=True, timeout=30)
        command = namespace + ["/usr/local/bin/python", "/submission/infer.py", "--request",
                               f"/request/{request.name}", "--output", "/output"]
    else:
        command = [sys.executable, str(package / "infer.py"), "--request", str(request), "--output", str(output)]
    env = dict(os.environ, RFD4_MODEL=str(assets / "rfd4"), DATA_PATH=str(assets),
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
               HF_HUB_OFFLINE="1", WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1")
    if gpu_info is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_info["uuid"]
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


def run_conditioned_program(package, assets, work, input_cif, request_id, slots, seed, *, private=False,
                            seconds_per_slot=SECONDS_PER_SLOT, metadata=None, gpu_info=None):
    work.mkdir(parents=True, exist_ok=False)
    seconds = slots * seconds_per_slot
    request = write_conditioned_request(work / "request", input_cif, request_id, slots, seed, seconds, metadata)
    output = work / "output"
    output.mkdir()
    costs = invoke_program(package, request, output, assets, seconds, work / "inference.log",
                           private=private, gpu_info=gpu_info)
    if costs["program_exit_code"] != 0 or costs["timed_out"]:
        return [], costs
    return collect_outputs(output, slots, 1024, work / "accepted"), costs
