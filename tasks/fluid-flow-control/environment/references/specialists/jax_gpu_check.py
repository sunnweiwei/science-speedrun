"""Minimal QR and loaded-library diagnostic; no physics or model training."""

import argparse
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    result = {"devices": str(jax.devices()), "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
              "XLA_FLAGS": os.environ.get("XLA_FLAGS"), "versions": {}, "checks": []}
    for name in ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "nvidia-cusolver-cu12",
                 "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12"):
        try:
            result["versions"][name] = version(name)
        except PackageNotFoundError:
            result["versions"][name] = None
    started = time.monotonic()
    try:
        for shape in ((64, 256), (256, 256), (256, 4)):
            def generate(key):
                array = jax.random.normal(key, shape)
                if shape[0] < shape[1]:
                    array = array.T
                return jnp.linalg.qr(array, mode="reduced")
            q, r = jax.jit(generate)(jax.random.PRNGKey(30))
            q.block_until_ready()
            # Verify QR independently in host FP64; default GPU matrix multiply
            # may itself use reduced mantissa precision and is not a QR oracle.
            host = np.asarray(q).astype(np.float64)
            error = float(np.max(np.abs(host.T @ host - np.eye(host.shape[1]))))
            gpu_error = float(jnp.max(jnp.abs(q.T @ q - jnp.eye(q.shape[1]))))
            result["checks"].append({"shape": shape, "q_finite": bool(np.isfinite(host).all()),
                "orthogonality_error": error if np.isfinite(error) else None,
                "default_gpu_matmul_error": gpu_error if np.isfinite(gpu_error) else None})
            if not np.isfinite(host).all() or error > 1e-4:
                raise RuntimeError("Nonfinite or inaccurate GPU QR")
        result["status"] = "gpu-qr-passed"
    except Exception as error:
        result.update(status="gpu-qr-failed", error=str(error))
    result["loaded_cuda_libraries"] = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
        if any(name in line for name in ("libcusolver", "libcublas", "libcudart", "libcuda.so", "libnvJitLink"))})
    result["wall_seconds"] = time.monotonic() - started
    (args.output / "diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "gpu-qr-passed":
        raise RuntimeError("GPU QR validation failed; see diagnostic.json")


if __name__ == "__main__":
    main()
