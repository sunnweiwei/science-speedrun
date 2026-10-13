"""Harbor's built-in Docker environment with GPU allocation on a local NVIDIA host.

Harbor 0.23's `docker` environment reports no GPU capability, so every task whose task.toml asks
for GPUs is refused, and `--extra-docker-compose` overlays are not applied to a separate verifier.
This subclass allocates the task's declared GPU count ([environment] gpus for the agent,
[verifier.environment] gpus for the verifier) to the `main` service of each environment.

    --env harbor_ext.local_gpu_docker:LocalGpuDockerEnvironment
    HARBOR_GPU_DEVICE_IDS=0,1,2,3   optional pool; the first N ids are used. Unset: the first N GPUs.
    HARBOR_GPU_CAP=1                agent-eval sessions: at most this many GPUs per environment.
    HARBOR_SKIP_VERIFIER=1          do not start the separate verifier environment (the trial records
                                    an error after the agent artifacts are collected); the saved
                                    package is scored later with the task's full verifier resources.
"""
import json
import os
import tempfile
from pathlib import Path

from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.docker.docker import DockerEnvironment


class LocalGpuDockerEnvironment(DockerEnvironment):
    @property
    def _is_verifier(self) -> bool:
        return "__verifier__" in str(getattr(self, "session_id", ""))

    async def start(self, force_build: bool):
        if self._is_verifier and os.environ.get("HARBOR_SKIP_VERIFIER") == "1":
            raise RuntimeError("verification deferred: HARBOR_SKIP_VERIFIER=1 (score the saved package "
                               "with the task's full verifier resources)")
        return await super().start(force_build)

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return super().capabilities.model_copy(update={"gpus": True})

    @property
    def _docker_compose_paths(self) -> list[Path]:
        paths = list(super()._docker_compose_paths)
        n = self._effective_gpus
        cap = os.environ.get("HARBOR_GPU_CAP")
        if cap:   # codex-eval: every session gets at most this many GPUs (1), whatever the task declares
            n = min(n, int(cap))
        if n <= 0:
            return paths
        pool = [x for x in os.environ.get("HARBOR_GPU_DEVICE_IDS", "").split(",") if x.strip()]
        if pool and len(pool) < n:
            raise RuntimeError(f"task needs {n} GPU(s); HARBOR_GPU_DEVICE_IDS lists {len(pool)}")
        device = {"driver": "nvidia", "capabilities": ["gpu"]}
        device.update({"device_ids": pool[:n]} if pool else {"count": n})
        overlay = {"services": {"main": {"deploy": {"resources": {"reservations": {"devices": [device]}}}}}}
        if getattr(self, "_gpu_overlay_dir", None) is None:
            self._gpu_overlay_dir = tempfile.TemporaryDirectory()
        path = Path(self._gpu_overlay_dir.name) / "docker-compose-gpus.json"
        path.write_text(json.dumps(overlay))
        return paths + [path]
