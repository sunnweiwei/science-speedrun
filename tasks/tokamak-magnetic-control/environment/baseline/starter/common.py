"""Shared pieces of the starter: observation scaling, NumPy actor (deployable), torch nets.

The deployable actor is a plain MLP evaluated with NumPy so that the published package
needs no PyTorch. Training uses torch; the exported weights are the same tensors.
"""

import json
from pathlib import Path

import numpy as np

# Observation layout of tokagym-mastu/v0.2 (see contract.md; unchanged since v0.1 apart from the target order).
N_FLOOP, N_PICKUP, N_COIL = 102, 398, 12
N_PROBE = N_FLOOP + N_PICKUP
SL_PROBES = slice(0, N_PROBE)
SL_COILS = slice(N_PROBE, N_PROBE + N_COIL)
SL_IP = N_PROBE + N_COIL
SL_TARGETS = slice(N_PROBE + N_COIL + 1, None)   # control points, X-points, flags, Ip target, time
N_OBS = 585


def make_scaler(probe_scale):
    """Fixed per-signal scales: probes by reference magnitude, currents 1e4 A, Ip 1e6 A, geometry 1 m."""
    scale = np.ones(N_OBS)
    scale[SL_PROBES] = np.asarray(probe_scale, dtype=float)
    scale[SL_COILS] = 1e4
    scale[SL_IP] = 1e6
    tail = np.ones(N_OBS - (N_PROBE + N_COIL + 1))
    tail[-2] = 1e6                      # target Ip
    scale[SL_TARGETS] = tail
    return scale


class Scaler:
    def __init__(self, scale, offset=None):
        self.scale = np.asarray(scale, dtype=np.float64)
        self.offset = np.zeros_like(self.scale) if offset is None else np.asarray(offset, dtype=np.float64)

    def __call__(self, obs):
        return (np.asarray(obs, dtype=np.float64) - self.offset) / self.scale

    def to_dict(self):
        return {"scale": self.scale.tolist(), "offset": self.offset.tolist()}


def mlp_forward(params, x, activation="relu"):
    """params: list of (W, b); returns pre-tanh output."""
    h = x
    for i, (W, b) in enumerate(params):
        h = h @ W + b
        if i < len(params) - 1:
            h = np.maximum(h, 0.0) if activation == "relu" else np.tanh(h)
    return h


class NumpyActor:
    """Deterministic actor: obs -> scaled features -> MLP -> tanh -> voltages.

    Inference runs in float32 on 1-D vectors (the weights are stored in float32): about 2x faster
    than float64 batches, which matters under the 0.5 ms latency gate on a loaded host.
    """

    def __init__(self, params, scaler, vmax, action_scale=1.0):
        self.params = [(np.asarray(W, dtype=np.float32), np.asarray(b, dtype=np.float32)) for W, b in params]
        self.scaler = scaler
        self.offset = np.asarray(scaler.offset, dtype=np.float32)
        self.inv_scale = (1.0 / np.asarray(scaler.scale, dtype=np.float64)).astype(np.float32)
        self.vmax = np.asarray(vmax, dtype=np.float64)
        self.action_scale = float(action_scale)
        self.out_scale = self.vmax * self.action_scale
        self.n_act = len(self.vmax)

    def __call__(self, obs):
        h = (np.asarray(obs, dtype=np.float32) - self.offset) * self.inv_scale
        last = len(self.params) - 1
        for i, (W, b) in enumerate(self.params):
            h = h @ W + b
            if i < last:
                np.maximum(h, 0.0, out=h)
        return np.tanh(h[: self.n_act].astype(np.float64)) * self.out_scale

    @classmethod
    def load(cls, package_dir):
        package_dir = Path(package_dir)
        cfg = json.loads((package_dir / "config.json").read_text())
        data = np.load(package_dir / "actor.npz")
        params = [(data[f"W{i}"], data[f"b{i}"]) for i in range(cfg["n_layers"])]
        scaler = Scaler(np.array(cfg["scaler"]["scale"]), np.array(cfg["scaler"]["offset"]))
        return cls(params, scaler, np.array(cfg["vmax"]), cfg.get("action_scale", 1.0))


def save_actor(package_dir, params, scaler, vmax, action_scale, extra=None):
    package_dir = Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for i, (W, b) in enumerate(params):
        arrays[f"W{i}"] = np.asarray(W, dtype=np.float32)
        arrays[f"b{i}"] = np.asarray(b, dtype=np.float32)
    cfg = {"n_layers": len(params), "scaler": scaler.to_dict(), "vmax": np.asarray(vmax).tolist(),
           "action_scale": action_scale, "n_obs": N_OBS, "n_act": len(vmax)}
    cfg.update(extra or {})
    tmp_cfg = package_dir / "config.json.tmp"
    tmp_cfg.write_text(json.dumps(cfg))
    tmp_cfg.replace(package_dir / "config.json")
    tmp_npz = package_dir / "actor.npz.tmp.npz"
    np.savez(tmp_npz, **arrays)
    tmp_npz.replace(package_dir / "actor.npz")   # atomic: readers see old or new file, never partial
    return cfg
