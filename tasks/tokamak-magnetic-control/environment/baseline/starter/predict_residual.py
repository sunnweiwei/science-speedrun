"""Resident tokamak-policy/v1 server for a residual policy: deployable base controller + NumPy actor residual.

Package layout: policy.py (this file), pid_core.py + its resources (the base package's policy.py
and files, e.g. baseline/shape_vc/package), actor.npz + config.json (residual actor), common.py.
Action = clip(base(obs) + tanh(actor(obs)) * residual_scale * vmax, limits).
"""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import NumpyActor  # noqa: E402

PROTOCOL = "tokamak-policy/v1"


def load_core():
    spec = importlib.util.spec_from_file_location("pid_core", str(HERE / "pid_core.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.Controller()


def main():
    cfg = json.loads((HERE / "config.json").read_text())
    scale = float(cfg.get("residual_scale", 0.2))
    actor = NumpyActor.load(HERE)
    core = load_core()
    low = high = None
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("protocol") != PROTOCOL:
            raise SystemExit("unsupported protocol")
        reply = {"request_id": req["request_id"]}
        if req["op"] == "reset":
            low, high = np.asarray(req["action_low"]), np.asarray(req["action_high"])
            core.reset(req["action_low"], req["action_high"])
            reply["ready"] = True
        elif req["op"] == "act":
            obs = np.asarray(req["observation"], dtype=np.float64)   # converted once, shared
            base = core.act(obs)
            res = actor(obs) * scale              # actor() already returns tanh(.) * vmax
            reply["action"] = np.clip(base + res, low, high).tolist()
        elif req["op"] == "close":
            reply["closed"] = True
            print(json.dumps(reply), flush=True)
            return
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
