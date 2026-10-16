"""Standalone frozen Flax actor; no CFD, author checkout or training state."""

import hashlib
import json
import math
from pathlib import Path
import sys

import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


class Actor(nn.Module):
    actions: int
    activation: str

    @nn.compact
    def __call__(self, observation):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        value = activation(nn.Dense(256, name="Dense_0")(observation))
        value = activation(nn.Dense(256, name="Dense_1")(value))
        return nn.Dense(self.actions, name="Dense_2")(value)


def main():
    root = Path(__file__).resolve().parent
    cfg = json.loads((root / "config.json").read_text())
    payload = (root / "model.msgpack").read_bytes()
    if hashlib.sha256(payload).hexdigest() != cfg["model_sha256"]:
        raise ValueError("Reference model checksum mismatch")
    jax.config.update("jax_enable_x64", cfg["float64"])
    weights = flax.serialization.msgpack_restore(payload)
    actor = Actor(cfg["action_size"], cfg["activation"])
    infer = jax.jit(lambda obs: jnp.clip(actor.apply(weights, obs), cfg["action_low"], cfg["action_high"]))
    ready = False
    for line in sys.stdin:
        request = json.loads(line)
        if request["protocol"] != "hydrogym-policy/v1":
            raise ValueError("Unsupported policy protocol")
        reply = {"request_id": request["request_id"]}
        if request["op"] == "reset":
            if (math.prod(request["observation_shape"]) != cfg["observation_size"]
                    or math.prod(request["action_shape"]) != cfg["action_size"]
                    or request["action_low"] != [cfg["action_low"]] * cfg["action_size"]
                    or request["action_high"] != [cfg["action_high"]] * cfg["action_size"]):
                raise ValueError("Native shapes/bounds differ from this reference actor")
            ready = True
            reply["ready"] = True
        elif request["op"] == "act":
            if not ready or len(request["observation"]) != cfg["observation_size"]:
                raise ValueError("Missing reset or incorrect observation shape")
            obs = np.asarray(request["observation"], dtype=np.float64 if cfg["float64"] else np.float32)
            if not np.isfinite(obs).all():
                raise ValueError("Nonfinite observation")
            action = np.asarray(infer(jnp.asarray(obs)))
            if not np.isfinite(action).all():
                raise ValueError("Nonfinite reference action")
            reply["action"] = action.tolist()
        elif request["op"] == "close":
            reply["closed"] = True
            print(json.dumps(reply, allow_nan=False), flush=True)
            return
        else:
            raise ValueError("Unknown operation")
        print(json.dumps(reply, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
