"""Standalone TorchScript baseline inference; stdin/stdout contract only."""

import hashlib
import json
import math
from pathlib import Path
import sys

import torch


def main():
    root = Path(__file__).resolve().parent
    cfg = json.loads((root / "config.json").read_text())
    if hashlib.sha256((root / "model.pt").read_bytes()).hexdigest() != cfg["model_sha256"]:
        raise ValueError("Reference model checksum mismatch")
    torch.set_num_threads(1)
    model = torch.jit.load(str(root / "model.pt"), map_location="cpu").eval()
    limits = None
    for line in sys.stdin:
        request = json.loads(line)
        if request["protocol"] != "hydrogym-policy/v1":
            raise ValueError("Unsupported policy protocol")
        reply = {"request_id": request["request_id"]}
        if request["op"] == "reset":
            if math.prod(request["observation_shape"]) != cfg["observation_size"] or math.prod(request["action_shape"]) != cfg["action_size"]:
                raise ValueError("Input shape is outside this reference actor's declared capacity")
            limits = request["action_low"], request["action_high"]
            if limits != ([-1.0] * cfg["action_size"], [1.0] * cfg["action_size"]):
                raise ValueError("Native action bounds differ from the trained reference")
            reply["ready"] = True
        elif request["op"] == "act":
            if limits is None or len(request["observation"]) != cfg["observation_size"]:
                raise ValueError("Missing reset or wrong observation shape")
            obs = torch.tensor(request["observation"], dtype=torch.float32)
            if not torch.isfinite(obs).all():
                raise ValueError("Nonfinite observation")
            with torch.inference_mode():
                action = model(obs.unsqueeze(0))[0]
            if not torch.isfinite(action).all():
                raise ValueError("Nonfinite action")
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
