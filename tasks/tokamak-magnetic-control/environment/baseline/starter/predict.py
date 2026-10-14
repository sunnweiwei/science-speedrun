"""Resident tokamak-policy/v1 server for the exported NumPy actor (no torch)."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NumpyActor  # noqa: E402

PROTOCOL = "tokamak-policy/v1"


def main():
    actor = NumpyActor.load(Path(__file__).resolve().parent)
    low = high = None
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("protocol") != PROTOCOL:
            raise SystemExit("unsupported protocol")
        reply = {"request_id": req["request_id"]}
        if req["op"] == "reset":
            low, high = req["action_low"], req["action_high"]
            if len(low) != len(actor.vmax) or req["observation_shape"][0] != len(actor.scaler.scale):
                raise SystemExit("package capacity mismatch")
            reply["ready"] = True
        elif req["op"] == "act":
            a = actor(req["observation"])
            reply["action"] = [float(min(max(x, lo), hi)) for x, lo, hi in zip(a, low, high)]
        elif req["op"] == "close":
            reply["closed"] = True
            print(json.dumps(reply), flush=True)
            return
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
