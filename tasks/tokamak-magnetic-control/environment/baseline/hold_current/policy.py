"""Hold-current reference policy: V_i = R_i * I_i(measured), clipped to the limits.

With constant coil currents the plasma is left to its own (unstable) dynamics; the
policy is a protocol/plumbing check and the natural 'no control' anchor. Set
TOKAGYM_REFERENCE=zero to send zero volts instead.
"""

import json
import os
import sys

# Coil resistances [Ohm] of the vendored MAST-U-like description (tokamak.coil_resist[:12]).
RESIST = [0.02133, 0.01147, 0.00483, 0.00490, 0.00415, 0.00395, 0.00403, 0.00406, 0.00415,
          0.00575, 0.00568, 0.00265]
N_ACTIVE = 12
MODE = os.environ.get("TOKAGYM_REFERENCE", "hold")


def main():
    low = high = None
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("protocol") != "tokamak-policy/v1":
            raise SystemExit("unsupported protocol")
        reply = {"request_id": req["request_id"]}
        if req["op"] == "reset":
            low, high = req["action_low"], req["action_high"]
            reply["ready"] = True
        elif req["op"] == "act":
            obs = req["observation"]
            currents = obs[500:500 + N_ACTIVE]
            if MODE == "zero":
                v = [0.0] * N_ACTIVE
            else:
                v = [r * i for r, i in zip(RESIST, currents)]
            reply["action"] = [min(max(x, a), b) for x, a, b in zip(v, low, high)]
        elif req["op"] == "close":
            reply["closed"] = True
            print(json.dumps(reply), flush=True)
            return
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
