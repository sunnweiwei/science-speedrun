"""Null policy for the latency gate: parses each request and answers zero volts (clipped to bounds).

The verifier times it in the same process and under the same load as the submission; the
difference of the two median round trips estimates the submission's own computation.
"""

import json
import sys


def main():
    low = high = None
    for line in sys.stdin:
        req = json.loads(line)
        reply = {"request_id": req["request_id"]}
        if req["op"] == "reset":
            low, high = req["action_low"], req["action_high"]
            reply["ready"] = True
        elif req["op"] == "act":
            reply["action"] = [min(max(0.0, a), b) for a, b in zip(low, high)]
        else:
            reply["closed"] = True
            print(json.dumps(reply), flush=True)
            return
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
