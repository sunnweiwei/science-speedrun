"""Zipapp entry point: a fixed shared PPO checkpoint, native environment actions."""

import io
import json
import math
import sys
import zipfile

import torch
from shared import build_policy, encode, native_action


def main():
    torch.set_num_threads(1)
    with zipfile.ZipFile(sys.argv[0]) as archive:
        config = json.loads(archive.read("config.json"))
        state = torch.load(io.BytesIO(archive.read("checkpoint.pt")), map_location="cpu", weights_only=True)
    policy = build_policy(config)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    policy.requires_grad_(False)
    episode = None
    for line in sys.stdin:
        request = json.loads(line)
        if request["protocol"] != "hydrogym-policy/v1":
            raise ValueError("Unsupported protocol")
        reply = {"request_id": request["request_id"]}
        if request["op"] == "reset":
            if math.prod(request["action_shape"]) > config["max_action"]:
                raise ValueError("Action exceeds starter capacity")
            episode = request
            reply["ready"] = True
        elif request["op"] == "act":
            if episode is None:
                raise ValueError("Missing reset")
            vector = encode(request["observation"], episode["context"]["environment"], config)
            with torch.inference_mode():
                action, _, _ = policy(torch.as_tensor(vector).unsqueeze(0), deterministic=True)
            reply["action"] = native_action(action[0].numpy(), episode["action_low"], episode["action_high"]).tolist()
        elif request["op"] == "close":
            reply["closed"] = True
            print(json.dumps(reply, allow_nan=False), flush=True)
            return
        else:
            raise ValueError("Unsupported operation")
        print(json.dumps(reply, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
