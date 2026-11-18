"""Read only encoder weights from a submitted checkpoint, before loading its architecture."""
import argparse
import tarfile
import zipfile

import torch


def load_model_weights(path):
    with open(path, "rb") as source:
        magic = source.read(4)
        source.seek(0)
        if magic == b"PK\x03\x04":
            with zipfile.ZipFile(source) as archive:
                if any(name.split("/")[-1] == "constants.pkl" for name in archive.namelist()):
                    raise ValueError("A TorchScript program is not an encoder state dict")
        elif tarfile.is_tarfile(source):
            # The task pins torch 2.4. Its legacy TAR reader can bypass the restricted unpickler.
            raise ValueError("Legacy TAR checkpoints cannot be loaded safely")
        source.seek(0)
        state = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not isinstance(state.get("model"), dict):
        raise ValueError('checkpoint.pt must contain a "model" state dict')
    return state["model"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    # Uni-Core's fine-tune loader reads only this locally rewritten state, using the task's native precision.
    torch.save({"model": load_model_weights(args.source)}, args.destination)
