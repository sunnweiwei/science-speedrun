"""Separate, explicitly labeled official full-data reference; never training init."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig

REVISION = "b2c9f41de3d8454cb37f0cd9cc3e79ec7c4af435"
SHA256 = "1adb5026bd520c54bc415a1282386954927623bab81d01be2f5b6379cc039035"
p = argparse.ArgumentParser()
p.add_argument("--weights", type=Path, required=True)
p.add_argument(
    "--random-package",
    type=Path,
    required=True,
    help="Verified official Large architecture and license",
)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise ValueError("Use a fresh reference package path")
with a.weights.open("rb") as f:
    if hashlib.file_digest(f, "sha256").hexdigest() != SHA256:
        raise ValueError("Public reference hash mismatch")
config = json.loads((a.random_package / "config.json").read_text())
if config["embedding_size"] != 1024 or config["depth"] != 24:
    raise ValueError("Expected official Large configuration")
state = torch.load(a.weights, map_location="cpu", weights_only=True)
encoder = {
    key.removeprefix("encoder."): value
    for key, value in state.items()
    if key.startswith("encoder.")
}
if not encoder:
    raise ValueError("No encoder in public checkpoint")
model = EncoderConfig.from_dict(config).build()
model.load_state_dict(encoder, strict=True)
a.output.mkdir(parents=True)
torch.save(encoder, a.output / "encoder.pt")
for name in ["config.json", "infer.py", "LICENSE-OlmoEarth"]:
    shutil.copyfile(a.random_package / name, a.output / name)
with (a.output / "encoder.pt").open("rb") as f:
    encoder_hash = hashlib.file_digest(f, "sha256").hexdigest()
provenance = {
    "origin": "official_public_pretrained_reference",
    "variant": "v1/large",
    "hf_revision": REVISION,
    "source_weights_sha256": SHA256,
    "checkpoint": f"https://huggingface.co/allenai/OlmoEarth-v1-Large/tree/{REVISION}",
    "initialization_seed": 12536,
    "parameters": sum(p.numel() for p in model.parameters()),
    "infer_sha256": hashlib.sha256((a.output / "infer.py").read_bytes()).hexdigest(),
    "encoder_sha256": encoder_hash,
    "training_recipe": None,
    "limitation": "Official full-data budget and potential imagery overlap; reference only, not our from-scratch result.",
}
(a.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
print(json.dumps(provenance, indent=2))
