"""MACE checkpoints used to generate the published energy hull and embedding references."""
import hashlib

CHECKPOINTS = {
    "macemp0b3mediummodel": "2f2be696351ac9e94fbe01cdfb6f017679acdbd2db7645209ef55fec9826b012",
    "macempa0mediummodel": "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638",
}


def validate_mace_assets(root):
    for filename, expected in CHECKPOINTS.items():
        path = root / "home" / ".cache" / "mace" / filename
        if not path.is_file():
            raise ValueError(f"Missing reference-matched MACE checkpoint {filename}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Wrong reference-matched MACE checkpoint {filename}")
