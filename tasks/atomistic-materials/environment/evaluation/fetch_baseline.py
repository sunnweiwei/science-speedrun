"""Download the pinned baseline using existing authorized Hugging Face access."""

import argparse
import json
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download

from common import digest, write_json


def fetch(output: Path):
    baseline = json.loads(Path(__file__).with_name("baseline.json").read_text())
    if output.exists():
        raise FileExistsError(output)
    downloaded = Path(
        hf_hub_download(
            baseline["repository"],
            baseline["filename"],
            revision=baseline["revision"],
        )
    )
    if digest(downloaded) != baseline["sha256"]:
        raise ValueError("Baseline checkpoint checksum mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, output)
    write_json(output.with_suffix(".json"), baseline)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    fetch(parser.parse_args().output)
