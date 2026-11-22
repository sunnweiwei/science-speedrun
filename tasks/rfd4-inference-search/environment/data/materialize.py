"""Controller-only downloads and sealed seed initialization."""
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
import tarfile
import tempfile
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
MODEL = "https://huggingface.co/nvidia/RFD4-Proteina/resolve/a06e2509c7033b010fc43614e9fd840303588502"
OF3_REPO = "https://huggingface.co/nvidia/RFD4-Proteina/resolve/598acab93d5d0aeacc4362f6e88dc20e7db85ef6"
REPO_RAW = "https://raw.githubusercontent.com/RosettaCommons/RFD4-Proteina/6306a210dd17dcae8e4e0f381b5b74a719d9b30e"
SOURCE = REPO_RAW + "/data/benchmarks/monomer"

FILES = {
    "rfd4/config.json": (MODEL + "/config.json", "154ea927ab8005b7eac2d047a792147fb4606ab4878e496aa8414b6b40dad368"),
    "rfd4/model.safetensors": (MODEL + "/model.safetensors", "5b549dd61adb2f6c6b5cb95bda13f40f5e31556b4e6c755edc0ca6f011c68fb0"),
    "af3/af3.bin.zst": ("https://storage.googleapis.com/alphafold3/af3.bin.zst?generation=1780568696389861", "74d0258616917cd122f5eab6d076afe4a8930e96823851e65e4f777dfb1f33ff"),
    "foldseek.tar.gz": ("https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/foldseek-linux-avx2.tar.gz", "e6a3e3860e76b0b7b858736f627fe680d4d7f0fc7298301d8a92b9f711d74ac7"),
}
for size, digest in (
    ("short", "5e66073220a3314d3426ca5d7828c02492e61d24321966d4d73dff08481cfa37"),
    ("medium", "06cb8093bcb51c8a923d352041db8bb9941b74236d0b2357d8c477204135ff50"),
    ("long", "26f764fa13f44288ca6214db3312505d646f2017a21e8329a40604731bd944bd"),
):
    name = f"monomer-{size}.cif.zst"
    FILES["monomers/" + name] = (SOURCE + "/" + name, digest)

BENCHMARK_MANIFEST = HERE / "figure2_benchmarks.json"
if BENCHMARK_MANIFEST.is_file():
    manifest = json.loads(BENCHMARK_MANIFEST.read_text())
    raw_base = f"https://raw.githubusercontent.com/{manifest['upstream_repo']}/{manifest['upstream_commit']}"
    for target_key, info in manifest["targets"].items():
        FILES[f"benchmarks/{target_key}.cif.zst"] = (f"{raw_base}/{info['upstream_rel']}", info["sha256"])


def verified(path, digest):
    if not path.is_file():
        return False
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest() == digest


def materialize(root):
    shared = root / "development/assets"
    for name, (url, sha) in FILES.items():
        path = shared / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if verified(path, sha):
            continue
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            download = Path(tmp) / "download"
            with urlopen(url, timeout=120) as response, download.open("wb") as f:
                shutil.copyfileobj(response, f)
            if not verified(download, sha):
                raise RuntimeError(f"Checksum mismatch for {name}")
            download.replace(path)
    with tarfile.open(shared / "foldseek.tar.gz") as archive, tempfile.TemporaryDirectory(dir=shared) as tmp:
        exe = Path(tmp) / "foldseek"
        with archive.extractfile("foldseek/bin/foldseek") as src, exe.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        exe.chmod(0o755)
        exe.replace(shared / "foldseek")
    private = root / "private"
    private.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Link the completed file once. Concurrent materializers and retries retain the same private suite.
    with tempfile.TemporaryDirectory(dir=private) as tmp:
        candidate = Path(tmp) / "seed"
        candidate.write_bytes(secrets.token_bytes(32))
        candidate.chmod(0o600)
        try:
            os.link(candidate, private / "seed")
        except FileExistsError:
            pass
    if (private / "seed").stat().st_size != 32:
        raise RuntimeError("Invalid existing private seed")
    print(f"Verified {len(FILES)} assets in {shared} and the private seed under {private}")


if __name__ == "__main__":
    materialize(Path(sys.argv[1]).resolve())
