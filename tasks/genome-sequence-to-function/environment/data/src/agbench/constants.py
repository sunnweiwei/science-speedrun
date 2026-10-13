"""Pinned release contract. Coordinates are zero-based, half-open."""

from __future__ import annotations

MODEL_VERSION = "FOLD_0"
ORGANISM = "HOMO_SAPIENS"
SPLIT_TO_FOLDS = {
    "train": tuple(f"fold{i}" for i in range(2, 8)),
    "valid": ("fold0",),
    "test": ("fold1",),
}
SPLIT_TO_UPSTREAM = {"train": "TRAIN", "valid": "VALID", "test": "TEST"}

SOURCE_BUCKET = "alphagenome-datasets"
SOURCE_PREFIX = "v1/train"
SOURCE_ROOT = f"gs://{SOURCE_BUCKET}/{SOURCE_PREFIX}"

UPSTREAM_RESEARCH_REVISION = "0db53bd4352c66d1e00a049a81da373a066e6670"
UPSTREAM_CLIENT_VERSION = "0.7.0"
UPSTREAM_CLIENT_REVISION = "209c439494fe6fccd060e617850fdf4616770be8"

FOLD_BED_URL = (
    "https://raw.githubusercontent.com/calico/borzoi/"
    "5c9358222b5026abb733ed5fb84f3f6c77239b37/"
    "data/sequences_human.bed.gz"
)
FOLD_BED_SHA256 = "13724966c9378c2fd024c62e6bb72b94850e12d96d23a419f60040ae50b98ba3"

OUTPUT_METADATA_URL = (
    "https://raw.githubusercontent.com/google-deepmind/alphagenome_research/"
    f"{UPSTREAM_RESEARCH_REVISION}/src/alphagenome_research/model/metadata/"
    "OutputMetadataResponse_ORGANISM_HOMO_SAPIENS.textproto"
)
OUTPUT_METADATA_SHA256 = "413b4063096bbf7a75a7fdf5560030f8e3866c6af2455fb313e58686ae3d49e8"

FASTA_URL = (
    "https://storage.googleapis.com/alphagenome/reference/gencode/hg38/"
    "GRCh38.p13.genome.fa?generation=1766084693379925"
)
FASTA_SIZE = 3_321_586_957
FASTA_MD5_BASE64 = "7e5UCDA/bBxrrhx2/9I2cQ=="
FAI_URL = (
    "https://storage.googleapis.com/alphagenome/reference/gencode/hg38/"
    "GRCh38.p13.genome.fa.fai?generation=1766086454720750"
)
FAI_SIZE = 22_027
FAI_MD5_BASE64 = "d4WMEKhe5McPSd5KKj2n9Q=="
FAI_SHA256 = "9293fb33f63b7f09d8fadc055d78e401d07c76de0211243fba778780ff836ff9"
