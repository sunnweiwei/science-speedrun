"""Versioned contract for the compact human 1 Mb / 11-output protocol."""

from __future__ import annotations

from dataclasses import dataclass


TARGET_STORAGE_SCHEMA = "alphagenome-human-compact/v2"
# Backwards-compatible name used by the target writer.  The target storage
# contract is deliberately independent from the inference/release contract:
# existing target shards must never need to be regenerated for a manifest-only
# protocol migration.
DATASET_SCHEMA = TARGET_STORAGE_SCHEMA
INFERENCE_SCHEMA = "alphagenome-human-compact-inference/v3"
RELEASE_SCHEMA = "alphagenome-human-compact-release/v3"
TRANSFORM_CONTRACT = "centered-source-coordinate-transforms/v1"
INPUT_BP = 1_048_576
TARGET_BP = 196_608
DENSE_BIN_BP = 128
DENSE_BINS = TARGET_BP // DENSE_BIN_BP
CONTACT_BIN_BP = 2_048
CONTACT_BINS = INPUT_BP // CONTACT_BIN_BP
SELECTION_SEED = "alphagenome-human-compact-v2-2026-09"
# Train keeps every eligible interval (stratified_select returns all rows when the budget is at
# least the eligible count); its targets are stored as 8-bit codes (quantize.py).
SPLIT_BUDGETS = {"train": 41_020, "valid": 200, "test": 1_000}
QUANTIZED_SPLITS = frozenset({"train"})
RELEASE_TARGET_GIB = 90
RELEASE_MAX_GIB = 500


@dataclass(frozen=True)
class BundleSpec:
    bundle: str
    family: str | None
    tensors: tuple[tuple[str, str], ...]
    representation: str
    native_resolution: int
    stored_resolution: int | None
    real_tracks: int | None


# Dtype strings are resolved to TensorFlow dtypes lazily so importing manifest
# tooling does not require TensorFlow.
BUNDLES: dict[str, BundleSpec] = {
    "ATAC": BundleSpec("ATAC", "ATAC", (("atac", "bfloat16"), ("atac_mask", "bool")), "dense_128", 1, 128, 167),
    "DNASE": BundleSpec("DNASE", "DNASE", (("dnase", "bfloat16"), ("dnase_mask", "bool")), "dense_128", 1, 128, 305),
    "PROCAP": BundleSpec("PROCAP", "PROCAP", (("procap", "bfloat16"), ("procap_mask", "bool")), "dense_128", 1, 128, 12),
    "CAGE": BundleSpec("CAGE", "CAGE", (("cage", "bfloat16"), ("cage_mask", "bool")), "dense_128", 1, 128, 546),
    "RNA_SEQ": BundleSpec(
        "RNA_SEQ",
        "RNA_SEQ",
        (("rna_seq", "bfloat16"), ("rna_seq_mask", "bool"), ("rna_seq_strand", "int32")),
        "dense_128",
        1,
        128,
        667,
    ),
    "CHIP_TF": BundleSpec("CHIP_TF", "CHIP_TF", (("chip_tf", "float32"), ("chip_tf_mask", "bool")), "dense_128", 128, 128, 1_617),
    "CHIP_HISTONE": BundleSpec(
        "CHIP_HISTONE",
        "CHIP_HISTONE",
        (("chip_histone", "float32"), ("chip_histone_mask", "bool")),
        "dense_128",
        128,
        128,
        1_116,
    ),
    "CONTACT_MAPS": BundleSpec("CONTACT_MAPS", "CONTACT_MAPS", (("contact_maps", "float32"),), "symmetric_upper_triangle", 2_048, 2_048, 28),
    "SPLICE_SITES": BundleSpec("SPLICE_SITES", "SPLICE_SITES", (("splice_sites", "bool"),), "base_or_event", 1, 1, 4),
    "SPLICE_SITE_USAGE": BundleSpec(
        "SPLICE_SITE_USAGE",
        "SPLICE_SITE_USAGE",
        (("splice_site_usage", "float16"),),
        "sparse_event",
        1,
        None,
        734,
    ),
    "SPLICE_JUNCTIONS": BundleSpec(
        "SPLICE_JUNCTIONS",
        "SPLICE_JUNCTIONS",
        (("splice_junctions", "float32"),),
        "sparse_pair",
        1,
        None,
        734,
    ),
    "SPLICE_SITE_POSITIONS": BundleSpec(
        "SPLICE_SITE_POSITIONS",
        None,
        (("splice_site_positions", "int32"),),
        "auxiliary_positions",
        1,
        None,
        None,
    ),
}

OUTPUT_FAMILIES = tuple(
    spec.family for spec in BUNDLES.values() if spec.family is not None
)
