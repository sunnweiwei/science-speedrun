"""Public GCS inventory for every AlphaGenome bundle."""

from __future__ import annotations

from dataclasses import asdict

from .constants import SPLIT_TO_UPSTREAM
from .errors import ContractError
from .gcs import ObjectInfo, _list_prefix, _SHARD_RE
from .constants import MODEL_VERSION, ORGANISM, SOURCE_BUCKET, SOURCE_PREFIX
from .protocol import BUNDLES


def list_shards(split: str, bundle: str) -> list[ObjectInfo]:
    if split not in SPLIT_TO_UPSTREAM:
        raise ValueError(f"Unknown split {split!r}")
    if bundle not in BUNDLES:
        raise ValueError(f"Unknown benchmark bundle {bundle!r}")
    prefix = "/".join(
        [SOURCE_PREFIX, MODEL_VERSION, ORGANISM, SPLIT_TO_UPSTREAM[split], bundle, ""]
    )
    shards: list[ObjectInfo] = []
    for item in _list_prefix(prefix):
        name = item["name"]
        match = _SHARD_RE.search(name)
        if not match:
            if not name.endswith("_$folder$"):
                raise ContractError(f"Unexpected object below the source prefix: {name}")
            continue
        shards.append(
            ObjectInfo(
                bundle=bundle,
                split=split,
                chromosome=match.group("chromosome"),
                shard=int(match.group("shard")),
                shard_count=int(match.group("count")),
                name=name,
                size=int(item["size"]),
                generation=item["generation"],
                md5_base64=item.get("md5Hash", ""),
                crc32c_base64=item.get("crc32c", ""),
            )
        )
    if not shards:
        raise ContractError(f"No benchmark TFRecords found at gs://{SOURCE_BUCKET}/{prefix}")
    shards.sort(key=lambda item: item.pair_key)
    return shards


def aligned_shards(split: str) -> list[dict[str, ObjectInfo]]:
    by_bundle = {
        bundle: {item.pair_key: item for item in list_shards(split, bundle)}
        for bundle in BUNDLES
    }
    keys = set(next(iter(by_bundle.values())))
    mismatches = {
        bundle: {"missing": sorted(keys - set(items)), "extra": sorted(set(items) - keys)}
        for bundle, items in by_bundle.items()
        if set(items) != keys
    }
    if mismatches:
        raise ContractError(f"Bundle shard keys are not aligned for {split}: {mismatches}")
    return [
        {bundle: by_bundle[bundle][key] for bundle in BUNDLES}
        for key in sorted(keys)
    ]


def build_inventory(splits: list[str]) -> dict[str, object]:
    objects: list[ObjectInfo] = []
    for split in splits:
        for bundle in BUNDLES:
            objects.extend(list_shards(split, bundle))
    return {
        "bucket": SOURCE_BUCKET,
        "prefix": SOURCE_PREFIX,
        "model_version": MODEL_VERSION,
        "organism": ORGANISM,
        "bundles": list(BUNDLES),
        "objects": [asdict(item) for item in objects],
        "source_bytes": {
            split: sum(item.size for item in objects if item.split == split)
            for split in splits
        },
    }
