"""Fail-closed CLI guard for Protenix training on the frozen native view."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from agent_pipeline.contract import NATIVE_VIEW_AUDIT_SCHEMA
from official_baselines.data_policy import UNIFIED_TRAIN_SHA256


CLI_OPTION = "--unified-native-view-audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pop_option(option: str, values: list[str]) -> str:
    found = None
    index = 1
    while index < len(values):
        if values[index] == option:
            if index + 1 >= len(values):
                raise ValueError(f"{option} requires a value")
            found = values[index + 1]
            del values[index : index + 2]
            continue
        if values[index].startswith(f"{option}="):
            found = values[index].split("=", 1)[1]
            del values[index]
            continue
        index += 1
    if found is None:
        raise ValueError(f"{option} is required for fixed-data training")
    return found


def upstream_options(values: list[str]) -> dict[str, str]:
    arguments = values[1:]
    if len(arguments) % 2:
        raise ValueError("upstream Protenix arguments must be --key value pairs")
    result = {}
    for key, value in zip(arguments[::2], arguments[1::2], strict=True):
        if not key.startswith("--"):
            raise ValueError(f"invalid upstream option: {key}")
        result[key] = value
    return result


def enforce(argv: list[str] | None = None) -> dict:
    values = sys.argv if argv is None else argv
    audit_path = Path(pop_option(CLI_OPTION, values))
    audit = json.loads(audit_path.read_text())
    schema = audit.get("schema_version")
    if schema not in {
        "foldingtrain-protenix-native-view-v1",
        NATIVE_VIEW_AUDIT_SCHEMA,
    }:
        raise ValueError(f"invalid Protenix native-view audit: {audit_path}")
    source_digest = (
        audit.get("source_train_sha256")
        if schema == "foldingtrain-protenix-native-view-v1"
        else audit.get("source_split_sha256")
    )
    if source_digest != UNIFIED_TRAIN_SHA256:
        raise ValueError("native-view audit is not bound to the frozen train split")
    if schema == "foldingtrain-protenix-native-view-v1":
        source_split = Path(audit.get("source_train_split", ""))
        if not source_split.is_file() or sha256(source_split) != UNIFIED_TRAIN_SHA256:
            raise ValueError(
                "frozen train split is missing or differs from the native-view audit"
            )
        if audit.get("outside_frozen_train") != 0:
            raise ValueError("native-view audit contains structures outside frozen train")
        admitted = {record["sha256"] for record in audit["native_indices"]}
    else:
        if audit.get("selection_policy") != "capability_only":
            raise ValueError("native-view audit uses a forbidden selection policy")
        if audit.get("accepted_examples", 0) + audit.get("rejected_examples", 0) != audit.get(
            "source_examples"
        ):
            raise ValueError("native-view audit does not account for every source example")
        admitted = {audit.get("native_manifest_sha256")}

    options = upstream_options(values)
    train_sets = options.get("--data.train_sets")
    if not train_sets:
        raise ValueError("--data.train_sets must be explicit for fixed-data training")
    requested_indices = []
    for name in train_sets.split(","):
        key = f"--data.{name}.base_info.indices_fpath"
        if key not in options:
            raise ValueError(f"{key} must be explicit for fixed-data training")
        path = Path(options[key])
        if not path.is_file():
            raise FileNotFoundError(f"Protenix native index not found: {path}")
        requested_indices.append((str(path), sha256(path)))
    unadmitted = [path for path, digest in requested_indices if digest not in admitted]
    if unadmitted:
        raise ValueError(f"Protenix indices absent from common-data audit: {unadmitted}")
    return audit
