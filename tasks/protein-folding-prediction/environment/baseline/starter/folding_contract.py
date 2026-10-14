"""Strict, model-independent FoldBench folding contract (v0).

The contract describes biological targets and prediction artifacts.  It does
not expose model-native tensors, MSA features, templates, or evaluator ground
truth to inference backends.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


REQUEST_SCHEMA = "foldbench-folding-request/v0"
RESULT_SCHEMA = "foldbench-folding-result/v0"
TRAINING_SCHEMA = "foldbench-folding-training-example/v0"
ENTITY_TYPES = frozenset({"protein", "rna", "dna", "ligand"})


class ContractError(ValueError):
    """Raised when a value does not satisfy the public folding contract."""


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContractError(f"{location} must be an array")
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{location} must be a non-empty string")
    return value


def _integer(value: Any, location: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{location} must be an integer >= {minimum}")
    return value


def _keys(
    value: Mapping[str, Any],
    location: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required - value.keys()
    unexpected = value.keys() - required - optional
    if missing:
        raise ContractError(f"{location} is missing fields: {sorted(missing)}")
    if unexpected:
        raise ContractError(f"{location} has unsupported fields: {sorted(unexpected)}")


def _validate_modifications(value: Any, location: str) -> None:
    for index, item in enumerate(_list(value, location)):
        item = _mapping(item, f"{location}[{index}]")
        _keys(item, f"{location}[{index}]", {"position", "ccd"})
        _integer(item["position"], f"{location}[{index}].position", minimum=1)
        _string(item["ccd"], f"{location}[{index}].ccd")


def _validate_entity(value: Any, location: str) -> str:
    entity = _mapping(value, location)
    _keys(
        entity,
        location,
        {"id", "type"},
        {"sequence", "copies", "modifications", "ccd_codes", "smiles"},
    )
    entity_id = _string(entity["id"], f"{location}.id")
    entity_type = _string(entity["type"], f"{location}.type")
    if entity_type not in ENTITY_TYPES:
        raise ContractError(
            f"{location}.type must be one of {sorted(ENTITY_TYPES)}, got {entity_type!r}"
        )
    _integer(entity.get("copies", 1), f"{location}.copies", minimum=1)

    if entity_type == "ligand":
        if "sequence" in entity or "modifications" in entity:
            raise ContractError(f"{location}: ligands cannot define sequence/modifications")
        ccd_codes = entity.get("ccd_codes")
        smiles = entity.get("smiles")
        if (ccd_codes is None) == (smiles is None):
            raise ContractError(
                f"{location}: ligand must define exactly one of ccd_codes or smiles"
            )
        if ccd_codes is not None:
            codes = _list(ccd_codes, f"{location}.ccd_codes")
            if not codes:
                raise ContractError(f"{location}.ccd_codes must not be empty")
            for index, code in enumerate(codes):
                _string(code, f"{location}.ccd_codes[{index}]")
    else:
        _string(entity.get("sequence"), f"{location}.sequence")
        if "ccd_codes" in entity or "smiles" in entity:
            raise ContractError(f"{location}: polymer entities cannot define ligand fields")
        _validate_modifications(entity.get("modifications", []), f"{location}.modifications")
    return entity_id


def _validate_atom_reference(value: Any, location: str) -> None:
    reference = _mapping(value, location)
    _keys(reference, location, {"entity_id", "position", "atom_name"})
    _string(reference["entity_id"], f"{location}.entity_id")
    _integer(reference["position"], f"{location}.position", minimum=1)
    _string(reference["atom_name"], f"{location}.atom_name")


def validate_target(value: Any, location: str = "target") -> None:
    target = _mapping(value, location)
    _keys(target, location, {"entities"}, {"covalent_bonds"})
    entities = _list(target["entities"], f"{location}.entities")
    if not entities:
        raise ContractError(f"{location}.entities must not be empty")
    entity_ids = [
        _validate_entity(entity, f"{location}.entities[{index}]")
        for index, entity in enumerate(entities)
    ]
    if len(entity_ids) != len(set(entity_ids)):
        raise ContractError(f"{location}.entities must have unique ids")

    for index, bond in enumerate(target.get("covalent_bonds", [])):
        bond = _mapping(bond, f"{location}.covalent_bonds[{index}]")
        _keys(bond, f"{location}.covalent_bonds[{index}]", {"left", "right"})
        _validate_atom_reference(bond["left"], f"{location}.covalent_bonds[{index}].left")
        _validate_atom_reference(bond["right"], f"{location}.covalent_bonds[{index}].right")
        for side in ("left", "right"):
            if bond[side]["entity_id"] not in entity_ids:
                raise ContractError(
                    f"{location}.covalent_bonds[{index}].{side}.entity_id is unknown"
                )


def validate_request(value: Any) -> None:
    request = _mapping(value, "request")
    _keys(request, "request", {"schema_version", "request_id", "target", "run"})
    if request["schema_version"] != REQUEST_SCHEMA:
        raise ContractError(f"unsupported request schema: {request['schema_version']!r}")
    _string(request["request_id"], "request.request_id")
    validate_target(request["target"], "request.target")
    run = _mapping(request["run"], "request.run")
    _keys(run, "request.run", {"seeds", "num_samples"})
    seeds = _list(run["seeds"], "request.run.seeds")
    if not seeds:
        raise ContractError("request.run.seeds must not be empty")
    for index, seed in enumerate(seeds):
        _integer(seed, f"request.run.seeds[{index}]")
    if len(seeds) != len(set(seeds)):
        raise ContractError("request.run.seeds must be unique")
    _integer(run["num_samples"], "request.run.num_samples", minimum=1)


def validate_training_example(value: Any) -> None:
    example = _mapping(value, "training_example")
    _keys(
        example,
        "training_example",
        {"schema_version", "example_id", "target", "ground_truth"},
    )
    if example["schema_version"] != TRAINING_SCHEMA:
        raise ContractError(f"unsupported training schema: {example['schema_version']!r}")
    _string(example["example_id"], "training_example.example_id")
    validate_target(example["target"], "training_example.target")
    ground_truth = _mapping(example["ground_truth"], "training_example.ground_truth")
    _keys(
        ground_truth,
        "training_example.ground_truth",
        {"structure_path", "format"},
        {"entity_mapping"},
    )
    _string(ground_truth["structure_path"], "training_example.ground_truth.structure_path")
    if ground_truth["format"] != "mmcif":
        raise ContractError("training_example.ground_truth.format must be 'mmcif'")
    if "entity_mapping" in ground_truth:
        _mapping(ground_truth["entity_mapping"], "training_example.ground_truth.entity_mapping")


def validate_result(value: Any) -> None:
    result = _mapping(value, "result")
    _keys(
        result,
        "result",
        {"schema_version", "request_id", "backend", "predictions", "provenance"},
    )
    if result["schema_version"] != RESULT_SCHEMA:
        raise ContractError(f"unsupported result schema: {result['schema_version']!r}")
    _string(result["request_id"], "result.request_id")
    backend = _mapping(result["backend"], "result.backend")
    _keys(backend, "result.backend", {"name", "model"}, {"revision"})
    _string(backend["name"], "result.backend.name")
    _string(backend["model"], "result.backend.model")
    if "revision" in backend:
        _string(backend["revision"], "result.backend.revision")

    predictions = _list(result["predictions"], "result.predictions")
    if not predictions:
        raise ContractError("result.predictions must not be empty")
    prediction_ids: set[str] = set()
    for index, prediction in enumerate(predictions):
        location = f"result.predictions[{index}]"
        prediction = _mapping(prediction, location)
        _keys(
            prediction,
            location,
            {
                "prediction_id",
                "structure_path",
                "format",
                "seed",
                "sample",
                "ranking_score",
                "confidence",
            },
        )
        prediction_id = _string(prediction["prediction_id"], f"{location}.prediction_id")
        if prediction_id in prediction_ids:
            raise ContractError("result prediction ids must be unique")
        prediction_ids.add(prediction_id)
        _string(prediction["structure_path"], f"{location}.structure_path")
        if prediction["format"] != "mmcif":
            raise ContractError(f"{location}.format must be 'mmcif'")
        _integer(prediction["seed"], f"{location}.seed")
        _integer(prediction["sample"], f"{location}.sample")
        ranking_score = prediction["ranking_score"]
        if ranking_score is not None and not isinstance(ranking_score, (int, float)):
            raise ContractError(f"{location}.ranking_score must be numeric or null")
        _mapping(prediction["confidence"], f"{location}.confidence")
    _mapping(result["provenance"], "result.provenance")
