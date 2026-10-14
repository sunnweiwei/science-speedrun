"""Inject frozen ESMC features without changing Protenix's data/model code.

The project-wide feature manifest is keyed by the exact protein sequence.  The
adapter leaves Protenix's official cropping and token-to-entity mapping in
charge, then supplies the one extra tensor consumed by its existing ESM input
path.  A missing or mismatched feature is fatal: silently replacing it with
zeros would make runs incomparable.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from first_party_baselines.hybrid_esmc_pairformer.config import FEATURE_ID


EMBEDDING_DIM = 256
CLI_OPTION = "--unified-esmc-manifest"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CachedESMCFeatures:
    """Validated sequence-to-safetensors index for the frozen common data."""

    def __init__(self, manifest: str | Path, *, allow_empty: bool = False) -> None:
        self.manifest = Path(manifest)
        if not self.manifest.is_file():
            raise FileNotFoundError(f"ESMC feature manifest not found: {self.manifest}")
        self.records: dict[str, dict[str, Any]] = {}
        self.skipped_by_status: dict[str, int] = {}
        with self.manifest.open() as stream:
            for line_number, line in enumerate(stream, 1):
                record = json.loads(line)
                sequence = record["sequence"]
                sequence_sha256 = hashlib.sha256(sequence.encode()).hexdigest()
                if record["sequence_sha256"] != sequence_sha256:
                    raise ValueError(
                        f"sequence digest mismatch at {self.manifest}:{line_number}"
                    )
                if record["feature_version"] != FEATURE_ID:
                    raise ValueError(
                        f"feature version {record['feature_version']} != {FEATURE_ID}"
                    )
                status = record.get("local_status", "missing")
                if status != "ready":
                    self.skipped_by_status[status] = (
                        self.skipped_by_status.get(status, 0) + 1
                    )
                    continue
                if record["shape"] != [len(sequence), EMBEDDING_DIM]:
                    raise ValueError(f"invalid feature shape for {sequence_sha256}")
                previous = self.records.setdefault(sequence_sha256, record)
                if previous["local_path"] != record["local_path"]:
                    raise ValueError(f"conflicting feature paths for {sequence_sha256}")
        if not self.records and not allow_empty:
            raise ValueError(f"empty ESMC feature manifest: {self.manifest}")

    def tensor(self, sequence: str):
        import torch
        from safetensors.torch import load_file

        sequence_sha256 = hashlib.sha256(sequence.encode()).hexdigest()
        try:
            record = self.records[sequence_sha256]
        except KeyError as error:
            raise KeyError(
                f"no frozen ESMC feature for sequence {sequence_sha256} "
                f"(length={len(sequence)})"
            ) from error
        path = Path(record["local_path"])
        if not path.is_file():
            raise FileNotFoundError(f"cached ESMC tensor not found: {path}")
        tensors = load_file(str(path), device="cpu")
        if set(tensors) != {"single"}:
            raise ValueError(f"unexpected tensor keys in {path}: {sorted(tensors)}")
        value = tensors["single"]
        expected = (len(sequence), EMBEDDING_DIM)
        if tuple(value.shape) != expected:
            raise ValueError(f"tensor shape {tuple(value.shape)} != {expected}: {path}")
        if not torch.isfinite(value).all():
            raise ValueError(f"non-finite cached ESMC feature: {path}")
        return value

    def token_features(self, token_array, atom_array, bioassembly_dict):
        import numpy as np
        import torch

        centre_indices = token_array.get_annotation("centre_atom_index")
        centre_atoms = atom_array[centre_indices]
        is_protein = centre_atoms.chain_mol_type == "protein"
        output = torch.zeros((len(token_array), EMBEDDING_DIM), dtype=torch.bfloat16)
        for entity_id in np.unique(centre_atoms.label_entity_id[is_protein]):
            entity_key = str(entity_id)
            try:
                sequence = bioassembly_dict["sequences"][entity_key]
            except KeyError as error:
                raise KeyError(
                    f"missing protein entity {entity_key} in "
                    f"{bioassembly_dict.get('pdb_id', '<unknown>')}"
                ) from error
            entity_mask = is_protein & (centre_atoms.label_entity_id == entity_id)
            residue_indices = centre_atoms.res_id[entity_mask].astype(int) - 1
            if (residue_indices < 0).any() or (residue_indices >= len(sequence)).any():
                raise IndexError(
                    f"cropped residue index outside entity {entity_key} sequence"
                )
            output[torch.from_numpy(entity_mask)] = self.tensor(sequence)[residue_indices]
        return output

    def audit(self) -> dict[str, Any]:
        return {
            "schema_version": "foldingtrain-esmc-cache-adapter-v1",
            "manifest": str(self.manifest),
            "manifest_sha256": file_sha256(self.manifest),
            "feature_version": FEATURE_ID,
            "embedding_dim": EMBEDDING_DIM,
            "unique_sequences": len(self.records),
            "manifest_rows_skipped_by_status": self.skipped_by_status,
            "missing_feature_policy": "fail-closed",
        }


def pop_manifest_argument(argv: list[str] | None = None) -> Path:
    """Remove the adapter-only option before the upstream CLI parses argv."""

    values = sys.argv if argv is None else argv
    found: str | None = None
    index = 1
    while index < len(values):
        argument = values[index]
        if argument == CLI_OPTION:
            if index + 1 >= len(values):
                raise ValueError(f"{CLI_OPTION} requires a path")
            found = values[index + 1]
            del values[index : index + 2]
            continue
        if argument.startswith(f"{CLI_OPTION}="):
            found = argument.split("=", 1)[1]
            del values[index]
            continue
        index += 1
    if not found:
        raise ValueError(
            f"{CLI_OPTION} is required; the hybrid must use the frozen common-data cache"
        )
    return Path(found)


def install_cached_feature_adapter(
    manifest: str | Path, *, allow_empty: bool = False
) -> CachedESMCFeatures:
    """Patch only the upstream feature-loading boundary, once per process."""

    cache = CachedESMCFeatures(manifest, allow_empty=allow_empty)
    from protenix.data.esm.esm_featurizer import ESMFeaturizer
    from protenix.data.pipeline.dataset import BaseSingleDataset

    marker = "_foldingtrain_esmc_adapter_installed"
    if getattr(BaseSingleDataset, marker, False):
        raise RuntimeError("cached ESMC adapter is already installed")

    original_get_bioassembly = BaseSingleDataset._get_bioassembly_data
    original_get_feature = BaseSingleDataset.get_feature_and_label

    def get_bioassembly_with_context(self, idx):
        result = original_get_bioassembly(self, idx)
        self._foldingtrain_esmc_bioassembly = result[1]
        return result

    def get_feature_with_esmc(
        self,
        idx,
        token_array,
        atom_array,
        msa_features,
        template_features,
        full_atom_array,
        is_spatial_crop=True,
        max_entity_mol_id=None,
    ):
        result = original_get_feature(
            self,
            idx,
            token_array,
            atom_array,
            msa_features,
            template_features,
            full_atom_array,
            is_spatial_crop,
            max_entity_mol_id,
        )
        try:
            bioassembly = self._foldingtrain_esmc_bioassembly
        except AttributeError as error:
            raise RuntimeError("ESMC adapter lost the active bioassembly context") from error
        result[0]["esm_token_embedding"] = cache.token_features(
            token_array, atom_array, bioassembly
        )
        return result

    def cached_load(_self, sequence):
        return cache.tensor(sequence).float()

    @staticmethod
    def register_inference_sequences(
        inputs, _model_name, _embedding_dir, sequence_fpath, _checkpoint_dir
    ):
        rows = []
        for sample in inputs:
            for index, wrapper in enumerate(sample["sequences"], 1):
                entity_type = next(iter(wrapper))
                if entity_type != "proteinChain":
                    continue
                sequence = wrapper[entity_type]["sequence"]
                # Resolve now so inference cannot silently begin with a partial cache.
                cache.tensor(sequence)
                label = f"{sample['name']}_{index}"
                rows.append(
                    {
                        "seq": sequence,
                        "pdb_entity_id": label,
                        "seq_label": label,
                        "part_id": label,
                    }
                )
        destination = Path(sequence_fpath)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["seq", "pdb_entity_id", "seq_label", "part_id"],
            )
            writer.writeheader()
            writer.writerows(rows)

    BaseSingleDataset._get_bioassembly_data = get_bioassembly_with_context
    BaseSingleDataset.get_feature_and_label = get_feature_with_esmc
    setattr(BaseSingleDataset, marker, True)
    ESMFeaturizer.load_esm_embedding = cached_load
    ESMFeaturizer.precompute_esm_embedding = register_inference_sequences
    return cache
