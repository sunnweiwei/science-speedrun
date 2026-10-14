"""Pinned identities and runtime configuration for the OpenDDE baseline."""

from pathlib import Path

UPSTREAM_REPOSITORY = "https://github.com/aurekaresearch/OpenDDE"
UPSTREAM_REVISION = "f607bb3c9ff299c0627ac20f5ef8e25d716ed46f"
RELEASE_ASSET_REVISION = "eddd563ce96571f784012edd8f045181c8f8627d"
UPSTREAM_IMAGE = "aurekaresearch/opendde:v1"
MODEL_NAME = "opendde_v1"
MODEL_PARAMETERS = 655_791_538
CHECKPOINT_SHA256 = "7b826620390afad877ee2babc6a4d0df81b94d3a0be030959853d6a7da0807cc"

MODEL_NAME_50M = "opendde_structural_50m_v1"
EXPECTED_PARAMETER_RANGE = (48_000_000, 52_000_000)


def configure_ccd(directory: Path) -> None:
    """Point the released featurizer at the bundle-local CCD artifacts.

    This deliberately lives in the lightweight runtime configuration module:
    contract inference must not import the training entry point (and therefore
    repository-only data-audit dependencies) from a portable ModelBundle.
    """

    from opendde.data.core import ccd

    components = directory / "components.cif"
    rdkit = directory / "components.cif.rdkit_mol.pkl"
    for path in (components, rdkit):
        if not path.is_file():
            raise FileNotFoundError(path)
    ccd.set_ccd_cache_paths(str(components), str(rdkit))


def scaled_config():
    """Return a config-only scale-down of the complete released module graph."""

    from opendde.config.inference import build_inference_config, deep_update
    from opendde.config.schema import OpenDDEConfig

    base = build_inference_config(model_name=MODEL_NAME).model_dump()
    overrides = {
        "c_s": 192,
        "c_z": 96,
        "c_s_inputs": 257,
        "c_atom": 96,
        "c_atompair": 16,
        "c_token": 192,
        "n_blocks": 12,
        # Keep the released OpenDDE activation-checkpointing policy.  This is
        # an execution policy (not a layer/parameter change) and is required
        # for structural-token training examples to fit on an 80GB H100.
        "blocks_per_ckpt": 1,
        "triangle_multiplicative": "torch",
        "triangle_attention": "torch",
        "model": {
            "N_cycle": 1,
            "input_embedder": {"c_atom": 96, "c_atompair": 16, "c_token": 192},
            "relative_position_encoding": {"c_z": 96},
            "template_embedder": {
                "c": 48,
                "c_z": 96,
                "n_blocks": 1,
                "blocks_per_ckpt": 1,
            },
            "msa_module": {
                "c_m": 96,
                "c_z": 96,
                "c_s_inputs": 257,
                "n_blocks": 3,
                "blocks_per_ckpt": 1,
            },
            "pairformer": {
                "n_blocks": 12,
                "c_z": 96,
                "c_s": 192,
                "n_heads": 8,
                "blocks_per_ckpt": 1,
            },
            "structural_token_expansion": {
                "structural_refiner": {
                    "n_blocks": 3,
                    "n_heads": 8,
                    "blocks_per_ckpt": 1,
                },
            },
            "diffusion_module": {
                "use_fine_grained_checkpoint": True,
                "c_token": 384,
                "c_atom": 96,
                "c_atompair": 16,
                "c_z": 96,
                "c_z_pair_diffusion": 96,
                "c_s": 192,
                "c_s_inputs": 257,
                "atom_encoder": {"n_blocks": 2, "n_heads": 4},
                "transformer": {"n_blocks": 13, "n_heads": 8},
                "atom_decoder": {"n_blocks": 2, "n_heads": 4},
                "blocks_per_ckpt": 1,
            },
            "confidence_head": {
                "c_z": 96,
                "c_s": 192,
                "c_s_inputs": 257,
                "n_blocks": 3,
                "blocks_per_ckpt": 1,
            },
            "distogram_head": {"c_z": 96},
        },
    }
    deep_update(base, overrides)
    return OpenDDEConfig.model_validate(base)
