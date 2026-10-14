"""Gradient-enabled orchestration around Biohub's released ESMFold2 modules.

The public top-level ``ESMFold2Model.forward`` is inference-only.  This module
uses its undecorated implementation to execute the exact released input,
PARCAE/CODA, and trunk dataflow, stops before ODE sampling, then calls the
released differentiable diffusion denoiser for one EDM training step.
"""

from __future__ import annotations

from typing import Any

from official_baselines.project_recipe import (
    ProjectRecipeConfig,
    add_coordinate_noise,
    edm_coordinate_loss,
    sample_noise_levels,
)


class _TrunkReady(RuntimeError):
    pass


def persistent_state_dict(state):
    """Drop official kernel caches that are regenerated from canonical weights.

    Biohub's fused SwiGLU path installs an ``nn.Module`` lazily.  Its tensors
    mirror LayerNorm/linear weights and therefore must not become independent
    checkpoint state.  All non-cache keys remain subject to strict loading.
    """

    return {
        name: value
        for name, value in state.items()
        if "._fused_swiglu." not in name
    }


def build_model():
    """Instantiate the scaled official class; ESMC is external to the folding graph."""

    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    from official_baselines.esmfold2.config import scaled_config

    return ESMFold2Model(scaled_config())


def initialize_frozen_lm_projection(model, hf_home):
    """Load the exact released projection used to create mixed-256 caches."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    from official_baselines.esmfold2.config import MODEL_REPOSITORY, MODEL_REVISION

    snapshot = snapshot_download(
        MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        repo_type="model",
        cache_dir=hf_home,
        local_files_only=True,
    )
    path = f"{snapshot}/model.safetensors"
    names = (
        "language_model.base_z_combine",
        "language_model.base_z_linear.0.bias",
        "language_model.base_z_linear.0.weight",
        "language_model.base_z_linear.1.weight",
    )
    parameters = dict(model.named_parameters())
    with safe_open(path, framework="pt", device="cpu") as source:
        for name in names:
            value = source.get_tensor(name)
            if name not in parameters or parameters[name].shape != value.shape:
                raise ValueError(f"released LM projection shape mismatch for {name}")
            parameters[name].data.copy_(value)
            parameters[name].requires_grad_(False)
    return {
        "model_revision": MODEL_REVISION,
        "frozen_projection_parameters": sum(parameters[name].numel() for name in names),
        "cache_boundary": "base_z_linear + base_z_combine; before base_z_mlp",
    }


def _run_released_trunk(model, features: dict[str, Any], *, num_loops: int):
    """Run the exact upstream forward body and capture the pre-sampling state."""

    import torch
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    captured: dict[str, Any] = {}

    def save(name):
        def hook(_module, _inputs, output):
            captured[name] = output

        return hook

    def stop(_module, _inputs, output):
        captured["distogram_logits"] = output
        raise _TrunkReady

    handles = [
        model.inputs_embedder.register_forward_hook(save("s_inputs")),
        model.rel_pos.register_forward_hook(save("relative_position_encoding")),
        model.parcae_coda.register_forward_hook(save("z_trunk")),
        model.distogram_head.register_forward_hook(stop),
    ]
    forward_features = dict(features)
    forward_features.pop("target_coords", None)
    forward_features.pop("target_atom_mask", None)
    mixed = forward_features.pop("lm_mixed256", None)
    input_ids = forward_features.pop("input_ids", None)
    original_language_forward = None
    if mixed is not None:
        import torch.nn.functional as functional

        if mixed.ndim != 3 or mixed.shape[-1] != model.config.d_pair:
            raise ValueError(f"invalid precomputed LM feature shape: {tuple(mixed.shape)}")
        original_language_forward = model.language_model.forward

        def precomputed_forward(value, *, lm_dropout=0.0):
            output = model.language_model.base_z_mlp(value)
            if lm_dropout > 0:
                output = functional.dropout(output, p=lm_dropout, training=True)
            return output

        model.language_model.forward = precomputed_forward
        forward_features["lm_hidden_states"] = mixed
        input_ids = None
    try:
        # ``__wrapped__`` removes only Biohub's top-level inference_mode.  All
        # layer implementations and orchestration remain upstream code.
        ESMFold2Model.forward.__wrapped__(
            model,
            **forward_features,
            input_ids=input_ids,
            num_loops=num_loops,
            num_diffusion_samples=1,
            num_sampling_steps=1,
        )
    except _TrunkReady:
        pass
    finally:
        if original_language_forward is not None:
            model.language_model.forward = original_language_forward
        for handle in handles:
            handle.remove()
    required = {
        "s_inputs",
        "relative_position_encoding",
        "z_trunk",
        "distogram_logits",
    }
    missing = required - captured.keys()
    if missing:
        raise RuntimeError(f"failed to capture released ESMFold2 trunk: {missing}")
    if not torch.is_grad_enabled():
        raise RuntimeError("ESMFold2 training forward requires gradients enabled")
    return captured


def _one_hot_reference_features(features: dict[str, Any]):
    import torch.nn.functional as functional
    from transformers.models.esmfold2.modeling_esmfold2_common import (
        CHAR_VOCAB_SIZE,
        MAX_ATOMIC_NUMBER,
    )

    atom_mask = features["atom_attention_mask"]
    mask_f = atom_mask.float()
    element = functional.one_hot(
        features["ref_element"].long(), num_classes=MAX_ATOMIC_NUMBER
    ).float()
    atom_name = functional.one_hot(
        features["ref_atom_name_chars"].long(), num_classes=CHAR_VOCAB_SIZE
    ).float()
    element = element * mask_f.unsqueeze(-1)
    atom_name = atom_name * mask_f.unsqueeze(-1).unsqueeze(-1)
    atom_to_token = features["atom_to_token"] * atom_mask.long()
    return element, atom_name, atom_to_token


def _distogram_loss(logits, target_coords, atom_mask, representative_atom_idx):
    import torch
    import torch.nn.functional as functional
    from transformers.models.esmfold2.modeling_esmfold2_common import (
        gather_rep_atom_coords,
    )

    representative_atom_idx = representative_atom_idx.long()
    representative = gather_rep_atom_coords(target_coords, representative_atom_idx)
    representative_mask = atom_mask.gather(1, representative_atom_idx)
    pair_mask = representative_mask[:, :, None] & representative_mask[:, None, :]
    pair_mask = pair_mask & ~torch.eye(
        pair_mask.shape[-1], device=pair_mask.device, dtype=torch.bool
    ).unsqueeze(0)
    distances = torch.cdist(
        representative.float(),
        representative.float(),
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    boundaries = torch.linspace(
        2.0, 22.0, logits.shape[-1] - 1, device=logits.device
    )
    labels = torch.bucketize(distances, boundaries)
    if not pair_mask.any():
        return logits.sum() * 0.0
    return functional.cross_entropy(logits[pair_mask], labels[pair_mask])


def training_forward(
    model,
    features: dict[str, Any],
    target_coords,
    target_atom_mask,
    *,
    recipe: ProjectRecipeConfig | None = None,
    num_loops: int = 0,
):
    """Return differentiable EDM/distogram losses for an official feature batch."""

    recipe = recipe or ProjectRecipeConfig()
    state = _run_released_trunk(model, features, num_loops=num_loops)
    atom_mask = target_atom_mask.bool() & features["atom_attention_mask"].bool()
    target = target_coords.float() * atom_mask.unsqueeze(-1)
    augmented, _ = model.structure_head._center_random_augmentation(
        target, atom_mask.float()
    )
    batch_size = augmented.shape[0]
    sigma = sample_noise_levels(batch_size, recipe, augmented.device)
    noisy = add_coordinate_noise(augmented, sigma)
    element, atom_name, atom_to_token = _one_hot_reference_features(features)

    diffusion = model.structure_head.diffusion_module(
        x_noisy=noisy,
        t_hat=sigma,
        ref_pos=features["ref_pos"],
        ref_charge=features["ref_charge"],
        ref_mask=features["atom_attention_mask"],
        ref_element=element,
        ref_atom_name_chars=atom_name,
        ref_space_uid=features["ref_space_uid"],
        tok_idx=atom_to_token,
        s_inputs=state["s_inputs"],
        s_trunk=None,
        z_trunk=state["z_trunk"].float(),
        relative_position_encoding=state["relative_position_encoding"],
        asym_id=features["asym_id"],
        residue_index=features["residue_index"],
        entity_id=features["entity_id"],
        token_index=features["token_index"],
        sym_id=features["sym_id"],
        token_attention_mask=features["token_attention_mask"],
        num_diffusion_samples=1,
    )
    predicted = diffusion["x_denoised"]
    diffusion_loss = edm_coordinate_loss(
        predicted,
        augmented,
        sigma,
        atom_mask,
        sigma_data=recipe.sigma_data,
    )
    distogram_loss = _distogram_loss(
        state["distogram_logits"],
        augmented,
        atom_mask,
        features["distogram_atom_idx"],
    )
    total = (
        recipe.diffusion_weight * diffusion_loss
        + recipe.distogram_weight * distogram_loss
    )
    return {
        "loss": total,
        "diffusion_loss": diffusion_loss,
        "distogram_loss": distogram_loss,
        "sigma_mean": sigma.mean().detach(),
        "predicted_coords": predicted,
    }
