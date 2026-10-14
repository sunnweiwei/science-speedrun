"""Training orchestration for the released OpenDDE module graph.

OpenDDE v1 exposes only inference and disables gradients around its recycling
loop.  The functions here preserve the exact released layer ordering while
running that orchestration with gradients and applying the shared project EDM
objective.  No network layer is reimplemented locally.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

from official_baselines.project_recipe import (
    ProjectRecipeConfig,
    add_coordinate_noise,
    edm_coordinate_loss,
    sample_noise_levels,
)


def build_model():
    from opendde.model.opendde import OpenDDE

    from official_baselines.opendde.config import scaled_config

    return OpenDDE(scaled_config())


def grad_enabled_pairformer_output(
    model,
    input_feature_dict: dict[str, Any],
    *,
    n_cycle: int = 1,
    chunk_size: int | None = None,
):
    """Released ``get_pairformer_output`` ordering with its no-grad guard removed."""

    import torch

    if n_cycle < 1:
        raise ValueError("OpenDDE requires at least one recycling cycle")
    if not torch.is_grad_enabled():
        raise RuntimeError("OpenDDE training forward requires gradients enabled")
    s_inputs = model.input_embedder(
        input_feature_dict, inplace_safe=False, chunk_size=chunk_size
    )
    s_init = model.linear_no_bias_sinit(s_inputs)
    z_init = (
        model.linear_no_bias_zinit1(s_init)[..., None, :]
        + model.linear_no_bias_zinit2(s_init)[..., None, :, :]
    )
    z_init = z_init + model.relative_position_encoding(input_feature_dict["relp"])
    z_init = z_init + model.linear_no_bias_token_bond(
        input_feature_dict["token_bonds"].unsqueeze(dim=-1)
    )
    z = torch.zeros_like(z_init)
    s = torch.zeros_like(s_init)
    for _ in range(n_cycle):
        z = z_init + model.linear_no_bias_z_cycle(model.layernorm_z_cycle(z))
        if model.template_embedder.n_blocks > 0:
            z = z + model.template_embedder(
                input_feature_dict,
                z,
                triangle_multiplicative=model.configs.triangle_multiplicative,
                triangle_attention=model.configs.triangle_attention,
                inplace_safe=False,
                chunk_size=chunk_size,
            )
        # The preview release routes MSAStack.forward to a slice-inplace
        # inference implementation even when autograd is on.  Use the exact
        # same two released sublayers in functional residual form for training.
        def grad_safe_msa_stack(stack, m, pair, **_kwargs):
            m = m + stack.msa_pair_weighted_averaging(m, pair)
            return m + stack.transition_m(m)

        original_msa_forwards = []
        for block in model.msa_module.blocks:
            stack = block.msa_stack
            original_msa_forwards.append((stack, stack.forward))
            stack.forward = MethodType(grad_safe_msa_stack, stack)
        try:
            z = model.msa_module(
                input_feature_dict,
                z,
                s_inputs,
                pair_mask=None,
                triangle_multiplicative=model.configs.triangle_multiplicative,
                triangle_attention=model.configs.triangle_attention,
                inplace_safe=False,
                chunk_size=chunk_size,
            )
        finally:
            for stack, forward in original_msa_forwards:
                stack.forward = forward
        s = s_init + model.linear_no_bias_s(model.layernorm_s(s))
        s, z = model.pairformer_stack(
            s,
            z,
            pair_mask=None,
            triangle_multiplicative=model.configs.triangle_multiplicative,
            triangle_attention=model.configs.triangle_attention,
            inplace_safe=False,
            chunk_size=chunk_size,
        )
    return s_inputs, s, z


def _distogram_loss(model, feature_dict, pair_z, target_coords, target_atom_mask):
    import torch
    import torch.nn.functional as functional

    logits = model.distogram_head(pair_z)
    rep_mask = feature_dict["distogram_rep_atom_mask"].to(target_coords.dtype)
    if rep_mask.ndim == 1:
        selected = rep_mask.bool()
        representative = target_coords[selected]
        counts = target_atom_mask[selected].to(rep_mask.dtype)
    elif rep_mask.ndim == 2:
        representative = rep_mask @ target_coords
        counts = rep_mask @ target_atom_mask.to(rep_mask.dtype)
    else:
        representative = torch.einsum("...ta,...ad->...td", rep_mask, target_coords)
        counts = torch.einsum(
            "...ta,...a->...t", rep_mask, target_atom_mask.to(rep_mask.dtype)
        )
    representative = representative / counts.clamp_min(1.0).unsqueeze(-1)
    pair_mask = (counts > 0).unsqueeze(-1) & (counts > 0).unsqueeze(-2)
    eye = torch.eye(pair_mask.shape[-1], device=pair_mask.device, dtype=torch.bool)
    pair_mask = pair_mask & ~eye
    distances = torch.cdist(
        representative.float(),
        representative.float(),
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    boundaries = torch.linspace(
        2.25, 25.75, logits.shape[-1] - 1, device=logits.device
    )
    labels = torch.bucketize(distances, boundaries)
    if not pair_mask.any():
        return logits.sum() * 0.0
    return functional.cross_entropy(logits[pair_mask], labels[pair_mask])


def training_forward(
    model,
    input_feature_dict: dict[str, Any],
    target_coords,
    target_atom_mask,
    *,
    recipe: ProjectRecipeConfig | None = None,
    n_cycle: int = 1,
    chunk_size: int | None = None,
):
    """Run one differentiable native OpenDDE denoising training example."""

    from opendde.model.opendde import update_input_feature_dict
    from opendde.model.utils import centre_random_augmentation

    recipe = recipe or ProjectRecipeConfig()
    features = dict(input_feature_dict)
    features = model.relative_position_encoding.generate_relp(features, lazy=False)
    features = update_input_feature_dict(features)
    residue_features = features
    residue_s_inputs, residue_s, residue_z = grad_enabled_pairformer_output(
        model,
        residue_features,
        n_cycle=n_cycle,
        chunk_size=chunk_size,
    )
    structural_features, structural_s_inputs, structural_s, structural_z = (
        model.expand_to_structural_tokens(
            input_feature_dict=residue_features,
            s_inputs=residue_s_inputs,
            s=residue_s,
            z=residue_z,
            inplace_safe=False,
            chunk_size=chunk_size,
        )
    )
    pair_features, _pair_s_inputs, _pair_s, pair_z = model.select_pair_output_branch(
        residue_feature_dict=residue_features,
        residue_s_inputs=residue_s_inputs,
        residue_s=residue_s,
        residue_z=residue_z,
        structural_feature_dict=structural_features,
        structural_s_inputs=structural_s_inputs,
        structural_s=structural_s,
        structural_z=structural_z,
    )

    atom_mask = target_atom_mask.bool() & structural_features["ref_mask"].bool()
    target = target_coords.float() * atom_mask.unsqueeze(-1)
    augmented = centre_random_augmentation(
        target,
        N_sample=1,
        mask=atom_mask.float(),
    )
    leading_batch = 1 if augmented.ndim == 3 else augmented.shape[0]
    sigma_flat = sample_noise_levels(leading_batch, recipe, augmented.device)
    sigma = sigma_flat if augmented.ndim == 3 else sigma_flat.unsqueeze(-1)
    noisy = add_coordinate_noise(augmented, sigma)
    predicted = model.diffusion_module(
        x_noisy=noisy,
        t_hat_noise_level=sigma,
        input_feature_dict=structural_features,
        s_inputs=structural_s_inputs,
        s_trunk=structural_s,
        z_trunk=structural_z,
        pair_z=None,
        p_lm=None,
        c_l=None,
        inplace_safe=False,
        chunk_size=chunk_size,
        use_conditioning=True,
        enable_efficient_fusion=False,
    )
    diffusion_loss = edm_coordinate_loss(
        predicted,
        augmented,
        sigma,
        atom_mask,
        sigma_data=recipe.sigma_data,
    )
    distogram_loss = _distogram_loss(
        model,
        pair_features,
        pair_z,
        target_coords.float(),
        target_atom_mask.bool(),
    )
    total = (
        recipe.diffusion_weight * diffusion_loss
        + recipe.distogram_weight * distogram_loss
    )
    return {
        "loss": total,
        "diffusion_loss": diffusion_loss,
        "distogram_loss": distogram_loss,
        "sigma_mean": sigma.float().mean().detach(),
        "predicted_coords": predicted,
    }
