"""Pinned provenance and the only architecture scaling used for DPLM-2 50M.

The implementation remains ByteDance's ``EsmForDPLM2``.  The values below are
standard ESM config dimensions; no layer or loss is reimplemented locally.
"""

from __future__ import annotations


MODEL_NAME = "dplm2_official_50m_v1"
UPSTREAM_REPOSITORY = "https://github.com/bytedance/dplm"
UPSTREAM_REVISION = "8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d"
UPSTREAM_DPLM_MODEL = "airkingbd/dplm_150m"
UPSTREAM_DPLM_MODEL_REVISION = "49b7125a5d28c6418fcc2f3c4fe799352ac1488b"
UPSTREAM_DPLM2_MODEL = "airkingbd/dplm2_150m"
UPSTREAM_DPLM2_MODEL_REVISION = "3451d984d06497f835ed49634bd68c9dfb54d730"
STRUCT_TOKENIZER_MODEL = "airkingbd/struct_tokenizer"

# The folding core is the multimodal DPLM-2 network.  The frozen structure
# tokenizer is an external decoder and is recorded separately from this budget.
SCALED_ARCHITECTURE = {
    "hidden_size": 512,
    "intermediate_size": 2048,
    "num_hidden_layers": 12,
    "num_attention_heads": 8,
    "max_position_embeddings": 1026,
}
DPLM_VOCAB_SIZE = 33
DPLM2_VOCAB_SIZE = 8229
EXPECTED_FOLDING_CORE_RANGE = (46_500_000, 47_500_000)

# Values are transcribed from configs/experiment/dplm2/dplm2_150m.yaml at the
# pinned revision.  The verification utility composes that Hydra config and
# checks these fields before a run is admitted as strict.
LOCKED_RECIPE = {
    "num_diffusion_timesteps": 500,
    "single_modality_ratio": 0.25,
    "folding_loss_ratio": 0.25,
    "inverse_folding_loss_ratio": 0.25,
    "joint_loss_ratio": 0.25,
    "independent_loss_ratio": 0.0,
    "lora_rank": 16,
    "lora_dropout": 0.1,
    "lora_alpha": 32,
    "lora_target_module": "(esm.encoder.layer.[0-9]*.attention.(self.query|self.key|self.value|output.dense).*|esm.encoder.layer.[0-9]*.(intermediate|output).dense.*)",
    "modules_to_save": "lm_head,esm.embeddings",
    "criterion": "byprot.modules.cross_entropy.StructAARDMCrossEntropyLoss",
    "label_smoothing": 0.0,
    "cal_constant_loss": True,
    "learning_rate": 1e-4,
    "adam_betas": [0.9, 0.98],
    "weight_decay": 0.01,
    "scheduler": "polynomial",
    "warmup_steps": 2000,
    "total_steps": 100_000,
    "end_learning_rate": 1e-5,
    "warmup_initial_learning_rate": 1e-7,
    "gradient_clip": 0.5,
    "seed": 42,
}

LOCKED_SEQUENCE_RECIPES = {
    "mlm": {
        "experiment": "dplm/mlm_150m",
        "objective": "masked language modeling",
        "mask_probability": 0.15,
        "learning_rate": 4e-4,
        "warmup_steps": 2000,
        "total_steps": 500_000,
        "end_learning_rate": 4e-5,
    },
    "dplm": {
        "experiment": "dplm/dplm_150m_stage2",
        "objective": "random-mask discrete diffusion",
        "num_diffusion_timesteps": 500,
        "learning_rate": 4e-5,
        "warmup_steps": 2000,
        "total_steps": 100_000,
        "end_learning_rate": 1e-5,
    },
}


def scaled_config(source: dict, *, multimodal: bool) -> dict:
    """Return an upstream ESM config with only exposed scale fields changed."""
    result = dict(source)
    result.update(SCALED_ARCHITECTURE)
    result.update(
        {
            "architectures": ["EsmForDPLM2" if multimodal else "EsmForDPLM"],
            "dplm_type": "dplm2_esm" if multimodal else "dplm_esm",
            "vocab_size": DPLM2_VOCAB_SIZE if multimodal else DPLM_VOCAB_SIZE,
        }
    )
    return result
