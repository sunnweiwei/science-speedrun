"""Approximately-50M Biohub ESMFold2 graph for final-hidden inputs."""

ESM_REPOSITORY = "https://github.com/Biohub/esm"
ESM_REVISION = "917af90b624535eed1e072d343c717e3ec11fef4"
TRANSFORMERS_REPOSITORY = "https://github.com/Biohub/transformers"
TRANSFORMERS_REVISION = "ef32577f55da19a4989cd7b22e004dc43a4998cb"
MODEL_REPOSITORY = "biohub/ESMFold2-Fast"
MODEL_REVISION = "b28d8ace5e05e61e5bec1e6820cfd3e221819d12"
ESMC_REPOSITORY = "biohub/ESMC-6B"
ESMC_REVISION = "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
DEFAULT_LOOPS = 3
DEFAULT_SAMPLING_STEPS = 50

# A configuration-only scale-down of the released ESMFold2 module graph. ESMC
# is external input data. Unlike the historical mixed-layer adapter, this model
# consumes one final normalized [L,2560] state and trains its own projection.
MODEL_NAME_50M = "esmfold2_final_hidden_50m_v1"
EXPECTED_PARAMETER_RANGE = (45_000_000, 55_000_000)


def scaled_config():
    """Build an approximately-49M ESMFold2 architecture without loading ESMC.

    Only upstream-exposed dimensions and block counts are changed.  Every
    trainable layer is still instantiated by Biohub's ``ESMFold2Model``.
    """

    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config

    return ESMFold2Config(
        type="release",
        d_single=256,
        d_pair=256,
        num_loops=3,
        num_diffusion_samples=1,
        lm_d_model=2560,
        # LanguageModelShim therefore expects [B,L,1,2560]. Its projection and
        # singleton mixture weight are randomly initialized candidate weights.
        lm_num_layers=0,
        esmc_id=ESMC_REPOSITORY,
        folding_trunk={"n_layers": 8, "n_heads": 8, "dropout": 0.0},
        lm_encoder={
            "enabled": True,
            "n_layers": 2,
            "lm_dropout": 0.25,
            "per_loop_lm_dropout": True,
        },
        parcae={
            "enabled": True,
            "poisson_mean": 3.0,
            "min_steps": 1,
            "max_steps": 6,
            "coda_n_layers": 2,
        },
        inputs={
            # Upstream InputsEmbedder emits d_token/2 + 2*33 + 1 channels.
            "d_inputs": 259,
            "atom_encoder": {
                "d_atom": 128,
                "d_token": 384,
                "n_blocks": 2,
                "n_heads": 4,
            },
        },
        structure_head={
            "diffusion_module": {
                "sigma_data": 16.0,
                "c_atom": 128,
                "c_token": 384,
                "c_z": 256,
                "c_s_inputs": 259,
                "fourier_dim": 128,
                "atom_num_blocks": 2,
                "atom_num_heads": 4,
                "token_num_blocks": 8,
                "token_num_heads": 8,
                "transition_multiplier": 2,
            },
            "distogram_bins": 128,
            "train_noise_log_mean": -1.2,
            "train_noise_log_std": 1.5,
        },
        confidence_head={
            "enabled": True,
            "folding_trunk": {"n_layers": 2, "n_heads": 8, "dropout": 0.0},
        },
        msa_encoder={"enabled": False},
    )
