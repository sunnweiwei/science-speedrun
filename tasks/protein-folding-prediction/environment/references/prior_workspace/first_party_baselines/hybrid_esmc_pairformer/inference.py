#!/usr/bin/env python3
"""Run official Protenix inference for the project-owned hybrid config."""

from first_party_baselines.hybrid_esmc_pairformer.config import (
    INFERENCE_MODEL_NAME,
    enforce_cli_model_name,
    register_model,
)
from first_party_baselines.hybrid_esmc_pairformer.data_adapter import (
    install_cached_feature_adapter,
    pop_manifest_argument,
)


def main() -> None:
    manifest = pop_manifest_argument()
    register_model(INFERENCE_MODEL_NAME)
    install_cached_feature_adapter(manifest)
    enforce_cli_model_name(INFERENCE_MODEL_NAME)
    from runner.inference import run as upstream_run

    upstream_run()


if __name__ == "__main__":
    main()
