#!/usr/bin/env python3
"""Run the official Protenix trainer for the project-owned hybrid config."""

from first_party_baselines.hybrid_esmc_pairformer.config import (
    enforce_cli_model_name,
    register_model,
)
from first_party_baselines.hybrid_esmc_pairformer.data_adapter import (
    install_cached_feature_adapter,
    pop_manifest_argument,
)
from official_baselines.protenix.data_guard import enforce as enforce_data_policy
from official_baselines.protenix.resume import install as install_resume


def main() -> None:
    manifest = pop_manifest_argument()
    enforce_data_policy()
    register_model()
    install_cached_feature_adapter(manifest)
    enforce_cli_model_name()
    install_resume()
    from runner.train import main as upstream_main

    upstream_main()


if __name__ == "__main__":
    main()
