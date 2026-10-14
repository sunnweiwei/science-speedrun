#!/usr/bin/env python3
"""Run the unmodified upstream Protenix trainer with the registered 50M config."""

from official_baselines.protenix.config import enforce_cli_model_name, register_model
from official_baselines.protenix.data_guard import enforce as enforce_data_policy
from official_baselines.protenix.resume import install as install_resume


def main() -> None:
    enforce_data_policy()
    register_model()
    enforce_cli_model_name()
    install_resume()
    from runner.train import main as upstream_main

    upstream_main()


if __name__ == "__main__":
    main()
