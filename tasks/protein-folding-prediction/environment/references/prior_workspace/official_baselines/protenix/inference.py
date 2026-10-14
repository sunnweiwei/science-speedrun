#!/usr/bin/env python3
"""Run unmodified upstream Protenix inference with the registered 50M config."""

from official_baselines.protenix.config import enforce_cli_model_name, register_model


def main() -> None:
    register_model()
    enforce_cli_model_name()
    from runner.inference import run as upstream_run

    upstream_run()


if __name__ == "__main__":
    main()
