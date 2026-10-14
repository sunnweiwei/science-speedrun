"""Install the local DPLM data adapter in fresh Lightning DDP ranks only."""

from __future__ import annotations

import os


if os.environ.get("AUTOFOLD_DPLM_DATA_ADAPTER") == "1":
    from official_baselines.dplm2.data_adapter import install

    install(os.environ.get("AUTOFOLD_DPLM_SEQUENCE_TOKENIZER") or None)
