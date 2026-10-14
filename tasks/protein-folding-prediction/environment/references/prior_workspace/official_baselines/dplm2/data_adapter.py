"""Runtime-only adapter from the unified local split to DPLM's batch schemas."""

from __future__ import annotations

import os
from pathlib import Path


def _load_local_split(data_path: str, split: str):
    from datasets import load_from_disk

    path = Path(data_path) / split
    if not path.is_dir():
        raise FileNotFoundError(f"unified DPLM dataset split is missing: {path}")
    return load_from_disk(str(path))


def install(sequence_tokenizer: str | Path | None = None) -> None:
    """Replace only upstream dataset lookup; model/training code is untouched."""
    import byprot.datamodules.dataset.tokenized_protein as tokenized_protein
    import byprot.datamodules.dataset.uniref_hf as uniref_hf

    tokenized_protein.load_dataset_from_hf = _load_local_split
    uniref_hf.load_dataset_from_hf = _load_local_split
    if sequence_tokenizer is not None:
        tokenizer_path = str(Path(sequence_tokenizer))
        original_collater_init = uniref_hf.DPLMCollater.__init__

        def local_collater_init(self, requested_path=None):
            # Upstream hard-codes the public ESM2-150M tokenizer when this
            # argument is omitted.  The vocabulary is identical, but the
            # frozen local config avoids network access and binds tokenization
            # to the scaled model artifact.
            original_collater_init(self, requested_path or tokenizer_path)

        uniref_hf.DPLMCollater.__init__ = local_collater_init


def install_subprocess_bootstrap(
    sequence_tokenizer: str | Path | None = None,
) -> None:
    """Make the adapter survive Lightning's fresh DDP Python processes.

    The one-GPU path can monkey-patch the imported upstream modules directly.
    Lightning's ``ddp`` strategy starts each rank in a fresh interpreter, so
    those children must install the same read-only adapter during Python
    startup.  The narrowly-scoped sitecustomize directory is added only to
    child ``PYTHONPATH`` and is activated by an explicit environment flag.
    """
    bootstrap = Path(__file__).with_name("ddp_bootstrap")
    if not (bootstrap / "sitecustomize.py").is_file():
        raise FileNotFoundError(f"DPLM DDP bootstrap is missing: {bootstrap}")
    current = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if str(bootstrap) not in entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(bootstrap), *entries])
    os.environ["AUTOFOLD_DPLM_DATA_ADAPTER"] = "1"
    if sequence_tokenizer is not None:
        os.environ["AUTOFOLD_DPLM_SEQUENCE_TOKENIZER"] = str(sequence_tokenizer)
