"""Build the pinned QRENND LSTM with the paper's experimental-noise widths.

The native model implementation and global-data-parity training objective are
unchanged. A circuit-derived detector parity converts between native targets and
official path-observable labels. Our detector representation and bounded training
protocol are task adaptations, so results are not directly comparable with
published ones.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tensorflow import keras


REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
UPSTREAM_REPOSITORY = "https://github.com/Hossein-D/qrennd"
UPSTREAM_COMMIT = "c2ed73e3ec6ee843293c8839bf059c63b2960e61"
PAPER_URL = "https://doi.org/10.1103/PhysRevResearch.7.013029"
DEFAULT_WIDTHS = {3: 64, 5: 253}
EXPECTED_PARAMETER_COUNTS = {3: 60_418, 5: 926_488}


def model_config(
    distance: int,
    learning_rate: float = 0.0005,
    width: int | None = None,
) -> dict[str, Any]:
    """Return the native model/train settings, suitable for recording as JSON."""
    if distance not in DEFAULT_WIDTHS:
        raise ValueError(f"Only distances 3 and 5 are supported, received {distance}")
    if width is None:
        width = DEFAULT_WIDTHS[distance]
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")

    return {
        "model": {
            "type": "LSTM",
            "LSTM": {"units": [width, width], "dropout_rates": [None, 0.05]},
            "main_eval": {
                "units": [width, 1],
                "dropout_rates": [0.05, None],
                "l2_factor": None,
            },
            "aux_eval": {
                "units": [width, 1],
                "dropout_rates": [0.05, None],
                "l2_factor": None,
            },
        },
        "train": {
            "optimizer": {"learning_rate": learning_rate},
            "loss": {
                "main_output": "binary_crossentropy",
                "aux_output": "binary_crossentropy",
            },
            "loss_weights": {"main_output": 1.0, "aux_output": 0.5},
            "metrics": {"main_output": "accuracy", "aux_output": "accuracy"},
        },
        "dataset": {"input": "defects"},
    }


def _native_qrennd() -> ModuleType:
    """Import our unchanged source without resolving upstream dependency pins."""
    expected_source = REFERENCE_DIR / "qrennd" / "__init__.py"
    existing = sys.modules.get("qrennd")
    if existing is None:
        sys.path.insert(0, str(REFERENCE_DIR))
        try:
            existing = importlib.import_module("qrennd")
        finally:
            sys.path.remove(str(REFERENCE_DIR))
    module_path = getattr(existing, "__file__", None)
    if module_path is None or Path(module_path).resolve() != expected_source.resolve():
        raise RuntimeError(
            "A different qrennd package was already imported; start a fresh Python "
            "process so the baseline uses the pinned vendored implementation."
        )
    return existing


def build_model(
    distance: int,
    seed: int = 0,
    learning_rate: float = 0.0005,
    width: int | None = None,
) -> keras.Model:
    """Return a compiled native QRENND model with main and auxiliary outputs.

    Inputs are ``rec_input`` of shape ``(batch, rounds, distance**2 - 1)``
    and ``eval_input`` of shape ``(batch, (distance**2 - 1) // 2)``. Both
    outputs train against the same native global-data-parity flip label. Only
    ``main_output`` is decoded, then converted to the official path observable.
    Seeding also resets Python and NumPy RNGs via Keras.
    """
    setup = model_config(distance, learning_rate, width)
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    native = _native_qrennd()
    config = native.Config(
        experiment="sycamore-surface-code",
        run=f"d{distance}-seed{seed}",
        setup=setup,
        data_dir=".",
        add_timestamp=False,
        add_unique_identifier=False,
        seed=seed,
    )
    return native.get_model(
        rec_features=distance**2 - 1,
        eval_features=(distance**2 - 1) // 2,
        config=config,
    )
