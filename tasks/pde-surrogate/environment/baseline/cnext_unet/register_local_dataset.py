"""Register a locally converted dataset with The Well's name check, then run its trainer.

The Well's `WellDataset` asserts the dataset name is one of the released ones
(`the_well.data.utils.WELL_DATASETS`), so a dataset converted into The Well's
layout by `convert_pdearena_to_well.py` cannot be trained with the upstream
recipe as it stands. This adds the extra names to that list and nothing else,
so the recipe, the model and the trainer are still upstream's:

    WELL_EXTRA_DATASETS=pdearena_navier_stokes_2d \
        python3 -c "import register_local_dataset" -m the_well.benchmark.train ...

`train_dataset.sh` does this for datasets that are not part of The Well.
"""
import os
import runpy
import sys

from the_well.data import utils as _utils

for name in os.environ.get("WELL_EXTRA_DATASETS", "").split(","):
    if name and name not in _utils.WELL_DATASETS:
        _utils.WELL_DATASETS.append(name)
        print(f"registered local dataset {name!r} with The Well", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "-m":
        module = sys.argv[2]
        sys.argv = [module] + sys.argv[3:]
        runpy.run_module(module, run_name="__main__")
