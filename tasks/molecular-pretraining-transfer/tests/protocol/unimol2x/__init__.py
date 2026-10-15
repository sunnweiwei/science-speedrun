"""Uni-Core user-dir for the fixed fine-tune protocol, around the official Uni-Mol2 code (Uni-Mol 90f52c4 unimol2/).
No official file is edited.

* Registers the official Uni-Mol2 tasks, models and losses.
* Imports the submission's own user-dir (UNIMOL2X_SUBMISSION_USER_DIR, from encoder.json) when it has one, so a
  package can register its own architecture.
* Adds target normalisation statistics (train-split mean/std) for the suite's tasks that the official
  `task_metainfo` lacks (COMPAS-1D, ExpansionRx, ASAP), read from the data directory's task_metainfo.json
  (UNIMOL2X_METAINFO, default /data/task_metainfo.json). Official entries (QM9) are kept as they are.
"""
import importlib.util
import json
import os
import sys

sys.path.insert(0, "/opt/Uni-Mol/unimol2")
import unimol2  # noqa: E402,F401  (registers the official tasks, models and losses)
from unimol2.tasks import unimol_finetune as _ft  # noqa: E402

_sub = os.environ.get("UNIMOL2X_SUBMISSION_USER_DIR")
if _sub:
    _sub = os.path.abspath(_sub)
    _spec = importlib.util.spec_from_file_location(
        "submission_user_dir", os.path.join(_sub, "__init__.py"), submodule_search_locations=[_sub])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["submission_user_dir"] = _mod
    _spec.loader.exec_module(_mod)

_meta = os.environ.get("UNIMOL2X_METAINFO", "/data/task_metainfo.json")
if os.path.exists(_meta):
    for _k, _v in json.load(open(_meta)).items():
        _ft.task_metainfo.setdefault(_k, _v)
