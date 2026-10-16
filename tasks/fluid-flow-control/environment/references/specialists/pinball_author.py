"""Read audited Re100 author observation statistics without importing legacy Gym."""

import ast
import hashlib
from pathlib import Path

import numpy as np


def load(author_root):
    path = Path(author_root) / "maiaGym/envs/pinball/pinball_environment.py"
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != "e3ae3696a32b5bdb313ca12c4e97483ac5918da3690055c70a72c7196ea2d320":
        raise ValueError("Pinball author source differs from audited commit")
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "Pinball")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    mapping = next(n.value for n in init.body if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Attribute) and n.targets[0].attr == "environment_default_values")
    case = next(v for k, v in zip(mapping.keys, mapping.values) if ast.literal_eval(k) == (100, 2))
    vectors = {}
    for key, value in zip(case.keys, case.values):
        name = ast.literal_eval(key)
        if name in ("obs_loc", "obs_scale"):
            if not isinstance(value, ast.Call) or ast.unparse(value.func) != "np.array":
                raise ValueError("Unexpected source observation-vector construction")
            vectors[name] = np.asarray(ast.literal_eval(value.args[0]), dtype=float)
    if set(vectors) != {"obs_loc", "obs_scale"} or any(v.shape != (98,) or not np.isfinite(v).all() for v in vectors.values()):
        raise ValueError("Expected 98 finite Re100 author statistics")
    if (vectors["obs_scale"] <= 0).any():
        raise ValueError("Invalid source observation scales")
    # Exact default np.meshgrid order: y is the outer dimension, not x.
    probes = [(float(x), float(y)) for y in np.linspace(-1.5, 1.5, 7)
              for x in np.linspace(1.5, 7.5, 7)]
    audit = {"status": "public-author-default-normalization-not-verified-final-figure",
        "author_revision": "20a483ecc6ea711f6b380fff05f212ca4d7e2c67",
        "source_sha256": sha, "reynolds": 100, "probe_order": "y-major meshgrid",
        "source": "maiaGym/envs/pinball/pinball_environment.py:Pinball.environment_default_values[(100,2)]"}
    return probes, vectors, audit
