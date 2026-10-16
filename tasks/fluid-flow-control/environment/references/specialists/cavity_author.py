"""Recover the author's 34-probe cavity example in current geometry units.

The old geometry has depth/referenceLength=20 and domainLength=90; current
assets have depth/referenceLength=1 and domainLength=4.5. Divide coordinates
by 20, not velocities, to preserve the author's nondimensional sensor positions.
This example is public, but is not identified as the final paper figure setup.
"""

import ast
import hashlib
from pathlib import Path

import numpy as np


def load(author_root):
    path = Path(author_root) / "maiaGym/maia_env_core.py"
    expected = "9c01ccb0c7e87a8ee00e424247acbda60d2b35ef812055f0f32814cf1440202e"
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Author cavity source differs from the audited commit")
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    vectors = {}
    for node in main.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in ("obs_loc", "obs_scale", "reward_loc", "reward_scale"):
                vectors[target.id] = np.asarray(ast.literal_eval(node.value), dtype=float)
    if set(vectors) != {"obs_loc", "obs_scale", "reward_loc", "reward_scale"}:
        raise ValueError("Author example normalization vectors were not found")
    if any(x.shape != (68,) or not np.isfinite(x).all() for x in vectors.values()):
        raise ValueError("Expected 68 finite author-example values")
    if any((vectors[k] <= 0).any() for k in ("obs_scale", "reward_scale")):
        raise ValueError("Invalid author-example normalization scales")
    points = []
    for x in np.linspace(26., 42., 5):
        fraction = (x - 26.) / 16.
        ymin = -.625 + fraction * (-1.5 + .625)
        ymax = .625 + fraction * (1.5 - .625)
        points.extend((x, y) for y in np.linspace(ymin, ymax, 5))
    # np.meshgrid(...).ravel() in the author's second, near-wall probe array.
    points.extend((x, y) for y in np.linspace(-5., -2.5, 3)
                  for x in np.linspace(43., 43.8, 3))
    probes = (np.asarray(points) / 20.).ravel().tolist()
    audit = {"status": "rescaled-public-author-example-not-verified-figure-setup",
        "source": "maiaGym/maia_env_core.py:main",
        "author_revision": "20a483ecc6ea711f6b380fff05f212ca4d7e2c67",
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "coordinate_divisor": 20., "velocity_vector_scale": 1.,
        "old_depth": 20., "new_depth": 1., "old_domain_length": 90., "new_domain_length": 4.5,
        "probe_count": 34, "observations": ["u", "v"],
        "unresolved": ["original target calibration duration and Reynolds number",
                       "figure-specific sensors and reward aggregation",
                       "paper prose domain differs from both public geometries"]}
    return probes, vectors, audit
