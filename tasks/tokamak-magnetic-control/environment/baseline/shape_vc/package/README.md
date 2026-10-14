# Built package: full-shape virtual-circuit controller (classical, deployable)

`tokamak-policy/v1`, NumPy only, ~0.06 ms per step in-process. Built by `../build_package.py` from
`../jacobians.py` (model.npz) and `../estimator.py` (estimator.npz, 3 reference shapes); gains in
`params.json`. Learned parameters: estimator normal equations, 3 x (513 x 513 + 513 x 69) numbers
(0.90 M); no neural network. Sealed result and provenance: `../RESULTS.md`, `../README.md`.
