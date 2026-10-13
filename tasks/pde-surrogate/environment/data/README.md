# PDE surrogate data

`/workspace/data` holds this file, `AUDIT.md`, the task's split (`split.json`, and `split_smoke.json`
for maintainer smoke runs), the source lock (`sources.lock.json`), and four read-only datasets
prepared before the session:

```text
/workspace/data/
  ape2d-full/<name>.hdf5, <name>.json              pretraining corpus (training heads of the split's cut files)
  pdearena-ns/NavierStokes2D_{train,valid}_*.h5    PDEArena Navier-Stokes pool shards
  well-active_matter/data/{train,valid}/*.hdf5     The Well: active matter
  well-turbulent_radiative_layer_2D/data/{train,valid}/*.hdf5
  materialize-receipt.json, release-receipt.json   what was downloaded, with checksums
```

The held-out cases' targets are not here: files the split cuts are truncated to their training
heads, and the transfer datasets' test files are absent. Every source is read with the task's
own reader, `/workspace/evaluation/pde_io.py`.

## Pinned sources

All sources are Hugging Face datasets pinned to a full commit SHA in
[`sources.lock.json`](sources.lock.json); `build_sources_lock.py` regenerates
the lock from the API. See [`AUDIT.md`](AUDIT.md) for how each source relates
to its paper and public code.

| id | repository | revision | license | distinct HDF5 | bytes |
| --- | --- | --- | --- | ---: | ---: |
| `ape2d` (reduced release; plumbing only, a different draw from the full release) | `thuerey-group/pde-transformer-ape2d` | `abd0a02dab9ee28dda6cc7ef5cbaa95ce414c942` | MIT | 27 | 9.28 GB |
| `ape2d-full` | `thuerey-group/pde-transformer-ape2d-full` | `fa711d56a1a58705c3d8e889830b459fd92733f0` | MIT | 24 | 78.65 GB |
| `well-active_matter` | `polymathic-ai/active_matter` | `dc3e9135a75b4e5a0d979086219fc97fe668f7a3` | CC-BY-4.0 | 82 | 55.81 GB |
| `well-turbulent_radiative_layer_2D` | `polymathic-ai/turbulent_radiative_layer_2D` | `2ee7756575ff6f90981d0308cbcb6a2ab5995bdc` | CC-BY-4.0 | 27 | 7.85 GB |
| `pdearena-ns` | `pdearena/NavierStokes-2D` | pinned in the lock | MIT | 156 shards | 42.96 GB |
| `well-rayleigh_benard` | `polymathic-ai/rayleigh_benard` | `10e143ebedae8a9b1c699b95d5b1eb8feaae09b5` | CC-BY-4.0 | 105 | 369.94 GB |
| `well-shear_flow` | `polymathic-ai/shear_flow` | `fc867f856f306905cf94c1f5df978cc518a2048c` | CC-BY-4.0 | 84 | 471.17 GB |

The pretraining corpus is the data of PDE-Transformer (Holzschuh, Liu, Kohl,
Thuerey, ICML 2025), generated with `exponax` 0.1.0. The transfer datasets are
from The Well (Ohana et al., NeurIPS 2024 D&B); PDEArena's Navier-Stokes set
(Gupta & Brandstetter, 2022) is the third transfer source. All are
redistributable under their licenses; this task republishes only what its split
creates (the truncated heads and held-out tails of nine corpus files, see "The
published release") and downloads everything else from the pinned upstreams.
Nothing under this directory is committed except code, the lock and
documentation. The full corpus release ships `hyp.hdf5` as a content copy of
`diff.hdf5`; the split excludes it (`AUDIT.md` §7).

## Array layouts

**Pretraining corpus** (PBDL format): one HDF5 per PDE; group `sims` with the
PDE metadata as attributes (`PDE`, `Fields`, `Fields Scheme`, `Resolution`,
`Time Steps`, `Dt`, `Domain Extent`, `Boundary Conditions`, `Constants`);
datasets `sims/sim<i>` of shape `(T, C, 256, 256)` float32 with the varying
constants as per-simulation attributes. Training files hold 30 steps; the
long-rollout test files hold 200 (`ks`, `decay_turb`, `kolm_flow`) or 100
(`gs_alpha`, `gs_beta`, `gs_gamma`, `gs_epsilon`) steps. Top-level `norm_*`
datasets are the upstream normalization buffers; this task computes its own.

**The Well**: one HDF5 per physical parameter combination with root attributes
`n_trajectories` and `simulation_parameters`; `t0_fields/<f>` `(n, T, X, Y)`,
`t1_fields/<f>` `(n, T, X, Y, 2)`, `t2_fields/<f>` `(n, T, X, Y, 2, 2)`;
`boundary_conditions/<name>/mask`; `dimensions/{time, x, y}`; `scalars`.
Channels are flattened in the Well's own order: t0 fields, then each t1 field
as `(x, y)`, then each t2 field as `(xx, xy, yx, yy)`.

| dataset | trajectory `(T, C, X, Y)` | files per split (train/valid/test) | trajectories per split |
| --- | --- | --- | --- |
| `active_matter` | `(81, 11, 256, 256)` | 45 / 16 / 21 | 175 / 24 / 26 |
| `turbulent_radiative_layer_2D` | `(101, 4, 128, 384)` | 9 / 9 / 9 | 72 / 9 / 9 |
| `rayleigh_benard` | `(200, 4, 512, 128)` | 35 / 35 / 35 | 1,400 / 175 / 175 |
| `shear_flow` | `(200, 4, 256, 512)` | 28 / 28 / 28 | 896 / 112 / 112 |

Trajectory counts per file vary (1 to 4 for active matter, 40 per training
file for Rayleigh-Bénard, 32 for shear flow); the file is the download unit.

**PDEArena** shards hold `u`, `vx`, `vy` of shape `(n, 14, 128, 128)` in
float64 (converted to float32), one root group named by split; a closed domain
with Dirichlet velocity and Neumann scalar boundaries, labelled `wall` in the
case metadata. Active matter is scored on all 11 channels, including the
orientation and strain tensors the PDE-Transformer paper dropped.

## Splits and visibility

The task's own trajectory-level split for training, validation and held-out
evaluation is `split.json` under this directory (`make_split.py` regenerates it
from the lock; `AUDIT.md` §5 records the decisions). Upstream data are public, so evaluation is
benchmark-visible: the verifier replays the delivered package on trajectories
the agent could in principle have downloaded, and the rules in
`instruction.md` forbid using them. The materializer can prepare development
mounts that omit held-out trajectories.

