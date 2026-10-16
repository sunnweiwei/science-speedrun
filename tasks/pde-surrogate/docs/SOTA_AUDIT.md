# Benchmark and model audit for the PDE surrogate task

Produced 2026-09-16 by an automated literature and API audit (eleven research passes, one per benchmark source or model family, followed by a synthesis). Every Hugging Face revision, byte count and license marked **[V]** was read from the Hub API on that date; **[A]** marks facts taken from the individual passes; **[U]** marks unverified claims. The task's own decisions are in `../data/AUDIT.md` and the task README; this document is the evidence they cite.

# `pde-surrogate` — Data & Model Synthesis for the Task Designer

**Verification date: 2026-09-16.** Every SHA, byte count and split size in §1 marked **[V]** was re-verified this session against `https://huggingface.co/api/datasets/<id>?blobs=true` (and `/paths-info/<sha>` for the duplication check). Items marked **[A]** are inherited from the prior audits (PDE-Transformer, The Well, DPOT, PDEBench, PDEArena, ShallowWater, FNO/TFNO, Poseidon, PINNs, PROSE, MeshGraphNets, ReactionDiffusion, NavierStokes-FNO) and were *not* independently re-fetched. Items marked **[U]** are explicitly unverified.

---

## 1. Candidate data sources for a multi-source pinned materializer

### 1a. INCLUDE

| Source (HF id) | PDEs | Host + pin (verified?) | Download for a sensible subset | License | Format | Recommendation & reason |
|---|---|---|---|---|---|---|
| `thuerey-group/pde-transformer-ape2d` | 12 scalar/vector families at 256² (`adv`, `adv_diff`, `burgers`, `decay_turb`, `diff`, `disp`, `fisher`, `hyp`, `kdv`, `kolm_flow`, `ks`, `sh`) + Gray-Scott in 8 parameter regimes (`gs_{alpha,beta,gamma,delta,epsilon,theta,iota,kappa}`) | HF, sha `abd0a02dab9ee28dda6cc7ef5cbaa95ce414c942` **[V]**; 83 files, **18.719 GB** repo total, ungated | **Unique payload is only 9.281 GB** — every `.hdf5` exists twice (top level and `<name>/<name>.hdf5`) with **identical LFS sha256** (verified: `adv.hdf5` and `adv/adv.hdf5` both `debd285a…eac50`; `gs_theta.hdf5` and `gs_theta/gs_theta.hdf5` both `7b8892d7…a207`) **[V]**. Per-PDE files 0.157 GB (`gs_*`) to 0.944 GB (`kdv`, `burgers`) **[V]**. Dedicated held-out files exist: `decay_turb_test`, `kolm_flow_test`, `ks_test` (0.262 GB each), `gs_{alpha,beta,epsilon,gamma}_test` (0.157 GB each) — 7 files, 2.989 GB **[V]** | MIT **[V]** | HDF5 + `meta_all.json` sidecar (`{PDE, Fields, Dt 0.01, isSingleFile}`) | **INCLUDE — primary backbone.** Only candidate that is simultaneously MIT, HF-SHA-pinned, uniform 256², multi-PDE, and cheap enough that an agent can hold the *whole* corpus. Ships its own `*_test.hdf5` files, so the held-out split is upstream rather than invented |
| `polymathic-ai/turbulent_radiative_layer_2D` | Radiatively-cooling turbulent mixing layer (compressible hydro), 128×384, 101 steps, 4 fields, 90 traj, 9 `t_cool` values | HF, sha `2ee7756575ff6f90981d0308cbcb6a2ab5995bdc` **[V]**; 34 files, **7.856 GB** total, ungated | **test split = 0.981 GB (9 files)**; train 5.889 GB; valid 0.981 GB **[V]**. One file per `t_cool` ∈ {0.03…3.16} → per-parameter granularity | CC-BY-4.0 **[V]** | HDF5 (Well spec: `t0_fields`/`t1_fields`, `boundary_conditions`, `scalars`) | **INCLUDE.** Smallest Well 2D set. It is the *only* candidate where published **VRMSE** exists in both the one-step and the K-step rollout form and where no baseline diverges: FNO 1.79/3.54, U-net 0.66/1.04, CNextU-net 0.54/1.01 at windows 6:12/13:30 **[A: 2412.00568v2 Table 3]**. That band straddles VRMSE = 1 and therefore actually discriminates |
| `polymathic-ai/active_matter` | Continuum active nematic (Doi-type), 256², 81 steps, 11 fields, 360 traj | HF, sha `dc3e9135a75b4e5a0d979086219fc97fe668f7a3` **[V]**; 88 files, **55.818 GB** total, ungated (prompt's "55.8 GB" confirmed) | **test = 6.988 GB (21 files)**; valid 6.300 GB; train 42.522 GB **[V]** | CC-BY-4.0 **[V]** | HDF5 (Well spec) | **INCLUDE (test-split-only materialization).** 11 fields makes per-field VRMSE meaningful rather than degenerate. Published VRMSE one-step FNO 0.3691 / TFNO 0.3598 / U-net 0.2489 / CNextU-net 0.1034 **[A: Table 2]**; rollout 6:12 FNO >10, TFNO 7.52, U-net 2.53, CNextU-net 2.11 **[A: Table 3]** |
| `pdearena/NavierStokes-2D` | 2D incompressible NS, buoyancy-driven smoke (PhiFlow), 128², 14 steps, dt 1.5 s, 3 fields (scalar `u` + `vx`,`vy`), ν=0.01, fixed buoyancy 0.5 | HF, sha `cd99556a883a20acb9102c1f4bfdfae66ae33495` **[V]**; 314 files, **42.957 GB**, ungated | **test = 7.160 GB = 52 shards × 137,685,784 B exactly** (each shard = 25 trajectories) **[V]**. A 25-trajectory smoke eval costs **0.138 GB**. Train 28.638 GB = 52 × 550,726,984 B (100 traj/shard) **[V]** | MIT **[V]** | HDF5, flat root, one shard per seed per split; root group named by split; datasets `u, vx, vy, t, dt, x, dx, y, dy, buo_y`; **float64** | **INCLUDE.** The cheapest high-quality 2D autoregressive data of anything audited, with the finest useful granularity. Independently corroborated by three separate findings (PDEArena, DPOT, PROSE-FD audits) |

### 1b. OPTIONAL

| Source | PDEs | Host + pin | Subset cost | License | Format | Recommendation & reason |
|---|---|---|---|---|---|---|
| `polymathic-ai/planetswe` | Forced hyperviscous rotating shallow water on a sphere w/ Earth topography, 256×512 equiangular, **1008 steps**, 3 fields, 120 traj (40 ICs × 3 year-segments) | HF, sha `33eabc3e7e33454644ce5c10aee76b2e22605abd` **[V]**; 137 files, **192.444 GB** | **test = 19.227 GB (12 files × 1.602 GB)** **[V]**. One trajectory per file | CC-BY-4.0 **[V]** | HDF5 + `planetswe.yaml`, `stats.yaml`, `full_stats.yaml` | **OPTIONAL — best rollout calibration in the corpus.** VRMSE 6:12/13:30: FNO 0.81/2.96, TFNO 0.29/0.55, U-net 1.18/1.92, CNextU-net 0.42/0.52 **[A]** — four baselines spread across VRMSE 0.29→2.96 with none censored. Two caveats: the grid is **spherical, non-square, `[OPEN, PERIODIC]`**, so periodic-square assumptions silently lose; and the forcing is strongly periodic (day = 24 steps, year = 1008), so short-context rollouts measure forcing-phase recovery. Note the README says 185.8 GB and the paper 186 GB — **the API says 192.444 GB**; cite the API |
| `polymathic-ai/gray_scott_reaction_diffusion` | Gray-Scott, 128², **1001 steps**, 2 fields, 6 (f,k) patterns × 200 ICs | HF, sha `913ef5679d5c4b4ccd6fcd52c055509dfbf190f9` **[V]**; 27 files, **157.992 GB** | **test = 15.905 GB (6 files × 2.651 GB)**; train 126.131 GB **[V]**. One file per pattern per split → a single pattern's test+valid is 5.30 GB | CC-BY-4.0 **[V]** | HDF5 (Well spec) | **OPTIONAL.** Built-in generalization axis across (f,k). But: the README lists trajectories that go **stationary at t-index ≈121–159** (gliders f=0.014 k=0.054, some spirals) **[A]**, which makes the VRMSE denominator collapse; and every Well baseline except CNextU-net exceeds 10 in the 13:30 window (0.89/1.54/0.57/0.29 at 6:12 → >10/>10/>10/7.62 at 13:30) **[A]**. Usable only with a short K and explicit stationary-trajectory exclusion |
| `polymathic-ai/shear_flow` | Incompressible NS shear layer + passive tracer, 256×512, 200 steps, 4 fields, 1120 traj | HF, sha `fc867f856f306905cf94c1f5df978cc518a2048c` **[V]**; **471.193 GB** | test 47.446 GB (28 files); train 376.279 GB **[V]** | CC-BY-4.0 **[V]** | HDF5 | **OPTIONAL (test-only).** Even the test split is 47 GB. All four Well baselines **diverge (>10)** at 13:30; only CNextU-net survives 6:12 at 2.33 **[A]** — so it is a difficulty ceiling, not a discriminator. Paper says 547 GB, API says 471.193 GB — a real discrepancy beyond GiB/GB **[V]** |
| `polymathic-ai/rayleigh_benard` | Boussinesq RB convection, 512×128, wall-bounded in z, 200 steps, 4 fields, 1750 traj | HF, sha `10e143ebedae8a9b1c699b95d5b1eb8feaae09b5` **[V]**; **369.955 GB** | test 37.875 GB (35 files) **[V]** | CC-BY-4.0 **[V]** | HDF5 | **OPTIONAL, low value.** **All four** Well baselines are >10 at both rollout windows **[A]** — the metric carries no signal here. Also wall-bounded, which structurally penalizes spectral entrants (The Well handled BCs "naively according to model architecture") |
| `thuerey-group/pde-transformer-ape2d-full` | Same 20 scenarios, ~10× trajectories | HF, sha `fa711d56a1a58705c3d8e889830b459fd92733f0`, **78.7 GB** **[A — prompt-supplied, not re-fetched]** | Per-PDE: `fisher` 4.720 GB, `sh` 4.720 GB, `gs_*` 1.573 GB each **[A]** | MIT **[A]** | HDF5 + json sidecar | **OPTIONAL — the training-scale companion.** Use `-ape2d` for eval, `-full` for whoever wants more training data. Re-verify the duplication question here: if `-full` is also double-stored, the real payload is ~39 GB **[U — not checked]** |
| `thuerey-group/apebench-scraped` | APEBench procedural scenarios; 2D RD subset `2d_phy_gs_theta`, `2d_phy_sh`, `2d_diff_fisher` at 160² | HF, sha `a1f3d3b2759a724c61d2a3f14553632eaac63c09`, 142 files, 30.46 GB **[A]** | 2D RD subset ≈ 3.5 GB; individual files 0.261–1.235 GB **[A]** | MIT **[A]** | `.npy` + JSON config sidecars; 50 train × 51 steps, 30 test × 201 steps **[A]** | **OPTIONAL.** Small and MIT, and the 201-step test trajectories are good for long rollouts. But only 50 train trajectories per scenario, axis order was **inferred from byte arithmetic, not read** **[A/U]**, and the README warns of JAX-version/backend variation |
| `jhagnberger/fno-vorticity` | FNO ICLR-2021 2D NS vorticity, 64², 1 field, ν=1e-4 (10000 traj, **50** stored steps) and ν=1e-5 (1200 traj, 20 steps) | HF, sha `825bb0bb6b402ea67a5ea305c575968e7d20fcb4` **[V]**; 8 files, **8.769 GB** | Fetch one folder: `1e-5/` = **0.413 GB**, `1e-4/` = **8.356 GB** **[V]** | `mit` on the card **[V]** — **but the uploader is not an author and the original Google-Drive data carry no license statement** **[A]** | `.npy`, float32, **MATLAB-reversed axes (T,X,Y,N)** — headers verified by HTTP Range **[A]** | **OPTIONAL with a license caveat.** The most-cited 2D AR benchmark in existence (FNO-2D 0.1556, U-Net 0.1982, Transolver 0.0900 on ν=1e-5 **[A]**), 64² and trivially cheap. Do **not** treat the MIT card as authoritative; if the repo requires attested redistribution rights, this fails |
| `pdearena/ShallowWater-2D` | SpeedyWeather.jl global SWE, 192×96 lat-lon, `u,v,vor,div,pres` | HF, sha `c4557b2eae797dd8c339200ce03aeefe3c397176`, 8403 files, **124.378 GB** **[A]** | Per-trajectory `.nc` ≈ 14.8 MB (non-uniform, compressed) **[A]** | `mit` card metadata only; **README is 21 bytes** (`---\
license: mit\
---`), no LICENSE file in repo **[A]** | netCDF4, one trajectory per file at `<split>/seed=<N>/run<NNNN>/output.nc` **[A]** | **OPTIONAL, materializer-hostile.** Finest granularity of anything audited, but a split means **thousands** of HTTPS requests; the reference loader needs `normstats.pt` (a **pickle** — do not `torch.load` it in a verifier); `nt` is taken from *commented-out* config blocks and **could not be verified from the file** **[A/U]**; axis order is quoted inconsistently (paper 192×96, SineNet 96×192) |

### 1c. EXCLUDE

| Source | Why excluded |
|---|---|
| **PDEBench 2D** (DaRUS `doi:10.18419/darus-2986` v8.0, CC-BY-4.0, 336 files / 3902.9 GB) **[A]** | Fails the HF-pinning rule outright. **No 2D time-dependent PDEBench data exists on Hugging Face in its original layout** — the official `pdebench` org hosts exactly three datasets, all 1D (`Advection` sha `33d90a24…`, `Burgers` sha `97bf5bb5…`, `1D-Compressible-Navier-Stokes` sha `a6d507e6…`) **[A]**. Pinning degrades to DaRUS *version* 8.0 + numeric file id + MD5; file ids have churned across versions (v7→v8: 594 files/6452.8 GB → 336 files/3902.9 GB; twelve 1D Burgers ids changed, e.g. `133136`→`281363`) **[A]**. Downloads 303-redirect to a presigned S3 URL with `X-Amz-Expires=172800`, `HEAD` returns 403 **[A]**. *Keep the byte-range trick in the back pocket*: the files are contiguous uncompressed HDF5 and Range works (206 verified), so full array metadata reads in ~10 requests / <10 KB **[A]** |
| `Nionio/PDEBench_2D_{diff-react,SWE,DarcyFlow}` **[A]** | Third-party, unattested provenance, re-encoded to **PLAID/CGNS parquet** (flat float lists over QUAD_4 cells) — the regular-grid structure must be reconstructed and the `plaid` library is required. The SWE card declares `num_bytes` exactly 2× the source HDF5, consistent with the IC field being replicated per timestep **[A/U]** |
| **PDEBench 2D Darcy**; **camlab-ethz** `SE-AF`, `Poisson-Gauss`, `Helmholtz` **[A]** | **Steady-state / time-independent** — no trajectory axis, unusable for a K-step rollout without redefining the problem |
| **Poseidon / PDEgym** (`camlab-ethz/*`, 20 datasets, 992.37 GB) **[A]** | **CC-BY-NC-4.0 on every dataset and every checkpoint**, and the `camlab-ethz/poseidon` code repo has **no LICENSE file at all** (GitHub API `license: null`). Strictly worse posture than MIT (`ape2d`) or CC-BY-4.0 (The Well). Also: finest downloadable unit is a ~4.8–5 GB monolithic NetCDF chunk; 128² only; README-vs-file mismatch verified (Wave-Gauss README says 15 timesteps, the file header says **16**) |
| **PDEBench 2D incompressible NS** (`ns_incom_inhom_2d_512-*`, 275 files, 2732.29 GB) **[A]** | 2.73 TB; **no published baseline of any kind**; no model config in the repo (open issue #12); `visualize_ns_incom()` is an empty stub; paper Table 1 says 256²/1000 samples while the files are 512²/1100 traj (issue #40, unanswered since 2023-06) |
| **MeshGraphNets / GNS** (GCS `dm-meshgraphnets` 291 GB; `learning-to-simulate-complex-physics` 334 GB) **[A]** | Unstructured triangle/tetra meshes and particle clouds, not regular grids; variable node counts (CylinderFlow 1732–2059). **No license is stated for the data anywhere** (no LICENSE object in the bucket, no license sentence in the README or project site; the Apache-2.0 covers code only). Rasterizing to a grid changes the benchmark and voids the published RMSEs |
| `JingminSun/prose` (373.5 GB, MIT) **[A]** | **1D only** (128 spatial points). Monolithic ~16.8 GB per-PDE files. Created 2026-01-14, long after the paper; identity with the paper's data unverified |
| **PINNacle** `i207M/PINNacle` **[A]** | Not on HF. In-repo `ref/*.dat` are **unstructured FEM point clouds** (`ns2d.dat` "% Nodes: 11980", `heat_complex.dat` "% Nodes: 13312"), one instance per file, no trajectory ensemble, no splits. Beware the name collisions: HF `PinnaclePDE/*` is an unrelated unlicensed 1D project; HF `tdc/PINNACLE` is a biology model; GitHub `apivich-h/pinnacle` is a different paper |
| `pdearena/NavierStokes-2D-conditoned` (MIT, sha `777ab1693937a050f4e22c271c4c4797b24952ab`, 81.75 GB) **[A]** | Three concrete defects: the repo id is **misspelled** (`conditoned`; the correct spelling returns **HTTP 401**, not 404, so autocorrection fails with a confusing auth error); shard `NavierStokes2D_train_496019_0.38774_32.h5` is **96 bytes / corrupt** (every other shard is exactly 704,729,352 B); and the upload is **incomplete** — 79/19/19 shards × 32 = 2,528/608/608 trajectories vs the 6,656/1,664/1,664 in its own `docs/data.md`. Only worth it if 56-step rollouts are essential, and then only with an explicit skip-list |
| `ZizhangC/2D-Kolmogorov-Flow-Multifidelity-Dataset`; `abelsr1710`/`viper333qaq/navier-stokes-2d-fno`; `scailab/hyper-navier-stokes-64-1000` **[A]** | Gated + unlicensed (ZizhangC, `cardData: null`); mislabeled (`abelsr1710` claims ν=1e-3 but ships a `[1200,64,64,20]` tensor — the *shape of the ν=1e-5 file*); no license stated (scailab) |
| `OneScience-Group/Kolmogorov_flow_2d`; `jorge-mif/kolmogorov`; `scaomath/navier-stokes-dataset` **[A]** | Usable in principle (Re=1000 Kolmogorov, long trajectories) but each has a defect: OneScience's README **disclaims the data license** ("verify its upstream source"); `jorge-mif`'s README says ν=1e-4/Re≈1e4 while the **filename says Re1000**; `scaomath` is 87 GB of undocumented pickles requiring `add_safe_globals`. Revisit only if more chaotic long-horizon data is needed |
| **ν=1e-3 N5000 T50 FNO file** | **No verified pinned copy exists anywhere** — HF name search (`V1e-3`, `N5000`, `ns_V1e`) and web search both empty **[A]**. Do not list it as available |

---

## 2. Model families for the SOTA audit and starter

| Model | Released weights (pin) | License | Datasets with published numbers | The numbers | Recommendation |
|---|---|---|---|---|---|
| **FNO / TFNO** (`neuraloperator`) | `polymathic-ai/FNO-<dataset>` ×15, `TFNO-<dataset>` ×15. Verified pins: `FNO-turbulent_radiative_layer_2D` sha `49357c80caeecf58a34964c648f8660db5764422` **[V]**; `FNO-active_matter` `8efefd7b…`, `FNO-planetswe` `08e75850…`, `FNO-shear_flow` `9ff091f4…`, `FNO-rayleigh_benard` `9975702d…`, `FNO-gray_scott…` `d7da44b7…` **[A]**. ~76 MB each, ≈19.01–19.02M fp32 params | Library **MIT** (pin 2.0.0 `b307030d…`, or **0.3.0 `43c76a4c…`** which the checkpoints require). **Weights: NO LICENSE — `license: None`, zero license tags, verified on `FNO-turbulent_radiative_layer_2D` and `UNetConvNext-turbulent_radiative_layer_2D` this session [V]** | All 6 candidate Well 2D datasets, in **VRMSE** — the task's own metric | One-step VRMSE (FNO/TFNO): active_matter 0.3691/0.3598; rayleigh_benard 0.8395/0.6566; shear_flow 1.189/1.472; turb_rad_layer 0.5001/0.5016; gray_scott 0.1365/0.3633; planetswe 0.1727/0.0853. Rollout 6:12 (FNO): turb_rad 1.79, planetswe 0.81, gray_scott 0.89, active_matter/RB/shear **>10** **[A: 2412.00568v2 Tables 2–3]** | **Reference baseline (floor).** The only family with published numbers already in VRMSE. Resolution-independent param count verified across 128², 128×384, 256², 256×512, 512×128. **Do not redistribute the weights** — reference by pinned revision or retrain from the MIT library |
| **U-net / CNextU-net** (`UNetClassic`, `UNetConvNext`) | `polymathic-ai/UNetConvNext-<dataset>` ×14, `UNetClassic-*` ×14. `UNetConvNext-active_matter` sha `7083c51f…`, ≈18.59M params **[A]**; `UNetConvNext-turbulent_radiative_layer_2D` sha `1eeb9df8def89ed3506d22ffd3c83bcd15ef32cc` **[V]** | **NO LICENSE (verified `license: None`) [V]** | Same 6 Well datasets, VRMSE | One-step (U-net/CNextU-net): active_matter 0.2489/0.1034; turb_rad 0.2418/0.1956; shear_flow 3.447/**0.8080**; planetswe 0.3620/0.3724; gray_scott 0.2252/0.1761; RB 1.4860/0.6699. Rollout 6:12/13:30 CNextU-net: turb_rad **0.54/1.01**, planetswe 0.42/0.52, active_matter 2.11/2.71 **[A]** | **Reference baseline (ceiling).** "CNextU-net outperforms the others on 8 of 17 experiments" **[A]**. Sets the realistic top of the band. Same unlicensed-weights problem |
| **DPOT** | `hzk17/DPOT` sha `2adec1cf9a55942f1456aa7463cd7ade908398d0` **[V]**, **apache-2.0 verified on both `cardData` and tags [V]**, 9 files / 20.409 GB (Ti 90.5 MB → H 12.37 GB; ~3× param size because **Adam state is bundled**) | **Weights Apache-2.0 [V]. Code: `thu-ml/DPOT` and `HaoZhongkai/DPOT` have NO LICENSE FILE** (`license: null`); `models/dpot.py` carries an NVIDIA 2020 header and `torch_utils/` an NVIDIA "strictly prohibited" header **[A]** | FNO-NS ν-series, PDEBench 2D CNS/SWE/DR, **PDEArena NS + NS-cond**, CFDBench — in L2RE | Zero-shot DPOT-H: FNO-1e-5 0.0174, PDB-M1 0.0138, SWE 0.00199, DR 0.0191, **PDEArena-NS 0.0379**, NS-cond 0.213. DPOT-Ti: 0.0976 / 0.0285 / 0.00560 / 0.0321 / **0.125** / 0.384. Fine-tuned L-500: PDEArena-NS 0.0278, NS-cond 0.170. Long rollout (Table 10): pretrained 0.0385 vs scratch 0.0912 at **500 steps** **[A: 2403.03542v4]** | **Starter option** — it is *exactly* the task's shape (fixed 128², channels-last `[B,H,W,T,C]`, 10-frame context, one-step target rolled out autoregressively). Ti is 90 MB and 0.408 ms/step. **But:** `PatchEmbed` asserts H=W=128, `in_channels` hard-fixed at 4 with one-padding, `normalize=0`; and `torch≥2.6` needs `weights_only=False` because `args` is a pickled `argparse.Namespace`. **Contamination: DPOT pretrained on PDEArena NS/NS-cond train splits** — if `pdearena/NavierStokes-2D` is in the task, DPOT is in-distribution |
| **Poseidon / scOT** | `camlab-ethz/Poseidon-T` sha `ec976ed5d25883ec9db4e486ebbeeefa9e08303b` **[V]** (20.8M, 83 MB safetensors), `-B` `702667ed…` (157.8M), `-L` `7ffaff43…` (628.6M) **[A]** | **cc-by-nc-4.0 verified on cardData + tags [V]**; code repo has **no LICENSE** **[A]** | 15 downstream PDEgym tasks — but **only EG/AG ratios are published; absolute median relative L1 exists only inside figures** **[A]** | Median EG / Mean AG over 15 tasks: Poseidon-L 49.8/9.58, B 24.4/6.71, T 19.5/6.49; CNO-FM 10.6/2.91; MPP-B 2.0/1.82 **[A: SM Table 9]**. **No absolute error number is citable** | **Related work + optional starter.** Cite the **all2all training recipe** (every ordered pair (t_i,t_j), i≤j → 66 pairs from 11 snapshots) — architecture-agnostic and directly useful to an agent. But NC license, no absolute numbers, 128²/4-channel lock-in, baked-in normalization constants (mean `[0.80,0,0,0]`, std `[0.31,0.391,0.356,0.185]`, lead time `t/20.0`), and `pip install scOT` installs an **unrelated EEG toolbox** |
| **MPP (AViT)** | Not audited for a weight pin here **[U]** | — | PDEBench 2D diff-react; PDEArena NS (pretraining) | DiffRe2D NRMSE: MPP-AViT-Ti 0.0168, S 0.0112, L 0.0098 **[A]**. Retrained by BCAT on PDEArena: MPP-B 5.71, MPP-L 4.64 (rel-L2 %) **[A]** | **Related work only.** MPP's stated PDEArena config is **internally inconsistent** (cites the 5,200/1,300/1,300 standard split, which is 14 steps at fixed b=0.5, while also claiming T=16 and variable b∈[0.2,0.5]) **[A]** |
| **BCAT** | `felix-lyx/bcat` sha `aa0c8a46a7c65c006f33ffede6c4f3889c547ba7`, MIT, `bcat.pth` 622.6 MB, 156M **[A]** | MIT **[A]** | PDEBench SWE/CNS/INS, PDEArena NS/NS-cond, CFDBench | Avg rel-L2 1.18% (vs PROSE-FD 3.32, DPOT-L 2.16, DPOT-M 3.47, MPP-L 3.91, ViT 4.10). PDEArena NS **1.59** / NS-cond **3.13** **[A: 2501.18972v2 Table 1]** | **Related work only** — but the most valuable such entry, because BCAT **retrained MPP-L, DPOT-L/M and PROSE-FD under a single metric**, which is the only apples-to-apples cross-model table in this whole audit |
| **PROSE-FD** | `felix-lyx/prose` sha `b185d5837693973e57bf8cd1675c723149d29591` **[V]**, **mit verified [V]**, `prose_fd.pth` 665 MB, 169M | MIT **[A]** | Same 6 families | SWE 0.28, CNS 1.53, INS 2.84, NS 6.34, NS-cond 10.76, CFDBench 0.54, avg 3.71 **[A]** | **Related work only.** It is a **one-shot 10→10 operator, not autoregressive** (rollout exists only as an ablation: 3.74 vs 3.71), and the released 2-to-1 weights **consume the governing equation as tokens** — incompatible with a field-only `INPUT_DIR`. Numbers also shift between papers (3.71 vs 3.32) |
| **PROSE-PDE** | **No checkpoints exist anywhere** **[A]** | Maintained code `JingminSun/prose_v1` has **no LICENSE** **[A]** | 1D only | 2-to-2 Known: rel-L2 0.92%, R² 0.998 (v3); v1 said 2.74% **[A]** | **Related work only** — 1D, symbolic modality, no weights |
| **SineNet** | Not audited for a weight pin **[U]** | — | PDEArena INS + SWE | SineNet-16: INS 1.46%/17.63%, SWE 0.92%/1.64% (1-step/rollout, per-field scaled-L2) **[A: 2403.19507 Table 1]** | **Related work only.** Its per-field scaled-L2 is the **closest published analogue to VRMSE** (divides by per-field ‖u‖₂ rather than per-field std) — useful for sanity-checking magnitudes, never for quoting |
| **PDEBench FNO / U-Net / PINN checkpoints** | DaRUS `doi:10.18419/darus-2987` v2.0, 47 files, 4.07 GB, CC-BY-4.0. 2D: `fno/2D_diff-react_NA_NA_FNO.pt` (11 MB), `fno/2D_rdb_NA_NA_FNO.pt`, `fno/2DCFD_FNO.tar`, `unet/2D_diff-react_NA_NA_Unet-{1-step,AR,PF-20}.pt`, `unet/2DCFD_Unet.tar` **[A]** | CC-BY-4.0 **[A]** | PDEBench 2D sets, nRMSE | e.g. 2D diff-react FNO 1.2e-1, U-Net 8.4e-1, PINN 1.6e0; 2D SWE FNO 4.4e-3, U-Net 8.3e-2, PINN 1.7e-2 **[A]** | **Reference baseline only if PDEBench enters the task** (which §1c recommends against). **U-Net numbers are suspect**: open issue #90 reports an indentation bug making `training_type='single'` train on one batch element per epoch, and the maintainer replied "Possibly yes… severely limited" **[A]** |
| **MeshGraphNets / GNS** | No DeepMind checkpoints. Only NVIDIA NGC `modulus_ns_meshgraphnet` (24.8 MB, DGL, **license not shown**) **[A]** | Code Apache-2.0; **data license unstated** **[A]** | CylinderFlow etc. — **absolute RMSE ×10⁻³**, not normalized | CylinderFlow 1-step 2.34±0.12, rollout-50 6.3±0.7, rollout-all 40.88±7.2 **[A]** | **Related work only.** No paper reports MGN rollout on a regular grid; the MGN authors' own regular-grid follow-up (Stachenfeld et al. 2022) **replaced the GNN processor with a dilated CNN**. Note: the claim that The Well benchmarks graph models is **false** — zero occurrences of MeshGraphNet/GNN/GNS/"graph network" in the v1 or v2 HTML **[A]** |
| **PINNs** | Per-instance; no reusable weights **[A]** | PINNacle code MIT; PDEBench code `NOASSERTION` **[A]** | PINNacle 22 tasks (L2RE); PDEBench 2D diff-react + SWE (nRMSE) | Vanilla PINN solves **10/22** PINNacle tasks at a 10% threshold; on long-time 2D cases **all 12 variants sit at ≈1.0 L2RE** (Heat2d-LT 9.99e-1, NS2d-LT 9.96e-1, KS 1.01e0) **[A]** | **Explicit anti-goal, documented as such.** A PINN has no cross-IC generalization, so `infer.sh` would have to re-solve per held-out trajectory (~3.05e3 s per NS2d-LT instance → **~42 GPU-hours per PDE per submission**, derived) **[A]**. **Do carve in PINO-style physics-informed losses** on a data-driven AR model — but first verify that per-trajectory PDE coefficients are actually present in the chosen files **[U]** |

---

## 3. Proposed common evaluation protocol

### 3.1 Contract

```
INPUT_DIR/   <source>/<traj_id>/input.npz    # fields[C, H, W, F] float32  (C context frames)
                                 meta.json    # source, dt, field_names, bc_type, grid_type
OUTPUT_DIR/  <source>/<traj_id>/pred.npz     # fields[K, H, W, F] float32
```

One model, one `bash infer.sh INPUT_DIR OUTPUT_DIR`, run once over all sources. Field count `F`, `H`, `W` vary by source — that heterogeneity *is* the task.

### 3.2 Context length and rollout start

- **C = 4 context frames.** Matches The Well's baseline convention (`dim_in = 4 × n_fields`, proven by the published `config.json`) **[A]**, so the published VRMSE table is the calibration. (For contrast: DPOT uses 10, PDEArena 4 for NS / 2 for SWE, PDEBench 10.)
- **Rollout always starts at the beginning of the trajectory** (`t = 0…C-1` given, predict `t = C…C+K-1`). This matches The Well's *rollout* table convention — the paper states longer rollouts "are always initiated from the beginning of the simulation" and **explicitly warns that its one-step numbers (sliding windows sampled from ground truth) are not comparable to its rollout numbers** **[A]**. Pick one and never mix.
- **Free-running**: the model's own output is fed back; no teacher forcing at any step.

### 3.3 Choice of K

The binding constraint is the shortest-horizon source:

| Source | Stored steps | Max K at C=4 |
|---|---|---|
| `pdearena/NavierStokes-2D` | 14 | **10** |
| `pde-transformer-ape2d` | 30 | 26 |
| `active_matter` | 81 | 77 |
| `turbulent_radiative_layer_2D` | 101 | 97 |
| `gray_scott` | 1001 | 997 |
| `planetswe` | 1008 | 1004 |

**Proposal: K = 10 as the primary score** (feasible on every included source), plus an **optional K = 26 long-horizon secondary** on the sources that support it. Do not chase a larger uniform K: The Well's 13:30 window already censors 3 of 4 baselines on three of the six datasets **[A]**, and at that point the metric stops discriminating.

### 3.4 Metric — VRMSE, specified to the bit

Copy The Well's definition verbatim (`the_well/benchmark/metrics/spatial.py`, `VRMSE → NRMSE.eval(norm_mode="std") → sqrt(NMSE)`) **[A]**:

```
spatial_dims = (-3, -2)                       # layout (..., H, W, F); spatial axes ONLY
MSE   = mean((pred - true)**2, dim=spatial_dims)        # -> [traj, t, field]
var   = std(true, dim=spatial_dims)**2                  # variance of the TARGET, per field
VRMSE = sqrt(MSE / (var + 1e-7))
```

Three implementation details a verifier **must** replicate or it will not match anything:
1. **Reduction is spatial only** — the field axis survives and per-field values are averaged *afterwards*. It is **not** a pooled RMSE over all channels.
2. **`torch.std` is the unbiased (n-1) estimator** — a NumPy reimplementation must pass `ddof=1`.
3. **`eps = 1e-7`**, added to the variance (not the std).

Interpretation: VRMSE ≈ 1 ⟺ no better than predicting the per-frame spatial mean. That gives the reward a natural, physically meaningful anchor.

### 3.5 Per-source normalization and aggregation

(Design notes from before the task was built. The grader that scores submissions is defined in `contract.md` and
`evaluation/score.py`: mean over cases, VRMSE clamped per step and channel; where this section differs, those win.)

VRMSE is already normalized per (trajectory, timestep, field), so **no additional per-source normalization is needed** — that is precisely why it beats nRMSE/L2RE for a multi-source reward. Fix the reduction **order** explicitly (mean-then-sqrt ≠ sqrt-then-mean):

```
1. spatial mean          -> V[traj, t, field]
2. mean over fields      -> V[traj, t]
3. mean over t in 0..K-1 -> V[traj]              # time-averaged, matching The Well's windows
4. median over traj      -> V_source             # median, not mean: robust to a single divergence
5. mean over sources     -> V_overall            # UNWEIGHTED per PDE family
```

Step 5 must be **unweighted per source**, not per trajectory — otherwise a 1750-trajectory source (`rayleigh_benard`) drowns a 90-trajectory one (`turbulent_radiative_layer_2D`). Step 4 uses the **median** because the dominant failure mode is a single trajectory blowing up to 1e30.

### 3.6 Divergence handling (mandatory, not optional)

Autoregressive rollout divergence is the *primary* failure mode, not an edge case — The Well censors entries as ">10" for FNO on active_matter/RB/shear_flow at 6:12 and for all four models on rayleigh_benard at both windows **[A]**. A verifier that does not handle this will emit `inf`/`NaN` and crash scoring.

```
V = nan_to_num(V, nan=VMAX, posinf=VMAX, neginf=VMAX)   # VMAX = 10.0
V = clip(V, 0.0, VMAX)
```

Apply the clamp at **step 3** (per trajectory, after time-averaging) so one bad step cannot poison a whole source. Reject outputs with wrong shape/dtype before scoring rather than silently padding.

### 3.7 Reward

VRMSE spans roughly 0.006 (PDE-Transformer 1-step on `gs_theta` **[A]**) to >10 (Well rollouts), so a linear map wastes almost all its range. Propose a **log-scaled band**:

```
R_source  = clip( (log10(VMAX) - log10(max(V_source, VFLOOR))) / (log10(VMAX) - log10(VFLOOR)), 0, 1 )
           with VFLOOR = 0.01, VMAX = 10.0
reward    = mean over sources of R_source
```

Anchors this produces: VRMSE 0.01 → 1.00; 0.1 → 0.67; 0.32 → 0.50; 1.0 (spatial-mean predictor) → 0.33; 3.2 → 0.17; ≥10 → 0.00. Publish the anchor table in the task card so the number is legible. *This mapping is a design proposal, not a published convention — the components (VRMSE, the >10 censoring threshold) are, the log-band is mine.*

### 3.8 Cross-walk — where published numbers use a different convention

**Only The Well's numbers are in VRMSE. Everything else must be recomputed. Do not place any of the following in the same table as a VRMSE score.**

| Convention | Definition | Why it is not VRMSE |
|---|---|---|
| **The Well VRMSE** ✅ | `sqrt(MSE / (std(y)² + 1e-7))`, spatial reduction | **The task metric.** Caveat: one-step numbers use sliding windows from GT, rollout numbers start at t=0 — the paper warns these "do not always line up" **[A]** |
| **PDEBench nRMSE** | `sqrt(mean((p-t)²)) / sqrt(mean(t²))`, spatial RMS of truth, **mean not removed** | = The Well's `NRMSE(norm_mode="norm")`. For fields with large nonzero mean (CNS density ≈1, pressure, SWE height ≈1) **VRMSE is substantially larger**. Two further traps: the *paper tables include the context frames in the error average* (maintainer confirmed, issue #41: "induces an overestimation of the model performance"), while current `metrics.py` slices them off — so **the shipped code no longer reproduces the paper** **[A]** |
| **PDE-Transformer nRMSE** | `mean_i sqrt(MSE(û,u)/MSE(0,u))` per step, then averaged | Same *form* as a per-step normalized RMSE and closest to the task, but the denominator is **uncentered** (MSE against zero, not against the spatial mean) **[A]** |
| **PDEArena SMSE** | Elementwise MSE → mean over space → **SUM over fields AND over 5 rollout steps**; unnormalized; SWE `pres`/`vor` **pre-normalized by `normstats.pt`** | Not normalized at all, and summed over time so it is ~5–8× the one-step number by construction. Absolute values are meaningless outside that pickle **[A]** |
| **SineNet scaled-L2** | Per-field `‖û-u‖₂/‖u‖₂`, averaged over fields, averaged over steps, in % | Nearest analogue to VRMSE, but divides by the **uncentered** L2 norm. Diverges materially on the strictly-positive smoke-concentration channel **[A]** |
| **BCAT / PROSE-FD rel-L2** | `(1/T) Σ ‖u-û‖₂/(‖u‖₂+1e-7)`, **whole state, channels pooled into one norm** | Channel pooling means a high-magnitude field dominates; VRMSE weights all fields equally **[A]** |
| **DPOT L2RE** | `‖p-t‖₂/(‖t‖₂+1e-8)` with **all rollout steps AND all space pooled into one ratio**, then averaged over *unmasked* channels | Pools time before taking the ratio. The paper never states whether Table 1 is the pooled (`test_l2_full`) or per-step variant; `evaluate.py` prints only the pooled one **[A]** |
| **FNO / Transolver rel-L2** | Full-trajectory relative L2, one norm ratio over all predicted steps | Weights later high-error steps by energy; per-step averaging weights every step equally **[A]** |
| **APEBench mean-nRMSE** | Per-step nRMSE, aggregated by **geometric mean** over 100 rollout steps, median over 20 seeds | Geometric, not arithmetic — systematically lower than a plain mean when errors grow **[A]** |
| **Poseidon** | **Median relative L1** at the **final lead time**; numerator and denominator each summed **across channels before dividing** | Large-magnitude channels dominate; and **only EG/AG ratios are published — no absolute number exists to compare against [A]**. Also chooses direct-vs-autoregressive inference **per task** (SM Table 6) |
| **MeshGraphNets** | Absolute RMSE ×10⁻³ on momentum/position, unnormalized, Airfoil inside an ROI | Not scale-free at all **[A]** |
| **PINNacle L2RE** | Relative L2 over all test points of **one instance**; 3 seeds on the same instance | No per-timestep or per-field structure; measures stability, not generalization **[A]** |

---

## 4. Open risks and what must be verified on real downloads

### 4.1 Blocking — must be resolved before "data-ready"

1. **Open every chosen HDF5 and record the real array layout.** Several trajectory counts across these audits are **inferred from byte arithmetic, not read**: `ape2d` (60 train / 10 test per file, inferred as 1/10 of `-full`) **[A/U]**; The Well `gray_scott` train count (160, from 21.022/2.651 ≈ 7.93) **[A/U]**; `apebench-scraped` axis order **[A/U]**. Byte arithmetic has been right every time it was checkable, but the task cannot ship on inference.
2. **`pde-transformer-ape2d` is double-stored — verified.** `adv.hdf5` and `adv/adv.hdf5` share LFS oid `debd285a…eac50`; `gs_theta.hdf5` and `gs_theta/gs_theta.hdf5` share `7b8892d7…a207` **[V]**. A naive `snapshot_download` pulls **18.719 GB for a 9.281 GB payload**. The materializer must use `allow_patterns` on exactly one copy. Check whether `-full` has the same structure **[U]**.
3. **Reconcile the PDE count for `ape2d`.** The task brief says "16 PDEs at 256²"; the pinned revision has **20 non-test `.hdf5` scenario files** = 12 distinct families + Gray-Scott in 8 parameter regimes, plus 7 `*_test.hdf5` **[V]**. Decide whether a family or a scenario is the unit before writing the task card.
4. **Reimplement VRMSE and unit-test it against The Well's `spatial.py`** on a synthetic tensor — specifically the `ddof=1` std, the `eps` inside the variance, and the spatial-only reduction. Then reproduce at least one published number end-to-end (recommend FNO on `turbulent_radiative_layer_2D`, the cheapest at 0.981 GB of test data) before trusting the harness.
5. **Decide and document the licensing posture for baseline weights.** `polymathic-ai/FNO-*` and `UNetConvNext-*` declare **no license at all** (verified `license: None`, no tags, on two repos this session **[V]**), while the code is BSD-3-Clause and the data CC-BY-4.0. **Do not ship these checkpoints inside a task image.** Either reference them by pinned revision, or retrain a baseline from the MIT `neuraloperator` library.
6. **Audit pretraining contamination per source.** If `pdearena/NavierStokes-2D` is included, **DPOT, MPP, BCAT and PROSE-FD all trained on its train split, and their "test" splits are the tail of the same files** **[A]**. An agent permitted to download foundation-model weights could submit a fine-tune of a model that has already seen the held-out distribution. Either forbid pretrained weights, or state plainly that the split is held out from the *agent* but not from the *public model ecosystem*.

### 4.2 High — will silently corrupt results if missed

7. **Axis-order and dtype traps.** `pdearena/NavierStokes-2D` stores **float64** (verified by exact arithmetic: 25×14×128×128×3×8 = 137,625,600 + 60,184 B overhead = 137,685,784 B) — the reference loader casts to float32 on read **[A]**. `jhagnberger/fno-vorticity` `.npy` arrays are **MATLAB-reversed `(T,X,Y,N)`**, and `scipy.io.loadmat` vs `h5py` on the `.mat` originals silently transpose relative to each other **[A]**.
8. **Degenerate VRMSE denominators.** `gray_scott` gliders (f=0.014, k=0.054) and some spirals go **stationary at t-index ≈121–159** (the README lists the indices) **[A]**. Near-constant targets make `std(y)² → 0` and the `eps=1e-7` floor takes over, producing either an explosion or a meaningless score. Any quiescent early-time window (e.g. `planetswe` spin-up) has the same problem. **Scan the actual chosen trajectories for low target variance and exclude or floor them explicitly.**
9. **Boundary conditions are not uniform and bias the comparison.** Included sources span periodic (`ape2d`), wall-bounded (`rayleigh_benard`), open (`turbulent_radiative_layer_2D`), Dirichlet/Neumann (`pdearena` NS), and spherical `[OPEN, PERIODIC]` (`planetswe`). The Well states BCs "were handled naively according to model architecture" — **Fourier convolutions implicitly assume periodicity** and are silently penalized on wall-bounded domains **[A]**. Either document that domain padding is permitted, or surface `bc_type` in `meta.json` (the Well HDF5 files carry `boundary_conditions/bc_type`).
10. **Resolution heterogeneity is a real difficulty axis, not an accident.** Candidates span 64² / 128² / 128×384 / 160² / 256² / 256×512 / 512×128. FNO's parameter count is resolution-independent (verified 19.01–19.02M across five shapes **[A]**), but DPOT's released weights **hard-assert 128×128** and Poseidon's **hard-code `image_size=128`, `window_size=16`** **[A]**. A single-model-across-shapes requirement rules those starters out zero-shot.
11. **Re-verify every SHA immediately before writing the manifest.** All SHAs here were live on 2026-09-16. Content churns: The Well repos were last touched 2025-04-10, `pdearena` 2023-08, but `Nionio/*` mirrors moved as recently as 2026-06-06 **[A]**.

### 4.3 Medium — document, then proceed

12. **Size figures disagree between paper and API.** `shear_flow`: paper 547 GB vs **API 471.193 GB [V]** — larger than a GiB/GB gap. `planetswe`: README 185.8, paper 186, **API 192.444 GB [V]**. `gray_scott`: README 153.8, paper 154, **API 157.992 GB [V]**. **Always cite the API figure and note the discrepancy.**
13. **Upstream documentation may not exist.** All three `pdearena` dataset READMEs contain **only `license: mit`** (21 bytes) **[A]**. The repository's "documented size and layout" requirement must be satisfied by the task's own docs; there is no upstream card to point at.
14. **The Well baselines are deliberately weak.** Time-boxed to **12 h on one H100** or 500 epochs, "defaults or commonly tuned", explicitly "should not be considered state-of-the-art". FNO got only **24 epochs** on `shear_flow` and **32** on `rayleigh_benard`, while `turbulent_radiative_layer_2D` hit the 500-epoch cap **[A]**. Quote the compute budget alongside every number or the difficulty is mis-set in both directions.
15. **These are per-dataset models, not one model.** Every published Well VRMSE comes from a **separately trained** network (`dim_in` baked to 4 × n_fields: 44 for active_matter, 16 for shear_flow/RB/turb_rad, 12 for planetswe) **[A]**. A single-model-across-PDEs task is *harder* than the published numbers imply, and no existence proof is being quoted.
16. **PINO carve-in needs a prerequisite check.** If the task README says physics-informed losses are allowed, verify first that **per-trajectory PDE coefficients** (viscosity, Mach, diffusivity, f/k) are actually exposed in the chosen files — the Well spec has a `scalars` group and `ape2d` a `meta_all.json`, but this was **not verified for the specific candidate files [U]**. Otherwise phrase it as permitted-without-guarantee.
17. **Dependency pinning for any vendored reference code.** `the-well[benchmark]` pins `neuraloperator==0.3.0` (Dec 2023) while PyPI is at 2.0.0 with an **incompatible `FNO.__init__` signature** — state_dict keys and kwargs will not line up **[A]**. `neuraloperator` also depends on `wandb`, which can touch the network at import; a hermetic `infer.sh` must set `WANDB_MODE=disabled`. `pdearena` is pinned to removed PyTorch-Lightning keys and `torchdata` datapipes — **read its loss/rollout functions and reimplement rather than importing** **[A]**.

---

### Bottom line

A four-source materializer of **`thuerey-group/pde-transformer-ape2d` (9.28 GB unique, MIT) + `polymathic-ai/turbulent_radiative_layer_2D` (7.86 GB, CC-BY-4.0) + `polymathic-ai/active_matter` test split (6.99 GB) + `pdearena/NavierStokes-2D` test split (7.16 GB, MIT)** gives **≈31 GB** of held-out-capable data across ~15 distinct 2D PDE families at four different grid shapes, every byte HF-hosted under a verified full commit SHA, every license permissive (MIT or CC-BY-4.0), with per-file granularity everywhere, and with published VRMSE baselines on two of the four sources for calibration. That is the only combination in this audit that satisfies every constraint the repository imposes simultaneously.",
    "new_findings": [
      {
        "name": "PDEBench (Takamoto et al., NeurIPS 2022 Datasets & Benchmarks)",
        "kind": "benchmark + simulation dataset corpus (1D/2D/3D PDEs on regular grids; 2D time-dependent subset is the relevant part)",
        "summary": "PDEBench is the canonical pre-Well benchmark for autoregressive PDE surrogates. Canonical data host is DaRUS (Dataverse, Univ. Stuttgart), doi:10.18419/darus-2986, CC BY 4.0, currently at version 8.0 (released 2024-02-13), 336 files, 3902.90 GB total (verified via Dataverse API). The 2D portion has five families: diffusion-reaction (13.24 GB, 1 file), shallow-water / radial dam break (6.63 GB, 1 file), Darcy flow (5 files x 1.31 GB, STEADY-STATE, no time dimension), compressible Navier-Stokes (8 files, 551.5 GB total), and inhomogeneous incompressible Navier-Stokes (275 files, 2732.29 GB). Files are plain uncompressed contiguous HDF5 (except ns_incom, which is lzf-chunked), so a materializer can pull a subset of trajectories by HTTP byte range — I verified this end to end by opening five of the 2D files remotely with h5py over a Range-request file object, reading full shapes with 9-15 requests and <10 KB transferred per file. CRITICAL for this repo's HF-preference rule: there is NO Hugging Face mirror of the 2D time-dependent data in its original layout. The official `pdebench` HF org holds only 1D sets (Advection, Burgers, 1D-Compressible-NS). The only 2D presence on HF is a third-party account `Nionio/*` that re-encodes 2D diff-react, 2D SWE and the five Darcy betas into PLAID/CGNS parquet (flat float lists over QUAD_4 cells) requiring the `plaid` library. 2D compressible NS and 2D incompressible NS are not on HF at all. Official baselines are FNO and U-Net for every 2D set, plus PINN for 2D diff-react and shallow-water only; no baseline at all for incompressible NS. The published metric is nRMSE = RMS-of-error / RMS-of-target, which is The Well's NRMSE(norm_mode="norm"), NOT VRMSE — the normalizer does not subtract the spatial mean, so PDEBench numbers are not comparable to VRMSE on fields with a large mean (density~1, pressure, water height).",
        "pdes_covered": [
          "2D diffusion-reaction (FitzHugh-Nagumo, activator/inhibitor, 2 channels, 101 steps, 128x128, Neumann BC)",
          "2D shallow-water (radial dam break, water height, 1 channel, 101 steps, 128x128)",
          "2D compressible Navier-Stokes (density/pressure/Vx/Vy, 4 channels, 21 steps, 128x128 or 512x512, periodic BC, Rand and Turb initial conditions, M in {0.1,1.0}, eta=zeta in {1e-8,0.01,0.1})",
          "2D incompressible Navier-Stokes, inhomogeneous (velocity 2ch + particles 1ch + static force 2ch, 1000 steps, 512x512, Dirichlet BC)",
          "2D Darcy flow (STEADY STATE, no time dimension: input nu -> output u, 128x128, beta in {0.01,0.1,1.0,10.0,100.0})",
          "also 1D advection, 1D Burgers, 1D diffusion-reaction, 1D diffusion-sorption, 1D compressible NS, 3D compressible NS (out of scope for a 2D task)"
        ],
        "resolution_and_splits": "VERIFIED HDF5 LAYOUTS (read remotely with h5py over HTTP Range against DaRUS v8.0 files; all float32):

(a) 2D_diff-react_NA_NA.h5 (file id 133017, 13,243,560,688 B = 13.2436 GB, md5 b8d0b86064193195ddc30c33be5dc949). 1000 top-level groups named "0000".."0999". Each group: `data` shape (101, 128, 128, 2) [t, x, y, channel] contiguous uncompressed, plus `grid/t` (101,), `grid/x` (128,), `grid/y` (128,), plus a `config` YAML string attr. Verified config: Du=0.001, Dv=0.005, k=0.005, t=5, tdim=501, xdim=ydim=512, domain [-1,1]^2 — i.e. simulated at 512^2 x 501 and stored downsampled 4x in space and 5x in time. Verified grid: t[0]=0.0, t[-1]=5.0 (dt=0.05); x[0]=-0.994140625, x[-1]=0.9902344.

(b) 2D_rdb_NA_NA.h5 (file id 133021, 6,626,098,972 B = 6.6261 GB, md5 75d838c47aa410694bdc912ea7f22282). 1000 groups "0000".."0999". Each: `data` (101, 128, 128, 1), `grid/{t,x,y}`. Verified config attr: name=radial_dam_break, T_end=1.0, n_time_steps=100 (=> 101 stored frames, dt=0.01), xdim=ydim=128, gravity=1.0, domain [-2.5,2.5]^2, per-sample random dam_radius and inner_height=2.0, seed = sample index.

(c) 2D_DarcyFlow_beta*.hdf5 (ids 133217/133218/133219/133220/133221 for beta 0.01/0.1/1.0/10.0/100.0, each 1,310,724,488-760 B = 1.3107 GB). Flat file: `nu` (10000, 128, 128), `tensor` (10000, 1, 128, 128), `x-coordinate` (128,), `y-coordinate` (128,); root attr beta. Verified x[0]=0.003906, x[-1]=0.996094 (cell-centred on [0,1]^2). NO time dimension — this is a steady-state operator-learning task, unusable as-is for a K-step rollout.

(d) 2D_CFD_{Rand,Turb}_M*_Eta*_Zeta*_periodic_{128,512}_Train.hdf5 (ids 164685-164692). Flat file: `density`, `pressure`, `Vx`, `Vy` each (Nb, 21, Nx, Ny); `t-coordinate` (22,); `x-coordinate` (Nx,); `y-coordinate` (Ny,); root attrs M, eta, zeta. VERIFIED: the 128-resolution files have Nb=10000 samples (55,050,245,208 B each); the 512-resolution files have Nb=1000 (88,080,392,280/528 B each). Verified t-coordinate: t[0]=0.0, dt=0.05, t[-1]=1.05 — note 22 time coordinates for 21 stored frames (off-by-one). Verified x cell-centred on [0,1]^2.

(e) ns_incom_inhom_2d_512-N.h5, N=0..274 (275 files, 9.873-9.995 GB each, 2,732,292,333,193 B = 2732.29 GB total). Verified on file -0 (id 133280): `velocity` (4, 1000, 512, 512, 2), `particles` (4, 1000, 512, 512, 1), `force` (4, 512, 512, 2), `t` (4, 1000); all lzf-compressed and chunked (chunks (1,1,512,512,2) etc.); root attrs include a config YAML with grid_size [512,512], domain_size [1,1]. DERIVED (not individually verified for all 275 files): 275 x 4 = 1100 trajectories.

OFFICIAL SPLIT CONVENTIONS (from pdebench/models/fno/utils.py, current main):
- `FNODatasetSingle` (used for single_file: True — 2D CFD, Darcy, all 1D): `test_idx = int(num_samples_max * test_ratio)` with `test_ratio=0.1`; if_test -> `data[:test_idx]` (FIRST 10%), train -> `data[test_idx:num_samples_max]` (last 90%).
- `FNODatasetMult` (used for single_file: False — 2D diff-react, 2D SWE, i.e. the group-per-sample .h5 files): `test_idx = int(len(data_list) * (1 - test_ratio))`; if_test -> `data_list[test_idx:]` (LAST 10%), train -> `data_list[:test_idx]`. The two loaders take opposite ends of the index range — an inconsistency to be aware of if you want to match published numbers.
- There is no separate validation split; the same "test" loader is used for validation during training.
- Config defaults (pdebench/models/config/args/): config_diff-react.yaml and config_rdb.yaml both use initial_step=10, t_train=101, reduced_resolution=1, reduced_resolution_t=1, ar_mode=True, pushforward=True, unroll_step=20. config_2DCFD.yaml uses initial_step=10, t_train=21, reduced_resolution=2 (so 128->64 or 512->256), filename `2D_CFD_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5` (note: the actual DaRUS filename contains an extra `Rand_`, so the shipped config filename does not match the shipped data filename). config_Darcy.yaml uses t_train=2, ar_mode=False.
- Rollout in `metrics.py`: full autoregressive unroll from t=initial_step to the last stored frame, feeding predictions back (`xx = cat((xx[...,1:,:], im), dim=-2)`).",
        "metric_conventions": "PDEBench nRMSE, exact code from pdebench/models/metrics.py `metric_func` (current main):

  pred = pred[..., initial_step:, :]; target = target[..., initial_step:, :]   # context frames sliced off
  err_mean = sqrt(mean((pred - target)**2 over all spatial cells))             # -> [nb, nc, nt]
  err_RMSE = mean(err_mean, axis=0)
  nrm      = sqrt(mean(target**2 over all spatial cells))                      # -> [nb, nc, nt]
  err_nRMSE = mean(err_mean / nrm, dim=0)                                      # per-sample ratio, then batch-mean
  reported = mean(err_nRMSE, dim=[0, -1])                                      # mean over channels and time

So the normalizer is the spatial RMS (second moment) of the GROUND TRUTH, per (sample, channel, timestep) — the spatial mean is NOT removed.

RELATION TO THE WELL (verified against the_well/benchmark/metrics/spatial.py on PolymathicAI/the_well master):
- The Well `NMSE(norm_mode="norm")` = MSE / mean(y**2); `NRMSE` = sqrt of that. This is mathematically PDEBench's per-sample nRMSE.
- The Well `VMSE` = NMSE(norm_mode="std") = MSE / (std(y)**2 + eps); `VRMSE` = sqrt of that — the spatial MEAN IS removed before normalizing.
=> PDEBench nRMSE == The Well NRMSE, NOT VRMSE. For fields with a large nonzero spatial mean (2D CFD density ~1 and pressure, SWE water height ~1) VRMSE will be substantially LARGER than the published PDEBench nRMSE; for zero-mean-ish fields (Vx, Vy) the two nearly coincide. Do not port published nRMSE numbers into a VRMSE-scored task as if they were baselines.
- Minor differences that still matter for exact reproduction: PDEBench averages the per-sample ratio over the batch (mean of ratios, not ratio of means), and has no eps in the denominator; The Well adds eps=1e-7.
- PDEBench also reports cRMSE (conserved value), max error, bRMSE (boundary), and fRMSE in three frequency bands; these are in the same tables.",
        "published_numbers": [
          {
            "model": "FNO",
            "dataset": "PDEBench 2D diffusion-reaction (2D_diff-react_NA_NA)",
            "metric": "nRMSE (PDEBench, full autoregressive rollout)",
            "value": "1.2e-01",
            "source": "Takamoto et al., NeurIPS 2022 D&B supplemental PDF Table 5; identical in arXiv:2210.07182 (ar5iv) Table 5"
          },
          {
            "model": "U-Net",
            "dataset": "PDEBench 2D diffusion-reaction (2D_diff-react_NA_NA)",
            "metric": "nRMSE",
            "value": "8.4e-01",
            "source": "NeurIPS 2022 supplemental Table 5 / arXiv:2210.07182 Table 5"
          },
          {
            "model": "PINN",
            "dataset": "PDEBench 2D diffusion-reaction (2D_diff-react_NA_NA)",
            "metric": "nRMSE",
            "value": "1.6e+00",
            "source": "NeurIPS 2022 supplemental Table 5 / arXiv:2210.07182 Table 5"
          },
          {
            "model": "FNO",
            "dataset": "PDEBench 2D diffusion-reaction",
            "metric": "RMSE",
            "value": "8.1e-03",
            "source": "NeurIPS 2022 supplemental Table 5"
          },
          {
            "model": "U-Net",
            "dataset": "PDEBench 2D diffusion-reaction",
            "metric": "RMSE",
            "value": "6.1e-02",
            "source": "NeurIPS 2022 supplemental Table 5"
          },
          {
            "model": "FNO",
            "dataset": "PDEBench 2D shallow-water (2D_rdb_NA_NA)",
            "metric": "nRMSE",
            "value": "4.4e-03",
            "source": "NeurIPS 2022 supplemental Table 5 / arXiv:2210.07182 Table 5"
          },
          {
            "model": "U-Net",
            "dataset": "PDEBench 2D shallow-water (2D_rdb_NA_NA)",
            "metric": "nRMSE",
            "value": "8.3e-02",
            "source": "NeurIPS 2022 supplemental Table 5 / arXiv:2210.07182 Table 5"
          },
          {
            "model": "PINN",
            "dataset": "PDEBench 2D shallow-water (2D_rdb_NA_NA)",
            "metric": "nRMSE",
            "value": "1.7e-02",
            "source": "NeurIPS 2022 supplemental Table 5 / arXiv:2210.07182 Table 5"
          },
          {
            "model": "FNO",
            "dataset": "PDEBench 2D shallow-water",
            "metric": "RMSE",
            "value": "4.5e-03",
            "source": "NeurIPS 2022 supplemental Table 5"
          },
          {
            "model": "U-Net",
            "dataset": "PDEBench 2D shallow-water",
            "metric": "RMSE",
            "value": "8.6e-02",
            "source": "NeurIPS 2022 supplemental Table 5"
          },
          {
            "model": "FNO",
            "dataset": "PDEBench 2D Darcy flow, beta=0.01 (steady state)",
            "metric": "nRMSE",
            "value": "2.5e+00",
            "source": "NeurIPS 2022 supplemental Table 8 / arXiv:2210.07182 Table 8"
          },
          {
            "model": "U-Net",
            "dataset": "PDEBench 2D Darcy flow, beta=0.01",
            "metric": "nRMSE",
            "value": "1.1e+00",
            "source": "NeurIPS 2022 supplemental Table 8"
          },
          {
            "model": "FNO",
            "dataset": "PDEBench 2D Darcy flow, beta=0.1",
            "metric": "nRMSE",
            "value": "2.2e-01",
            "source": "NeurIPS 2022 supplemental Table 8"
          },
          {
            "model": "U-Net",
            "dataset": "PDEBench 2D Darcy flow, beta=0.1",
            "metric": "nRMSE",
            "value": "1.8e-01",
            "source": "NeurIPS 2022 supplemental Table 8"
          },
          {
            "model": "FNO",
            "dataset": "PDEBench 2D Darcy flow, beta=1.0",
            "metric": "nRMSE",
            "value": "6.4e-02",
            "source": "NeurIPS 2022 supplemental Table 8"
          },
          {
            "model": "U-Net",
            "dataset": "PDEBench 2D Darcy flow, beta=1.0",
            "metric": "nRMSE",
            "value": "3.3e-02",
            "source": "NeurIPS 2022 supplemental Table 8"
          },
          {
            "model": "FNO",
            "dataset": "PDEBench 2D Darcy flow, beta=10.0",
            "metric": "nRMSE",
            "value": "1.2e-02",
            "source": "NeurIPS 2022 supplemental Table 8"
          },
          {
            "model": "U-Net",
            "dataset": "PDEBench 2D Darcy flow, beta=10.0",
            "metric": "nRMSE",
            "value": "8.2e-03",
            "source": "NeurIPS 2022 supplemental Table 8"
          },
          {
            "model": "FNO",
            "dataset": "PDEBench 2D Darcy flow, beta=100.0",
            "metric": "nRMSE",
            "value": "6.4e-03",
            "source": "NeurIPS 2022 supplemental Table 8"
          },
          {
            "model": "U-Net",
            "dataset": "PDEBench 2D Darcy flow, beta=100.0",
            "metric": "nRMSE",
            "value": "4.4e-03",
            "source": "NeurIPS 2022 supplemental Table 8"
          },
          {
            "model": "FNO",
            "dataset": "PDEBench 2D compressible NS, M=0.1 inviscid Rand periodic",
            "metric": "nRMSE",
            "value": "2.8e-01",
            "source": "NeurIPS 2022 supplemental Table 11 / arXiv:2210.07182 Table 11"
          },
          {
            "model": "U-Net",
            "dataset": "PDEBench 2D compressible NS, M=0.1 inviscid Rand periodic",
            "metric": "nRMSE",
            "value": "6.6e-01",
            "source": "NeurIPS 2022 supplemental Table 11"
          },
          {
            "model": "FNO",
            "dataset": "PDEBench 2D compressible NS, M=0.1 eta=zeta=0.01 Rand periodic",
            "metric": "nRMSE",
            "value": "1.7e-01",
            "source": "NeurIPS 2022 supplemental Table 11"
          },
          {
            "model": "U-Net",
            "dataset": "PDEBench 2D compressible NS, M=0.1 eta=zeta=0.01 Rand periodic",
            "metric": "nRMSE",
            "value": "7.1e-01",
            "source": "NeurIPS 2022 supplemental Table 11"
          },
          {
            "model": "FNO",
            "dataset": "PDEBench 2D compressible NS, M=0.1 eta=zeta=0.1 Rand periodic",
            "metric": "nRMSE",
            "value": "3.6e-01",
            "source": "NeurIPS 2022 supplemental Table 11"
          },
          {
            "model": "U-Net",
            "dataset": "PDEBench 2D compressible NS, M=0.1 eta=zeta=0.1 Rand periodic",
            "metric": "nRMSE",
            "value": "5.1e+00",
            "source": "NeurIPS 2022 supplemental Table 11"
          },
          {
            "model": "FNO",
            "dataset": "PDEBench 2D compressible NS, M=1.0 inviscid Rand periodic",
            "metric": "nRMSE",
            "value": "3.5e-01",
            "source": "NeurIPS 2022 supplemental Table 11"
          },
          {
            "model": "U-Net",
            "dataset": "PDEBench 2D compressible NS, M=1.0 inviscid Rand periodic",
            "metric": "nRMSE",
            "value": "4.7e-01",
            "source": "NeurIPS 2022 supplemental Table 11"
          },
          {
            "model": "FNO",
            "dataset": "PDEBench 2D compressible NS, M=1.0 eta=zeta=0.01 Rand periodic",
            "metric": "nRMSE",
            "value": "9.6e-02",
            "source": "NeurIPS 2022 supplemental Table 12 / arXiv:2210.07182 Table 12"
          },
          {
            "model": "U-Net",
            "dataset": "PDEBench 2D compressible NS, M=1.0 eta=zeta=0.01 Rand periodic",
            "metric": "nRMSE",
            "value": "3.6e-01",
            "source": "NeurIPS 2022 supplemental Table 12"
          },
          {
            "model": "FNO",
            "dataset": "PDEBench 2D compressible NS, M=1.0 eta=zeta=0.1 Rand periodic",
            "metric": "nRMSE",
            "value": "9.8e-02",
            "source": "NeurIPS 2022 supplemental Table 12"
          },
          {
            "model": "U-Net",
            "dataset": "PDEBench 2D compressible NS, M=1.0 eta=zeta=0.1 Rand periodic",
            "metric": "nRMSE",
            "value": "9.2e-01",
            "source": "NeurIPS 2022 supplemental Table 12"
          },
          {
            "model": "FNO",
            "dataset": "PDEBench 2D compressible NS, M=0.1 inviscid Turb periodic",
            "metric": "nRMSE",
            "value": "1.6e-01",
            "source": "NeurIPS 2022 supplemental Table 12"
          },
          {
            "model": "U-Net",
            "dataset": "PDEBench 2D compressible NS, M=0.1 inviscid Turb periodic",
            "metric": "nRMSE",
            "value": "1.9e-01",
            "source": "NeurIPS 2022 supplemental Table 12"
          },
          {
            "model": "FNO",
            "dataset": "PDEBench 2D compressible NS, M=1.0 inviscid Turb periodic",
            "metric": "nRMSE",
            "value": "1.3e-01",
            "source": "NeurIPS 2022 supplemental Table 12"
          },
          {
     

[truncated for the repository copy; the full audit text is retained in the PR description history]

