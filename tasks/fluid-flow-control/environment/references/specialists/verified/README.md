# Verified specialist baselines

Snapshot: 2026-09-19T21:50:35.613629+00:00; 39/39 complete.

Same frozen Research v2 physics. Uncontrolled means zero physical action, not a random model.
Models are per-case comparisons, not valid shared-model submissions. No paper-convergence or confidence-interval claim.
Results and provenance: [results.json](results.json). Code: environment/references/specialists/RESEARCH_BASELINES.md in Git, or /workspace/references/specialists/README.md in the agent image.

| Case | Metric | Uncontrolled | Best trained | Algorithm / step | Recipe | Training budget | Paper same-metric reference |
| --- | --- | ---: | ---: | --- | --- | ---: | --- |
| Channel_3D_Retau180 | wall_shear ↓ | 0.2476 | 0.2474 | PPO / 12000 | public | 20000 | — |
| Cube_3D_Re300 | drag_cd ↓ | 0.8596 | 0.8609 | PPO / 4000 | public | 4000 | — |
| Cube_3D_Re3700 | drag_cd ↓ | 1.1395 | 1.1304 | PPO / 4000 | public | 4000 | — |
| Cylinder_2D_Re1000 | drag_cd ↓ | 1.5313 | 1.3544 | PPO / 16000 | public | 20000 | ≈1.18 |
| Cylinder_2D_Re200 | drag_cd ↓ | 1.3519 | 1.1395 | PPO / 12000 | public | 20000 | — |
| Cylinder_2D_Re3900 | drag_cd ↓ | 1.5994 | 1.1236 | PPO / 20000 | public | 20000 | ≈1.18 |
| Cylinder_3D_Re200 | drag_cd ↓ | 1.3281 | 1.1755 | PPO / 4000 | public | 4000 | — |
| Cylinder_3D_Re3900 | drag_cd ↓ | 1.1114 | 1.2067 | PPO / 4000 | public | 4000 | — |
| Kolmogorov_2D | tke ↓ | 3.3023 | 3.3280 | PPO / 4000 | public | 4000 | — |
| NACA0012Gust_2D_Re1000_AOA20 | gust_load_error ↓ | 0.5022 | 0.5200 | PPO / 4000 | public | 20000 | — |
| NACA0012Gust_2D_Re1000_AOA40 | gust_load_error ↓ | 0.9399 | 0.9196 | PPO / 20000 | public | 20000 | — |
| NACA0012Gust_2D_Re100_AOA20 | gust_load_error ↓ | 0.2174 | 0.2245 | PPO / 4000 | public | 20000 | — |
| NACA0012Gust_2D_Re100_AOA40 | gust_load_error ↓ | 0.3253 | 0.3282 | PPO / 8000 | public | 20000 | — |
| NACA0012Gust_3D_Re1000_AOA20 | gust_load_error ↓ | 0.3982 | 0.4606 | PPO / 4000 | public | 4000 | — |
| NACA0012Gust_3D_Re1000_AOA40 | gust_load_error ↓ | 0.5315 | 0.4513 | PPO / 4000 | public | 4000 | — |
| NACA0012Gust_3D_Re100_AOA20 | gust_load_error ↓ | 0.3513 | 0.3444 | PPO / 4000 | public | 4000 | — |
| NACA0012Gust_3D_Re100_AOA40 | gust_load_error ↓ | 0.0894 | 0.0655 | PPO / 4000 | public | 4000 | — |
| NACA0012Gust_3D_Re50000_AOA12 | gust_load_error ↓ | 0.3347 | 0.3871 | PPO / 4000 | public | 4000 | — |
| NACA0012_2D_Re1000_AOA20 | lift_drag ↑ | 1.9768 | 3.5905 | PPO / 20000 | public | 20000 | — |
| NACA0012_2D_Re1000_AOA40 | lift_drag ↑ | 1.1113 | 1.1536 | PPO / 20000 | public | 20000 | — |
| NACA0012_2D_Re100_AOA20 | lift_drag ↑ | 1.3564 | 1.5469 | PPO / 20000 | public | 20000 | — |
| NACA0012_2D_Re100_AOA40 | lift_drag ↑ | 0.9453 | 1.0465 | PPO / 20000 | public | 20000 | — |
| NACA0012_3D_Re1000_AOA20 | lift_drag ↑ | 2.0103 | 2.7033 | PPO / 4000 | public | 4000 | — |
| NACA0012_3D_Re100_AOA20 | lift_drag ↑ | 1.3459 | 1.4781 | PPO / 4000 | public | 4000 | — |
| NACA0012_3D_Re100_AOA40 | lift_drag ↑ | 0.9364 | 1.0019 | PPO / 4000 | public | 4000 | — |
| Nozzle2D_coarse | nozzle_angle_mae ↓ | 6.1857 | 6.1283 | PPO / 4000 | public | 4000 | — |
| Nozzle2D_fine | nozzle_angle_mae ↓ | 6.1854 | 6.1863 | PPO / 4000 | ppo-positive-small-start | 4000 | — |
| Pinball_2D_Re100 | drag_cd ↓ | 2.9054 | 0.2487 | DDPG / 16000 | public | 20000 | ≈0.05 |
| Pinball_2D_Re150 | drag_cd ↓ | 2.8828 | 0.1115 | PPO / 20000 | public | 20000 | — |
| Pinball_2D_Re30 | drag_cd ↓ | 3.5270 | 0.7387 | PPO / 20000 | public | 20000 | — |
| Pinball_2D_Re75 | drag_cd ↓ | 3.0221 | 0.3917 | PPO / 20000 | public | 20000 | — |
| Pinball_3D_Re100 | drag_cd ↓ | 2.8885 | 1.6991 | PPO / 4000 | public | 4000 | — |
| Pinball_3D_Re150 | drag_cd ↓ | 2.6520 | 1.8010 | PPO / 4000 | public | 4000 | ≈0.11 |
| Pinball_3D_Re30 | drag_cd ↓ | 3.4896 | 2.2794 | PPO / 4000 | public | 4000 | — |
| Pinball_3D_Re75 | drag_cd ↓ | 2.9893 | 1.8730 | PPO / 4000 | public | 4000 | — |
| Sphere_3D_Re300 | drag_cd ↓ | 0.6617 | 0.6601 | PPO / 4000 | public | 4000 | — |
| SquareCylinder_2D_Re1000 | drag_cd ↓ | 2.4806 | 2.3735 | PPO / 4000 | ppo-small-start | 4000 | — |
| SquareCylinder_2D_Re200 | drag_cd ↓ | 1.4891 | 1.1468 | PPO / 20000 | public | 20000 | ≈1.14 |
| SquareCylinder_3D_Re200 | drag_cd ↓ | 1.3504 | 1.3731 | PPO / 4000 | public | 4000 | — |

Paper references are approximate readings of published test curves, with different configurations/budgets; see JSON figure provenance.
All cases remain in the benchmark, including the pending Cylinder_3D_Re3900. Missing results are not zeros.
Historical paper-profile results are separately recorded in JSON and are never pooled with Research v2.
