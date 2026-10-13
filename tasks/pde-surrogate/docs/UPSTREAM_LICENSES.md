# Upstream licenses and attribution

| Resource | Use in this task | License | Notes |
| --- | --- | --- | --- |
| `thuerey-group/pde-transformer-ape2d`, `-ape2d-full` (Holzschuh, Liu, Kohl, Thuerey, ICML 2025) | pretraining corpus, pinned download; the truncated heads and held-out tails of nine files are republished with attribution in the task's data release (`environment/data/PUBLICATION.md`) | MIT | generated with `exponax` 0.1.0; the full release's `hyp.hdf5` duplicates `diff.hdf5` and is excluded |
| `thuerey-group/pde-transformer` checkpoints | external-target replay by the maintainers; not available to agents (the task forbids pretrained assets) | MIT | pinned `d5889aab16b1fe82e82940b612d54a8f0773e542` |
| `pdetransformer` package (tum-pbs/pde-transformer) | reference replay code | MIT | requires `transformers<5` |
| The Well datasets `polymathic-ai/{turbulent_radiative_layer_2D, active_matter}` (Ohana et al., NeurIPS 2024 D&B) | transfer datasets, pinned download | CC-BY-4.0 | |
| `the_well` package (PolymathicAI/the_well) | CNextU-net reference retraining; metric definition | BSD-3-Clause | the published baseline checkpoints `polymathic-ai/FNO-*`, `UNetConvNext-*` declare **no license** and are not used or redistributed |
| `pdearena/NavierStokes-2D` (Gupta & Brandstetter, 2022) | transfer dataset, pinned download | MIT | upstream README holds only the license tag; layout documented in this task |
| `hzk17/DPOT` weights (Hao et al., ICML 2024) | external-reference replay by the maintainers; not available to agents | Apache-2.0 | the DPOT code repositories carry no license file and NVIDIA-restricted headers; the task ships no DPOT code |
| `neuraloperator` | dependency of `the_well` models | MIT | |
| PDEBench, Poseidon/PDEgym, MeshGraphNets, PROSE | audited, excluded | CC-BY-4.0 / CC-BY-NC-4.0 / unstated / MIT | reasons in `environment/references/SOTA_AUDIT.md` |

Cite the source papers when publishing results on this task. Contributor:
Deqing Fu (T0-RSI); task engineering assisted by Claude Code.
