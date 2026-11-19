# Sources and protocol decisions

Corley, Didi, Kazman, Brent et al. describe RFD4-Proteina in a bioRxiv preprint. Supplementary Table S15 is the closest published comparison for this task. It holds the monomer lengths and number of trials fixed and compares sampler efficiency. Section S12 specifies the Foldseek protocol. The paper’s headline design-task leaderboard uses different cohorts and denominators. [Preprint](https://doi.org/10.64898/2026.11.12.766037) · [Supplementary methods and Table S15](https://www.biorxiv.org/content/biorxiv/early/2026/11/15/2026.11.12.766037/DC1/embed/media-1.pdf?download=true)

| Sampler in Table S15 | Steps | Clustered pass rate | Reported time (s) |
| --- | ---: | ---: | ---: |
| Stochastic TSR | 60 | 34.7% | 2.99 |
| Stochastic TSR, prior schedule | 60 | 0.0% | 2.97 |
| ODE TSR | 60 | 33.3% | 3.04 |
| Full-length SDE | 400 | 37.3% | 17.68 |
| Shortened SDE | 60 | 0.0% | 2.95 |

These paper values describe 150 trials per row. The 400-step result illustrates why a fixed-cost task needs a compute bound. The table gives neither the original per-design seeds nor a unique mapping from that experiment to a public checkpoint. Baseline directories report new runs of the frozen public release.

The released default uses stochastic TSR and a medium recycle schedule. Beam search is optional. The authors also provide a monomer example that ranks rollouts by refolder confidence. The full example uses additional refolder calls during search. Here the independent final AF3 budget is one invocation per output slot, and intermediate geometry is computed directly from the decoded monomer. The upstream monomer YAML and BSD license are in `upstream/`. The installed `rfproteina` package contains the native source and example YAML for offline inspection. [Default config](https://github.com/RosettaCommons/RFD4-Proteina/blob/6306a210dd17dcae8e4e0f381b5b74a719d9b30e/configs/inference.yaml) · [Monomer beam example](https://github.com/RosettaCommons/RFD4-Proteina/blob/6306a210dd17dcae8e4e0f381b5b74a719d9b30e/configs/inference_experiment/inference_beam_example.yaml) · [Native search implementation](https://github.com/RosettaCommons/RFD4-Proteina/blob/6306a210dd17dcae8e4e0f381b5b74a719d9b30e/src/rfproteina/search/beam_search.py)

## Related work

- Proteína reports designability, diversity and novelty for unconditional samples. A structurally clustered numerator prevents repeated instances of one successful fold from dominating this benchmark. Novelty relative to a training corpus would require a separate reference database. [Geffner et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f4e9121ad30cd4e5528042fbfd835b3f-Abstract-Conference.html)
- FrameFlow compares designability and the number of denoising steps. RFD4 also recycles inside a step, so our cost ledger counts individual trunk evaluations and the suffixes used for rollout scoring. [Yim et al., FrameFlow](https://arxiv.org/abs/2310.05297)
- Ma et al. compare search over completed samples with search over intermediate diffusion paths. This motivates allowing a terminal-only checkpoint and intermediate branching under the same budget. Their experiments also motivate reporting a separate final endpoint. [Ma et al., CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Ma_Scaling_Inference_Time_Compute_for_Diffusion_Models_CVPR_2025_paper.html)
- AlphaFold 3 provides an independent sequence-to-structure prediction. We use its public implementation with one seed, one sample, five recycles, no templates and empty MSAs. The released RFD4 oracle wrapper also defaults to five recycles and one diffusion sample. Table S15 specifies the RMSD threshold without listing every AF3 flag. [Abramson et al., Nature 2024](https://doi.org/10.1038/s41586-024-07487-w) · [RFD4 refolding wrapper](https://github.com/RosettaCommons/RFD4-Proteina/blob/6306a210dd17dcae8e4e0f381b5b74a719d9b30e/src/rfproteina/af3/run_alphafold_batch.py)
- Foldseek supplies the structural clustering executable. We pin release 10-941cd33 as in §S12. We cluster valid monomers within length and count representatives with at least one passing member. [van Kempen et al., Nature Biotechnology](https://doi.org/10.1038/s41587-023-01773-0) · [Pinned release](https://github.com/steineggerlab/foldseek/releases/tag/10-941cd33)
