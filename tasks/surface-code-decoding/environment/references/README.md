# References

The anchors and prior work behind the score. The numbers here are published or audited values,
not re-run in this workspace.

- **Lower anchor, P = 0.71605:** audited minimum-weight perfect matching (MWPM) on the same test
  shots.
- **Upper anchor, P = 0.76152:** the strongest same-test released tensor-network decoder result.
- **AlphaQubit** (Bausch et al., "Learning high-accuracy error decoding for quantum processors",
  Nature 2024): the recurrent-transformer decoder that the baseline reproduces independently.
- **Data:** Google Quantum AI, "Suppressing quantum errors by scaling a surface code logical qubit",
  Nature 2023 (Zenodo 6804040).

Learned decoders may be combined with matching decoders. Pretrained or external checkpoints,
historical per-shot decoder answers and test labels are not allowed (`/workspace/contract.md`).
