# Starter: reproduced results

The starter was trained from scratch with `train.sh` defaults: 49,198,221 folding parameters, one
epoch over the 100K pool. The run took 3.7 h on six H100s, including one resume. It was published
through the starter's own publication path and replayed by the task's verifier on the full
visible suite (1,522 requests).

| track | metric | starter | frozen ESMFold2-Fast (reference, `/workspace/data/models`) | native structures (oracle) |
| --- | --- | ---: | ---: | ---: |
| monomer_protein | lDDT | 0.000 | 0.822 | 1.000 |
| monomer_rna | lDDT | 0.000 | 0.569 | 1.000 |
| monomer_dna | lDDT | 0.000 | 0.372 | 1.000 |
| interface_protein_protein | DockQ success | 0.057 | 0.552 | 0.993 |
| interface_antibody_antigen | DockQ success | 0.017 | 0.291 | 0.965 |
| interface_protein_peptide | DockQ success | 0.118 | 0.608 | 0.941 |
| interface_protein_rna | DockQ success | 0.086 | 0.529 | 1.000 |
| interface_protein_dna | DockQ success | 0.058 | 0.649 | 0.964 |
| interface_protein_ligand | ligand success | 0.000 | 0.511 | 0.998 |
| **reward** (unweighted mean) | | **0.037** | **0.545** | 0.985 |

Interface success also requires CAPRI acceptable geometry (ligand RMSD ≤ 10 Å or interface RMSD
≤ 4 Å) since 2026-10-07. An agent package collapsed every chain onto the others: all native
contacts were recovered (DockQ ≈ 0.36 from fnat alone) at interface RMSD ≈ 15 Å and ligand RMSD ≈ 30 Å,
and the interface tracks scored 0.86–1.0 (reward 0.524) with every monomer at 0. Under the rule the
same package scores 0.011. On the raw scores of an ESMFold2-Fast replay the interface rates fall by
0.01–0.04 (reward 0.569 → 0.557); the native-structure oracle is unchanged (0.985). The table above
predates the rule; the starter's interface rates have not been rescored yet.

- The starter's monomer TM-score is 0.53, but its lDDT is zeroed by the scorer's stereochemistry
  checks. The coordinates are near the fold, but the atoms are not chemically valid yet.
- A one-epoch starter is a floor, not a model. The research objective is to beat the frozen
  ESMFold2-Fast reference on every track with a from-scratch 45M–55M model. That reference
  checkpoint may be run for analysis, but never copied into a candidate.
- The ESMFold2-Fast published monomer lDDT is 0.8375; this replay measured 0.822.
