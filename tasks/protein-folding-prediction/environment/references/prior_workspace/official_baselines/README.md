# Strict official-code baselines

For the autonomous task, use
[`agent_workspace/docs/REFERENCES.md`](../agent_workspace/docs/REFERENCES.md)
as the navigation guide.  The implementations and results below are references
for design and comparison, not restrictions on the submitted architecture.

A baseline belongs here only when all of the following are true:

1. The upstream source revision and runtime image are pinned.
2. Trainable network modules execute from the upstream package; local code may
   register configuration and adapt the common contract, but may not rewrite
   the architecture.
3. The project-wide frozen FoldingTrain dataset and split are used through a
   deterministic native-schema adapter. The upstream losses, permutation
   logic, optimizer, scheduler, checkpoint format, and inference runner are
   retained; upstream dataset sources and sampling mixtures are not copied.
4. Scaling changes only dimensions/counts exposed by the upstream model config.
5. Reported quality numbers come from the pinned FoldBench evaluator and are
   labelled with the evaluated target set, seeds, samples, and inference budget.

The parameter budget applies to the folding model. A separately frozen protein
embedding model is recorded but is not included in that budget.

| Candidate | Strict status | Reason |
|---|---|---|
| Protenix / Pairformer diffusion | Completed | The 48.68M released graph completed one common-data pass, exact ordinary+EMA resume, portable bundle gate, and all nine controlled FoldBench tracks. |
| DPLM-2 | Completed | The official 47.05M core completed the controlled three-stage common-data run, was exported as a portable ModelBundle, and was scored by the pinned FoldBench/OpenStructure evaluator. |
| ESMFold2-Fast | Completed | The 49.20M pinned Biohub graph completed its common-data pass, portable bundle gate, and controlled protein-monomer FoldBench score; its training Recipe is explicitly project-owned. |
| OpenDDE | Completed | The 48.75M released graph completed one common-data pass, a portable offline/read-only bundle gate, and all nine controlled FoldBench tracks. Its training Recipe is explicitly project-owned. |
| Hybrid PLM + Pairformer | First-party completed | ESMC-6B mixed-256 features feed the official Protenix ESM input path and 48.76M folding core; one common-data pass, an offline/read-only bundle gate, and all nine controlled FoldBench tracks completed. |

The old implementations under `reference/` are lightweight pipeline fixtures,
not entries in this strict set.

“Official structure” and “official training Recipe” are separate claims. The
ESMFold2 and OpenDDE public releases still do not provide a complete official
trainer. We reuse their official network modules and fill the missing training
orchestration with the common, inspectable `foldingtrain_edm_v1` Recipe.
Reports always label the architecture upstream-owned and the Recipe
project-owned; they must never be described as official training reproductions.

The word “Recipe” in this directory excludes training-data selection. Data is
an experiment control shared by every compatible model. A capability-limited
model (for example protein-monomer-only DPLM-2) trains on the explicitly audited
supported projection of that same split and never replaces unsupported records
with examples from its original paper dataset.

The exact boundary and per-model status are frozen in
[`RECIPE_SCOPE.md`](RECIPE_SCOPE.md).
