# Continuously improve one 50M folding model

## Task and goal

Design, train, evaluate and improve one general biomolecular structure-prediction model for all
nine FoldBench tracks:

- monomers: protein, RNA, DNA;
- interfaces: protein-protein, antibody-antigen, protein-peptide, protein-RNA, protein-DNA,
  protein-ligand.

The folding model must contain **45M--55M learned parameters; target about 50M**. The fixed
ESMC-6B input features are data and are not counted. Every learned feature projection, trunk,
structure module, confidence head and modality-specific component in the candidate is counted.

Each track is scored by the pinned official FoldBench scorer. Monomers are scored by lDDT and
interfaces by DockQ or ligand success rates, each in [0, 1]. An interface succeeds with DockQ ≥ 0.23
and CAPRI acceptable geometry (ligand RMSD ≤ 10 Å or interface RMSD ≤ 4 Å). The reward is the unweighted mean of the
nine track scores. The research objective is a significant improvement over the pinned frozen
ESMFold2-Fast reference (reward 0.545) on **every** FoldBench track, not only in the mean. The
from-scratch starter currently reaches 0.037.

## Workspace

- `/workspace/contract.md`: the independently replayed delivery interface.
- `/workspace/data/`: `README.md` describes the frozen 100K training pool and the fixed-feature
  semantics. The pool, structures, ESMC-6B features, the visible FoldBench suite and the
  ESMFold2-Fast reference checkpoint are mounted read-only next to it.
- `/workspace/baseline/starter/`: the executable approximately-50M starter (an ESMFold2 graph
  trained from scratch) and its reproduced results next to the reference, per track.
- `/workspace/references/`: runnable approximately-50M architectures, pinned upstream sources and
  prior controlled experiments, not re-run by the integrator.
- `/workspace/evaluation/`: direct development evaluation on the visible suite.
- `/workspace/research/`: writable space for experiments, checkpoints and resume state.

First verify the starter's parameter audit, one-step training path, publication path and
independent replay. Then use it, or another supplied reference, as a serious starting point.
You may replace or deeply change the architecture, data processing, training method, objectives,
optimizer, sampling, checkpoint selection and inference-time compute. This is full model R&D,
not merely an architecture search. A toy model may be useful for an isolated diagnostic. It is
not a valid 45M--55M candidate, and hand-written geometric scaffolds must not replace the main
research line.

## Rules

- All 100,000 supplied records form the training pool. Choose data projections, crops, sampling,
  augmentation, derived training targets and internal validation splits as part of the research.
- The ESMC-6B final-layer groupwise-INT8 tensors are fixed, evaluator-supplied input **data**.
  ESMC-6B was run before the session; its checkpoint is not mounted, is not part of the
  candidate, and is not needed at inference. The canonical ESMC tensors are final normalized
  `[L,2560]` features. They contain no ESMFold2 projection or layer mixture. Any projection the
  candidate uses is part of the 45M--55M model. Learn any feature projection from scratch.
- The pinned ESMFold2-Fast checkpoint under `/workspace/data/models` is a baseline/reference data
  asset. You may:
  - inspect its architecture;
  - run it when compatible inputs are available;
  - compare against its results;
  - use it for legitimate analysis or target construction on the 100K training pool.

  You must not copy, initialize, transform, average or otherwise transfer any of its parameters
  into the candidate.
- Do not use any other pretrained model weights. Supplied source code, configurations,
  algorithms and architectural ideas are references. All submitted learned parameters must
  originate from random initialization and training in this session. Train from scratch, or
  resume only checkpoints from your own candidate lineage.
- No MSA or template cache is supplied or required. Do not build MSA databases or use MSA or
  template inputs.
- The complete FoldBench inputs, native structures, fixed inputs and evaluation code are visible
  for analysis, repeated evaluation and global model selection. There is no organizer-defined
  hidden development partition.
  - Do not train on benchmark examples: not their inputs, labels, per-example transformations,
    augmentations or distillation targets.
  - Do not fit parameters at test time.
  - General changes motivated by aggregate errors are allowed only when learned on legitimate
    training data.
  - Do not implement native-coordinate lookups, cached benchmark predictions, ID or sequence
    answer tables, or manually corrected benchmark structures. Native coordinates are for analysis
    and scoring, never predictor inputs.
- Maintain one model with one fixed parameter set across every track. Independently trained,
  fine-tuned or selected per-track models are forbidden, including behind a common wrapper.
  Shared multimodal components and input-dependent computation inside one jointly trained model
  are allowed.
- There is no network.

## Submission

`/workspace/submission` is only the currently selected replay package, never an experiment
archive. At every stable publication it contains:

- exactly one current candidate's inference-time learned parameter set;
- only the code, configuration and owned assets needed to replay that candidate.

Never keep `candidate-vN` directories, rollback copies, old weights, optimizer states or ongoing
experiments there; keep research history and resume state under `/workspace/research`. The
package may be read at any moment, so publish atomically, as the starter's publication path
does.

The entry point is `infer.sh`, documented in the contract. It must:

- load the package independently of the live trainer;
- use the same 45M--55M parameter set on every track;
- handle unseen conforming inputs.

Keep a concise README in the package: architecture, exact parameter count, from-scratch training
provenance, data used, benchmark-feedback use and known limitations. Publish a usable candidate
early; the first trained checkpoint of the starter is enough.

## Continuous research

The baselines have already been trained and scored in this setup: `RESULTS.md` in each
`/workspace/baseline/<name>/` has their numbers and, where present, `reproduction/` their
training logs and validation outputs. Do not retrain a baseline to confirm its numbers; check your
own pipeline with a short run instead.

This is a continuing research task, not a finite deliverable. Repeatedly:

1. establish or update a measured reference and form a research hypothesis;
2. implement the change in a scalable learned folding system;
3. train from scratch or resume only checkpoints from your own candidate lineage;
4. evaluate and analyze every affected FoldBench track;
5. publish the strongest complete, compliant, independently replayable candidate when it
   improves the evidence;
6. immediately begin the next experiment.

Do not return a final answer or declare the task complete. A completed run, a published model, an
empty todo list or a reached score target is only a milestone. Reaching the research objective
still does not end the research loop.
