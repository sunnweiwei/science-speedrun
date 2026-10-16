# FoldBench: validation and publication records

Moved from the README when the task adopted TASK_STANDARD.md. Paths below predate the move
(`/data` is now `/workspace/data`, `environment/starter` is now `environment/baseline/starter`).

## Validation (2026-08-30)

- Both Dockerfiles build from pinned public revisions; the ESMFold2 CCD is
  checksum-verified, and the starter graph audits to exactly 49,198,221 learned
  parameters without loading weights.
- A real 100K index reported 98,002 train / 1,998 group-disjoint validation
  views and zero missing fixed protein features. No MSA paths enter the index.
- Eight H100s completed a BF16 forward/backward, finite-gradient synchronization,
  optimizer step, native checkpoint, and current-candidate publication. A second
  optimizer step completed after strict resume.
- The 196,871,800-byte step-1 package replayed with networking disabled in the
  independent verifier image on coordinate-free request `5sbj-assembly1`; its
  generated all-atom mmCIF passed contract validation and the pinned
  OpenStructure scorer. This is a pipeline smoke test, not a benchmark result.
- Interface tests cover placeholder failure, path containment, canonical INT8
  decoding and payload provenance, fixed-feature coverage, pretrained-checkpoint
  rejection, relocated model loading, fixed denominators, and a real
  native-scorer fixture.
  That optional fixture copies one existing native as a plumbing oracle; it is
  **not a trained model, an agent experiment, or a benchmark score**.
- The organizer's local cache was fully generated and audited: 91,366 training
  features and 1,568 suite features, with no ESMC checkpoint in Agent-visible
  data. A scientifically trained long run is separate from these pipeline checks.
  No large data or learned weights are stored in this GitHub repository.

## Data publication (2026-09-19)

See [publication validation](environment/data/PUBLICATION.md) for the frozen
payload breakdown, download pin and materialization tests. This publication
changes storage and download wiring, not the training pool, evaluation suite,
metrics, feature representation, research prompt or submission contract.

## Verifier isolation note (moved from contract.md)

This visible verifier runs predictor and scorer sequentially in one container;
it is not a hardened boundary against malicious code reading native structures.
A private evaluator must isolate input-only inference from label-bearing scoring.
