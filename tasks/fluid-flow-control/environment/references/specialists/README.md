# Specialist campaign

Formerly the task-root `baselines/` directory; dated documents below still use that path.

Current entrypoints:

- [Research v2 specialist trainer and reproduction instructions](RESEARCH_BASELINES.md).
- [Verified same-metric result table](verified/README.md) and [evidence/configs](verified/results.json).
- [Agent's shared-policy starter](../../baseline/starter/README.md).

The specialist implementations are learning/comparison resources, not a
compliant multi-task submission. Only their curated runnable subset and small
results are copied into the agent image; checkpoint/data payloads stay out of Git.

Other files in this directory are dated organizer experiment history: older
paper configurations, cavity/JAX reconstruction, local scheduling, performance
diagnostics and audits. Their machine paths, budgets and old reward tables are
**not current task entrypoints or an extension of the frozen 39-case data**.
Keep them for provenance rather than silently relabeling historical experiments.
