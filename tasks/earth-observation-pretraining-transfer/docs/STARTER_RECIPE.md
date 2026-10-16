# Starter recipe

The agent-facing baseline documentation is organized under
[environment/baseline/starter](../environment/baseline/starter/README.md):

- [Recipe, cache preparation, training and export](../environment/baseline/starter/README.md).
- [Recorded results, measured costs and limitations](../environment/baseline/starter/RESULTS.md).
- [Fixed evaluation protocol and commands](../environment/evaluation/README.md).

Inside the prepared agent environment, these guides are available at
`/workspace/baseline/starter/README.md`,
`/workspace/baseline/starter/RESULTS.md` and
`/workspace/evaluation/README.md`. Runtime and source-data preparation are
controller responsibilities; the baseline writes derived data, training runs
and exported research packages under `/workspace/research`.

Repository Python source lives in `environment/baseline/starter/`, next to the
baseline guides; the environment build copies the directory to `/workspace/baseline/starter/`.
