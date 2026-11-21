# Executable starter measurements

On one RTX A6000, a public request for two 119-residue monomers produced two AF3 passes in two distinct Foldseek clusters. The matched native Low reference also produced two passing clusters. Generation and final AF3 took 136.17 seconds for the starter and 139.54 seconds for Low. This two-slot run checks the executable interface. The standard 30- and 150-slot measurements of the original packaged policy are in the [stochastic TSR results](../stochastic-tsr/RESULTS.md).

Generation used seed 727119 and the final AF3 seed was 727000. The paired report records both programs’ counts, timings and GPU. [Public paired report](reproduction/public-119.json) · [Starter inference log](reproduction/public-119.log)
