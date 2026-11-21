# References for the research workspace

The executable evaluator runs a matched human reference on each request. The standard report also includes the archived baseline points at the same sample count. Compare actual GPU time, including final AF3, alongside clustered quality. [Submission and comparison protocol](../contract.md)

[Official suite-level operating points](overall/operating_points.csv) summarize the Low, Medium, High and Search results for each public Figure 2 / S29 suite. Use them as historical starting points. [Published comparator field](overall/comparison_field.csv) · [Generation throughput](overall/quality-throughput.png) · [Throughput including final AF3](overall/quality-throughput-af3.png)

The live research workspace evaluates monomer policies on development draws. The baseline directories include public development run records. Use these results to choose policies. The controller evaluates a frozen submission on private draws. Current monomer development and verification use the same three lengths.

The local S15 comparison measures five unconditional samplers on the same lengths, seeds and RTX A6000 hardware. It includes warm generation speed, throughput including final AF3 and clustered pass rates for 150 attempts per sampler. Each 60-step arm used 180 trunk passes per slot. The 400-step arm used 1,199 and exceeds the legacy policy allowance. Executable programs are timed under the request budget. [Measured S15 operating points](s15/summary.csv)

The [native monomer search example](upstream/inference_beam_example.yaml) and its [license](upstream/LICENSE) are available offline. See the published study for the interpretation of other benchmarks. [RFD4-Proteina paper](https://doi.org/10.64898/2026.11.12.766037)
