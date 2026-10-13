---
license: cc-by-4.0
pretty_name: Sycamore Surface-code Decoding — Private Evaluator Labels
tags:
- quantum-error-correction
- benchmark-evaluation
---

# Trusted evaluator distribution

This repository contains `grader-labels.tar.gz`, a private operator distribution
for the [public materialized benchmark](https://huggingface.co/datasets/sunweiwei/ai4sci-surface-code-decoding),
and `reference-model.tar.gz`, the frozen reference AlphaQubit submission operators
replay as a known answer (P=0.75506 on the full test). Neither is ever mounted for agents.
It holds official-path binary logical-flip labels and opaque sample IDs for
3,250,000 odd-indexed Sycamore shots across 130 fixed conditions. Its manifest
binds labels to the exact input-only verifier manifest via SHA-256.

Only the trusted grader may mount the extracted `grader/` directory. The
prediction container and development agent must never receive it. The grader
never imports submitted model source or executes submitted code. This private
packaging controls runtime access; the upstream historical data is public.

Original source: [Google Quantum AI, Zenodo 6804040](https://zenodo.org/records/6804040),
CC-BY-4.0. Attribute Google Quantum AI and retain the source citation when
redistributing derived data. The fixed score configuration uses same-test
MWPM/TN anchors, preserves raw performance, and clips normalized score to [0,1].
Use the immutable revision and archive SHA-256 pinned in the GitHub task.
