# Third-party notices

`environment/evaluation/physics.py` adapts FAIRChem calculation recipes at
commit a1605489fc8ffe875108721b2c25d08b70375f5b.

MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Training and data

The baselines' YAML templates are copied from FAIRChem release 2.16.0, commit
`5ff7bac1052e384f408584d6b472c4b34d0de8f0`. The full MIT license is retained at
`environment/baseline/uma-mp-finetune/templates/LICENSE`. Task-owned preparation/export wrappers
use the public FAIRChem APIs.

The historical reference-convention investigation followed the published
[hyllios/utils phonon benchmark code](https://github.com/hyllios/utils/tree/4e86e82ae2eec3762f66e6228a70f67b69570789/benchmark_ph).
It independently reconstructs mesh, conversion and harmonic-mode contributions;
it does not copy that repository's implementation. MPTrj, sAlex and public MPA
reference provenance and terms are documented in `environment/data/README.md`.
