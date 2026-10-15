# Final request suite

The final request suite is operator-held and is not committed. Build it from an
operator-owned random seed file containing at least 32 random bytes:

```bash
python3 crystal-structure-generation/tests/requests/build_suite.py \
  --seed-file /secure/crystalgen/private/request-seed \
  --output-root /secure/crystalgen/final-suite
```

The output contains one 2,500-sample unconditional request and a
`crystalgen-final-suite/v1` manifest binding its hash and sample count. Use a
fresh output directory: the builder refuses to overwrite an existing suite.
The algorithm and schema are public; only the final seed is held back until
replay. The same seed file reproducibly creates the same request.

No MP-20 test CSV, chemical-system labels, or target densities are needed.
There is no hidden answer key: generated structures are scored by physics and
reference matching. Holding back the seed limits tuning to a particular final
sample batch. Development uses exactly the same scorer and request schema.
