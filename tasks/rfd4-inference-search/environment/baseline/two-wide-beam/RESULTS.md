# Baseline measurements

Single RTX A6000, native RFD4 0.0.3, independent AF3 3.0.2 (five recycles, one seed and one sample), Foldseek 10-941cd33. The two public suite sizes use the same seed. Keep their results separate.

## Public development

**11/30 distinct passing folds (36.7%).** 20/30 individual structures passed AF3.

| Length | Slots | AF3 passes | Passing clusters | Trunk passes |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 10 | 7 | 2 | 1800 |
| 150 | 10 | 8 | 5 | 1800 |
| 250 | 10 | 5 | 4 | 1800 |

[Raw metrics](reproduction/development.json) · [Slot measurements](reproduction/development-slots.tsv) · [Run log](reproduction/development-run.log)

## Public paper-size replay

**46/150 distinct passing folds (30.7%).** 101/150 individual structures passed AF3.

| Length | Slots | AF3 passes | Passing clusters | Trunk passes |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 50 | 41 | 11 | 9000 |
| 150 | 50 | 34 | 15 | 9000 |
| 250 | 50 | 26 | 20 | 9000 |

[Raw metrics](reproduction/paper-size.json) · [Slot measurements](reproduction/paper-size-slots.tsv) · [Run log](reproduction/paper-size-run.log)

AF3-derived measurements are subject to the [AlphaFold 3 Output Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md).
