#!/usr/bin/env bash
# fetch_variant_evals.sh OUT_DIR -- download the pinned published evaluation tables (public, no auth)
set -euo pipefail
out=${1:?usage: fetch_variant_evals.sh OUT_DIR}
mkdir -p "$out/ag-evals" "$out/traitgym"
ag=https://storage.googleapis.com/alphagenome/evals
for name in clinvar_splice_site_region clinvar_noncoding clinvar_missense sqtl_variant_causality_gene_human \
    mfass_splicing eqtl_variant_borzoi_sign_human eqtl_variant_catalogue_causality_gene_balanced_human \
    eqtl_variant_borzoi_coefficient_human enhancer_gene_linking_e2g caqtl_african_variant_causality_human \
    bqtl_spi1_variant_causality_human; do
  curl -fsSL -o "$out/ag-evals/$name.feather" "$ag/${name}_predictions.feather"
done
tg=https://huggingface.co/datasets/songlab/TraitGym/resolve/1fde19555fe8c0a55b1382bdf4c6f7082209f566
for set in mendelian_traits_matched_9 complex_traits_matched_9; do
  curl -fsSL -o "$out/traitgym/$set.test.parquet" "$tg/$set/test.parquet"
  for model in Borzoi_L2_L2.plus.all Enformer_L2_L2.plus.all CADD.plus.RawScore GPN-MSA_absLLR.plus.score; do
    curl -fsSL -o "$out/traitgym/$set.$model.parquet" "$tg/$set/preds/all/$model.parquet"
  done
done
curl -fsSL -o "$out/gencode.v46.annotation.gtf.gz.feather" \
  https://storage.googleapis.com/alphagenome/reference/gencode/hg38/gencode.v46.annotation.gtf.gz.feather
here=$(cd "$(dirname "$0")" && pwd); (cd "$out" && sha256sum -c "$here/variant_evals.sha256")
