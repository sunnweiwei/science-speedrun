#!/usr/bin/env python3
"""CPU-only native graph/mask shape audit; never runs the neural network."""
import argparse
import json
import math
from pathlib import Path

from weathernext.utils import data_modalities, mesh_transformer, sparse_transformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = []
    for splits in (5, 6):
        mesh = data_modalities.TriangularMeshData.with_icosahedral_mesh(splits_list=[2]*splits)
        adjacency = mesh_transformer._get_adjacency_matrix(mesh, add_self_edges=True)
        mask = adjacency ** 32
        width = int(sparse_transformer.get_mask_block_size(mask))
        blocks = math.ceil(mask.shape[0]/width)
        elements = 3*blocks*6*width*width
        results.append(dict(mesh_splits=splits, nodes=mask.shape[0], attention_hops=32,
                            allowed_pairs=int(mask.nnz), block_width=width, blocks=blocks,
                            padded_nodes=blocks*width, heads=6, batch=1,
                            three_logit_blocks_elements=elements,
                            three_logit_blocks_bf16_bytes=elements*2,
                            three_softmax_blocks_fp32_bytes=elements*4,
                            scope='Logical tensor sizes before compiler fusion/reuse; not measured peak memory'))
        del mask, adjacency, mesh
    args.output.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
