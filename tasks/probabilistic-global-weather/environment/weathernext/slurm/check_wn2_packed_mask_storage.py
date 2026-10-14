#!/usr/bin/env python3
"""Check that compiled masks keep byte-packed constants and exact values."""
import argparse
import json
import re
from pathlib import Path
import jax
import numpy as np
from scipy import sparse
from wn2_boolean_masks import mask_block_diags as boolean
from wn2_packed_masks import mask_block_diags as packed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    width=257;nodes=513
    graph=sparse.csr_matrix(np.random.default_rng(4).random((nodes,nodes))>.7)
    compiled=jax.jit(lambda:packed(graph,1,width)).lower().compile()
    np.testing.assert_array_equal(np.asarray(compiled()),np.asarray(boolean(graph,1,width)))
    hlo=compiled.as_text()
    constants=[line.strip() for line in hlo.splitlines() if ' constant(' in line]
    shapes=[re.search(r'= (\w+\[[^]]*\])',line).group(1) for line in constants]
    assert f'u8[{(width*width+7)//8}]' in shapes,shapes
    assert f'pred[{width},{width}]' not in shapes,shapes
    args.output.write_text(json.dumps(dict(status='passed',backend=jax.default_backend(),
        exact_values=True,constant_shapes=shapes,logical_mask_block_bytes=width*width,
        packed_block_bytes=(width*width+7)//8,
        scope='Small compiled mask only; full native executable serialization not established'),indent=2)+'\n')


if __name__=='__main__':main()
