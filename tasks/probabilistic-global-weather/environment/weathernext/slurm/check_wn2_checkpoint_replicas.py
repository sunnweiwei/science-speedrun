"""Verify exact replica checks without indexing a remote GPU through JAX."""
import argparse
import json
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from wn2_distributed import save_state, replicate_from_host


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    devices = jax.devices()
    assert len(devices) >= 2 and all(d.platform == 'gpu' for d in devices)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'state.npz'
        good = jax.device_put_replicated(np.array([1., 0.], np.float32), devices)
        jax.block_until_ready(good)
        audit = dict(devices=[str(d) for d in devices],
                     full=np.asarray(good).tolist(),
                     shards=[dict(device=str(shard.device), shape=list(shard.data.shape),
                                  values=np.asarray(shard.data).tolist(),
                                  bits=np.asarray(shard.data).tobytes().hex())
                             for shard in good.addressable_shards])
        independent = jax.device_put_sharded([np.array([1., 0.], np.float32)
                                             for _ in devices], devices)
        audit['independent_full'] = np.asarray(independent).tolist()
        audit['independent_shards'] = [dict(device=str(shard.device),
            values=np.asarray(shard.data).tolist(), bits=np.asarray(shard.data).tobytes().hex())
            for shard in independent.addressable_shards]
        args.output.with_name('transfer-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        good = replicate_from_host(np.array([1., 0.], np.float32), devices)
        save_state(path, {'x': good}, {'moment': good}, {})
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['leaf_0000'], [1., 0.])
        path.unlink()
        bad = jax.device_put_sharded([
            np.array([1., -0. if i == 1 else 0.], np.float32)
            for i in range(len(devices))], devices)
        for params, state in [({'x': bad}, {'moment': good}),
                              ({'x': good}, {'moment': bad})]:
            try:
                save_state(path, params, state, {})
            except ValueError as error:
                assert 'replicas differ' in str(error)
            else:
                raise AssertionError('Divergent replica checkpoint accepted')
            assert not path.exists()
    # Exercise zeros after pmap outputs, as at the next training update.
    tree = replicate_from_host([np.ones(n, np.float32) for n in [1, 3, 32, 768, 2304, 4096]], devices)
    fill = jax.pmap(lambda p: jax.tree.map(lambda x: x+1, p), devices=devices)
    zero = jax.pmap(lambda p: jax.tree.map(jnp.zeros_like, p), devices=devices)
    zero_audit = []
    for step in range(3):
        tree = fill(tree)
        old = jax.tree.map(jnp.zeros_like, tree)
        new = zero(tree)
        bad_old = []
        for index, leaf in enumerate(old):
            for shard in leaf.addressable_shards:
                if np.any(np.asarray(shard.data) != 0):
                    bad_old.append(dict(leaf=index, device=str(shard.device)))
        for leaf in new:
            for shard in leaf.addressable_shards:
                assert np.all(np.asarray(shard.data) == 0)
        zero_audit.append(dict(step=step, original_nonzero_shards=bad_old))
    args.output.with_name('zero-audit.json').write_text(json.dumps(zero_audit, indent=2)+'\n')
    rank_values = np.arange(1, len(devices)+1, dtype=np.float32)
    identity = jax.pmap(lambda x: x, devices=devices)
    np.testing.assert_array_equal(np.asarray(identity(rank_values)), rank_values)
    mean = jax.pmap(lambda x: jax.lax.pmean(x, 'rank'), axis_name='rank', devices=devices)
    np.testing.assert_array_equal(np.asarray(mean(rank_values)), np.full(len(devices), rank_values.mean()))
    args.output.write_text(json.dumps(dict(
        status='passed', devices=len(devices), equal_replicas_saved=True, host_argument_distribution_exact=True, pmean_exact=True,
        parameter_and_optimizer_bit_differences_rejected=True), indent=2)+'\n')


if __name__ == '__main__':
    main()
