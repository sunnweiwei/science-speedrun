"""Physical cavity fluctuation diagnostics, not recovered paper aggregation.

Use each probe's temporal mean for its fluctuation, over the complete declared
episode. Report all probes and explicitly named spatial summaries, not a chosen
sensor/window or an implied match to the paper's unspecified plotted statistic.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from compare_trajectories import selected


def analyze(rows, protocol):
    horizon = protocol['horizon']
    if horizon <= 1 or len(rows) != horizon or [r['step'] for r in rows] != list(range(1, horizon + 1)):
        raise ValueError('Require one complete ordered episode with at least two samples')
    recording = protocol['recorded_probe_velocity']
    if recording != dict(enabled=True, units='lattice_velocity', order='all_u_then_all_v'):
        raise ValueError('Missing explicit raw velocity recording convention')
    if protocol['environment'] != 'Cavity_2D_Re4200' or len(protocol['probe_locations']) % 2:
        raise ValueError('Only the audited 2D cavity sensor layout is supported')
    count = len(protocol['probe_locations']) // 2
    values = np.asarray([r['raw_probe_velocity'] for r in rows], dtype=np.float64)
    mach = protocol['initial_state']['mach']
    if isinstance(mach, bool) or not math.isfinite(mach) or mach <= 0:
        raise ValueError('Require finite positive source Mach number')
    if count == 0 or values.shape != (horizon, 2 * count) or not np.isfinite(values).all():
        raise ValueError('Invalid raw probe velocity shape or finite values')
    returns = [r['reward'] for r in rows]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in returns):
        raise ValueError('Invalid raw rewards')
    normalized = values / (mach / math.sqrt(3))
    if not np.isfinite(normalized).all():
        raise ValueError('Velocity normalization overflowed')
    mean = normalized.mean(axis=0)
    rms = np.sqrt(np.square(normalized - mean).mean(axis=0))
    if not np.isfinite(mean).all() or not np.isfinite(rms).all():
        raise ValueError('Nonfinite velocity statistics')
    components = {}
    for name, segment in (('u', slice(0, count)), ('v', slice(count, 2 * count))):
        components[name] = dict(mean_per_probe=mean[segment].tolist(),
            temporal_rms_per_probe=rms[segment].tolist(),
            spatial_mean_of_temporal_rms=float(rms[segment].mean()),
            spatial_rms_of_temporal_rms=float(np.sqrt(np.square(rms[segment]).mean())))
    return dict(status='physical-velocity-diagnostics-not-paper-aggregation-certified',
        window='complete-declared-episode', samples=horizon, probes=count,
        velocity_units='freestream_velocity', fluctuation_reference='each_probe_own_episode_mean',
        temporal_variance_ddof=0, raw_return=math.fsum(returns), components=components)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--training-step', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, source = selected(args.trajectory, args.episode, args.training_step)
    result = analyze(rows, json.loads(args.protocol.read_text()))
    result['trajectory'] = source
    import hashlib
    result['protocol_sha256'] = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
