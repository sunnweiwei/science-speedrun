"""Frozen per-case physical reporting metrics; never read candidate scores.

Native rewards remain available to learners/diagnostic logs. This specification
does not define a cross-case aggregate or change any source physics.
"""
import hashlib
import json
import math
from pathlib import Path

from protocol import InfrastructureError

VERSION = 'hydrogym-physical-metrics/v1'
DEFINITIONS = {
    'drag_cd': dict(name='平均阻力系数 CD', units='dimensionless', direction='lower_is_better',
                    window='last_half', quantity='CD', reduction='mean'),
    'lift_drag': dict(name='平均升阻比 L/D', units='dimensionless', direction='higher_is_better',
                     window='last_half', quantity='lift_drag', reduction='mean'),
    'gust_load_error': dict(name='平均加权载荷偏差', units='coefficient', direction='lower_is_better',
                           window='full_episode', quantity='gust_load_error', reduction='mean'),
    'nozzle_angle_mae': dict(name='推力方向平均绝对误差', units='degree', direction='lower_is_better',
                            window='full_episode', quantity='angle_error_deg', reduction='mean'),
    'tke': dict(name='平均扰动动能 TKE', units='native_nondimensional', direction='lower_is_better',
               window='last_half', quantity='TKE', reduction='mean'),
    'wall_shear': dict(name='平均壁面剪切应力', units='native_nondimensional', direction='lower_is_better',
                      window='last_half', quantity='wall_shear', reduction='mean'),
}
GROUPS = {
    'drag_cd': '''Cube_3D_Re300 Cube_3D_Re3700
        Cylinder_2D_Re1000 Cylinder_2D_Re200 Cylinder_2D_Re3900
        Cylinder_3D_Re200 Cylinder_3D_Re3900
        Pinball_2D_Re100 Pinball_2D_Re150 Pinball_2D_Re30 Pinball_2D_Re75
        Pinball_3D_Re100 Pinball_3D_Re150 Pinball_3D_Re30 Pinball_3D_Re75
        Sphere_3D_Re300 SquareCylinder_2D_Re1000 SquareCylinder_2D_Re200 SquareCylinder_3D_Re200''',
    'lift_drag': '''NACA0012_2D_Re1000_AOA20 NACA0012_2D_Re1000_AOA40
        NACA0012_2D_Re100_AOA20 NACA0012_2D_Re100_AOA40
        NACA0012_3D_Re1000_AOA20 NACA0012_3D_Re100_AOA20 NACA0012_3D_Re100_AOA40''',
    'gust_load_error': '''NACA0012Gust_2D_Re1000_AOA20 NACA0012Gust_2D_Re1000_AOA40
        NACA0012Gust_2D_Re100_AOA20 NACA0012Gust_2D_Re100_AOA40
        NACA0012Gust_3D_Re1000_AOA20 NACA0012Gust_3D_Re1000_AOA40
        NACA0012Gust_3D_Re100_AOA20 NACA0012Gust_3D_Re100_AOA40 NACA0012Gust_3D_Re50000_AOA12''',
    'nozzle_angle_mae': 'Nozzle2D_coarse Nozzle2D_fine',
    'tke': 'Kolmogorov_2D',
    'wall_shear': 'Channel_3D_Retau180',
}
CASE_METRICS = {name: key for key, names in GROUPS.items() for name in names.split()}


def definition(name):
    try:
        key = CASE_METRICS[name]
    except KeyError as error:
        raise InfrastructureError(f'No frozen primary metric for {name}') from error
    result = dict(version=VERSION, id=key, **DEFINITIONS[key])
    if key == 'gust_load_error':
        configs = json.loads((Path(__file__).resolve().parent.parent / 'data/native-configs.json').read_text())
        config = next(c['config']['maia'] for c in configs['configs'] if c['name'] == name)
        result['target'] = dict(CL=config['unperturbed_avg_lift'],
                                CD=config['unperturbed_avg_drag'], omega=config['omega'])
    return result


def registry():
    return {name: definition(name) for name in sorted(CASE_METRICS)}


def registry_hash():
    return hashlib.sha256(json.dumps(registry(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sample(env, case, info, reward):
    """Read physical outputs after one unchanged native step (no second CFD step).

    Pinball already exposes nondimensional coefficients; other MAIA classes
    expose raw lattice forces. Sphere uses pi*D^2/4, not the prism area.
    Channel's pinned reward is exactly -wss, with no action penalty; this one
    identity preserves its existing, hash-frozen actuator-clock adapter.
    """
    import numpy as np
    name = case['environment']
    metric = case.get('primary_metric') or definition(name)
    key = metric['id']
    base = getattr(env, 'unwrapped', env)
    if key in ('drag_cd', 'lift_drag', 'gust_load_error'):
        forces = np.asarray(info['forces'], dtype=np.float64)
        if forces.ndim == 1:
            forces = forces[None, :]
        if forces.ndim != 2 or forces.shape[0] == 0 or forces.shape[1] < 2 or not np.isfinite(forces).all():
            raise InfrastructureError('Missing/invalid per-body force measurements')
        if not name.startswith('Pinball_'):
            area = (math.pi / 4 * (base.referenceLength / base.dX) ** 2
                    if name.startswith('Sphere_') else base.referenceLength / base.dX * base.zLength / base.dX)
            forces = np.asarray(base.compute_nondim_coefficients(forces=forces, density=1.0,
                referenceVelocity=base.Ma / math.sqrt(3), projectionLength=area))
        values = dict(CD=float(forces[:, 0].sum()), CL=float(forces[:, 1].sum()))
        if forces.shape[1] > 2:
            values['CS'] = float(forces[:, 2].sum())
        if key == 'lift_drag':
            # Keep the source's admissible lift/drag domain; do not replace an
            # inadmissible physical ratio by its artificial -100 training penalty.
            valid = values['CD'] >= 1e-6 and values['CL'] >= 0
            values['lift_drag_valid'] = float(valid)
            values['lift_drag'] = values['CL'] / values['CD'] if valid else 0.0
        elif key == 'gust_load_error':
            target = metric['target']
            actual = dict(CL=float(base.unperturbed_avg_lift), CD=float(base.unperturbed_avg_drag), omega=float(base.omega))
            if actual != target:
                raise InfrastructureError('Gust target/weight changed from frozen source')
            values['lift_error'] = abs(values['CL'] - target['CL'])
            values['drag_error'] = abs(values['CD'] - target['CD'])
            values['gust_load_error'] = values['lift_error'] + target['omega'] * values['drag_error']
    elif key == 'nozzle_angle_mae':
        values = dict(angle_error_deg=float(abs(np.asarray(info['target_angle']).item()
                                               - np.asarray(info['thrust_angle']).item()) * 180 / math.pi))
    elif key == 'tke':
        values = dict(TKE=float(info['mean_tke']))
    elif key == 'wall_shear':
        values = dict(wall_shear=-float(reward))
    else:
        raise InfrastructureError('Unimplemented primary metric')
    if not all(math.isfinite(v) for v in values.values()):
        raise InfrastructureError('Nonfinite physical measurement')
    return values


def summarize(metric, rows, horizon):
    if len(rows) != horizon or horizon <= 0:
        raise InfrastructureError('Primary metric requires a complete episode')
    start = horizon // 2 if metric['window'] == 'last_half' else 0
    selected = rows[start:]
    result = dict(metric, samples=len(selected), first_control=start + 1, last_control=horizon)
    if metric['id'] == 'lift_drag' and any(r['lift_drag_valid'] != 1 for r in selected):
        return dict(result, status='invalid_operating_regime', value=None,
                    reason='L/D requires CD >= 1e-6 and CL >= 0 at every scored sample')
    values = [r[metric['quantity']] for r in selected]
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
        raise InfrastructureError('Missing/nonfinite primary metric samples')
    value = math.fsum(values) / len(values)
    if not math.isfinite(value):
        raise InfrastructureError('Nonfinite primary metric mean')
    return dict(result, status='ok', value=value)


def aggregate(metric, episodes):
    """Equal-weight complete episode metrics, never reward or terminal info."""
    measured = [e.get('primary_metric', {}) for e in episodes]
    if not measured or any(any(m.get(k) != v for k, v in metric.items()) for m in measured):
        raise InfrastructureError('Missing/mismatched physical metric definition')
    for m in measured:
        if m.get('status') == 'invalid_operating_regime':
            if metric['id'] != 'lift_drag' or m.get('value') is not None:
                raise InfrastructureError('Invalid operating-regime outcome')
        elif m.get('status') != 'ok' or type(m.get('value')) not in (int, float) or not math.isfinite(m['value']):
            raise InfrastructureError('Invalid physical metric episode value')
    if any(m.get('status') == 'invalid_operating_regime' for m in measured):
        return dict(metric, status='invalid_operating_regime', value=None, episodes=len(measured))
    values = [m['value'] for m in measured]
    return dict(metric, status='ok', value=math.fsum(values) / len(values), episodes=len(values))
