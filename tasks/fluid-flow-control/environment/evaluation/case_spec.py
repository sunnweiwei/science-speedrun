"""Executable native-replay profiles derived from the immutable Research v2 data.

These are organizer profiles, not recovered paper protocols or score certification.
Both development replay and the Harbor verifier use these exact profiles.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from protocol import InfrastructureError
from metrics import CASE_METRICS, VERSION as METRIC_VERSION, definition, registry_hash

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'data'))
from public_release import load_release, digest
from materialize import file_digest

PROFILE_VERSION = 'hydrogym-native-replay/v1'
BLIND_SEEDS_ENV = 'HYDROGYM_BLIND_SEEDS_FILE'
DATA = Path(__file__).resolve().parent.parent / 'data'


def load_blind_seeds(path=None):
    target = path if path is not None else os.environ.get(BLIND_SEEDS_ENV, '').strip()
    if not target:
        return None
    seed_path = Path(target)
    if not seed_path.is_file():
        raise InfrastructureError(f'Missing sealed blind seed roster: {seed_path}')
    payload = json.loads(seed_path.read_text())
    if payload.get('schema') != 'hydrogym-blind-seeds/v1' or not isinstance(payload.get('cases'), dict):
        raise InfrastructureError(f'Invalid blind seed roster schema in {seed_path}')
    for case_id, seeds in payload['cases'].items():
        if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds)
                or any(type(s) is not int or s < 0 or s == 123 for s in seeds)):
            raise InfrastructureError(
                f'Invalid blind seeds for {case_id}: require unique non-negative ints != 123')
    return payload


def specifications(blind_seeds_path=None):
    release = load_release()
    if {c['id'] for c in release['cases']} != set(CASE_METRICS):
        raise InfrastructureError('Frozen data and physical metric coverage differ')
    blind_roster = load_blind_seeds(blind_seeds_path)
    cases = []
    evidence = {c['case']: c for c in json.loads((DATA / 'cost-evidence-coverage-v1.json').read_text())['cases']}
    for row in release['cases']:
        profile = row['runtime_profile']
        name, backend = row['id'], row['backend']
        horizon = row['selection_review']['native_episode_controls_for_costing']
        observation_shape = profile.get('observation_shape', [128 if name.startswith('Channel') else 64])
        action_shape = profile.get('action_shape', [24 if name.startswith('Channel') else 4])
        if blind_roster is not None:
            if name not in blind_roster['cases']:
                raise InfrastructureError(f'Sealed blind seed roster missing case: {name}')
            case_seeds = list(blind_roster['cases'][name])
            init_note = (f'Native reset and supplied source initial states; {len(case_seeds)} sealed '
                         'blind perturbation seeds (masked from policy.reset).')
        else:
            case_seeds = [123]
            init_note = 'Native reset and supplied source initial states; seed 123. No independent-flow-sample claim.'
        case = dict(id=name, environment=name, family=row['family'], backend=backend,
            observation_size=math.prod(observation_shape), action_size=math.prod(action_shape),
            observation_shape=observation_shape, action_shape=action_shape,
            horizon=horizon, seeds=case_seeds, solver_processes=row['required_solver_gpus'],
            timeout_sec=max(1800, math.ceil(900 + 2 * len(case_seeds) * (evidence[name]['startup_seconds']
                + horizon * evidence[name]['seconds_per_control']))),
            runtime_profile=profile, runtime_profile_sha256=digest(profile),
            initialization=init_note,
            provenance='Organizer native replay profile adopted unchanged from the approved source/cost specification; not paper certification.',
            episode_end='native', strict_horizon=True, primary_metric=definition(name))
        if backend == 'maia':
            case['env_kwargs'] = {'probe_locations': profile['probe_locations'],
                                  'obs_normalization_strategy': 'none'}
            # Verify these defaults at runtime; do not silently override source physics.
            case['native_settings'] = {}
        cases.append(case)
    return {'profile_version': PROFILE_VERSION, 'status': 'executable-not-benchmark-certified',
        'metric_version': METRIC_VERSION, 'metrics_sha256': registry_hash(),
        'data_release_id': release['release_id'], 'data_release_sha256': digest(release),
        'source_revision': release['source']['revision'], 'cases': cases}


# The scored verifier suite: every 2D case, and one 3D case per family (cube included). Side by side on
# 8 H100s it takes about 2.3 h, set by Cube_3D_Re3700 (3.9 h when other jobs share the GPUs). Every family, dimension and solver
# backend stays scored; the dropped 3D instances repeat a family at 2.0-5.2 H100-hours each. The complete
# 39-case suite stays available for development replays (`--case all`).
VERIFIER_CASES = (
    'Cylinder_2D_Re200', 'Cylinder_2D_Re1000', 'Cylinder_2D_Re3900',
    'Nozzle2D_coarse', 'Nozzle2D_fine', 'Kolmogorov_2D',
    'NACA0012_2D_Re100_AOA20', 'NACA0012_2D_Re100_AOA40', 'NACA0012_2D_Re1000_AOA20', 'NACA0012_2D_Re1000_AOA40',
    'NACA0012Gust_2D_Re100_AOA20', 'NACA0012Gust_2D_Re100_AOA40',
    'NACA0012Gust_2D_Re1000_AOA20', 'NACA0012Gust_2D_Re1000_AOA40',
    'Pinball_2D_Re30', 'Pinball_2D_Re75', 'Pinball_2D_Re100', 'Pinball_2D_Re150',
    'SquareCylinder_2D_Re200', 'SquareCylinder_2D_Re1000',
    'Channel_3D_Retau180', 'Pinball_3D_Re100', 'NACA0012Gust_3D_Re100_AOA20', 'NACA0012Gust_3D_Re1000_AOA20',
    'Sphere_3D_Re300', 'Cube_3D_Re3700',
)


def estimated_seconds(case_id):
    """Single-replay cost from the frozen cost evidence (startup + controls x seconds per control)."""
    evidence = {c['case']: c for c in json.loads((DATA / 'cost-evidence-coverage-v1.json').read_text())['cases']}
    row = {c['id']: c for c in load_release()['cases']}[case_id]
    horizon = row['selection_review']['native_episode_controls_for_costing']
    return evidence[case_id]['startup_seconds'] + horizon * evidence[case_id]['seconds_per_control']


def select_cases(names, blind_seeds_path=None):
    spec = specifications(blind_seeds_path=blind_seeds_path)
    indexed = {c['id']: c for c in spec['cases']}
    if names == ['all']:
        names = list(indexed)
    elif names == ['verifier']:
        if set(VERIFIER_CASES) - indexed.keys():
            raise InfrastructureError('Verifier suite names a case outside the frozen release')
        names = [n for n in indexed if n in VERIFIER_CASES]
    if not names or len(set(names)) != len(names) or set(names) - indexed.keys():
        raise InfrastructureError('Select unique frozen case IDs, or --case all')
    return spec, [indexed[name] for name in names]


def verify_resources(data, cases):
    data = data.resolve()
    release = load_release()
    if not data.is_dir() or json.loads((data / 'release.json').read_text()) != release:
        raise InfrastructureError('Mounted resources differ from the frozen release')
    receipt = json.loads((data / 'materialized.json').read_text())
    index = json.loads((DATA / 'source-files.json').read_text())
    names = {c['environment'] for c in cases}
    if (receipt.get('data_release_id') != release['release_id'] or
            receipt.get('data_release_sha256') != digest(release) or
            receipt.get('source_index_sha256') != digest(index) or
            receipt.get('revision') != release['source']['revision'] or
            not names.issubset(receipt.get('verified_directories', []))):
        raise InfrastructureError('Missing verified materialization identity or selected case')
    # Large inputs were fully hashed by materialization and must be mounted read-only.
    # Recheck all sizes and small configuration hashes on each independent replay.
    for item in index['files']:
        if item['rfilename'].split('/')[0] not in names:
            continue
        path = data / 'environments' / item['rfilename']
        if path.resolve() != path or not path.is_file() or path.stat().st_size != item['size']:
            raise InfrastructureError(f'Missing/truncated/redirected resource: {path}')
        if item['size'] <= 1024 * 1024:
            lfs = item.get('lfs')
            if file_digest(path, 'sha256' if lfs else 'sha1') != (lfs['sha256'] if lfs else item['blobId']):
                raise InfrastructureError(f'Changed source configuration: {path}')
    for case in cases:
        if case['environment'] == 'Channel_3D_Retau180':
            adapter = Path(__file__).with_name('native_channel.py')
            if hashlib.sha256(adapter.read_bytes()).hexdigest() != case['runtime_profile']['adapter_sha256']:
                raise InfrastructureError('Frozen Channel actuator-clock adapter changed')


def local_only(data, case, scratch):
    import os
    import hydrogym.data_manager as manager
    manager.HF_AVAILABLE = False
    expected = data / 'environments' / case['environment']
    for kind in ('maiagym', 'jaxfluidsgym'):
        cached = Path.home() / '.cache' / kind / case['environment']
        if cached.exists() and cached.resolve() != expected.resolve():
            raise InfrastructureError(f'Legacy cache shadows pinned source: {cached}')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_CACHE'] = str(scratch / 'unused-hub-cache')
