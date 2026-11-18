"""Root controller; inference runs per-subject/per-session as an unprivileged UID without label access."""
from collections import defaultdict
from functools import lru_cache
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import numpy as np

local_evaluation = Path(__file__).resolve().parents[1] / 'environment/evaluation'
sys.path.insert(0, str(local_evaluation if local_evaluation.exists() else Path('/opt/bci/evaluation')))
from bci_score import (
    AGGREGATE_TRACKS,
    CLASSES,
    DEPRECATED_TRACKS,
    aggregate,
    composite_score,
    read_predictions,
    score,
)
from data_identity import split_identity

SUMMARY_REWARD_KEYS = frozenset({'score', 'reward', 'signed_score'})
AUTHORIZED_PRETRAINED_SHA256 = {
    'cbramod': 'a939ace9aa1e229f08391ad8bb2d197b507ae2c519a50addf087f0151b2df5c3',
    'codebrain': 'eeba68250b8ea61efbf3e91cf4f90242faf90dcee8eba4d1b8a6e50907a46d18',
}
DISALLOWED_PRETRAINED_FAMILIES = frozenset({'biot'})
DISALLOWED_PRETRAINED_SHA256 = {
    'a218f4054ae29962c6fd9d23d94c446d354c4f53e805a6771c85749ccc20baaa': (
        'biot (disallowed in V4 due to direct TUEV pretraining corpus overlap)'
    ),
}
WEIGHT_SUFFIXES = frozenset({'.safetensors', '.pt', '.pth', '.bin', '.ckpt'})


def terminate_users():
    subprocess.run(
        ['pkill', '-KILL', '-u', '10001'],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def package_hash(source: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(source.rglob('*')):
        if p.is_symlink():
            raise ValueError('package symlink is not allowed')
        if p.is_file():
            h.update(str(p.relative_to(source)).encode())
            h.update(b'\0')
            with p.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
    return h.hexdigest()


def validate_pretrained_policy(source: Path) -> None:
    """Reject submissions using BIOT or unapproved external pretrained weights."""
    allowed_hashes = set(AUTHORIZED_PRETRAINED_SHA256.values())
    for meta_name in ('config.json', 'training.json'):
        meta_path = source / meta_name
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        if not isinstance(meta, dict):
            raise ValueError(f'{meta_name} must be a JSON object')
        for field in ('family', 'model'):
            val = str(meta.get(field, '')).strip().lower()
            if val in DISALLOWED_PRETRAINED_FAMILIES:
                raise ValueError(
                    f'disallowed model/pretrained family {val!r} in {meta_name}: '
                    f'BIOT is prohibited due to TUEV pretraining corpus overlap'
                )
        init = meta.get('initialization')
        if isinstance(init, dict):
            repo = str(init.get('repo', '')).lower()
            sha = str(init.get('sha256', '')).lower()
            if 'biot' in repo or sha in DISALLOWED_PRETRAINED_SHA256:
                raise ValueError('disallowed BIOT pretrained checkpoint declared in metadata')
            if init.get('initialization') == 'pretrained' and sha not in allowed_hashes:
                raise ValueError(f'unauthorized pretrained checkpoint sha256 in {meta_name}: {sha}')

    for p in sorted(source.rglob('*')):
        if not p.is_file():
            continue
        lower_name = p.name.lower()
        if lower_name == 'bci-biot.safetensors':
            raise ValueError('disallowed bci-biot.safetensors weight file in submission')
        if p.suffix.lower() in WEIGHT_SUFFIXES:
            digest = file_sha256(p)
            if digest in DISALLOWED_PRETRAINED_SHA256:
                raise ValueError(
                    f'disallowed pretrained checkpoint detected ({DISALLOWED_PRETRAINED_SHA256[digest]})'
                )


def isolation_group_key(track: str, row: dict) -> tuple[str, str]:
    """Group evaluation windows by `(subject, recording_session)` so inference cannot whiten across subjects."""
    subject = str(row.get('subject') or '').strip()
    session = str(row.get('session') or '1').strip() or '1'
    if track == 'tuev' and subject in ('', 'official-test', 'unknown'):
        src = str(row.get('source') or '').strip()
        if src:
            stem = Path(src).stem
            rec = stem.rsplit('-', 1)[0]
            parts = rec.split('_')
            subject = parts[0] if parts[0] else rec
            session = parts[1] if len(parts) > 1 and parts[1] else '1'
        else:
            subject = str(row['id'])
    if not subject or subject == 'unknown':
        subject = str(row['id'])
    return (subject, session)


def evaluate(
    track,
    submission,
    private,
    limit,
    timeout,
    *,
    enforce_root: bool = True,
    temp_root: Path | None = None,
):
    if track not in CLASSES:
        raise ValueError(f'unsupported or deprecated BCI track: {track}')
    truth_path = private / track / 'truth.jsonl'
    public = private / track / 'public'
    rows = [json.loads(l) for l in truth_path.read_text().splitlines()]
    if limit:
        # Select complete subjects for grouped metrics, classes for event metrics.
        if track in ('ad65', 'openbmi', 'longitudinal'):
            selected = set()
            for label in sorted({r['label'] for r in rows}):
                selected.add(next(r['subject'] for r in rows if r['label'] == label))
            rows = [r for r in rows if r['subject'] in selected]
        else:
            keep = []
            for label in sorted({r['label'] for r in rows}):
                keep.extend([r for r in rows if r['label'] == label][:limit])
            rows = keep
    if not rows:
        raise RuntimeError('empty verifier data')

    base = Path(tempfile.mkdtemp(prefix='bci-verify-', dir=str(temp_root or Path('/tmp'))))
    os.chmod(base, 0o755)
    pkg = base / 'package'
    manifest = json.loads((public / 'manifest.json').read_text())
    ids = {r['id'] for r in rows}
    public_samples = [r for r in manifest['samples'] if r['id'] in ids]
    if any(set(r) != {'id', 'path', 'index'} for r in public_samples):
        shutil.rmtree(base, ignore_errors=True)
        raise RuntimeError('labels in public manifest')
    if {r['id'] for r in public_samples} != ids:
        shutil.rmtree(base, ignore_errors=True)
        raise RuntimeError('public/truth sample ID mismatch')
    public_by_id = {r['id']: r for r in public_samples}
    for name in {r['path'] for r in public_samples}:
        resolved_shard = (public / name).resolve()
        if not resolved_shard.is_relative_to(public.resolve()) or not resolved_shard.is_file():
            shutil.rmtree(base, ignore_errors=True)
            raise RuntimeError('invalid public path')
    try:
        data_id = split_identity(public, truth_path)
    except OSError as exc:
        shutil.rmtree(base, ignore_errors=True)
        raise RuntimeError(f'missing verifier split identity asset: {exc}') from exc

    @lru_cache(maxsize=64)
    def load_public_shard(name: str):
        return np.load((public / name).resolve(), mmap_mode='r', allow_pickle=False)

    started = time.monotonic()
    try:
        source = submission.resolve()
        if not source.is_dir():
            raise NotADirectoryError('submission directory does not exist')
        for p in source.rglob('*'):
            if p.is_symlink():
                raise ValueError('package symlink is not allowed')
        if not (source / 'infer.sh').is_file():
            raise FileNotFoundError('submission missing infer.sh')
        validate_pretrained_policy(source)
        shutil.copytree(source, pkg)
        for p in [pkg, *pkg.rglob('*')]:
            os.chmod(p, 0o755 if p.is_dir() or p.suffix == '.sh' else 0o644)

        units_map: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in rows:
            units_map[isolation_group_key(track, r)].append(r)
        ordered_keys = sorted(
            units_map,
            key=lambda k: hashlib.sha256(f'{track}|{k[0]}|{k[1]}'.encode()).hexdigest(),
        )

        predictions_by_id: dict[str, np.ndarray] = {}
        for u_idx, u_key in enumerate(ordered_keys):
            unit_rows = sorted(units_map[u_key], key=lambda r: r['id'])
            unit_inp = base / f'unit-{u_idx:04d}-in'
            unit_out = base / f'unit-{u_idx:04d}-out'
            unit_inp.mkdir()
            unit_out.mkdir()
            if enforce_root:
                os.chown(unit_out, 10001, 10001)

            # Materialize a single-unit shard containing ONLY this (subject, session) unit's windows.
            unit_arr = np.stack(
                [load_public_shard(public_by_id[r['id']]['path'])[public_by_id[r['id']]['index']] for r in unit_rows]
            )
            np.save(unit_inp / 'shard-000000.npy', unit_arr, allow_pickle=False)
            unit_manifest = {
                'schema': manifest.get('schema', 'bci-input-v1'),
                'track': track,
                'config': manifest.get('config', {}),
                'samples': [
                    {'id': r['id'], 'path': 'shard-000000.npy', 'index': i}
                    for i, r in enumerate(unit_rows)
                ],
            }
            (unit_inp / 'manifest.json').write_text(json.dumps(unit_manifest))

            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise ValueError('inference timeout')

            if enforce_root:
                cmd = [
                    'setpriv',
                    '--reuid=10001',
                    '--regid=10001',
                    '--clear-groups',
                    '--bounding-set=-all',
                    '--no-new-privs',
                    'bash',
                    str(pkg / 'infer.sh'),
                    str(unit_inp),
                    str(unit_out),
                ]
            else:
                cmd = ['bash', str(pkg / 'infer.sh'), str(unit_inp), str(unit_out)]

            env = {
                'PATH': os.environ['PATH'],
                'HOME': str(unit_out),
                'TMPDIR': str(unit_out),
                'PYTHONDONTWRITEBYTECODE': '1',
                'OMP_NUM_THREADS': '4',
                'OPENBLAS_NUM_THREADS': '4',
            }
            for key in ('LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'NVIDIA_DRIVER_CAPABILITIES'):
                if key in os.environ:
                    env[key] = os.environ[key]

            log_path = base / f'inference-{u_idx:04d}.log'
            with open(log_path, 'wb') as log:
                proc = subprocess.Popen(
                    cmd,
                    cwd=pkg,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    code = proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    raise ValueError('inference timeout')
            if enforce_root:
                terminate_users()
            if code:
                raise ValueError('inference failed: ' + log_path.read_text(errors='replace')[-2000:])

            pred = unit_out / 'predictions.jsonl'
            fd = os.open(pred, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, 'rb') as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > 1024**3:
                    raise ValueError('invalid output file')
                trusted = base / f'trusted-predictions-{u_idx:04d}.jsonl'
                trusted.write_bytes(stream.read())

            unit_probs = read_predictions(trusted, unit_rows, CLASSES[track])
            for r, p_vec in zip(unit_rows, unit_probs):
                predictions_by_id[r['id']] = p_vec

            shutil.rmtree(unit_inp, ignore_errors=True)
            shutil.rmtree(unit_out, ignore_errors=True)
            trusted.unlink(missing_ok=True)
            log_path.unlink(missing_ok=True)

        probs = np.stack([predictions_by_id[r['id']] for r in rows])
        result = score(track, rows, probs)
        result['status'] = 'ok'
        result['seconds'] = time.monotonic() - started
        result['sample_count'] = len(rows)
        result['subject_count'] = len({r.get('subject') for r in rows})
        result['inference_isolation'] = 'per_subject_session'
        result['isolated_inference_units'] = len(ordered_keys)
        result['data_identity'] = data_id
        result['package_sha256'] = package_hash(source)
        return result
    except (ValueError, KeyError, TypeError, FileNotFoundError, NotADirectoryError, PermissionError) as e:
        return {'reward': 0.0, 'status': 'invalid_submission', 'error': str(e), 'seconds': time.monotonic() - started}
    finally:
        load_public_shard.cache_clear()
        if enforce_root:
            terminate_users()
        shutil.rmtree(base, ignore_errors=True)


def write_receipts(output: Path, result: dict) -> None:
    track_rewards = {track: float(detail['reward']) for track, detail in result['tracks'].items()}
    comp = (
        composite_score(track_rewards)
        if result.get('scope') == 'full' and set(AGGREGATE_TRACKS).issubset(track_rewards)
        else None
    )
    if comp is not None:
        reward_payload = {
            'score': comp['score'],
            'reward': comp['reward'],
            'signed_score': comp['signed_score'],
            **track_rewards,
        }
    else:
        reward_payload = dict(track_rewards)
    (output / 'reward.json').write_text(json.dumps(reward_payload, indent=2, allow_nan=False) + '\n')
    (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


def run_verification(
    submission: Path,
    private: Path,
    output: Path,
    tracks: list[str] | tuple[str, ...] = AGGREGATE_TRACKS,
    smoke_limit: int = 0,
    timeout: int = 7200,
    *,
    enforce_root: bool = True,
    temp_root: Path | None = None,
) -> dict:
    if any(t not in CLASSES for t in tracks):
        bad = [t for t in tracks if t not in CLASSES]
        raise ValueError(f'unsupported or deprecated BCI tracks requested: {bad}')
    if enforce_root:
        if os.getuid() != 0:
            raise RuntimeError('verifier controller must be root')
        for path in (Path('/private'), Path('/tests'), output):
            path.mkdir(parents=True, exist_ok=True)
            os.chmod(path, 0o700)
        for p in [private / t / 'truth.jsonl' for t in tracks]:
            if not p.is_file():
                raise RuntimeError('missing infrastructure data ' + str(p))
            r = subprocess.run(
                [
                    'setpriv',
                    '--reuid=10001',
                    '--regid=10001',
                    '--clear-groups',
                    '--bounding-set=-all',
                    '--no-new-privs',
                    'test',
                    '-r',
                    str(p),
                ]
            )
            if r.returncode == 0:
                raise RuntimeError('private labels accessible')
    else:
        output.mkdir(parents=True, exist_ok=True)
        for p in [private / t / 'truth.jsonl' for t in tracks]:
            if not p.is_file():
                raise RuntimeError('missing infrastructure data ' + str(p))

    try:
        root_hash = package_hash(submission.resolve())
    except (OSError, ValueError):
        root_hash = None

    scope = 'smoke' if smoke_limit else 'full'
    result = {
        'schema': 'bci-shared-v4',
        'aggregate': None,
        'scope': scope,
        'inference_isolation': 'per_subject_session',
        'tracks': {},
        'package_sha256': root_hash,
    }
    for track in tracks:
        result['tracks'][track] = evaluate(
            track,
            submission,
            private,
            smoke_limit,
            timeout,
            enforce_root=enforce_root,
            temp_root=temp_root,
        )

    track_rewards = {t: d['reward'] for t, d in result['tracks'].items()}
    comp = composite_score(track_rewards) if scope == 'full' and set(AGGREGATE_TRACKS).issubset(track_rewards) else None
    result['aggregate'] = {
        'score': None if comp is None else comp['score'],
        'reward': None if comp is None else comp['reward'],
        'signed_score': None if comp is None else comp['signed_score'],
        'terms': None if comp is None else comp['terms'],
        'tracks': list(AGGREGATE_TRACKS),
        'deprecated_removed': DEPRECATED_TRACKS,
        'formula': 'clip(mean over (tuev, ad65, openbmi, longitudinal) of (ba_subject_session - chance_k)/(1 - chance_k), 0.0, 1.0)',
    }
    write_receipts(output, result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--submission', type=Path, default=Path('/workspace/submission'))
    ap.add_argument('--private', type=Path, default=Path('/private/data'))
    ap.add_argument('--output', type=Path, default=Path('/logs/verifier'))
    ap.add_argument('--tracks', nargs='+', choices=CLASSES, default=list(AGGREGATE_TRACKS))
    ap.add_argument('--smoke-limit', type=int, default=0)
    ap.add_argument('--timeout', type=int, default=7200)
    a = ap.parse_args()
    result = run_verification(
        a.submission,
        a.private,
        a.output,
        a.tracks,
        a.smoke_limit,
        a.timeout,
        enforce_root=True,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
