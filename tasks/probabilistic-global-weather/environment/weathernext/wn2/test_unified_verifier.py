"""Acceptance gates for the unified paper-subset entry point."""
import copy
import json
from pathlib import Path

import pytest
import xarray as xr

from verify_track import primary_reward


def fixture():
    protocol = dict(evaluation_initializations=['2022-01-01', '2022-02-01'],
                    variables={'temperature': {'pressure_levels': True}},
                    pressure_levels_hpa=[500, 850], lead_hours=[12, 24], members=56)
    reports = [dict(status='passed', members=56, per_case=[
        dict(case=date, variable='temperature', level_hpa=level, lead_hours=lead, crps=2.)
        for level in [500, 850] for lead in [12, 24]])
        for date in protocol['evaluation_initializations']]
    scales = xr.Dataset({'temperature': ('level', [4., 4.])}, coords={'level': [500, 850]})
    return reports, protocol, scales


def test_primary_reward_requires_every_date_level_and_lead():
    reports, protocol, scales = fixture()
    assert primary_reward(reports, protocol, scales)['reward'] == pytest.approx(2 / 3)
    for index in range(4):
        missing = copy.deepcopy(reports)
        missing[1]['per_case'].pop(index)
        with pytest.raises(ValueError, match='Missing primary panels'):
            primary_reward(missing, protocol, scales)


def test_primary_reward_rejects_duplicates_and_incomplete_ensemble():
    reports, protocol, scales = fixture()
    duplicate = copy.deepcopy(reports)
    duplicate[0]['per_case'].append(duplicate[0]['per_case'][0])
    with pytest.raises(ValueError, match='Duplicate'):
        primary_reward(duplicate, protocol, scales)
    reports[0]['members'] = 8
    with pytest.raises(ValueError, match='ensemble'):
        primary_reward(reports, protocol, scales)


def test_protocol_has_one_canonical_copy_and_complete_paper_horizon():
    root = Path(__file__).resolve().parents[1]
    for name in ['weather', 'precipitation']:
        candidate = root / 'paper-comparison' / f'{name}-protocol.json'
        canonical = root / 'wn2' / ('protocol.json' if name == 'weather' else 'precipitation-protocol.json')
        assert candidate.resolve() == canonical.resolve()
    protocol = json.loads((root / 'wn2/protocol.json').read_text())
    assert len(protocol['evaluation_initializations']) == 8
    assert protocol['members'] == 56
    assert protocol['lead_hours'] == list(range(12, 361, 12))


def test_untrusted_store_root_cannot_redirect_verifier_reads(tmp_path):
    from zarr_safety import audit_zarr
    outside = tmp_path / 'outside'
    outside.mkdir()
    store = tmp_path / 'forecast.zarr'
    store.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='ordinary directory'):
        audit_zarr(store)


def make_case(output, index=0):
    from verify_track import ENDPOINTS
    reports, protocol, scales = fixture()
    for endpoint in ENDPOINTS:
        path = output / endpoint / f'case-{index:03d}.json'
        path.parent.mkdir(exist_ok=True)
        report = reports[index] if endpoint == 'scores' else {'status': 'passed'}
        path.write_text(json.dumps(dict(report, protocol_sha256='protocol')))
    predictions = output / 'predictions'
    predictions.mkdir(exist_ok=True)
    for member in range(protocol['members']):
        stem = f'case-{index:03d}-member-{member:03d}'
        (predictions / (stem + '.zarr')).mkdir()
        (predictions / (stem + '.complete.json')).write_text('{}')
    replay = output / f'replay-{index:03d}'
    replay.mkdir()
    (replay / 'relocated-model').mkdir()
    (replay / 'untrusted-output').mkdir()
    (replay / 'replay.json').write_text('{}')
    return protocol, scales


@pytest.mark.parametrize('endpoint', ['scores', 'pooling-scores', 'spectra', 'cyclone-scores'])
def test_failed_endpoint_preserves_forecasts(tmp_path, endpoint):
    from verify_track import finish_case
    protocol, scales = make_case(tmp_path)
    path = tmp_path / endpoint / 'case-000.json'
    report = json.loads(path.read_text())
    report['status'] = 'failed'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Incomplete suite'):
        finish_case(tmp_path, 0, protocol, 'protocol', scales, False)
    assert len(list((tmp_path / 'predictions').glob('*.zarr'))) == 56
    assert (tmp_path / 'replay-000/relocated-model').is_dir()


def test_missing_panel_prevents_cleanup(tmp_path):
    from verify_track import finish_case
    protocol, scales = make_case(tmp_path)
    path = tmp_path / 'scores/case-000.json'
    report = json.loads(path.read_text())
    report['per_case'].pop()
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Missing primary panels'):
        finish_case(tmp_path, 0, protocol, 'protocol', scales, False)
    assert len(list((tmp_path / 'predictions').glob('*.zarr'))) == 56


@pytest.mark.parametrize('keep', [False, True])
def test_cleanup_preserves_reward_receipts_and_other_cases(tmp_path, keep):
    from verify_track import finish_case
    protocol, scales = make_case(tmp_path, 0)
    make_case(tmp_path, 1)
    paths = sorted((tmp_path / 'scores').glob('*.json'))
    before = primary_reward([json.loads(p.read_text()) for p in paths], protocol, scales)
    finish_case(tmp_path, 0, protocol, 'protocol', scales, keep)
    assert primary_reward([json.loads(p.read_text()) for p in paths], protocol, scales) == before
    assert len(list((tmp_path / 'predictions').glob('case-000-*.zarr'))) == (56 if keep else 0)
    assert len(list((tmp_path / 'predictions').glob('case-001-*.zarr'))) == 56
    assert len(list((tmp_path / 'predictions').glob('*.complete.json'))) == 112
    assert (tmp_path / 'replay-000/replay.json').is_file()
    assert (tmp_path / 'replay-000/cleanup.json').exists() != keep


def test_tracker_cleanup_requires_matching_receipts_and_keeps_tracks(tmp_path):
    from verify_track import discard_tracker_input
    (tmp_path / 'tracking').mkdir()
    (tmp_path / 'input.nc').write_bytes(b'large input')
    (tmp_path / 'input.json').write_text(json.dumps(dict(status='passed', sha256='expected')))
    receipt = tmp_path / 'tracking/complete.json'
    receipt.write_text(json.dumps(dict(status='passed', input_sha256='wrong')))
    tracks = tmp_path / 'tracking/tracks.csv'
    tracks.write_text('track data')
    with pytest.raises(ValueError, match='unverified'):
        discard_tracker_input(tmp_path)
    assert (tmp_path / 'input.nc').exists()
    receipt.write_text(json.dumps(dict(status='passed', input_sha256='expected')))
    discard_tracker_input(tmp_path)
    assert not (tmp_path / 'input.nc').exists()
    assert tracks.read_text() == 'track data'
    assert receipt.exists() and (tmp_path / 'input.json').exists()


def test_pipeline_applies_backpressure_until_scoring_finishes():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from verify_track import bounded_case_pipeline
    lock = threading.Lock()
    two_resident = threading.Event()
    release = threading.Event()
    resident, seen = set(), []
    def infer(index, device):
        with lock:
            resident.add(index)
            seen.append(index)
            assert len(resident) <= 2
            if len(resident) == 2:
                two_resident.set()
    def score(index):
        assert release.wait(5)
        with lock:
            resident.remove(index)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(bounded_case_pipeline, 8, ['0', '1', '2', '3'], 2, 1, infer, score)
        try:
            assert two_resident.wait(5)
            with lock:
                assert sorted(seen) == [0, 1]
        finally:
            release.set()
        future.result(timeout=5)
    assert sorted(seen) == list(range(8)) and not resident


def test_pipeline_stops_after_failed_case():
    from verify_track import bounded_case_pipeline
    seen = []
    def fail(index):
        raise RuntimeError('scoring failed')
    with pytest.raises(RuntimeError, match='scoring failed'):
        bounded_case_pipeline(8, ['0'], 1, 1, lambda i, d: seen.append(i), fail)
    assert seen == [0]


def test_split_forecast_requires_retention(monkeypatch, tmp_path):
    import sys
    from verify_track import main
    monkeypatch.setattr(sys, 'argv', ['verify_track', '--package', str(tmp_path),
        '--benchmark', str(tmp_path), '--output', str(tmp_path / 'out'), '--phase', 'forecast'])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert not (tmp_path / 'out').exists()


def test_cleanup_rejects_symlink_before_deleting_any_case_data(tmp_path):
    from verify_track import discard_case_forecasts
    make_case(tmp_path)
    path = tmp_path / 'predictions/case-000-member-055.zarr'
    path.rmdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'keep').write_text('untouched')
    path.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='ordinary output directory'):
        discard_case_forecasts(tmp_path, 0, 56)
    assert (tmp_path / 'predictions/case-000-member-000.zarr').exists()
    assert (outside / 'keep').read_text() == 'untouched'


def test_score_phase_rejects_cleaned_run(monkeypatch, tmp_path):
    import sys
    import verify_track
    from training_data import sha256
    protocol = Path(verify_track.__file__).with_name('protocol.json')
    p = json.loads(protocol.read_text())
    package = tmp_path / 'package'
    package.mkdir()
    (package / 'package.json').write_text('{}')
    benchmark = tmp_path / 'benchmark'
    (benchmark / 'tracker/truth').mkdir(parents=True)
    (benchmark / 'tracker/named-truth.json').write_text('{}')
    _, _, scales = fixture()
    scales.to_netcdf(benchmark / 'training-scales.nc')
    output = tmp_path / 'out'
    output.mkdir()
    hashes = {}
    for index in range(len(p['evaluation_initializations'])):
        replay = output / f'replay-{index:03d}'
        replay.mkdir()
        (replay / 'replay.json').write_text('{}')
        hashes[str(index)] = sha256(replay / 'replay.json')
    (output / 'forecast-status.json').write_text(json.dumps(dict(status='passed',
        predictions_retained=False, protocol_sha256=sha256(protocol),
        package_sha256=sha256(package / 'package.json'), replay_receipts=hashes)))
    monkeypatch.setattr(sys, 'argv', ['verify_track', '--package', str(package),
        '--benchmark', str(benchmark), '--output', str(output), '--phase', 'score', '--legacy-case-mode'])
    with pytest.raises(ValueError, match='Predictions were cleaned'):
        verify_track.main()
    assert json.loads((output / 'status.json').read_text())['status'] == 'failed'
    assert json.loads((output / 'reward.json').read_text())['reward'] == 0
