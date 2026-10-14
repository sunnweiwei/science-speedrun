import json
from pathlib import Path
import hashlib
import pytest
import wn2_parallel_score

ROOT = Path(__file__).resolve().parents[1]

def setup_run(tmp_path, monkeypatch):
    output = tmp_path / 'output'; output.mkdir()
    package = tmp_path / 'package'; package.mkdir()
    (package / 'package.json').write_text('{}')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    identity = digest(ROOT / 'wn2/protocol.json')
    receipts = {}
    for i in range(8):
        w = output / f'replay-{i:03d}'; w.mkdir()
        (w / 'replay.json').write_text('{}')
        receipts[str(i)] = digest(w / 'replay.json')
    (output / 'forecast-status.json').write_text(json.dumps(dict(status='passed', protocol_sha256=identity,
        package_sha256=digest(package / 'package.json'), replay_receipts=receipts)))
    monkeypatch.setattr('sys.argv', ['score', '--source', str(ROOT), '--benchmark', str(tmp_path),
        '--output', str(output), '--package', str(package), '--endpoint', 'finalize'])
    return output, identity


def test_changed_forecast_receipt_cannot_finalize(tmp_path, monkeypatch):
    output, _ = setup_run(tmp_path, monkeypatch)
    (output / 'replay-000/replay.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='Changed independent replay'):
        wn2_parallel_score.main()
    assert not (output / 'reward.json').exists()


def test_core_scores_alone_cannot_finalize(tmp_path, monkeypatch):
    output, identity = setup_run(tmp_path, monkeypatch)
    (output / 'scores').mkdir()
    for i in range(8):
        (output / 'scores' / f'case-{i:03d}.json').write_text(json.dumps(dict(status='passed', protocol_sha256=identity)))
    with pytest.raises(FileNotFoundError):
        wn2_parallel_score.main()
    assert not (output / 'status.json').exists()
    assert not (output / 'reward.json').exists()
