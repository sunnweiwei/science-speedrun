"""Validate the verifier's flat Harbor reward against its detailed report."""
import json
import math
from pathlib import Path


def checked_reward(directory: Path, track: str) -> float:
    directory = Path(directory)
    reward = json.loads((directory / 'reward.json').read_text())
    report = json.loads((directory / 'report.json').read_text())
    detail = report.get('tracks', {}).get(track, {})
    value = reward.get(track)
    digest = report.get('package_sha256')
    if (detail.get('status') != 'ok' or isinstance(value, bool)
            or not isinstance(value, (int, float)) or not math.isfinite(value)
            or not 0 <= value <= 1
            or report.get('scope') not in ('smoke', 'full')
            or not isinstance(digest, str) or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)
            or detail.get('package_sha256') != digest
            or not math.isclose(value, detail.get('reward', float('nan')), abs_tol=1e-12)):
        raise ValueError(f'{track}: inconsistent or unsuccessful verifier receipt')
    return float(value)


def checked_rewards(directory, tracks):
    """Require exactly the expected tracks in both producer files."""
    directory = Path(directory)
    reward = json.loads((directory / 'reward.json').read_text())
    report = json.loads((directory / 'report.json').read_text())
    if set(reward) != set(tracks) or set(report.get('tracks', {})) != set(tracks):
        raise ValueError('unexpected or missing track in verifier receipt')
    return {track: checked_reward(directory, track) for track in tracks}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--tracks', nargs='+', required=True)
    args = parser.parse_args()
    print(json.dumps(checked_rewards(args.directory, args.tracks), allow_nan=False))
