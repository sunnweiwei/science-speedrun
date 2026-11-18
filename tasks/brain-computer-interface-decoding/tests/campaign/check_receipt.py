"""Validate the verifier's Harbor reward receipt against its detailed report."""
import json
import math
from pathlib import Path

SUMMARY_KEYS = frozenset({'score', 'reward', 'signed_score'})
FULL_TRACKS = frozenset({'tuev', 'ad65', 'openbmi', 'longitudinal'})


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
    """Require exactly the expected tracks in both producer files (plus composite keys on full 4-track runs)."""
    directory = Path(directory)
    reward = json.loads((directory / 'reward.json').read_text())
    report = json.loads((directory / 'report.json').read_text())
    track_keys = set(reward) - SUMMARY_KEYS
    if track_keys != set(tracks) or set(report.get('tracks', {})) != set(tracks):
        raise ValueError('unexpected or missing track in verifier receipt')
    if report.get('scope') == 'full' and set(tracks) == FULL_TRACKS and 'score' in reward:
        agg = report.get('aggregate') or {}
        if (not isinstance(reward.get('score'), (int, float)) or isinstance(reward.get('score'), bool)
                or not math.isfinite(reward['score'])
                or not math.isclose(reward['score'], reward.get('reward', float('nan')), abs_tol=1e-12)
                or (agg.get('score') is not None
                    and not math.isclose(reward['score'], agg['score'], abs_tol=1e-12))):
            raise ValueError('composite V3/V4 score mismatch between reward.json and report.json')
    return {track: checked_reward(directory, track) for track in tracks}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--tracks', nargs='+', required=True)
    args = parser.parse_args()
    print(json.dumps(checked_rewards(args.directory, args.tracks), allow_nan=False))
