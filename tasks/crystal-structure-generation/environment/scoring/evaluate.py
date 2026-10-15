"""Shared unconditional CrystalGen evaluation orchestration."""

import json
import math
from pathlib import Path
import tempfile
import time

from .contract import load_output, load_requests
from .upstream import normalize_result, run_lemat
from .relaxation import relax_records


def harbor_reward(values):
    """Harbor parses reward.json as a flat object of finite numbers; strings, None and nested details
    belong in report.json / status.json."""
    import math as _math
    return {k: v for k, v in values.items() if isinstance(v, (int, float)) and _math.isfinite(v)}


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def evaluate(
    input_root: Path,
    generated_root: Path,
    scorer_root: Path,
    scorer_python: Path,
    fixture_results: Path | None = None,
    artifact_root: Path | None = None,
    skip_relaxation: bool = False,
) -> tuple[dict, dict]:
    requests = load_requests(input_root)
    records, output_errors = load_output(generated_root, requests)
    fixture = json.loads(fixture_results.read_text()) if fixture_results else None
    rewards = {}
    report = {"schema": "crystalgen-report/v2", "output_errors": output_errors, "tracks": {}}
    with tempfile.TemporaryDirectory(prefix="crystalgen-score-") as directory:
        work = Path(directory)
        if fixture is not None:
            report["relaxation"] = {"status": "fixture_not_run"}
        elif skip_relaxation:
            report["relaxation"] = {"status": "skipped_development_diagnostic"}
        else:
            records, report["relaxation"] = relax_records(
                records, sum(request["n_samples"] for request in requests.values()),
                (artifact_root or work) / "relaxation", scorer_python,
            )
        scoring_started = time.monotonic()
        for mode in ("unconditional",):
            selected = {key for key, request in requests.items() if request["mode"] == mode}
            expected = sum(requests[key]["n_samples"] for key in selected)
            if not selected:
                continue
            if fixture is not None:
                rates, details = normalize_result(fixture[mode], expected)
            else:
                rates, details = run_lemat(
                    records, selected, expected, work / mode, scorer_root, scorer_python
                )
            for name, value in rates.items():
                rewards[f"{mode}_{name}"] = value
            report["tracks"][mode] = details
        # TASK_STANDARD.md reward: the headline SUN + MetaSUN fraction, already in [0, 1].
        rewards["reward"] = rewards.get("unconditional_sun_metasun", 0.0)
        report["scoring_seconds"] = time.monotonic() - scoring_started
    return clean_json(rewards), clean_json(report)


def write_results(output_root: Path, rewards: dict, report: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "reward.json").write_text(json.dumps(harbor_reward(rewards), indent=2, sort_keys=True) + "\n")
    (output_root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
