#!/usr/bin/env python3
"""Reproducible HMAC-SHA256 generator for the sealed tokamak perturbation suite (7 cases x 15 seeds = 105 episodes).

Usage:
    SSI_TOKAMAK_SEALED_KEY="<secret-hex-or-passphrase>" python tests/generate_sealed_perturbations.py \
        --output tests/sealed/perturbations.json

Never commit or embed the secret key inside development-visible assets. Given a private key in
``SSI_TOKAMAK_SEALED_KEY`` (or ``--key-file``), this script deterministically derives 15 distinct,
non-consecutive 31-bit integer seeds with zero overlap with development seeds (0..10000), public
policy reset seeds (100000..100999), or legacy v0.3 seeds (1001..1015), preserving the frozen
FAIR-MAST-grounded randomization physics parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
LEGACY_FIXTURE = HERE / "fixtures" / "legacy_perturbations_1001_1015.json"
DEFAULT_OUTPUT = HERE / "sealed" / "perturbations.json"

N_CASES = 7
N_SEEDS = 15
EXPECTED_EPISODES = N_CASES * N_SEEDS
MIN_SEED = 1_000_000
MAX_SEED_EXCLUSIVE = 1 << 31  # positive signed 32-bit int (31-bit unsigned)
MIN_PAIRWISE_GAP = 5_000_000
FORBIDDEN_SEEDS = frozenset(range(0, 10_001)) | frozenset(range(100_000, 101_000))


def derive_sealed_seeds(secret_key: bytes, count: int = N_SEEDS) -> list[int]:
    """Derive ``count`` non-consecutive 31-bit positive integer seeds via HMAC-SHA256."""
    if not isinstance(secret_key, (bytes, bytearray)) or len(secret_key) < 16:
        raise ValueError("secret_key must be at least 16 bytes of entropy")
    if count != N_SEEDS:
        raise ValueError(f"Tokamak V4 protocol requires exactly {N_SEEDS} seeds (got {count})")

    seeds: list[int] = []
    counter = 0
    while len(seeds) < count:
        msg = f"ssi-v4:tokamak-magnetic-control:sealed-seed:v1:{counter}".encode("utf-8")
        digest = hmac.new(bytes(secret_key), msg, hashlib.sha256).digest()
        counter += 1
        candidate = struct.unpack(">I", digest[:4])[0] & 0x7FFFFFFF
        if candidate < MIN_SEED or candidate >= MAX_SEED_EXCLUSIVE:
            continue
        if candidate in FORBIDDEN_SEEDS:
            continue
        if any(abs(candidate - prev) < MIN_PAIRWISE_GAP for prev in seeds):
            continue
        seeds.append(int(candidate))
    return seeds


def build_sealed_payload(secret_key: bytes, template_path: Path = LEGACY_FIXTURE) -> dict:
    template = json.loads(Path(template_path).read_text())
    seeds = derive_sealed_seeds(secret_key, count=N_SEEDS)
    if len(set(seeds)) != N_SEEDS:
        raise AssertionError("Derived seeds are not unique")
    return {
        "name": "sealed-v0.4",
        "note": (
            "Wider than the visible training randomization and with plant mechanisms the visible plant lacks; "
            "15 blind 31-bit perturbation seeds (7 cases x 15 seeds = 105 scheduled episodes) unseen in development. "
            "Lives only in the verifier image / private evaluation mount. Data-grounded values: FAIR-MAST "
            "double-null flat-tops (environment/data/fair_mast/derived_ranges.json, CC BY-SA 4.0, UKAEA), "
            "5-95% ranges and p75 sensor residuals, mapped through environment/data/fair_mast/profile_calibration.json; "
            "drifts span the within-shot interquartile range over one 0.35 s episode. Supply, flux-loop drift, "
            "passive-filament and profile-constraint values match the frozen v0.3/v0.4 physics specification."
        ),
        "seeds": seeds,
        "randomization": template["randomization"],
    }


def _load_key(key_file: Path | None) -> bytes:
    if key_file is not None:
        raw = key_file.read_bytes().strip()
    else:
        env_val = os.environ.get("SSI_TOKAMAK_SEALED_KEY", "").strip()
        raw = env_val.encode("utf-8")
    if len(raw) < 16:
        raise SystemExit(
            "Error: provide at least 16 bytes of secret key material via "
            "SSI_TOKAMAK_SEALED_KEY or --key-file."
        )
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--key-file", type=Path, default=None)
    parser.add_argument("--template", type=Path, default=LEGACY_FIXTURE)
    args = parser.parse_args()

    secret_key = _load_key(args.key_file)
    payload = build_sealed_payload(secret_key, template_path=args.template)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"Wrote {args.output} ({payload['name']}): {len(payload['seeds'])} seeds -> "
        f"{N_CASES * len(payload['seeds'])} scheduled episodes"
    )


if __name__ == "__main__":
    main()
