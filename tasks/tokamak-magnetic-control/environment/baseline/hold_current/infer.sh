#!/usr/bin/env bash
# Trivial reference: hold each coil current at its observed value (V = R * I).
# Protocol conformance check only; not a trained candidate.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
exec python3 policy.py
