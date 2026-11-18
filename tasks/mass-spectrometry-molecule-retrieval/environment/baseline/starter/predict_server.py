#!/usr/bin/env python3
"""Persistent single-spectrum streaming server for the Fourier-DeepSets starter.

Implements `massspecgym-retrieval-stream/v2`: loads the declared model once and
processes one query spectrum + its candidate SMILES pool per request over
line-delimited JSON (stdin/stdout by default, or a Unix domain socket via
`--socket`), emitting strictly `{"query_id": ..., "ranking": [...]}` with
deterministic content-derived tie-breaking and zero cross-query state.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys


def _prepare_stdio():
    """Reserve original FD 1 exclusively for protocol JSON lines; route stdout to stderr."""
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(protocol_fd, "w", buffering=1, encoding="utf-8")


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("manifest path escapes package")
    return path


def load_starter_model(package: Path, device_name: str | None = None):
    import torch

    from common import sha256
    from model import DeepSetsFingerprint

    manifest = json.loads((package / "model_manifest.json").read_text(encoding="utf-8"))
    checkpoint_path = contained(package, manifest["checkpoint"]["path"])
    if sha256(checkpoint_path) != manifest["checkpoint"]["sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["model_config"]
    declared_config = json.loads(
        contained(package, manifest["model_config"]).read_text(encoding="utf-8")
    )
    if declared_config != config:
        raise ValueError("checkpoint and declared model config disagree")
    model = DeepSetsFingerprint(
        hidden_size=config["hidden_size"],
        fingerprint_size=config["fingerprint_size"],
        layers_per_mlp=config["layers_per_mlp"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    resolved_device = device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(resolved_device)
    model.to(device).eval()
    return model, config, device


def rank_single_query(row: dict, model, config: dict, device) -> dict:
    """Statelessly rank one spectrum's candidate pool with deterministic tie-breaking."""
    import torch
    import torch.nn.functional as F

    from common import molecule_id, morgan_fingerprint, spectrum_tensor

    if not isinstance(row, dict):
        raise ValueError("query request must be a JSON object")
    query_id = row.get("query_id")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("missing or invalid query_id")
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 256:
        raise ValueError(f"invalid candidate pool for {query_id}")

    validated = []
    seen_ids = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {"candidate_id", "smiles"}:
            raise ValueError("invalid candidate fields")
        cid = candidate["candidate_id"]
        smiles = candidate["smiles"]
        if not isinstance(cid, str) or not isinstance(smiles, str) or cid != molecule_id(smiles):
            raise ValueError("candidate ID does not match SMILES")
        if cid in seen_ids:
            raise ValueError("duplicate candidate ID")
        seen_ids.add(cid)
        validated.append((cid, smiles))

    # Canonicalize candidate ordering before tensor stacking so GEMV/GEMM kernels
    # and tie-breaking are 100% invariant to caller candidate permutation.
    validated.sort(key=lambda item: item[0])

    with torch.no_grad():
        spectrum = spectrum_tensor(
            row,
            max_peaks=config["max_peaks"],
            mz_min=config["mz_min"],
            mz_max=config["mz_max"],
        ).unsqueeze(0).to(device)
        prediction = F.normalize(model(spectrum), dim=-1).squeeze(0)
        candidate_fps = torch.stack([
            morgan_fingerprint(
                smiles,
                size=config["fingerprint_size"],
                radius=config["fingerprint_radius"],
                allow_unparseable=True,
            )
            for _, smiles in validated
        ]).to(device)
        candidate_fps = F.normalize(candidate_fps, dim=-1)
        scores = (candidate_fps @ prediction).cpu().tolist()

    order = sorted(
        range(len(validated)),
        key=lambda index: (-scores[index], validated[index][0]),
    )
    return {
        "query_id": query_id,
        "ranking": [validated[index][0] for index in order],
    }


def serve_stream(in_stream, out_stream, model, config: dict, device) -> None:
    from common import compact_json

    for line_number, raw_line in enumerate(in_stream, 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON request at line {line_number}") from error
        response = rank_single_query(request, model, config, device)
        out_stream.write(compact_json(response) + "\n")
        out_stream.flush()


def serve_unix_socket(socket_path: Path, model, config: dict, device) -> None:
    if socket_path.exists():
        socket_path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(socket_path))
        server.listen(1)
        conn, _ = server.accept()
        with conn:
            reader = conn.makefile("r", encoding="utf-8")
            writer = conn.makefile("w", encoding="utf-8", buffering=1)
            serve_stream(reader, writer, model, config, device)
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--socket",
        type=Path,
        default=None,
        help="Optional Unix domain socket path; defaults to stdin/stdout JSONL.",
    )
    args = parser.parse_args()

    protocol_out = _prepare_stdio() if args.socket is None else None
    try:
        package = Path(__file__).resolve().parent
        model, config, device = load_starter_model(package, args.device)
        if args.socket is not None:
            serve_unix_socket(args.socket, model, config, device)
        else:
            serve_stream(sys.stdin, protocol_out, model, config, device)
    finally:
        if protocol_out is not None:
            protocol_out.close()


if __name__ == "__main__":
    main()
