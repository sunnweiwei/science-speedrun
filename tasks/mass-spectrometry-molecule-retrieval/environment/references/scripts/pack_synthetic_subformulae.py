#!/usr/bin/env python3
"""Stream the synthetic tarball into an inode-efficient SQLite store."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tarfile
import time
from pathlib import Path


EXPECTED_ROWS = 2_333_510


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    labels_path = args.output_root / "labels.csv"
    database_path = args.output_root / "subformulae.sqlite"
    started = time.time()

    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA locking_mode=EXCLUSIVE;
        PRAGMA page_size=65536;
        CREATE TABLE subformulae (
          spec TEXT PRIMARY KEY,
          payload BLOB NOT NULL
        ) WITHOUT ROWID;
        """
    )
    labels_found = False
    label_lines = 0
    batch: list[tuple[str, bytes]] = []
    inserted = 0
    with tarfile.open(args.archive, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read tar member {member.name}")
            if member.name.endswith("/labels.csv"):
                labels_found = True
                with labels_path.open("wb") as target:
                    for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
                        label_lines += block.count(b"\n")
                        target.write(block)
                continue
            if "/subformulae/" not in member.name or not member.name.endswith(".json"):
                continue
            batch.append((Path(member.name).stem, source.read()))
            if len(batch) >= args.batch_size:
                connection.executemany(
                    "INSERT INTO subformulae(spec, payload) VALUES (?, ?)", batch
                )
                connection.commit()
                inserted += len(batch)
                batch.clear()
                if inserted % 100_000 == 0:
                    print(
                        json.dumps(
                            {
                                "subformulae": inserted,
                                "elapsed_seconds": time.time() - started,
                            }
                        ),
                        flush=True,
                    )
    if batch:
        connection.executemany(
            "INSERT INTO subformulae(spec, payload) VALUES (?, ?)", batch
        )
        connection.commit()
        inserted += len(batch)
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    connection.close()

    if not labels_found:
        raise ValueError("labels.csv was not found in the synthetic archive")
    label_rows = label_lines - 1
    if label_rows != args.expected_rows or inserted != args.expected_rows or integrity != "ok":
        raise ValueError(
            f"invalid packed corpus: labels={label_rows}, payloads={inserted}, "
            f"integrity={integrity!r}"
        )
    report = {
        "schema": "massspecgym-retrieval-packed-synthetic/v1",
        "archive": str(args.archive),
        "archive_md5": file_md5(args.archive),
        "labels": str(labels_path),
        "database": str(database_path),
        "rows": inserted,
        "database_bytes": database_path.stat().st_size,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_root / "packing_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
