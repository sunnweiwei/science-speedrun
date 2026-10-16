"""Full-release, hash-priority MPTrj+sAlex selection for the 500k task."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import heapq
import json
from pathlib import Path
import shutil
import tarfile
import time

import ijson
import requests
from ase.db import connect

from records import (
    MPTRJ_URL,
    SALEX_URL,
    mptrj_record,
    validate,
    protected_inventory,
    structure_key,
)
from common import digest, encode_atoms, write_json, write_rows
from splits import composition_key

VERSION = "adaptation-standard-500k-v1"
QUOTAS = {"train": 250000, "val": 2500}  # per source
PARENT_CAP = 4
SOURCE_INFO = {
    "mptrj": {
        "url": MPTRJ_URL,
        "bytes": 12188168685,
        "md5": "50ead5f27f9a4f6beb7564c4188f1e9f",
        "name": "mptrj.json",
    },
    "salex": {"url": SALEX_URL, "bytes": 8071921954, "name": "salex-train.tar.gz"},
}


def priority(value):
    return int(hashlib.sha256(f"{VERSION}:{value}".encode()).hexdigest(), 16)


class PrioritySelection:
    """Lowest hash priorities under source quotas and a per-parent frame cap.

    Keeps a bounded reservoir, scans the entire source, and is invariant to
    source order. Labels never influence selection.
    """

    def __init__(self, quotas=None, protected=None, parent_cap=PARENT_CAP):
        self.quotas = quotas or QUOTAS
        self.protected, self.blocked_ids = protected or protected_inventory()
        self.parent_cap = parent_cap
        self.rows = {s: {} for s in self.quotas}
        self.heaps = {s: [] for s in self.quotas}
        self.parents = defaultdict(dict)
        self.excluded = Counter()

    def split(self, composition):
        return "val" if priority("split:" + composition) % 16 == 0 else "train"

    def _worst(self, split):
        heap = self.heaps[split]
        while heap and heap[0][1] not in self.rows[split]:
            heapq.heappop(heap)
        return (-heap[0][0], heap[0][1]) if heap else None

    def eligible(self, sid, parent, composition):
        if composition in self.protected or parent in self.blocked_ids:
            return False
        split = self.split(composition)
        p = priority(sid)
        selected = self.parents.get(parent, {})
        if len(selected) >= self.parent_cap and p >= max(selected.values()):
            return False
        worst = self._worst(split)
        return len(self.rows[split]) < self.quotas[split] or p < worst[0]

    def _remove(self, split, sid):
        old = self.rows[split].pop(sid)
        parent = old["parent"]
        self.parents[parent].pop(sid)
        if not self.parents[parent]:
            del self.parents[parent]

    def offer(self, row, composition):
        sid, parent = row["id"], row["parent"]
        if not self.eligible(sid, parent, composition):
            return
        split = self.split(composition)
        validate(row)
        if sid in self.rows[split]:
            raise ValueError(f"Duplicate selected frame {sid}")
        selected = self.parents.get(parent, {})
        if len(selected) == self.parent_cap:
            self._remove(split, max(selected, key=selected.get))
        row["composition_group"] = composition
        self.rows[split][sid] = row
        self.parents[parent][sid] = priority(sid)
        heapq.heappush(self.heaps[split], (-priority(sid), sid))
        if len(self.rows[split]) > self.quotas[split]:
            self._remove(split, self._worst(split)[1])
        if len(self.heaps[split]) > max(100, 3 * self.quotas[split]):
            self.heaps[split] = [(-priority(k), k) for k in self.rows[split]]
            heapq.heapify(self.heaps[split])

    def counts(self):
        return {s: len(rows) for s, rows in self.rows.items()}


def fetch_source(source, cache, publish):
    item = SOURCE_INFO[source]
    path = Path(cache) / item["name"]
    receipt = path.with_suffix(path.suffix + ".receipt.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and receipt.exists():
        record = json.loads(receipt.read_text())
        if path.stat().st_size != item["bytes"] or digest(path) != record["sha256"]:
            raise ValueError("Cached full source corrupted")
        return path, record
    temp = path.with_suffix(path.suffix + ".partial")
    sha, md5 = hashlib.sha256(), hashlib.md5()
    count, last = 0, time.monotonic()
    if source == "salex":
        parts = path.with_suffix(path.suffix + ".parts")
        parts.mkdir(exist_ok=True)
        blocksize = 256 << 20
        ranges = [
            (i, start, min(start + blocksize, item["bytes"]) - 1)
            for i, start in enumerate(range(0, item["bytes"], blocksize))
        ]

        def fetch_range(job):
            i, start, end = job
            target = parts / f"{i:04d}"
            if target.exists() and target.stat().st_size == end - start + 1:
                return target
            for attempt in range(3):
                try:
                    with requests.get(
                        item["url"],
                        headers={"Range": f"bytes={start}-{end}"},
                        stream=True,
                        timeout=(30, 180),
                    ) as r:
                        r.raise_for_status()
                        if (
                            r.status_code != 206
                            or r.headers.get("Content-Range")
                            != f"bytes {start}-{end}/{item['bytes']}"
                        ):
                            raise ValueError("Server did not honor exact byte range")
                        temporary = target.with_suffix(".partial")
                        with temporary.open("wb") as f:
                            for block in r.iter_content(4 << 20):
                                f.write(block)
                    if temporary.stat().st_size != end - start + 1:
                        raise ValueError("Truncated range download")
                    temporary.replace(target)
                    return target
                except (requests.RequestException, ValueError):
                    if attempt == 2:
                        raise

        finished_bytes = 0
        with ThreadPoolExecutor(max_workers=12) as pool:
            for future in as_completed(
                [pool.submit(fetch_range, job) for job in ranges]
            ):
                finished_bytes += future.result().stat().st_size
                publish(
                    {
                        "phase": "downloading_ranges",
                        "bytes": finished_bytes,
                        "expected_bytes": item["bytes"],
                        "parallel_connections": 12,
                    }
                )
        with temp.open("wb") as stream:
            for i, _, _ in ranges:
                with (parts / f"{i:04d}").open("rb") as part:
                    for block in iter(lambda: part.read(4 << 20), b""):
                        stream.write(block)
                        sha.update(block)
                        md5.update(block)
                        count += len(block)
        shutil.rmtree(parts)
    else:
        with requests.get(item["url"], stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with temp.open("wb") as stream:
                for block in response.iter_content(4 << 20):
                    stream.write(block)
                    sha.update(block)
                    md5.update(block)
                    count += len(block)
                    if time.monotonic() - last > 20:
                        publish(
                            {
                                "phase": "downloading",
                                "bytes": count,
                                "expected_bytes": item["bytes"],
                            }
                        )
                        last = time.monotonic()
    if count != item["bytes"] or (item.get("md5") and md5.hexdigest() != item["md5"]):
        raise ValueError("Full source size/MD5 mismatch")
    temp.replace(path)
    record = {
        **item,
        "sha256": sha.hexdigest(),
        "actual_md5": md5.hexdigest(),
        "entire_source_downloaded": True,
        "publisher_checksum_verified": "md5" in item,
    }
    write_json(receipt, record)
    return path, record


def prepare_source(source, root, cache, publish=lambda value: None):
    output = Path(root) / "sources" / source
    if (output / "manifest.json").exists():
        result = json.loads((output / "manifest.json").read_text())
        for name, sha in result["files"].items():
            if digest(output / name) != sha:
                raise ValueError("Source subset corrupted")
        return result
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    path, provenance = fetch_source(source, cache, publish)
    selection = PrioritySelection()
    frames_scanned, parents_scanned, members = 0, 0, []
    if source == "mptrj":
        with path.open("rb") as stream:
            for parent, frames in ijson.kvitems(stream, "", use_float=True):
                parents_scanned += 1
                frames_scanned += len(frames)
                if not frames:
                    continue
                composition = structure_key(next(iter(frames.values()))["structure"])
                # All frames of a trajectory are present together; only its
                # lowest-priority four can ever enter the constrained reservoir.
                for frame in sorted(frames, key=lambda k: priority(f"mptrj/{k}"))[
                    :PARENT_CAP
                ]:
                    sid = f"mptrj/{frame}"
                    if selection.eligible(sid, parent, composition):
                        selection.offer(
                            mptrj_record(parent, frame, frames[frame]), composition
                        )
                if parents_scanned % 2000 == 0:
                    publish(
                        {
                            "phase": "scanning",
                            "parents_scanned": parents_scanned,
                            "frames_scanned": frames_scanned,
                            "counts": selection.counts(),
                        }
                    )
    elif source == "salex":
        compositions = {}
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".aselmdb"):
                    continue
                extracted = output / "current.aselmdb"
                with extracted.open("wb") as stream:
                    shutil.copyfileobj(archive.extractfile(member), stream)
                item = {
                    "name": member.name,
                    "bytes": extracted.stat().st_size,
                    "sha256": digest(extracted),
                }
                with connect(extracted) as db:
                    item["frames"] = len(db)
                    for row in db.select():
                        frames_scanned += 1
                        sid = row.data["sid"]
                        parent = sid.split("_")[0]
                        if parent not in compositions:
                            compositions[parent] = composition_key(Counter(row.symbols))
                        composition = compositions[parent]
                        if selection.eligible(f"salex/{sid}", parent, composition):
                            atoms = row.toatoms()
                            selection.offer(
                                dict(
                                    id=f"salex/{sid}",
                                    parent=parent,
                                    source=source,
                                    atoms=encode_atoms(atoms),
                                    energy=float(atoms.get_potential_energy()),
                                    forces=atoms.get_forces().tolist(),
                                    stress=atoms.get_stress(voigt=False).tolist(),
                                ),
                                composition,
                            )
                        if frames_scanned % 50000 == 0:
                            publish(
                                {
                                    "phase": "scanning",
                                    "frames_scanned": frames_scanned,
                                    "members_scanned": len(members),
                                    "counts": selection.counts(),
                                }
                            )
                members.append(item)
                extracted.unlink()
                publish(
                    {
                        "phase": "scanning",
                        "frames_scanned": frames_scanned,
                        "members_scanned": len(members),
                        "counts": selection.counts(),
                    }
                )
        parents_scanned = len(compositions)
    else:
        raise ValueError(source)
    if selection.counts() != QUOTAS:
        raise ValueError(f"Insufficient eligible {source} data: {selection.counts()}")
    for split, rows in selection.rows.items():
        write_rows(
            output / f"{split}.jsonl", (rows[k] for k in sorted(rows, key=priority))
        )
    result = dict(
        source=source,
        version=VERSION,
        counts=selection.counts(),
        provenance=provenance,
        entire_source_scanned=True,
        frames_scanned=frames_scanned,
        parents_scanned=parents_scanned,
        members=members,
        preparation_seconds=time.time() - started,
        parent_cap=PARENT_CAP,
        files={f"{split}.jsonl": digest(output / f"{split}.jsonl") for split in QUOTAS},
    )
    write_json(output / "manifest.json", result)
    return result


def assemble(root):
    # Reuse the existing exact source/label/overlap checks with explicit quotas.
    import records
    from splits import DEFAULT_SPLITS

    result = records.assemble(root, quotas=QUOTAS)
    result.update(
        protocol=VERSION,
        selection="Full source scan; lowest SHA256 priorities; max 4 frames per source-parent; composition-hash train/dev; no atom cap",
        parent_cap=PARENT_CAP,
        split_manifest_sha256=digest(DEFAULT_SPLITS / "manifest.json"),
        limitations=[
            "Public pretrained UMA and elemental references may have seen evaluation materials",
            "Equal source quotas and three epochs are a bounded AI4Sci adaptation recipe, not full-paper reproduction",
        ],
    )
    write_json(Path(root) / "data/manifest.json", result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--source", choices=["mptrj", "salex", "assemble"], required=True
    )
    args = parser.parse_args()
    if args.source == "assemble":
        result = assemble(args.root)
    else:
        result = prepare_source(
            args.source,
            args.root,
            args.cache,
            publish=lambda value: print(json.dumps(value), flush=True),
        )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["members", "sources"]},
            indent=2,
        )
    )
