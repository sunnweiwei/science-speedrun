"""Extract the two official human track tables from pinned textproto metadata."""
from __future__ import annotations

import re
import pandas as pd

from .errors import ContractError


def _balanced_blocks(text: str, field: str) -> list[str]:
    blocks: list[str] = []
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(field)}\s*\{{")
    for match in pattern.finditer(text):
        depth = 0
        in_string = False
        escaped = False
        for index in range(match.end() - 1, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(text[match.start() : index + 1])
                    break
        else:
            raise ContractError(f"Unterminated {field} block in output metadata")
    return blocks


def _string(block: str, name: str) -> str | None:
    match = re.search(
        rf'(?<![A-Za-z0-9_]){re.escape(name)}:\s*"((?:[^"\\]|\\.)*)"', block
    )
    if not match:
        return None
    return bytes(match.group(1), "utf-8").decode("unicode_escape")


def _token(block: str, name: str) -> str | None:
    match = re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(name)}:\s*([^\s}}]+)", block
    )
    return match.group(1) if match else None


def _nested(block: str, name: str) -> str:
    items = _balanced_blocks(block, name)
    return items[0] if items else ""


def _ontology_curie(block: str) -> str | None:
    ontology = _nested(block, "ontology_term")
    kind = _token(ontology, "ontology_type")
    ident = _token(ontology, "id")
    if not kind or ident is None:
        return None
    prefix = kind.removeprefix("ONTOLOGY_TYPE_")
    widths = {"CL": 7, "EFO": 7, "UBERON": 7}
    return f"{prefix}:{int(ident):0{widths.get(prefix, 7)}d}"


def parse_all_track_tables(text: str) -> dict[str, pd.DataFrame]:
    """Parse every released human output block without legacy channel assumptions."""
    result: dict[str, pd.DataFrame] = {}
    for output in _balanced_blocks(text, "output_metadata"):
        output_type = _token(output, "output_type")
        family = (output_type or "").removeprefix("OUTPUT_TYPE_")
        if not family:
            continue
        container = "junctions" if family == "SPLICE_JUNCTIONS" else "tracks"
        tracks_block = _nested(output, container)
        rows = []
        for source_index, track in enumerate(_balanced_blocks(tracks_block, "metadata")):
            biosample = _nested(track, "biosample")
            name = _string(track, "name")
            rows.append(
                {
                    "source_index": source_index,
                    "name": name,
                    "output_type": family,
                    "strand": (_token(track, "strand") or "").removeprefix("STRAND_"),
                    "ontology_curie": _ontology_curie(track),
                    "biosample_name": _string(biosample, "name"),
                    "biosample_type": (_token(biosample, "type") or "").removeprefix("BIOSAMPLE_TYPE_"),
                    "biosample_life_stage": _string(biosample, "stage"),
                    "assay": _string(track, "assay"),
                    "data_source": _string(track, "data_source"),
                    "endedness": (_token(track, "endedness") or "").removeprefix("ENDEDNESS_"),
                    "genetically_modified": _token(track, "genetically_modified"),
                    "nonzero_mean": _token(track, "nonzero_mean"),
                    "is_padding": (name or "").lower() == "padding",
                    "metadata_textproto": track,
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ContractError(f"Output metadata block {family} contains no tracks")
        real = frame.loc[~frame["is_padding"]].copy().reset_index(drop=True)
        real.insert(0, "track_index", range(len(real)))
        result[family] = real
    return result
