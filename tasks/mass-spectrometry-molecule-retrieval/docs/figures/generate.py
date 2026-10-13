"""Draw the committed validation spectrum, molecules and real retrieval outcome."""
from __future__ import annotations

import argparse
import html
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cairosvg
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

INK = "#182534"
MUTED = "#566675"
BLUE = "#276382"
GRID = "#d9e0e5"


def text(x: float, y: float, value: str, size: int = 20, color: str = INK,
         weight: int = 400, anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(value)}</text>'


def rect(x: float, y: float, width: float, height: float, fill: str, stroke: str = "none") -> str:
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" stroke="{stroke}"/>'


def line(x1: float, y1: float, x2: float, y2: float, color: str = GRID,
         width: float = 1.5, arrow: bool = False) -> str:
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{marker}/>'


def molecule(smiles: str, x: float, y: float, width: int, height: int) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid source molecule")
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().bondLineWidth = 1.7
    drawer.drawOptions().padding = 0.07
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    root = ET.fromstring(drawer.GetDrawingText())
    # Keep RDKit bonds and atom glyphs as native, editable vector elements.
    inner = "".join(ET.tostring(child, encoding="unicode") for child in root)
    return f'<g transform="translate({x},{y})">{inner}</g>'


def render(example: dict[str, Any]) -> str:
    assert example["split"] == "validation"
    query = example["query"]
    candidates = example["shown_candidates"]
    assert len(candidates) == 3
    assert candidates[0]["candidate_id"] == example["correct_candidate_id"]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="860" viewBox="0 0 1600 860" role="img" aria-labelledby="title description">',
             '<title id="title">Mass spectrum retrieval: identify a molecule from its fragmentation spectrum</title>',
             '<desc id="description">A real MassSpecGym validation spectrum, three actual candidate molecules, and an archived GLACIER ranking. The true molecule is ranked twenty-second of 256, illustrating the challenge of molecular identification. No test data or synthetic prediction is displayed.</desc>',
             '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#566675"/></marker></defs>',
             '<g font-family="DejaVu Sans, sans-serif">', rect(0, 0, 1600, 860, "#ffffff"),
             text(60, 62, "Identify a molecule from its fragmentation spectrum", 33, weight=600),
             text(60, 101, "Different molecular structures can share nearly the same mass. MS/MS fragments help distinguish them.", 21, MUTED),
             line(60, 129, 1540, 129),
             text(60, 182, "A   Measured spectrum", 24, weight=600),
             text(60, 216, f"Precursor m/z {query['precursor_mz']:.4f} · {query['adduct']}", 18, MUTED),
             text(585, 182, "B   Candidate structures", 24, weight=600),
             text(585, 216, f"3 of {example['candidate_count']} mass-compatible candidates", 18, MUTED),
             text(1050, 182, "C   Retrieval output", 24, weight=600),
             text(1050, 216, "Actual archived GLACIER validation ranking", 18, MUTED)]
    x0, y0, width, height = 105, 530, 405, 228
    lo, hi = 40.0, 205.0
    parts += [line(x0, y0, x0 + width, y0, MUTED), line(x0, y0, x0, y0 - height, MUTED),
              text(90, 269, "Relative intensity", 17, MUTED),
              text(x0 + width / 2, y0 + 59, "m/z", 18, MUTED, anchor="middle")]
    for value in (0, 0.5, 1):
        y = y0 - value * height
        parts += [line(x0, y, x0 + width, y, GRID, 0.8), text(x0 - 13, y + 5, str(value), 15, MUTED, anchor="end")]
    for mz in (50, 100, 150, 200):
        x = x0 + (mz - lo) / (hi - lo) * width
        parts += [line(x, y0, x, y0 + 6, MUTED), text(x, y0 + 28, str(mz), 15, MUTED, anchor="middle")]
    maximum = max(query["intensities"])
    for mz, intensity in zip(query["mzs"], query["intensities"], strict=True):
        assert lo <= mz <= hi
        x = x0 + (mz - lo) / (hi - lo) * width
        parts.append(line(x, y0, x, y0 - intensity / maximum * height, BLUE, 1.7))
    top_index = max(range(len(query["mzs"])), key=lambda i: query["intensities"][i])
    top_mz = query["mzs"][top_index]
    parts += [text(x0 + (top_mz - lo) / (hi - lo) * width, y0 - height - 11, f"{top_mz:.2f}", 15, BLUE, anchor="middle"),
              text(60, 622, f"{len(query['mzs'])} measured peaks · {query['instrument_type']} · validation", 17, MUTED),
              line(523, 417, 566, 417, MUTED, 2, True)]
    for index, candidate in enumerate(candidates):
        y = 247 + index * 126
        parts += [text(593, y + 55, candidate["display_name"], 23, BLUE, 600),
                  molecule(candidate["smiles"], 626, y, 340, 105)]
        if index < 2:
            parts.append(line(588, y + 116, 965, y + 116))
    parts += [text(585, 622, "Candidate IDs label structures, not model scores.", 16, MUTED),
              line(984, 417, 1024, 417, MUTED, 2, True),
              rect(1050, 258, 490, 75, "#eef4f7", GRID),
              text(1295, 289, "Rank candidates using fragment evidence", 20, BLUE, 600, "middle"),
              text(1295, 319, "Spectrum + metadata + candidate molecules", 17, MUTED, anchor="middle"),
              line(1295, 341, 1295, 365, MUTED, 1.8, True),
              text(1074, 402, "RANK", 15, MUTED, 600), text(1190, 402, "CANDIDATE", 15, MUTED, 600)]
    ordered = sorted(candidates, key=lambda item: item["rank"])
    for index, candidate in enumerate(ordered):
        y = 440 + index * 51
        is_target = candidate["candidate_id"] == example["correct_candidate_id"]
        if is_target:
            parts.append(rect(1050, y - 30, 490, 43, "#eef4f7"))
        parts += [text(1080, y, str(candidate["rank"]), 23, BLUE if is_target else INK, 600),
                  text(1190, y, candidate["display_name"] + ("  —  validation target" if is_target else ""), 22, BLUE if is_target else INK),
                  line(1050, y + 17, 1540, y + 17)]
    parts += [text(1050, 588, f"True structure ranks {example['correct_candidate_rank']} / {example['candidate_count']}: a retrieval error.", 18),
              text(1050, 622, "One example; not an estimate of overall performance.", 16, MUTED),
              line(60, 655, 1540, 655),
              text(60, 696, "Scientific value", 22, weight=600),
              text(60, 733, "Accurate molecular identification supports metabolomics, chemical analysis, and the discovery of unknown small molecules.", 20),
              text(60, 768, "Benchmark scope: rank supplied, mass-compatible structures from MS/MS data; the primary metric is exact-candidate Hit@1.", 18, MUTED),
              text(60, 818, f"Example: {query['query_id']} (validation). Source: MassSpecGym developers, 2024, MIT. Molecules and ranking are real; see example.json for provenance.", 15, MUTED),
              '</g></svg>']
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, default=Path(__file__).with_name("example.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    svg = render(json.loads(args.example.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "science-introduction.svg"
    path.write_text(svg)
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(path.with_suffix(".png")))
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(path.with_suffix(".pdf")))


if __name__ == "__main__":
    main()
