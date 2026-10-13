"""Render the committed validation example as editable SVG and portable PNG/PDF."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import cairosvg

INK = "#182534"
MUTED = "#566675"
BLUE = "#276382"
PALE = "#eef4f7"
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


def render(example: dict[str, Any]) -> str:
    assert example["split"] == "validation"
    assert example["distance"] == 3 and example["rounds"] == 9
    assert example["model_probability"] is None
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="860" viewBox="0 0 1600 860" role="img" aria-labelledby="title description">',
             '<title id="title">Surface-code decoding: from noisy stabilizer measurements to a logical-flip decision</title>',
             '<desc id="description">A real Sycamore distance-three validation shot is displayed on its hardware geometry and as an eight-check, ten-time-slice detection-event matrix. The decoder output is symbolic; no numerical prediction is claimed. Better decoding supports reliable logical memory.</desc>',
             '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#566675"/></marker></defs>',
             '<g font-family="DejaVu Sans, sans-serif">', rect(0, 0, 1600, 860, "#ffffff")]
    parts += [text(60, 62, "Recover a logical bit from noisy quantum measurements", 33, weight=600),
              text(60, 101, "Repeated stabilizer checks reveal error patterns; a decoder infers whether the stored logical observable flipped.", 21, MUTED),
              line(60, 129, 1540, 129),
              text(60, 182, "A   Quantum memory", 24, weight=600),
              text(60, 216, "Sycamore · distance 3 · Z basis", 18, MUTED),
              text(490, 182, "B   Detection-event input", 24, weight=600),
              text(490, 216, "One real shot, measured over 9 rounds", 18, MUTED),
              text(1050, 182, "C   Logical decoding", 24, weight=600),
              text(1050, 216, "Model input: events + known geometry", 18, MUTED)]
    position = lambda xy: (110 + (xy[0] - 1) * 52, 290 + (xy[1] - 3) * 52)
    for data in example["data_qubits"]:
        for check in example["check_qubits"]:
            if sum(abs(a - b) for a, b in zip(data, check, strict=True)) == 1:
                parts.append(line(*position(data), *position(check), GRID, 2))
    for data in example["data_qubits"]:
        x, y = position(data)
        parts.append(f'<circle cx="{x}" cy="{y}" r="11" fill="white" stroke="{BLUE}" stroke-width="2.5"/>')
    for check in example["check_qubits"]:
        x, y = position(check)
        parts.append(rect(x - 8, y - 8, 16, 16, "#8797a4"))
    parts += [f'<circle cx="84" cy="552" r="7" fill="white" stroke="{BLUE}" stroke-width="2"/>',
              text(102, 558, "9 data qubits", 17, MUTED), rect(241, 545, 14, 14, "#8797a4"),
              text(266, 558, "8 checks", 17, MUTED),
              text(60, 598, "Checks probe parity without revealing", 17, MUTED),
              text(60, 623, "the stored quantum information.", 17, MUTED),
              line(404, 384, 453, 384, MUTED, 2, True)]
    checks = sorted(tuple(xy) for xy in example["check_qubits"])
    events = {(tuple(item["coords"][:2]), int(item["coords"][2])): item["event"] for item in example["detectors"]}
    for row, check in enumerate(checks):
        y = 273 + row * 32
        parts.append(text(558, y + 20, f"({int(check[0])},{int(check[1])})", 16, MUTED, anchor="end"))
        for time in range(10):
            value = events.get((check, time))
            fill = BLUE if value == 1 else "white" if value == 0 else "#e6e9ec"
            parts.append(rect(574 + time * 34, y, 32, 30, fill, GRID))
    for time in range(10):
        parts.append(text(590 + time * 34, 556, "final" if time == 9 else str(time + 1), 15, MUTED, anchor="middle"))
    parts += [text(742, 583, "Measurement round / final readout", 17, MUTED, anchor="middle"),
              rect(496, 605, 15, 15, BLUE), text(518, 618, "event", 16, MUTED),
              rect(596, 605, 15, 15, "white", GRID), text(618, 618, "no event", 16, MUTED),
              rect(735, 605, 15, 15, "#e6e9ec", GRID), text(757, 618, "not defined", 16, MUTED),
              line(959, 384, 1018, 384, MUTED, 2, True),
              rect(1050, 264, 490, 80, PALE, GRID),
              text(1295, 296, "Decoder", 23, BLUE, 600, "middle"),
              text(1295, 324, "e.g., an AlphaQubit recurrent Transformer", 17, MUTED, anchor="middle"),
              line(1295, 350, 1295, 372, MUTED, 1.8, True),
              text(1295, 406, "p = P(logical flip | events, geometry)", 20, BLUE, anchor="middle"),
              rect(1050, 430, 490, 73, "#f8fafb", GRID),
              text(1295, 462, "Decision: flip if p ≥ 0.5", 22, weight=600, anchor="middle"),
              text(1295, 489, "Apply to the logical readout / Pauli frame", 17, MUTED, anchor="middle"),
              text(1050, 548, f"Validation target for this shot: flip = {example['target_logical_flip']}", 19),
              text(1050, 578, "Target shown for explanation; never a model input.", 16, MUTED),
              text(1050, 615, "Output above is symbolic, not a measured prediction.", 16, MUTED),
              line(60, 655, 1540, 655),
              text(60, 696, "Scientific value", 22, weight=600),
              text(60, 733, "Better decoding reduces logical readout errors, a bottleneck for reliable quantum memory and fault-tolerant computation.", 20),
              text(60, 768, "Benchmark scope: real hard-readout Sycamore data, distances 3 and 5; score held-out logical-flip decisions.", 18, MUTED),
              text(60, 818, "Example: d3 / Z / center (3,5) / 9 rounds / source shot 39760 (validation). Source: Google Quantum AI, Zenodo 6804040, CC BY 4.0.", 15, MUTED),
              '</g></svg>']
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, default=Path(__file__).with_name("example.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    example = json.loads(args.example.read_text())
    svg = render(example)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "science-introduction.svg"
    path.write_text(svg)
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(path.with_suffix(".png")))
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(path.with_suffix(".pdf")))


if __name__ == "__main__":
    main()
