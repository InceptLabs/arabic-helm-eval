#!/usr/bin/env python3
"""Build two-phase curriculum datasets (phase1: InstAr, phase2: grammar)."""

import argparse
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
FINAL_DIR = SCRIPT_DIR / "final"


def _read_rows(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(FINAL_DIR / "decontaminated.jsonl"))
    parser.add_argument("--phase1-out", default=str(FINAL_DIR / "phase1.jsonl"))
    parser.add_argument("--phase2-out", default=str(FINAL_DIR / "phase2.jsonl"))
    parser.add_argument("--curriculum-out", default=str(FINAL_DIR / "curriculum.yaml"))
    parser.add_argument("--include-instar-in-phase2", action="store_true")
    parser.add_argument("--phase2-instar-fraction", type=float, default=0.2)
    args = parser.parse_args()

    rows = _read_rows(Path(args.input))

    instar = []
    cidar = []
    synthetic = []
    other = []

    for row in rows:
        source = row.get("meta", {}).get("source", "unknown")
        if source == "instar":
            instar.append(row)
        elif source == "cidar":
            cidar.append(row)
        elif source == "synthetic_madinah":
            synthetic.append(row)
        else:
            other.append(row)

    phase1 = instar[:]
    phase2 = cidar + synthetic

    if args.include_instar_in_phase2 and instar:
        add_count = int(len(phase2) * args.phase2_instar_fraction)
        phase2.extend(instar[:add_count])

    _write_rows(Path(args.phase1_out), phase1)
    _write_rows(Path(args.phase2_out), phase2)

    curriculum = {
        "phase1": {
            "description": "Broad Arabic comprehension (InstAr-500k dominant)",
            "sources": {"instar": len(instar)},
            "rows": len(phase1),
        },
        "phase2": {
            "description": "Grammar injection (CIDAR Ask-the-Teacher + synthetic)",
            "sources": {"cidar": len(cidar), "synthetic_madinah": len(synthetic)},
            "rows": len(phase2),
        },
        "notes": [
            "Apply LR decay before phase2.",
            "Focus on grammar and morphology after broad adaptation.",
        ],
    }

    Path(args.curriculum_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.curriculum_out, "w", encoding="utf-8") as f:
        yaml_lines = []
        yaml_lines.append("phase1:\n")
        yaml_lines.append(f"  rows: {len(phase1)}\n")
        yaml_lines.append("  sources:\n")
        yaml_lines.append(f"    instar: {len(instar)}\n")
        yaml_lines.append("phase2:\n")
        yaml_lines.append(f"  rows: {len(phase2)}\n")
        yaml_lines.append("  sources:\n")
        yaml_lines.append(f"    cidar: {len(cidar)}\n")
        yaml_lines.append(f"    synthetic_madinah: {len(synthetic)}\n")
        yaml_lines.append("notes:\n")
        for note in curriculum["notes"]:
            yaml_lines.append(f"  - {note}\n")
        f.write("".join(yaml_lines))

    print(f"Phase1 rows: {len(phase1):,} -> {args.phase1_out}")
    print(f"Phase2 rows: {len(phase2):,} -> {args.phase2_out}")
    print(f"Curriculum: {args.curriculum_out}")


if __name__ == "__main__":
    main()
