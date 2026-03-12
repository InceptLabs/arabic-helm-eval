#!/usr/bin/env python3
"""Download and export core datasets as JSONL (streaming).

Outputs:
  raw/instar.jsonl
  raw/cidar_ask_teacher.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

from datasets import get_dataset_config_names, load_dataset


SCRIPT_DIR = Path(__file__).parent
RAW_DIR = SCRIPT_DIR / "raw"

INSTAR_DATASET = "ClusterlabAi/InstAr-500k"
CIDAR_DATASET = "arbml/CIDAR"

DEFAULT_CIDAR_REGEX = r"(ask the teacher|ask_teacher|ask-the-teacher|اسأل المعلم|اسال المعلم)"


def _iter_streaming_splits(dataset_name: str, config: Optional[str]):
    ds = load_dataset(dataset_name, name=config, split=None, streaming=True)
    if isinstance(ds, dict):
        for split, stream in ds.items():
            yield split, stream
    else:
        yield "train", ds


def _get_configs(dataset_name: str):
    try:
        return get_dataset_config_names(dataset_name)
    except Exception:
        return []


def _row_matches_regex(row: dict, pattern: re.Pattern) -> bool:
    for _, value in row.items():
        if isinstance(value, str) and pattern.search(value):
            return True
    return False


def _flatten_row(source: str, config: Optional[str], split: str, row: dict):
    return {
        "source": source,
        "source_config": config,
        "split": split,
        "row": row,
    }


def export_instar(out_path: Path, max_rows: Optional[int]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    configs = _get_configs(INSTAR_DATASET) or [None]
    total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for config in configs:
            for split, stream in _iter_streaming_splits(INSTAR_DATASET, config):
                for row in stream:
                    f.write(json.dumps(_flatten_row("instar", config, split, row), ensure_ascii=False) + "\n")
                    total += 1
                    if max_rows and total >= max_rows:
                        return total
    return total


def export_cidar(out_path: Path, max_rows: Optional[int], filter_regex: str, allow_fallback: bool):
    pattern = re.compile(filter_regex, re.IGNORECASE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matched = 0
    total = 0
    configs = _get_configs(CIDAR_DATASET) or [None]
    with open(out_path, "w", encoding="utf-8") as f:
        for config in configs:
            for split, stream in _iter_streaming_splits(CIDAR_DATASET, config):
                for row in stream:
                    total += 1
                    if _row_matches_regex(row, pattern):
                        f.write(json.dumps(_flatten_row("cidar", config, split, row), ensure_ascii=False) + "\n")
                        matched += 1
                    if max_rows and total >= max_rows:
                        break
                if max_rows and total >= max_rows:
                    break
            if max_rows and total >= max_rows:
                break

    if matched == 0 and not allow_fallback:
        print(
            "WARNING: No CIDAR rows matched the filter. "
            "Use --cidar-filter-regex or --allow-cidar-fallback to export all rows.",
            file=sys.stderr,
        )
    if matched == 0 and allow_fallback:
        total = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for config in configs:
                for split, stream in _iter_streaming_splits(CIDAR_DATASET, config):
                    for row in stream:
                        f.write(json.dumps(_flatten_row("cidar", config, split, row), ensure_ascii=False) + "\n")
                        total += 1
                        if max_rows and total >= max_rows:
                            return total, total
        return total, total
    return total, matched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None, help="Cap total rows per dataset")
    parser.add_argument("--cidar-filter-regex", default=DEFAULT_CIDAR_REGEX)
    parser.add_argument("--allow-cidar-fallback", action="store_true")
    parser.add_argument("--instar-out", default=str(RAW_DIR / "instar.jsonl"))
    parser.add_argument("--cidar-out", default=str(RAW_DIR / "cidar_ask_teacher.jsonl"))
    parser.add_argument("--skip-instar", action="store_true")
    parser.add_argument("--skip-cidar", action="store_true")
    args = parser.parse_args()

    if not args.skip_instar:
        print("Exporting InstAr-500k...", file=sys.stderr)
        instar_count = export_instar(Path(args.instar_out), args.max_rows)
        print(f"  InstAr rows: {instar_count:,} -> {args.instar_out}", file=sys.stderr)

    if not args.skip_cidar:
        print("Exporting CIDAR (Ask the Teacher subset)...", file=sys.stderr)
        cidar_total, cidar_matched = export_cidar(
            Path(args.cidar_out), args.max_rows, args.cidar_filter_regex, args.allow_cidar_fallback
        )
        print(
            f"  CIDAR total scanned: {cidar_total:,}, matched: {cidar_matched:,} "
            f"-> {cidar_matched:,} rows written to {args.cidar_out}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
