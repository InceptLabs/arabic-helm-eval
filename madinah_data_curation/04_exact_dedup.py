#!/usr/bin/env python3
"""Exact deduplication on ShareGPT JSONL."""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
INTERMEDIATE_DIR = SCRIPT_DIR / "intermediate"
REPORTS_DIR = SCRIPT_DIR / "reports"

THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
TASHKEEL = re.compile(r"[\u064B-\u065F\u0670]")
WHITESPACE = re.compile(r"\s+")

ALEF_MAP = str.maketrans({
    "\u0622": "\u0627",
    "\u0623": "\u0627",
    "\u0625": "\u0627",
    "\u0671": "\u0627",
})

YA_TA_MAP = str.maketrans({
    "\u0649": "\u064A",
    "\u0629": "\u0647",
})

PUNCT_MAP = str.maketrans({
    "\u060C": ",",
    "\u061B": ";",
    "\u061F": "?",
    "\u066B": ".",
    "\u066C": ",",
    "\u06D4": ".",
    "\u200F": "",
    "\u200E": "",
    "\u200B": "",
    "\u00A0": " ",
    "\uFEFF": "",
})


def normalize_whitespace(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()


def strip_think(text: str) -> str:
    return THINK_PATTERN.sub("", text)


def arabic_canonicalize(text: str) -> str:
    text = text.translate(ALEF_MAP)
    text = text.translate(YA_TA_MAP)
    text = text.translate(PUNCT_MAP)
    text = TASHKEEL.sub("", text)
    text = normalize_whitespace(text)
    return text.lower()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_user_assistant(row: dict) -> tuple[str, str]:
    user_text = ""
    asst_text = ""
    for m in row.get("messages", []):
        if m.get("role") == "user":
            user_text = m.get("content", "")
        elif m.get("role") == "assistant":
            asst_text = m.get("content", "")
    return user_text, asst_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INTERMEDIATE_DIR / "sharegpt.jsonl"))
    parser.add_argument("--out", default=str(INTERMEDIATE_DIR / "exact_deduped.jsonl"))
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    level_clusters = {
        "L1_raw_pair": defaultdict(list),
        "L2_ws_pair": defaultdict(list),
        "L3_think_stripped_pair": defaultdict(list),
        "L4_user_only": defaultdict(list),
        "L5_user_arabic_canon": defaultdict(list),
    }

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            user_text, asst_text = extract_user_assistant(row)
            l1 = sha256(user_text + "\x00" + asst_text)
            ws_user = normalize_whitespace(user_text)
            ws_asst = normalize_whitespace(asst_text)
            l2 = sha256(ws_user + "\x00" + ws_asst)
            ts_asst = normalize_whitespace(strip_think(asst_text))
            l3 = sha256(ws_user + "\x00" + ts_asst)
            l4 = sha256(ws_user)
            l5 = sha256(arabic_canonicalize(user_text))

            level_clusters["L1_raw_pair"][l1].append(idx)
            level_clusters["L2_ws_pair"][l2].append(idx)
            level_clusters["L3_think_stripped_pair"][l3].append(idx)
            level_clusters["L4_user_only"][l4].append(idx)
            level_clusters["L5_user_arabic_canon"][l5].append(idx)

    remove_indices = set()
    for indices in level_clusters["L3_think_stripped_pair"].values():
        if len(indices) > 1:
            remove_indices.update(indices[1:])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            if idx in remove_indices:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    report = {"total_rows": len(rows), "removed": len(remove_indices), "levels": {}}
    for level, clusters in level_clusters.items():
        dup_clusters = [v for v in clusters.values() if len(v) > 1]
        dup_rows = sum(len(v) - 1 for v in dup_clusters)
        report["levels"][level] = {
            "unique_hashes": len(clusters),
            "duplicate_clusters": len(dup_clusters),
            "duplicate_rows_removable": dup_rows,
        }

    report_path = REPORTS_DIR / "04_exact_dedup_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    ids_path = REPORTS_DIR / "04_exact_dedup_ids.json"
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "removed_indices": sorted(remove_indices),
                "level": "L3_think_stripped_pair",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Exact dedup complete: kept {kept:,} / {len(rows):,}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
