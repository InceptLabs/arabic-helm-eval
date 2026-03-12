#!/usr/bin/env python3
"""Stratified sampling for manual audit."""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
FINAL_DIR = SCRIPT_DIR / "final"
REPORTS_DIR = SCRIPT_DIR / "reports"

ARABIC_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
LATIN_RANGE = re.compile(r"[A-Za-z]")
MCQ_OPTION_PATTERN = re.compile(r"\n\s*[أبجدهـA-E][\)\.]\s")


def classify_language(text: str) -> str:
    ar = len(ARABIC_RANGE.findall(text))
    la = len(LATIN_RANGE.findall(text))
    total = ar + la
    if total == 0:
        return "other"
    frac = ar / total
    if frac >= 0.85:
        return "arabic"
    if frac <= 0.15:
        return "english"
    return "mixed"


class ReservoirSampler:
    def __init__(self, k: int):
        self.k = k
        self.items = []
        self.n = 0

    def add(self, item):
        self.n += 1
        if len(self.items) < self.k:
            self.items.append(item)
        else:
            j = random.randint(0, self.n - 1)
            if j < self.k:
                self.items[j] = item


class TopKCollector:
    def __init__(self, k: int, highest: bool = True):
        self.k = k
        self.highest = highest
        self.items = []

    def add(self, item, score: int):
        self.items.append((score, item))
        if len(self.items) > self.k * 3:
            self._prune()

    def _prune(self):
        self.items.sort(key=lambda x: x[0], reverse=self.highest)
        self.items = self.items[: self.k]

    def values(self):
        self._prune()
        return [item for _, item in self.items]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(FINAL_DIR / "decontaminated.jsonl"))
    parser.add_argument("--out-dir", default=str(REPORTS_DIR / "09_samples"))
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    uniform = ReservoirSampler(args.sample_size)
    by_lang = defaultdict(lambda: ReservoirSampler(args.sample_size))
    by_type = defaultdict(lambda: ReservoirSampler(args.sample_size))
    longest_user = TopKCollector(args.sample_size, highest=True)
    shortest_user = TopKCollector(args.sample_size, highest=False)
    longest_asst = TopKCollector(args.sample_size, highest=True)

    with open(args.input, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            user_text = ""
            asst_text = ""
            for m in row.get("messages", []):
                if m.get("role") == "user":
                    user_text = m.get("content", "")
                elif m.get("role") == "assistant":
                    asst_text = m.get("content", "")

            sample_row = {"row_index": idx, **row}

            uniform.add(sample_row)
            by_lang[classify_language(user_text)].add(sample_row)

            is_mcq = row.get("is_mcq", False) or bool(MCQ_OPTION_PATTERN.search(user_text))
            by_type["mcq" if is_mcq else "open_ended"].add(sample_row)

            longest_user.add(sample_row, len(user_text))
            shortest_user.add(sample_row, len(user_text))
            longest_asst.add(sample_row, len(asst_text))

    slices = {
        "uniform_random": uniform.items,
        **{f"lang_{k}": v.items for k, v in by_lang.items()},
        **{f"type_{k}": v.items for k, v in by_type.items()},
        "longest_user": longest_user.values(),
        "shortest_user": shortest_user.values(),
        "longest_assistant": longest_asst.values(),
    }

    index = {}
    for name, items in slices.items():
        out_path = out_dir / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        index[name] = {
            "file": str(out_path),
            "count": len(items),
            "row_indices": [item.get("row_index") for item in items],
        }

    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Samples written to {out_dir}")


if __name__ == "__main__":
    main()
