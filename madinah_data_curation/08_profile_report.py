#!/usr/bin/env python3
"""Profile dataset quality: language mix, lengths, MCQ compliance."""

import argparse
import json
import math
import re
from collections import Counter
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


class StreamingStats:
    def __init__(self):
        self.n = 0
        self.total = 0
        self.sq_total = 0.0
        self.mn = float("inf")
        self.mx = float("-inf")

    def add(self, v: int):
        self.n += 1
        self.total += v
        self.sq_total += v * v
        self.mn = min(self.mn, v)
        self.mx = max(self.mx, v)

    def to_dict(self):
        if self.n == 0:
            return {"count": 0}
        mean = self.total / self.n
        var = max(0, self.sq_total / self.n - mean * mean)
        return {
            "count": self.n,
            "mean": round(mean, 1),
            "std": round(math.sqrt(var), 1),
            "min": self.mn,
            "max": self.mx,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(FINAL_DIR / "decontaminated.jsonl"))
    parser.add_argument("--out", default=str(REPORTS_DIR / "08_profile_report.json"))
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lang_user = Counter()
    lang_asst = Counter()
    user_len = StreamingStats()
    asst_len = StreamingStats()
    mcq_count = 0
    total = 0
    mcq_answer_ok = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            user_text = ""
            asst_text = ""
            for m in row.get("messages", []):
                if m.get("role") == "user":
                    user_text = m.get("content", "")
                elif m.get("role") == "assistant":
                    asst_text = m.get("content", "")

            lang_user[classify_language(user_text)] += 1
            lang_asst[classify_language(asst_text)] += 1
            user_len.add(len(user_text))
            asst_len.add(len(asst_text))

            is_mcq = row.get("is_mcq", False) or bool(MCQ_OPTION_PATTERN.search(user_text))
            if is_mcq:
                mcq_count += 1
                if asst_text.strip() in {"أ", "ب", "ج", "د", "هـ"}:
                    mcq_answer_ok += 1

    report = {
        "total_rows": total,
        "language_user": lang_user,
        "language_assistant": lang_asst,
        "user_length": user_len.to_dict(),
        "assistant_length": asst_len.to_dict(),
        "mcq_rows": mcq_count,
        "mcq_answer_letter_only": mcq_answer_ok,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Profile report saved to {args.out}")


if __name__ == "__main__":
    main()
