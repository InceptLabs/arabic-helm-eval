#!/usr/bin/env python3
"""Format normalized data into ShareGPT-style JSONL with system prompt."""

import argparse
import json
import re
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).parent
INTERMEDIATE_DIR = SCRIPT_DIR / "intermediate"
PROMPT_PATH = SCRIPT_DIR / "prompts" / "madinah_curriculum.yaml"

ANSWER_LETTER_RE = re.compile(r"[أبجدهـ]")
LATIN_ANSWER_RE = re.compile(r"\b([A-E])\b")

LATIN_TO_AR = {"A": "أ", "B": "ب", "C": "ج", "D": "د", "E": "هـ"}


def _load_system_prompt(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("system_prompt", "").strip()


def _normalize_answer(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    match = ANSWER_LETTER_RE.search(text)
    if match:
        return match.group(0)
    match = LATIN_ANSWER_RE.search(text)
    if match:
        return LATIN_TO_AR.get(match.group(1), match.group(1))
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INTERMEDIATE_DIR / "normalized.jsonl"))
    parser.add_argument("--out", default=str(INTERMEDIATE_DIR / "sharegpt.jsonl"))
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--prompt-config", default=str(PROMPT_PATH))
    args = parser.parse_args()

    sys_prompt = args.system_prompt.strip()
    if not sys_prompt:
        sys_prompt = _load_system_prompt(Path(args.prompt_config))

    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages", [])
            is_mcq = row.get("is_mcq", False)
            meta = row.get("meta", {})

            if not messages:
                continue

            if is_mcq:
                for m in messages:
                    if m.get("role") == "assistant":
                        m["content"] = _normalize_answer(m.get("content", ""))

            if sys_prompt and (not messages or messages[0].get("role") != "system"):
                messages = [{"role": "system", "content": sys_prompt}] + messages

            out_row = {
                "messages": messages,
                "is_mcq": is_mcq,
                "meta": meta,
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept:,} rows to {out_path}")


if __name__ == "__main__":
    main()
