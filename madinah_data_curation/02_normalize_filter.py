#!/usr/bin/env python3
"""Normalize, language-filter, and standardize raw datasets."""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
RAW_DIR = SCRIPT_DIR / "raw"
INTERMEDIATE_DIR = SCRIPT_DIR / "intermediate"

ARABIC_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
LATIN_RANGE = re.compile(r"[A-Za-z]")

MCQ_OPTION_PATTERN = re.compile(r"\n\s*[أبجدهـA-E][\)\.]\s")


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _arabic_ratio(text: str) -> float:
    ar = len(ARABIC_RANGE.findall(text))
    lat = len(LATIN_RANGE.findall(text))
    total = ar + lat
    if total == 0:
        return 0.0
    return ar / total


def _load_fasttext(model_path: str):
    try:
        import fasttext
    except Exception:
        return None
    return fasttext.load_model(model_path)


def _predict_fasttext(model, text: str):
    labels, scores = model.predict(text.replace("\n", " "), k=1)
    return labels[0], scores[0]


def _extract_messages(record: dict):
    row = record.get("row", record)
    meta = {
        "source": record.get("source", "unknown"),
        "source_config": record.get("source_config"),
        "split": record.get("split"),
    }

    if "messages" in row and isinstance(row["messages"], list):
        return row["messages"], meta

    if "conversations" in row and isinstance(row["conversations"], list):
        messages = []
        for m in row["conversations"]:
            role = m.get("role") or m.get("from")
            content = m.get("content") or m.get("value")
            if role in {"human", "user"}:
                role = "user"
            if role in {"gpt", "assistant"}:
                role = "assistant"
            if role and content:
                messages.append({"role": role, "content": content})
        return messages, meta

    if "instruction" in row and "output" in row:
        user = row.get("instruction", "")
        if row.get("input"):
            user = f"{user}\n{row.get('input')}"
        return [{"role": "user", "content": user}, {"role": "assistant", "content": row.get("output", "")}], meta

    if "prompt" in row and "response" in row:
        return [
            {"role": "user", "content": row.get("prompt", "")},
            {"role": "assistant", "content": row.get("response", "")},
        ], meta

    if "question" in row and "answer" in row:
        user = row.get("question", "")
        options = row.get("options")
        if isinstance(options, dict):
            user = (
                f"{user}\n"
                f"أ) {options.get('أ', '')}\n"
                f"ب) {options.get('ب', '')}\n"
                f"ج) {options.get('ج', '')}\n"
                f"د) {options.get('د', '')}"
            )
        return [{"role": "user", "content": user}, {"role": "assistant", "content": row.get("answer", "")}], meta

    if "user" in row and "assistant" in row:
        return [
            {"role": "user", "content": row.get("user", "")},
            {"role": "assistant", "content": row.get("assistant", "")},
        ], meta

    return [], meta


def _is_mcq(text: str) -> bool:
    return bool(MCQ_OPTION_PATTERN.search(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", default=[
        str(RAW_DIR / "instar.jsonl"),
        str(RAW_DIR / "cidar_ask_teacher.jsonl"),
        str(RAW_DIR / "synthetic_grammar.jsonl"),
    ])
    parser.add_argument("--out", default=str(INTERMEDIATE_DIR / "normalized.jsonl"))
    parser.add_argument("--min-arabic-ratio", type=float, default=0.85)
    parser.add_argument("--fasttext-model", default="")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
    dropped = 0
    ft_model = None

    with open(out_path, "w", encoding="utf-8") as fout:
        for input_path in args.inputs:
            path = Path(input_path)
            if not path.exists():
                print(f"Skipping missing input: {path}", file=sys.stderr)
                continue
            with open(path, "r", encoding="utf-8") as fin:
                for line in fin:
                    if args.max_rows and total >= args.max_rows:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        dropped += 1
                        continue

                    messages, meta = _extract_messages(record)
                    if not messages:
                        dropped += 1
                        continue

                    for m in messages:
                        m["content"] = _normalize_text(m.get("content", ""))

                    user_text = next((m["content"] for m in messages if m["role"] == "user"), "")
                    asst_text = next((m["content"] for m in messages if m["role"] == "assistant"), "")

                    if not user_text or not asst_text:
                        dropped += 1
                        continue

                    if args.fasttext_model:
                        if ft_model is None:
                            ft_model = _load_fasttext(args.fasttext_model)
                        if ft_model is None:
                            raise RuntimeError("fasttext not available; install or disable --fasttext-model")
                        label, _score = _predict_fasttext(ft_model, user_text)
                        if not label.endswith("__ar"):
                            dropped += 1
                            continue
                    else:
                        if _arabic_ratio(user_text) < args.min_arabic_ratio:
                            dropped += 1
                            continue

                    row_out = {
                        "messages": messages,
                        "is_mcq": _is_mcq(user_text),
                        "meta": meta,
                    }
                    fout.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                    kept += 1

    print(
        f"Processed {total:,} rows, kept {kept:,}, dropped {dropped:,}. "
        f"Output: {out_path}"
    )


if __name__ == "__main__":
    main()
