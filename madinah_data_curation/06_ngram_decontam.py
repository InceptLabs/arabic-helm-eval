#!/usr/bin/env python3
"""N-gram decontamination against benchmark evaluation sets."""

import argparse
import dataclasses
import hashlib
import json
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
INTERMEDIATE_DIR = SCRIPT_DIR / "intermediate"
FINAL_DIR = SCRIPT_DIR / "final"
REPORTS_DIR = SCRIPT_DIR / "reports"

ARABIC_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
LATIN_RANGE = re.compile(r"[A-Za-z]")
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
    "\u060C": " ",
    "\u061B": " ",
    "\u061F": " ",
    "\u066B": " ",
    "\u066C": " ",
    "\u06D4": " ",
    "\u200F": "",
    "\u200E": "",
    "\u200B": "",
    "\u00A0": " ",
    "\uFEFF": "",
})


def _normalize(text: str) -> str:
    text = text.translate(ALEF_MAP)
    text = text.translate(YA_TA_MAP)
    text = text.translate(PUNCT_MAP)
    text = TASHKEEL.sub("", text)
    text = text.lower()
    text = WHITESPACE.sub(" ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    text = _normalize(text)
    return [tok for tok in text.split(" ") if tok]


def _ngrams(tokens: list[str], n: int):
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


def _hash_ngram(ngram: str) -> str:
    return hashlib.sha1(ngram.encode("utf-8")).hexdigest()


def _extract_texts(row: dict) -> list[str]:
    texts = []
    for _, value in row.items():
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, list):
            texts.extend([v for v in value if isinstance(v, str)])
        elif isinstance(value, dict):
            for v in value.values():
                if isinstance(v, str):
                    texts.append(v)
                elif isinstance(v, list):
                    texts.extend([vv for vv in v if isinstance(vv, str)])
    return [t for t in texts if t]


def _load_hf_texts(dataset_name: str, splits: list[str]) -> list[str]:
    from datasets import load_dataset

    texts = []
    for split in splits:
        try:
            ds = load_dataset(dataset_name, split=split, streaming=True)
        except Exception:
            continue
        for row in ds:
            texts.extend(_extract_texts(row))
    return texts


def _load_paths_texts(paths: list[str]) -> list[str]:
    texts = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    texts.extend(_extract_texts(row))
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, dict):
                        texts.extend(_extract_texts(row))
            elif isinstance(data, dict):
                texts.extend(_extract_texts(data))
    return texts


def _load_helm_texts(modules: list[str], attr: str) -> list[str]:
    import importlib

    texts = []
    for mod_path in modules:
        mod = importlib.import_module(mod_path)
        scenario = getattr(mod, attr, None)
        if scenario is None and hasattr(mod, "get_scenario"):
            scenario = mod.get_scenario()
        if scenario is None:
            raise RuntimeError(f"Could not find scenario in {mod_path} (attr={attr})")

        if hasattr(scenario, "get_instances"):
            instances = scenario.get_instances()
        elif hasattr(scenario, "get_instances_for_split"):
            instances = scenario.get_instances_for_split("test")
        else:
            raise RuntimeError(f"Scenario {mod_path} has no instance loader")

        for inst in instances:
            if dataclasses.is_dataclass(inst):
                row = dataclasses.asdict(inst)
            elif isinstance(inst, dict):
                row = inst
            else:
                row = inst.__dict__
            texts.extend(_extract_texts(row))
    return texts


def _build_ngram_set(texts: list[str], n: int) -> set[str]:
    grams = set()
    for text in texts:
        tokens = _tokenize(text)
        if len(tokens) < n:
            continue
        for ng in _ngrams(tokens, n):
            grams.add(_hash_ngram(ng))
    return grams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INTERMEDIATE_DIR / "deduped.jsonl"))
    parser.add_argument("--out", default=str(FINAL_DIR / "decontaminated.jsonl"))
    parser.add_argument("--benchmark-source", choices=["hf", "helm", "paths"], default="hf")
    parser.add_argument("--hf-madinah", default="MadinahQA")
    parser.add_argument("--hf-mmlu", default="arabic_mmlu")
    parser.add_argument("--hf-exams", default="arabic_exams")
    parser.add_argument("--hf-splits", default="test,validation")
    parser.add_argument("--helm-modules", nargs="*", default=[])
    parser.add_argument("--helm-attr", default="SCENARIO")
    parser.add_argument("--paths", nargs="*", default=[])
    parser.add_argument("--ngram-size", type=int, default=13)
    parser.add_argument("--min-overlap-ngrams", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.hf_splits.split(",") if s.strip()]

    benchmark_sets = {}
    if args.benchmark_source == "hf":
        benchmark_sets["madinahqa"] = _build_ngram_set(_load_hf_texts(args.hf_madinah, splits), args.ngram_size)
        benchmark_sets["arabic_mmlu"] = _build_ngram_set(_load_hf_texts(args.hf_mmlu, splits), args.ngram_size)
        benchmark_sets["arabic_exams"] = _build_ngram_set(_load_hf_texts(args.hf_exams, splits), args.ngram_size)
    elif args.benchmark_source == "helm":
        if not args.helm_modules:
            raise RuntimeError("--helm-modules is required for helm source")
        texts = _load_helm_texts(args.helm_modules, args.helm_attr)
        benchmark_sets["helm_combined"] = _build_ngram_set(texts, args.ngram_size)
    else:
        if not args.paths:
            raise RuntimeError("--paths is required for paths source")
        texts = _load_paths_texts(args.paths)
        benchmark_sets["paths_combined"] = _build_ngram_set(texts, args.ngram_size)

    report = {
        "benchmark_source": args.benchmark_source,
        "ngram_size": args.ngram_size,
        "min_overlap_ngrams": args.min_overlap_ngrams,
        "benchmarks": {k: len(v) for k, v in benchmark_sets.items()},
        "removed": {},
        "total_rows": 0,
    }

    for name, grams in benchmark_sets.items():
        if not grams:
            print(f"WARNING: no n-grams loaded for {name}. Check dataset names/splits.", file=sys.stderr)

    removed_total = 0
    kept_total = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if args.max_rows and idx >= args.max_rows:
                break
            line = line.strip()
            if not line:
                continue
            report["total_rows"] += 1
            row = json.loads(line)
            user_text = ""
            for m in row.get("messages", []):
                if m.get("role") == "user":
                    user_text = m.get("content", "")
                    break
            tokens = _tokenize(user_text)
            if len(tokens) < args.ngram_size:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept_total += 1
                continue

            row_ngrams = {_hash_ngram(ng) for ng in _ngrams(tokens, args.ngram_size)}
            hit_benchmarks = []
            for bname, bset in benchmark_sets.items():
                overlap = row_ngrams.intersection(bset)
                if len(overlap) >= args.min_overlap_ngrams:
                    hit_benchmarks.append(bname)
            if hit_benchmarks:
                removed_total += 1
                for b in hit_benchmarks:
                    report["removed"][b] = report["removed"].get(b, 0) + 1
                continue

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept_total += 1

    report["kept"] = kept_total
    report["removed_total"] = removed_total

    report_path = REPORTS_DIR / "06_decontam_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Decontamination complete. Kept {kept_total:,}, removed {removed_total:,}.")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
