#!/usr/bin/env python3
"""Semantic near-duplicate removal using embeddings + FAISS."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from typing import Optional


SCRIPT_DIR = Path(__file__).parent
INTERMEDIATE_DIR = SCRIPT_DIR / "intermediate"
REPORTS_DIR = SCRIPT_DIR / "reports"
CREDS_PATH = SCRIPT_DIR.parent / "credentials.conf"

OPENAI_MODEL = "text-embedding-3-small"
LOCAL_MODEL = "intfloat/multilingual-e5-base"


def _load_rows(path: Path):
    rows = []
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            user_text = ""
            for m in row.get("messages", []):
                if m.get("role") == "user":
                    user_text = m.get("content", "")
                    break
            texts.append(user_text)
    return rows, texts


def _embed_openai(texts: list[str], api_key: str, api_base: Optional[str]):
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=api_base)
    vectors = []
    batch_size = 500
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        resp = client.embeddings.create(model=OPENAI_MODEL, input=batch)
        vectors.extend([item.embedding for item in resp.data])
        print(f"  Embedded {min(start + batch_size, len(texts)):,}/{len(texts):,}", file=sys.stderr)
    return np.array(vectors, dtype=np.float32)


def _read_key_from_credentials(api_base: str) -> Optional[str]:
    if not CREDS_PATH.exists():
        return None
    key_name = "openaiApiKey"
    if "fireworks.ai" in api_base:
        key_name = "fireworksApiKey"
    with open(CREDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith(f"{key_name}:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
    return None


def _embed_local(texts: list[str]):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(LOCAL_MODEL)
    embeddings = model.encode(
        [f"query: {t}" for t in texts],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def _find_duplicates(embeddings: np.ndarray, threshold: float, k: int):
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    remove = set()
    clusters = []
    for i in range(embeddings.shape[0]):
        if i in remove:
            continue
        D, I = index.search(embeddings[i : i + 1], k)
        neighbors = []
        for score, idx in zip(D[0], I[0]):
            if idx == i:
                continue
            if score >= threshold:
                neighbors.append(idx)
        if neighbors:
            for idx in neighbors:
                remove.add(idx)
            clusters.append({"representative": i, "removed": neighbors})
        if i % 50000 == 0 and i > 0:
            print(f"  Processed {i:,} rows", file=sys.stderr)
    return remove, clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INTERMEDIATE_DIR / "exact_deduped.jsonl"))
    parser.add_argument("--out", default=str(INTERMEDIATE_DIR / "deduped.jsonl"))
    parser.add_argument("--backend", choices=["local", "openai"], default="local")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--openai-key", default="")
    parser.add_argument("--openai-base", default="https://api.openai.com/v1")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rows, texts = _load_rows(Path(args.input))
    if not rows:
        print("No rows to process.", file=sys.stderr)
        return

    print("Embedding rows...", file=sys.stderr)
    if args.backend == "openai":
        api_key = args.openai_key.strip()
        if not api_key:
            api_key = _read_key_from_credentials(args.openai_base) or ""
        if not api_key:
            raise RuntimeError("--openai-key not provided and not found in credentials.conf")
        embeddings = _embed_openai(texts, api_key, args.openai_base or None)
        embeddings = _normalize(embeddings)
    else:
        embeddings = _embed_local(texts)

    print("Finding near-duplicates...", file=sys.stderr)
    remove_indices, clusters = _find_duplicates(embeddings, args.threshold, args.k)

    out_path = Path(args.out)
    kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            if idx in remove_indices:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    cluster_path = REPORTS_DIR / "05_semantic_clusters.json"
    with open(cluster_path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)

    report_path = REPORTS_DIR / "05_semantic_dedup_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_rows": len(rows),
                "removed": len(remove_indices),
                "kept": kept,
                "threshold": args.threshold,
                "backend": args.backend,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Semantic dedup complete: kept {kept:,} / {len(rows):,}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
