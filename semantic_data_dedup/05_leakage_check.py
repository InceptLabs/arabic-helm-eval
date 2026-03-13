#!/usr/bin/env python3
"""Benchmark leakage / contamination check.

Streams the deduped JSONL and flags rows that look like they may have been
derived from one of the seven HELM Arabic benchmarks evaluated by this project:

  aratrust, arabic_mmlu, alghafa, arabic_exams, arabic_mmmlu, alrage, madinah_qa

Two detection layers:

  Layer 1 — Heuristic (regex / keyword pattern matching):
    1. Structural fingerprints: option formatting, system prompts, scoring rubrics
    2. Keyword anchors: benchmark-specific terms appearing in user or assistant text
    3. Direct benchmark mentions: rows that name the benchmark by name
    4. Answer-pattern leakage: assistant text that reproduces scoring/rubric language

  Layer 2 — Embedding cross-search (high precision):
    1. Load actual benchmark questions from HuggingFace
    2. Embed them with OpenAI text-embedding-3-small (same model as step 4)
    3. Cross-search against training embeddings using FAISS cosine similarity
    4. Flag training rows above a configurable threshold

Outputs:
  reports/05_leakage_report.json  — combined report from both layers
  reports/05_leakage_samples/     — calibration sample pairs at similarity bands
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "03_deduped.jsonl"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "reports"
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"

THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
WHITESPACE_PATTERN = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Embedding constants (must match step 4)
# ---------------------------------------------------------------------------

OPENAI_MODEL = "text-embedding-3-small"
OPENAI_DIM = 1536
OPENAI_BATCH_SIZE = 200
MAX_EMBED_CHARS = 5000

EMBEDDING_SIMILARITY_BANDS = [
    ("0.96+", 0.96, 1.01),
    ("0.93-0.96", 0.93, 0.96),
    ("0.90-0.93", 0.90, 0.93),
    ("0.88-0.90", 0.88, 0.90),
    ("0.85-0.88", 0.85, 0.88),
]
BAND_SAMPLE_SIZE = 30

# ---------------------------------------------------------------------------
# Benchmark registry — HuggingFace dataset IDs and field mappings
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY = {
    "arabic_mmlu": {
        "hf_id": "MBZUAI/ArabicMMLU",
        "splits": ["test"],
        "question_field": "Question",
        "option_fields": ["Option 1", "Option 2", "Option 3", "Option 4"],
        "description": "Arabic MMLU (14.5K questions across 40 subjects)",
    },
    "alghafa": {
        "hf_id": "OALL/AlGhafa-Arabic-LLM-Benchmark-Native",
        "splits": ["test"],
        "question_field": "query",
        "option_fields": None,  # varies by subset, use sol1/sol2/sol3/sol4
        "description": "AlGhafa NLU benchmark (sentiment, facts, dialects)",
    },
    "aratrust": {
        "hf_id": "asas-ai/AraTrust",
        "splits": ["test"],
        "question_field": "Question",
        "option_fields": ["A", "B", "C"],
        "description": "AraTrust safety/ethics (522 questions)",
    },
    "arabic_exams": {
        "hf_id": "OALL/Arabic_EXAMS",
        "splits": ["test"],
        "question_field": "question",
        "option_fields": ["A", "B", "C", "D"],
        "description": "Arabic school exam questions (562 questions)",
    },
    "arabic_mmmlu": {
        "hf_id": "MBZUAI/human_translated_arabic_mmlu",
        "splits": ["test"],
        "question_field": "question",
        "option_fields": None,  # uses "choices" list field
        "description": "Arabic Massive MMLU (14K questions, 57 subjects)",
    },
    "alrage": {
        "hf_id": "OALL/ALRAGE",
        "splits": ["train"],  # ALRAGE only has a train split
        "question_field": "question",
        "option_fields": None,  # generation benchmark, no options
        "description": "ALRAGE generation benchmark (2.1K questions)",
    },
    "madinah_qa": {
        "hf_id": "MBZUAI/MadinahQA",
        "splits": ["test"],
        "question_field": "Question",
        "option_fields": ["Option 1", "Option 2", "Option 3", "Option 4"],
        "description": "MadinahQA Arabic language (615 questions)",
    },
}

# ---------------------------------------------------------------------------
# Heuristic benchmark fingerprints (Layer 1)
# ---------------------------------------------------------------------------

BENCHMARK_PATTERNS = {
    "aratrust": {
        "description": "AraTrust safety/ethics benchmark",
        "user_patterns": [
            re.compile(r"(هل يمكنك|اكتب لي).*(إساءة|عنصري|تحرش|خطاب كراهية|عنف|تمييز)", re.DOTALL),
            re.compile(r"(toxicity|bias|offensive|harmful|hate speech)", re.IGNORECASE),
            re.compile(r"(خصوصية|بيانات شخصية|privacy|personal data)", re.IGNORECASE),
            re.compile(r"(أخلاقيات|ethical|morality).*(ذكاء اصطناعي|AI|artificial)", re.IGNORECASE | re.DOTALL),
        ],
        "asst_patterns": [
            re.compile(r"(لا أستطيع|لا يمكنني|أرفض).*(مساعدتك في|تقديم).*(ضار|عنصري|مسيء)", re.DOTALL),
        ],
    },
    "arabic_mmlu": {
        "description": "Arabic MMLU (translated MMLU subjects)",
        "user_patterns": [
            re.compile(
                r"(anatomy|biology|chemistry|physics|mathematics|algebra|history|psychology|"
                r"law|medicine|economics|sociology|philosophy|computer science|astronomy|"
                r"التشريح|الأحياء|الكيمياء|الفيزياء|الرياضيات|التاريخ|علم النفس|"
                r"القانون|الطب|الاقتصاد|علم الاجتماع|الفلسفة|علوم الحاسب|الفلك)",
                re.IGNORECASE,
            ),
        ],
        "asst_patterns": [],
    },
    "alghafa": {
        "description": "AlGhafa NLU benchmark (sentiment, facts, dialects)",
        "user_patterns": [
            re.compile(r"رأي (سلبي|إيجابي|محايد)", re.IGNORECASE),
            re.compile(r"(sentiment|إيجابي جدًا|سلبي جدًا)", re.IGNORECASE),
            re.compile(r"(صحيح|خطأ)\s*$", re.MULTILINE),
            re.compile(r"اختر رقمًا واحدًا فقط من الخيارات أعلاه", re.IGNORECASE),
        ],
        "asst_patterns": [],
    },
    "arabic_exams": {
        "description": "Arabic school exam questions",
        "user_patterns": [
            re.compile(r"(امتحان|اختبار|سؤال.*اختيار|مقرر|منهج|curriculum)", re.IGNORECASE),
            re.compile(r"(exam|test question)", re.IGNORECASE),
            re.compile(
                r"السؤال التالي هو سؤال متعدد الإختيارات",
                re.IGNORECASE,
            ),
        ],
        "asst_patterns": [],
    },
    "arabic_mmmlu": {
        "description": "Arabic Massive MMLU (57 subjects)",
        "user_patterns": [
            re.compile(
                r"(abstract_algebra|clinical_knowledge|college_biology|"
                r"econometrics|electrical_engineering|formal_logic|"
                r"jurisprudence|machine_learning|virology|world_religions)",
                re.IGNORECASE,
            ),
        ],
        "asst_patterns": [],
    },
    "alrage": {
        "description": "AlRAGE generation benchmark",
        "user_patterns": [
            re.compile(r"(اشرح بالتفصيل|وضح بالتفصيل|اكتب مقالا عن|explain in detail)", re.IGNORECASE),
            re.compile(r"(لخص النص التالي|summarize the following)", re.IGNORECASE),
        ],
        "asst_patterns": [],
    },
}

DIRECT_MENTION_PATTERN = re.compile(
    r"\b(aratrust|ara.?trust|arabic.?mmlu|alghafa|al.?ghafa|arabic.?exams|"
    r"arabic.?mmmlu|alrage|al.?rage|HELM|benchmark|leaderboard)\b",
    re.IGNORECASE,
)

RUBRIC_LEAK_PATTERNS = [
    re.compile(r"(الدرجة|score|الإجابة النموذجية|model answer|rubric)", re.IGNORECASE),
    re.compile(r"(الإجابة الصحيحة هي|الجواب الصحيح هو|The correct answer is)", re.IGNORECASE),
]

MCQ_OPTION_PATTERN = re.compile(r"\n\s*[أبجدهـA-E][\)\.]\s")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def extract_texts(row: dict) -> tuple:
    msgs = row.get("messages", [])
    user_text = ""
    asst_text = ""
    sys_text = ""
    for m in msgs:
        role = m.get("role", "")
        if role == "user":
            user_text = m.get("content", "")
        elif role == "assistant":
            asst_text = m.get("content", "")
        elif role == "system":
            sys_text = m.get("content", "")
    asst_clean = THINK_PATTERN.sub("", asst_text)
    return sys_text, user_text, asst_text, asst_clean


def _normalize_text(text: str) -> str:
    """Normalize text for embedding (same as step 4's normalize_prompt)."""
    text = THINK_PATTERN.sub("", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Embedding utilities (adapted from 04_semantic_dedup.py)
# ---------------------------------------------------------------------------

def _read_api_key() -> str:
    """Read OpenAI API key from credentials.conf or environment."""
    cred_path = SCRIPT_DIR.parent / "credentials.conf"
    if cred_path.exists():
        with open(cred_path) as f:
            for line in f:
                if line.strip().startswith("openaiApiKey:"):
                    key = line.split(":", 1)[1].strip().strip('"').strip("'")
                    if key and key != "lm-studio":
                        return key
    env_key = os.environ.get("OPENAI_API_KEY", "")
    if env_key:
        return env_key
    raise RuntimeError(
        "No OpenAI API key found. Set OPENAI_API_KEY or add to ../credentials.conf"
    )


def _truncate_text(text: str) -> str:
    if len(text) > MAX_EMBED_CHARS:
        return text[:MAX_EMBED_CHARS]
    return text


def _send_batch(client, batch: list[str], label: str, depth: int = 0) -> list:
    """Send a single embedding batch with retries. Splits on token-limit errors."""
    batch = [_truncate_text(t) for t in batch]
    for attempt in range(5):
        try:
            resp = client.embeddings.create(model=OPENAI_MODEL, input=batch)
            return [item.embedding for item in resp.data]
        except Exception as e:
            err_str = str(e)
            is_token_error = any(k in err_str for k in (
                "max_tokens", "maximum context length",
                "maximum request size", "maximum input length",
                "tokens per request",
            ))
            if is_token_error:
                if len(batch) > 1:
                    mid = len(batch) // 2
                    if depth < 3:
                        print(f"  {label}: splitting batch of {len(batch)}", file=sys.stderr)
                    left = _send_batch(client, batch[:mid], f"{label}L", depth + 1)
                    right = _send_batch(client, batch[mid:], f"{label}R", depth + 1)
                    return left + right
                else:
                    halved = batch[0][:len(batch[0]) // 2]
                    print(f"  {label}: single text too long ({len(batch[0])} chars), halving", file=sys.stderr)
                    batch = [halved]
                    continue
            if "rate" in err_str.lower() or "429" in err_str:
                wait = min(2 ** (attempt + 2), 60)
            else:
                wait = 2 ** attempt
            print(f"  {label}: API error (attempt {attempt+1}): {e}, retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Failed to embed {label} after retries")


# ---------------------------------------------------------------------------
# Benchmark loading from HuggingFace (Layer 2)
# ---------------------------------------------------------------------------

def _extract_question_text(row: dict, config: dict) -> str:
    """Extract question + options from a benchmark row for embedding."""
    qfield = config["question_field"]
    question = row.get(qfield, "")
    if not question:
        # Fallback: try all string fields
        for v in row.values():
            if isinstance(v, str) and len(v) > 20:
                question = v
                break

    parts = [question]

    # Try explicit option fields
    opt_fields = config.get("option_fields")
    if opt_fields:
        for i, field in enumerate(opt_fields):
            opt_val = row.get(field, "")
            if opt_val:
                label = chr(ord("A") + i)
                parts.append(f"{label}) {opt_val}")

    # Try "choices" list field (arabic_mmmlu uses this)
    if not opt_fields and "choices" in row:
        choices = row["choices"]
        if isinstance(choices, list):
            for i, c in enumerate(choices):
                if isinstance(c, str):
                    label = chr(ord("A") + i)
                    parts.append(f"{label}) {c}")

    # Try sol1/sol2/sol3/sol4 fields (alghafa uses this)
    if not opt_fields and "choices" not in row:
        for i in range(1, 6):
            sol = row.get(f"sol{i}", "")
            if sol:
                label = chr(ord("A") + i - 1)
                parts.append(f"{label}) {sol}")

    return "\n".join(parts)


def load_benchmark_questions(
    name: str,
    config: dict,
    splits: list[str] | None = None,
) -> list[dict]:
    """Load questions from a HuggingFace benchmark dataset.

    Returns list of dicts with keys: benchmark, text, raw_question, config_name, index.
    """
    from datasets import get_dataset_config_names, load_dataset

    hf_id = config["hf_id"]
    use_splits = splits or config.get("splits", ["test"])
    questions = []

    # Get all configs/subsets for the dataset
    try:
        configs = get_dataset_config_names(hf_id)
    except Exception:
        configs = [None]

    if not configs:
        configs = [None]

    for cfg in configs:
        for split in use_splits:
            try:
                kwargs = {"path": hf_id, "split": split, "streaming": True}
                if cfg is not None:
                    kwargs["name"] = cfg
                ds = load_dataset(**kwargs)
            except Exception as e:
                print(f"    Skipping {hf_id}/{cfg}/{split}: {e}", file=sys.stderr)
                continue

            for row_idx, row in enumerate(ds):
                raw_question = row.get(config["question_field"], "")
                full_text = _extract_question_text(row, config)
                normalized = _normalize_text(full_text)
                if len(normalized) < 10:
                    continue
                questions.append({
                    "benchmark": name,
                    "text": normalized,
                    "raw_question": raw_question[:500] if raw_question else normalized[:500],
                    "config_name": cfg,
                    "index": row_idx,
                })

    return questions


def load_all_benchmarks(
    benchmarks: list[str] | None = None,
    splits: list[str] | None = None,
) -> tuple[list[dict], dict]:
    """Load questions from all (or selected) benchmarks."""
    to_load = benchmarks or list(BENCHMARK_REGISTRY.keys())
    all_questions = []
    stats = {}

    for bname in to_load:
        if bname not in BENCHMARK_REGISTRY:
            print(f"  Warning: unknown benchmark '{bname}', skipping", file=sys.stderr)
            continue
        config = BENCHMARK_REGISTRY[bname]
        print(f"  Loading {bname} ({config['hf_id']})...", file=sys.stderr)
        questions = load_benchmark_questions(bname, config, splits)
        stats[bname] = {
            "description": config["description"],
            "questions_loaded": len(questions),
        }
        all_questions.extend(questions)
        print(f"    Loaded {len(questions):,} questions", file=sys.stderr)

    return all_questions, stats


# ---------------------------------------------------------------------------
# Embedding + FAISS cross-search (Layer 2)
# ---------------------------------------------------------------------------

def embed_benchmark_questions(
    texts: list[str],
    cache_path: Path,
    skip_embed: bool = False,
) -> "np.ndarray":
    """Embed benchmark question texts using OpenAI text-embedding-3-small.

    Writes to a numpy file at cache_path. Resumes from existing cache if skip_embed.
    Returns L2-normalized np.ndarray of shape (n, 1536).
    """
    import numpy as np

    total = len(texts)
    dim = OPENAI_DIM

    if skip_embed and cache_path.exists():
        print(f"  Loading cached benchmark embeddings from {cache_path}", file=sys.stderr)
        emb = np.load(cache_path)
        if emb.shape[0] == total:
            return emb
        print(f"  Cache shape mismatch ({emb.shape[0]} vs {total}), re-embedding", file=sys.stderr)

    from openai import OpenAI
    client = OpenAI(api_key=_read_api_key())

    all_vecs = []
    for start in range(0, total, OPENAI_BATCH_SIZE):
        batch = texts[start:start + OPENAI_BATCH_SIZE]
        vecs = _send_batch(client, batch, f"bench@{start}")
        all_vecs.extend(vecs)

        done = min(start + OPENAI_BATCH_SIZE, total)
        if done % 2000 < OPENAI_BATCH_SIZE or done == total:
            print(f"  Embedded {done:,}/{total:,} benchmark questions", file=sys.stderr)

    embeddings = np.array(all_vecs, dtype=np.float32)

    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    np.save(cache_path, embeddings)
    print(f"  Benchmark embeddings saved to {cache_path} ({embeddings.shape})", file=sys.stderr)
    return embeddings


def cross_search_faiss(
    training_emb: "np.ndarray",
    benchmark_emb: "np.ndarray",
    top_k: int = 5,
) -> tuple["np.ndarray", "np.ndarray"]:
    """For each benchmark question, find top-k most similar training rows.

    Builds FAISS IndexFlatIP from training embeddings, queries with benchmark embeddings.
    Returns (scores, indices) each of shape (n_bench, top_k).
    """
    import faiss
    import numpy as np

    n_train, d = training_emb.shape
    n_bench = benchmark_emb.shape[0]
    print(f"  Building FAISS index from {n_train:,} training vectors ({d}d)...", file=sys.stderr)

    index = faiss.IndexFlatIP(d)

    # Add training vectors in chunks to keep memory manageable
    chunk_size = 50000
    train_contiguous = np.ascontiguousarray(training_emb, dtype=np.float32)
    for start in range(0, n_train, chunk_size):
        end = min(start + chunk_size, n_train)
        index.add(train_contiguous[start:end])

    print(f"  Searching {n_bench:,} benchmark questions (top-{top_k})...", file=sys.stderr)

    batch_size = 1000
    all_scores = []
    all_indices = []
    for start in range(0, n_bench, batch_size):
        end = min(start + batch_size, n_bench)
        query = np.ascontiguousarray(benchmark_emb[start:end], dtype=np.float32)
        scores, indices = index.search(query, top_k)
        all_scores.append(scores)
        all_indices.append(indices)

    return np.vstack(all_scores), np.vstack(all_indices)


def find_contaminated(
    questions: list[dict],
    scores: "np.ndarray",
    indices: "np.ndarray",
    threshold: float,
) -> tuple[list[dict], set[int]]:
    """Extract (benchmark_question, training_row) pairs above threshold.

    Returns (matches, flagged_training_indices).
    """
    matches = []
    flagged = set()

    for qi in range(len(questions)):
        best_score = float(scores[qi, 0])
        if best_score < threshold:
            continue
        for ki in range(scores.shape[1]):
            score = float(scores[qi, ki])
            if score < threshold:
                break
            train_idx = int(indices[qi, ki])
            flagged.add(train_idx)
            matches.append({
                "benchmark": questions[qi]["benchmark"],
                "bench_config": questions[qi].get("config_name"),
                "bench_idx": questions[qi]["index"],
                "train_idx": train_idx,
                "score": round(score, 4),
                "bench_preview": questions[qi]["raw_question"][:300],
            })

    return matches, flagged


def build_embedding_histogram(scores: "np.ndarray") -> dict:
    """Build histogram of best-match scores for each benchmark question."""
    import numpy as np

    best_scores = scores[:, 0]  # top-1 match per benchmark question
    bins = [0.0, 0.50, 0.60, 0.70, 0.80, 0.85, 0.88, 0.90, 0.93, 0.96, 0.99, 1.01]
    counts, _ = np.histogram(best_scores, bins=bins)

    histogram = {}
    for i in range(len(bins) - 1):
        label = f"{bins[i]:.2f}-{bins[i+1]:.2f}"
        histogram[label] = int(counts[i])

    return {
        "histogram": histogram,
        "stats": {
            "mean": round(float(best_scores.mean()), 4),
            "median": round(float(np.median(best_scores)), 4),
            "p90": round(float(np.percentile(best_scores, 90)), 4),
            "p95": round(float(np.percentile(best_scores, 95)), 4),
            "p99": round(float(np.percentile(best_scores, 99)), 4),
            "max": round(float(best_scores.max()), 4),
        },
    }


def collect_embedding_samples(
    questions: list[dict],
    scores: "np.ndarray",
    indices: "np.ndarray",
    training_texts: list[str],
) -> dict:
    """Collect sample match pairs at different similarity bands for manual review."""
    band_pairs = {label: [] for label, _, _ in EMBEDDING_SIMILARITY_BANDS}

    for qi in range(len(questions)):
        best_score = float(scores[qi, 0])
        best_train_idx = int(indices[qi, 0])

        for label, lo, hi in EMBEDDING_SIMILARITY_BANDS:
            if lo <= best_score < hi and len(band_pairs[label]) < BAND_SAMPLE_SIZE:
                train_text = ""
                if 0 <= best_train_idx < len(training_texts):
                    train_text = training_texts[best_train_idx]
                band_pairs[label].append({
                    "score": round(best_score, 4),
                    "benchmark": questions[qi]["benchmark"],
                    "bench_question": questions[qi]["raw_question"][:500],
                    "train_text": train_text[:500],
                    "train_idx": best_train_idx,
                })
                break

    return band_pairs


def _load_training_texts(input_file: Path) -> list[str]:
    """Load normalized user prompts from the deduped JSONL (for previews)."""
    texts = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                for m in row.get("messages", []):
                    if m.get("role") == "user":
                        texts.append(_normalize_text(m.get("content", "")))
                        break
                else:
                    texts.append("")
            except json.JSONDecodeError:
                texts.append("")
    return texts


# ---------------------------------------------------------------------------
# Main check_leakage function
# ---------------------------------------------------------------------------

def check_leakage(
    input_file=None,
    report_dir=None,
    data_dir=None,
    embedding_leakage=True,
    leakage_threshold=0.88,
    skip_embed=False,
    benchmarks=None,
    splits=None,
):
    INPUT_FILE = Path(input_file) if input_file else DEFAULT_INPUT
    REPORT_DIR = Path(report_dir) if report_dir else DEFAULT_REPORT_DIR
    DATA_DIR = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # Layer 1: Heuristic / regex pattern matching
    # ===================================================================
    print("Layer 1: Heuristic pattern matching...", file=sys.stderr)

    benchmark_flags = {name: [] for name in BENCHMARK_PATTERNS}
    direct_mention_rows = []
    rubric_leak_rows = []
    mcq_option_count = 0
    total = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            sys_text, user_text, asst_text, asst_clean = extract_texts(row)
            combined = user_text + " " + asst_clean

            if MCQ_OPTION_PATTERN.search(user_text):
                mcq_option_count += 1

            for bname, bconf in BENCHMARK_PATTERNS.items():
                matched = False
                for pat in bconf["user_patterns"]:
                    if pat.search(user_text):
                        matched = True
                        break
                if not matched:
                    for pat in bconf.get("asst_patterns", []):
                        if pat.search(asst_clean):
                            matched = True
                            break
                if matched:
                    benchmark_flags[bname].append(idx)

            if DIRECT_MENTION_PATTERN.search(combined):
                direct_mention_rows.append(idx)

            for pat in RUBRIC_LEAK_PATTERNS:
                if pat.search(asst_clean):
                    rubric_leak_rows.append(idx)
                    break

            if total % 50000 == 0:
                print(f"  ... checked {total:,} rows", file=sys.stderr)

    # Build heuristic report section
    report = {
        "total_rows": total,
        "mcq_option_rows": mcq_option_count,
        "direct_benchmark_mentions": {
            "count": len(direct_mention_rows),
            "sample_indices": direct_mention_rows[:100],
        },
        "rubric_leak_rows": {
            "count": len(rubric_leak_rows),
            "sample_indices": rubric_leak_rows[:100],
        },
        "benchmark_flags": {},
    }

    all_flagged = set()
    for bname in BENCHMARK_PATTERNS:
        flags = benchmark_flags[bname]
        all_flagged.update(flags)
        report["benchmark_flags"][bname] = {
            "description": BENCHMARK_PATTERNS[bname]["description"],
            "flagged_count": len(flags),
            "flagged_fraction": round(len(flags) / max(total, 1), 4),
            "sample_indices": flags[:50],
        }

    report["summary"] = {
        "total_flagged_unique": len(all_flagged),
        "total_flagged_fraction": round(len(all_flagged) / max(total, 1), 4),
        "note": (
            "These are heuristic pattern matches, not confirmed contamination. "
            "Many flagged rows are legitimately about these topics. "
            "Direct benchmark mentions and rubric leaks are higher-confidence signals."
        ),
    }

    # Collect sample rows for the most-flagged benchmarks
    sample_rows = {}
    top_benchmarks = sorted(
        report["benchmark_flags"].items(),
        key=lambda x: x[1]["flagged_count"],
        reverse=True,
    )[:3]

    needed = set()
    for bname, info in top_benchmarks:
        needed.update(info["sample_indices"][:10])

    if needed:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx in needed:
                    try:
                        row = json.loads(line.strip())
                        _, user_text, _, _ = extract_texts(row)
                        sample_rows[idx] = user_text[:500]
                    except json.JSONDecodeError:
                        pass
                if len(sample_rows) == len(needed):
                    break

    for bname, info in top_benchmarks:
        previews = []
        for i in info["sample_indices"][:10]:
            if i in sample_rows:
                previews.append({"row_index": i, "user_preview": sample_rows[i]})
        report["benchmark_flags"][bname]["sample_previews"] = previews

    # Print Layer 1 summary
    print("\n" + "=" * 60)
    print("LAYER 1: HEURISTIC LEAKAGE CHECK")
    print("=" * 60)
    print(f"Total rows:           {total:,}")
    print(f"MCQ-formatted rows:   {mcq_option_count:,}")
    print(f"Direct mentions:      {len(direct_mention_rows):,}")
    print(f"Rubric leak rows:     {len(rubric_leak_rows):,}")
    print(f"\nPer-benchmark flags:")
    for bname, info in report["benchmark_flags"].items():
        count = info["flagged_count"]
        pct = info["flagged_fraction"] * 100
        print(f"  {bname:20s}: {count:>7,} ({pct:.1f}%)")
    print(f"\nTotal unique flagged: {len(all_flagged):,} "
          f"({100*len(all_flagged)/max(total,1):.1f}%)")

    # ===================================================================
    # Layer 2: Embedding cross-search
    # ===================================================================
    if embedding_leakage:
        import numpy as np

        print("\n" + "=" * 60)
        print("LAYER 2: EMBEDDING CROSS-SEARCH")
        print("=" * 60)

        # Check that training embeddings exist
        train_emb_path = DATA_DIR / "04_embeddings.npy"
        if not train_emb_path.exists():
            print(f"  WARNING: Training embeddings not found at {train_emb_path}", file=sys.stderr)
            print(f"  Run step 4 (semantic dedup) first to generate embeddings.", file=sys.stderr)
            print(f"  Skipping embedding leakage check.", file=sys.stderr)
            report["embedding_leakage"] = {
                "skipped": True,
                "reason": "Training embeddings not found. Run step 4 first.",
            }
        else:
            # Load training embeddings (memmap)
            print(f"Loading training embeddings from {train_emb_path}...", file=sys.stderr)
            training_emb = np.memmap(
                train_emb_path, dtype=np.float32, mode="r",
                shape=(total, OPENAI_DIM),
            )

            # Load training texts for previews
            print("Loading training texts for previews...", file=sys.stderr)
            training_texts = _load_training_texts(INPUT_FILE)

            # Load benchmark questions from HuggingFace
            print("\nLoading benchmark questions from HuggingFace...", file=sys.stderr)
            questions, bench_stats = load_all_benchmarks(benchmarks, splits)
            print(f"\nTotal benchmark questions loaded: {len(questions):,}", file=sys.stderr)

            if not questions:
                print("  No benchmark questions loaded. Skipping embedding check.", file=sys.stderr)
                report["embedding_leakage"] = {
                    "skipped": True,
                    "reason": "No benchmark questions could be loaded from HuggingFace.",
                }
            else:
                # Embed benchmark questions
                bench_texts = [q["text"] for q in questions]
                bench_cache = DATA_DIR / "05_benchmark_embeddings.npy"
                print(f"\nEmbedding {len(bench_texts):,} benchmark questions...", file=sys.stderr)
                bench_emb = embed_benchmark_questions(bench_texts, bench_cache, skip_embed)

                # Cross-search
                print("\nCross-searching benchmark vs training...", file=sys.stderr)
                top_k = 5
                scores, indices = cross_search_faiss(training_emb, bench_emb, top_k)

                # Find contaminated pairs
                print(f"Finding matches above threshold={leakage_threshold}...", file=sys.stderr)
                matches, flagged_indices = find_contaminated(
                    questions, scores, indices, leakage_threshold,
                )

                # Build histogram
                histogram = build_embedding_histogram(scores)

                # Collect calibration samples
                print("Collecting calibration samples...", file=sys.stderr)
                band_samples = collect_embedding_samples(
                    questions, scores, indices, training_texts,
                )

                # Write calibration samples
                samples_dir = REPORT_DIR / "05_leakage_samples"
                samples_dir.mkdir(parents=True, exist_ok=True)
                for label, pairs in band_samples.items():
                    out_path = samples_dir / f"band_{label.replace('+', 'plus')}.jsonl"
                    with open(out_path, "w", encoding="utf-8") as f:
                        for p in pairs:
                            f.write(json.dumps(p, ensure_ascii=False) + "\n")
                    print(f"  Band {label}: {len(pairs)} sample pairs", file=sys.stderr)

                # Build per-benchmark stats
                per_bench = {}
                bench_train_sets = defaultdict(set)
                bench_match_counts = Counter()
                for m in matches:
                    bname = m["benchmark"]
                    bench_train_sets[bname].add(m["train_idx"])
                    bench_match_counts[bname] += 1

                for bname, bstats in bench_stats.items():
                    train_flagged = bench_train_sets.get(bname, set())
                    # Get top-10 matches for this benchmark
                    top_matches = sorted(
                        [m for m in matches if m["benchmark"] == bname],
                        key=lambda x: x["score"],
                        reverse=True,
                    )[:10]
                    # Add training text previews to top matches
                    for m in top_matches:
                        tidx = m["train_idx"]
                        if 0 <= tidx < len(training_texts):
                            m["train_preview"] = training_texts[tidx][:300]

                    per_bench[bname] = {
                        "description": bstats["description"],
                        "questions_loaded": bstats["questions_loaded"],
                        "questions_with_match": bench_match_counts.get(bname, 0),
                        "training_rows_flagged": len(train_flagged),
                        "top_matches": top_matches,
                    }

                report["embedding_leakage"] = {
                    "threshold": leakage_threshold,
                    "embedding_model": OPENAI_MODEL,
                    "benchmark_questions_total": len(questions),
                    "contaminated_training_rows": len(flagged_indices),
                    "contaminated_fraction": round(
                        len(flagged_indices) / max(total, 1), 4
                    ),
                    "flagged_training_indices": sorted(flagged_indices),
                    "per_benchmark": per_bench,
                    "score_histogram": histogram,
                }

                # Print Layer 2 summary
                print(f"\n" + "-" * 40)
                print(f"Embedding leakage summary:")
                print(f"  Benchmark questions:       {len(questions):,}")
                print(f"  Threshold:                 {leakage_threshold}")
                print(f"  Contaminated train rows:   {len(flagged_indices):,} "
                      f"({100*len(flagged_indices)/max(total,1):.2f}%)")
                print(f"  Total match pairs:         {len(matches):,}")
                print(f"\n  Per-benchmark:")
                for bname, info in per_bench.items():
                    print(f"    {bname:20s}: {info['training_rows_flagged']:>6,} training rows "
                          f"(from {info['questions_loaded']:,} questions)")
                print(f"\n  Score stats: {histogram['stats']}")
    else:
        report["embedding_leakage"] = {
            "skipped": True,
            "reason": "Embedding leakage check disabled via --no-embedding-leakage.",
        }

    # ===================================================================
    # Write combined report
    # ===================================================================
    report_path = REPORT_DIR / "05_leakage_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark leakage / contamination check (heuristic + embedding)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Detection layers:
  Layer 1  Heuristic regex/keyword matching (always runs)
  Layer 2  Embedding cross-search against actual benchmark questions (optional)

Examples:
  python 05_leakage_check.py
  python 05_leakage_check.py --no-embedding-leakage
  python 05_leakage_check.py --benchmarks madinah_qa --leakage-threshold 0.90
  python 05_leakage_check.py --skip-embed  # reuse cached benchmark embeddings
""",
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="Path to deduped JSONL file (default: %(default)s)")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR),
                        help="Directory for reports (default: %(default)s)")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="Directory for data files (default: %(default)s)")

    emb_group = parser.add_argument_group("Embedding leakage (Layer 2)")
    emb_group.add_argument("--no-embedding-leakage", action="store_true",
                           help="Skip embedding-based leakage check (run only heuristic)")
    emb_group.add_argument("--leakage-threshold", type=float, default=0.88,
                           help="Cosine similarity threshold for flagging (default: %(default)s)")
    emb_group.add_argument("--skip-embed", action="store_true",
                           help="Reuse cached benchmark embeddings from data/05_benchmark_embeddings.npy")
    emb_group.add_argument("--benchmarks", nargs="*", default=None,
                           help="Specific benchmarks to check (default: all 7)")
    emb_group.add_argument("--splits", nargs="*", default=None,
                           help="HF splits to load (default: per-benchmark defaults)")

    args = parser.parse_args()
    check_leakage(
        input_file=args.input,
        report_dir=args.report_dir,
        data_dir=args.data_dir,
        embedding_leakage=not args.no_embedding_leakage,
        leakage_threshold=args.leakage_threshold,
        skip_embed=args.skip_embed,
        benchmarks=args.benchmarks,
        splits=args.splits,
    )
