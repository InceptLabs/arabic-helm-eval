#!/usr/bin/env python3
"""Run the Madinah data prep pipeline end-to-end.

Defaults: skips step 4 (format), as requested.
Provides a small set of high-signal overrides.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent
PY = sys.executable


def _run(script: str, args_list: list[str]):
    cmd = [PY, str(ROOT / script), *args_list]
    print(">>", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-format", action="store_true", help="Run step 4 (format)")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-synth", action="store_true")
    parser.add_argument("--skip-normalize", action="store_true")
    parser.add_argument("--skip-curriculum", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--skip-sample", action="store_true")
    # High-signal overrides (keep minimal)
    parser.add_argument("--fetch-skip-instar", action="store_true")
    parser.add_argument("--fetch-allow-cidar-fallback", action="store_true")
    parser.add_argument("--synth-model", default="")
    parser.add_argument("--synth-api-base", default="")
    parser.add_argument("--synth-max-examples", type=int, default=None)
    parser.add_argument("--normalize-min-arabic-ratio", type=float, default=None)
    args = parser.parse_args()

    if not args.skip_fetch:
        fetch_args = []
        if args.fetch_allow_cidar_fallback:
            fetch_args.append("--allow-cidar-fallback")
        if args.fetch_skip_instar:
            fetch_args.append("--skip-instar")
        _run("00_fetch_datasets.py", fetch_args)

    if not args.skip_synth:
        if not args.synth_model:
            raise SystemExit("Missing --synth-model (required unless --skip-synth).")
        synth_args = ["--model", args.synth_model]
        if args.synth_api_base:
            synth_args += ["--api-base", args.synth_api_base]
        if args.synth_max_examples is not None:
            synth_args += ["--max-examples", str(args.synth_max_examples)]
        _run("01_generate_synthetic_grammar.py", synth_args)

    if not args.skip_normalize:
        normalize_args = []
        if args.normalize_min_arabic_ratio is not None:
            normalize_args += ["--min-arabic-ratio", str(args.normalize_min_arabic_ratio)]
        _run("02_normalize_filter.py", normalize_args)

    last_output = ROOT / "intermediate" / "normalized.jsonl"

    if args.run_format:
        format_args = []
        _run("03_format_sharegpt.py", format_args)
        last_output = ROOT / "intermediate" / "sharegpt.jsonl"

    if not args.skip_curriculum:
        _run("07_build_curriculum.py", ["--input", str(last_output)])

    if not args.skip_profile:
        _run("08_profile_report.py", ["--input", str(last_output)])

    if not args.skip_sample:
        _run("09_sample_audit.py", ["--input", str(last_output)])


if __name__ == "__main__":
    main()
