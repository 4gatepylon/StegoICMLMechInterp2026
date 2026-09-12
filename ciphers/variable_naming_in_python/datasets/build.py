#!/usr/bin/env python3
"""
Combine multiple HuggingFace code-problem datasets into one canonical format.

Schema (all rows):
  question                 str
  answer                   str   (first available solution, or "")
  answers                  str   (JSON list[str] — all solutions collected across sources)
  expected_inputs_outputs  str   (JSON dict | null)
  metadata                 str   (JSON dict)

Deduplication: by question text, case-insensitive (.lower().strip()).
When a duplicate is found, its answers are merged into the first occurrence.

Sources:
  BAAI/TACO, codeparrot/apps, deepmind/code_contests,
  greengerong/leetcode, nvidia/OpenCodeReasoning

Usage:
  python datasets/build.py --output ./my_dataset
  python datasets/build.py --output hf --hf-args repo_id=myuser/dataset private=true

TODO(hadriano): please test this, read this, etc...
"""

import argparse
import json

from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import HfApi

# ── Helpers ────────────────────────────────────────────────────────────────────


def safe_json_loads(s):
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


def make_entry(question, answers, io, metadata):
    question = (question or "").strip()
    if not question:
        return None
    return {
        "question": question,
        "answer": answers[0] if answers else "",
        "answers": answers,
        "expected_inputs_outputs": io if isinstance(io, dict) else None,
        "metadata": metadata,
    }


# ── Per-source converters ─────────────────────────────────────────────────────


def _convert_with_solutions_column(row, source_dataset, extra_meta):
    """Shared logic for BAAI/TACO and codeparrot/apps (same column names)."""
    answers = safe_json_loads(row.get("solutions", "[]")) or []
    io = safe_json_loads(row.get("input_output", "null"))
    return make_entry(
        row.get("question"),
        answers,
        io,
        {"source_dataset": source_dataset, **extra_meta},
    )


def convert_baai_taco(row):
    return _convert_with_solutions_column(
        row,
        "BAAI/TACO",
        {
            "difficulty": row.get("difficulty"),
            "url": row.get("url"),
            "tags": row.get("tags"),
            "source": row.get("source"),
            "starter_code": row.get("starter_code"),
        },
    )


def convert_codeparrot_apps(row):
    return _convert_with_solutions_column(
        row,
        "codeparrot/apps",
        {
            "difficulty": row.get("difficulty"),
            "url": row.get("url"),
            "problem_id": row.get("problem_id"),
        },
    )


def convert_deepmind_code_contests(row):
    solutions = row.get("solutions") or {}
    answers = [
        s
        for s, lang in zip(
            solutions.get("solution", []),
            solutions.get("language", []),
        )
        if lang in (1, 3)  # PYTHON=1, PYTHON3=3
    ]
    tests = row.get("public_tests") or {}
    io = {"inputs": tests.get("input", []), "outputs": tests.get("output", [])}
    return make_entry(
        row.get("description"),
        answers,
        io,
        {
            "source_dataset": "deepmind/code_contests",
            "name": row.get("name"),
            "difficulty": row.get("difficulty"),
            "cf_tags": row.get("cf_tags"),
            "cf_rating": row.get("cf_rating"),
            "cf_contest_id": row.get("cf_contest_id"),
            "cf_index": row.get("cf_index"),
            "source": row.get("source"),
        },
    )


def convert_greengerong_leetcode(row):
    python_sol = (row.get("python") or "").strip()
    answers = [python_sol] if python_sol else []
    return make_entry(
        row.get("content"),
        answers,
        None,
        {
            "source_dataset": "greengerong/leetcode",
            "id": row.get("id"),
            "slug": row.get("slug"),
            "title": row.get("title"),
            "difficulty": row.get("difficulty"),
        },
    )


def convert_nvidia_opencodereasoning(row):
    solution = (row.get("solution") or "").strip()
    answers = [solution] if solution else []
    return make_entry(
        row.get("input"),
        answers,
        None,
        {
            "source_dataset": "nvidia/OpenCodeReasoning",
            "source": row.get("source"),
            "difficulty": row.get("difficulty"),
            "dataset": row.get("dataset"),
        },
    )


# ── Source registry ────────────────────────────────────────────────────────────

SOURCES = [
    {"name": "BAAI/TACO", "splits": ["train", "test"], "convert": convert_baai_taco},
    {
        "name": "codeparrot/apps",
        "splits": ["train", "test"],
        "convert": convert_codeparrot_apps,
    },
    {
        "name": "deepmind/code_contests",
        "splits": ["train", "test", "valid"],
        "convert": convert_deepmind_code_contests,
        "drop_columns": ["time_limit"],
    },
    {
        "name": "greengerong/leetcode",
        "splits": ["train"],
        "convert": convert_greengerong_leetcode,
    },
    {
        "name": "nvidia/OpenCodeReasoning",
        "splits": ["train"],
        "convert": convert_nvidia_opencodereasoning,
        "config": "split_0",
    },
]


def load_source(spec):
    config = spec.get("config")
    drop = spec.get("drop_columns", [])
    parts = []
    for split in spec["splits"]:
        kwargs = {"split": split, "trust_remote_code": True}
        ds = load_dataset(spec["name"], config, **kwargs) if config else load_dataset(spec["name"], **kwargs)
        if drop:
            ds = ds.remove_columns([c for c in drop if c in ds.column_names])
        parts.append(ds)
    return concatenate_datasets(parts) if len(parts) > 1 else parts[0]


# ── Build pipeline ─────────────────────────────────────────────────────────────


def build_dataset(sources=None) -> Dataset:
    sources = sources or SOURCES
    by_key = {}
    key_order = []

    for spec in sources:
        name = spec["name"]
        convert = spec["convert"]
        print(f"[{name}] loading...")
        raw = load_source(spec)
        print(f"[{name}] {len(raw)} rows")

        n_new = 0
        for row in raw:
            entry = convert(row)
            if entry is None:
                continue
            key = entry["question"].lower().strip()
            if key in by_key:
                by_key[key]["answers"].extend(entry["answers"])
            else:
                by_key[key] = entry
                key_order.append(key)
                n_new += 1
        print(f"[{name}] {n_new} new unique questions")

    rows = []
    for key in key_order:
        entry = by_key[key]
        seen_ans = set()
        unique_answers = []
        for a in entry["answers"]:
            if a and a not in seen_ans:
                seen_ans.add(a)
                unique_answers.append(a)
        entry["answers"] = json.dumps(unique_answers)
        entry["expected_inputs_outputs"] = json.dumps(entry["expected_inputs_outputs"])
        entry["metadata"] = json.dumps(entry["metadata"])
        rows.append(entry)

    print(f"\nTotal: {len(rows)} unique questions")
    return Dataset.from_list(rows)


# ── Output ─────────────────────────────────────────────────────────────────────


def parse_kv_args(raw_args):
    result = {}
    for item in raw_args or []:
        if "=" not in item:
            raise ValueError(f"Expected key=value, got: {item!r}")
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        if v.lower() in ("true", "1", "yes"):
            v = True
        elif v.lower() in ("false", "0", "no"):
            v = False
        result[k] = v
    return result


def save_dataset(ds, output, hf_args_raw):
    if output in ("hf", "huggingface"):
        hf_args = parse_kv_args(hf_args_raw)
        repo_id = hf_args.pop("repo_id", None)
        if not repo_id:
            raise ValueError("--hf-args must include repo_id=<user/dataset-name>")
        api = HfApi()
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        ds.push_to_hub(repo_id, **hf_args)
        print(f"Uploaded to https://huggingface.co/datasets/{repo_id}")
    else:
        ds.save_to_disk(output)
        print(f"Saved to {output}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default="combined_code_dataset",
        help="Local path, or 'hf'/'huggingface' to upload to HuggingFace Hub",
    )
    parser.add_argument(
        "--hf-args",
        nargs="*",
        default=[],
        help="key=value pairs for HF upload (e.g. repo_id=user/dataset private=true)",
    )
    args = parser.parse_args()
    ds = build_dataset()
    save_dataset(ds, args.output, args.hf_args)
