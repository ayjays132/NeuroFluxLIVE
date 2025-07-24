from __future__ import annotations

"""Dataset quality evaluation utilities."""

from typing import Any, Dict, Iterable, Optional, Tuple
from collections import defaultdict
import json
import argparse
import os

from datasets import Dataset, load_dataset


def score_record(
    record: Dict[str, Any],
    seen: Optional[set[str]] = None,
    numeric_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[float, list[str]]:
    """Score a single record for quality.

    Parameters
    ----------
    record:
        Dictionary representing a single data example.
    seen:
        Optional set of serialized records used to detect duplicates.
    numeric_ranges:
        Optional mapping of numeric field names to ``(min, max)`` tuples.

    Returns
    -------
    tuple
        ``(score, issues)`` where ``issues`` is a list of strings describing
        detected problems.
    """
    issues: list[str] = []
    score = 1.0

    if seen is not None:
        key = json.dumps(record, sort_keys=True)
        if key in seen:
            issues.append("duplicate")
            score -= 0.5
        else:
            seen.add(key)

    for col, val in record.items():
        if val is None or val == "" or val == []:
            issues.append(f"empty:{col}")
            score -= 0.1

    if numeric_ranges:
        for col, (lo, hi) in numeric_ranges.items():
            if col in record:
                val = record[col]
                if isinstance(val, (int, float)) and not (lo <= val <= hi):
                    issues.append(f"out_of_range:{col}")
                    score -= 0.2

    return max(score, 0.0), issues


def evaluate_dataset(
    dataset: Dataset,
    *,
    key_columns: Optional[Iterable[str]] = None,
    numeric_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """Evaluate an entire ``Dataset`` for duplicates and missing values."""
    keys = list(key_columns) if key_columns else list(dataset.column_names)
    all_cols = list(dataset.column_names)
    seen: set[tuple] = set()
    empty_counts: Dict[str, int] = defaultdict(int)
    oor_counts: Dict[str, int] = defaultdict(int)
    duplicates = 0

    for example in dataset:
        key = tuple(example.get(k) for k in keys)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)

        for col in all_cols:
            val = example.get(col)
            if val is None or val == "" or val == []:
                empty_counts[col] += 1
            if numeric_ranges and col in numeric_ranges:
                lo, hi = numeric_ranges[col]
                if isinstance(val, (int, float)) and not (lo <= val <= hi):
                    oor_counts[col] += 1

    return {
        "total_records": len(dataset),
        "duplicates": duplicates,
        "empty_fields": dict(empty_counts),
        "out_of_range": dict(oor_counts),
    }


def load_dataset_from_arg(path_or_name: str, split: str = "train") -> Dataset:
    """Load dataset from a local path or HF dataset name."""
    if os.path.exists(path_or_name):
        return load_dataset("json", data_files=path_or_name, split="train")
    return load_dataset(path_or_name, split=split)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check dataset quality")
    parser.add_argument("dataset")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    ds = load_dataset_from_arg(args.dataset, args.split)
    report = evaluate_dataset(ds)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
