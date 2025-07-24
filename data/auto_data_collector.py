"""Utilities for automatically fetching datasets from the Hugging Face Hub."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

# Importing `datasets` lazily so the module can be imported without the
# dependency installed. Functions will raise an informative error if the
# library is missing.
from importlib import import_module


def _datasets():
    try:
        return import_module("datasets")
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' library is required for AutoDataCollector."
        ) from exc


logger = logging.getLogger(__name__)


def search_hf_datasets(query: str, limit: int = 10) -> List[str]:
    """Return dataset names containing the query string."""
    query = query.lower()
    ds_mod = _datasets()
    names = [name for name in ds_mod.list_datasets() if query in name.lower()]
    return names[:limit]


def download_dataset(name: str, split: str, output_dir: str | Path) -> Path:
    """Download a split from the hub and save it as JSONL.

    Args:
        name: Dataset identifier on Hugging Face.
        split: Which split to download.
        output_dir: Directory to save the JSONL file.

    Returns:
        Path to the saved file.
    """
    ds_mod = _datasets()
    ds: Dataset = ds_mod.load_dataset(name, split=split)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name.replace('/', '_')}_{split}.jsonl"
    ds.to_json(str(path))
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download datasets from HF Hub")
    parser.add_argument("--query", required=True, help="Search term")
    parser.add_argument("--max_datasets", type=int, default=3)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="datasets")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    names = search_hf_datasets(args.query, args.max_datasets)
    logger.info("Found %d dataset(s) for '%s'", len(names), args.query)
    for name in names:
        logger.info("Downloading %s...", name)
        path = download_dataset(name, args.split, args.output_dir)
        logger.info("Saved to %s", path)


if __name__ == "__main__":
    main()
