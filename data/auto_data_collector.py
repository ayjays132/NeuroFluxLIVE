"""Utilities for automatically fetching datasets from the Hugging Face Hub."""
from __future__ import annotations

import logging
import asyncio
import json
import shutil
from pathlib import Path
from typing import List, Optional

import aiohttp
from analysis.dataset_quality import score_record

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


async def _download_file(session: aiohttp.ClientSession, url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` asynchronously."""
    async with session.get(url) as resp:
        resp.raise_for_status()
        dest.write_bytes(await resp.read())


def search_hf_datasets(query: str, limit: int = 10) -> List[str]:
    """Return dataset names containing the query string."""
    query = query.lower()
    ds_mod = _datasets()
    names = [name for name in ds_mod.list_datasets() if query in name.lower()]
    return names[:limit]


async def download_dataset(
    name: str, split: str, output_dir: str | Path, session: Optional[aiohttp.ClientSession] = None
) -> Path:
    """Download a dataset split supporting text, image, audio and sensor data.

    Text and sensor columns are stored in JSONL files while image and audio
    columns are downloaded using ``aiohttp`` where possible.

    Args:
        name: Dataset identifier on Hugging Face.
        split: Which split to download.
        output_dir: Directory to save files under.
        session: Optional shared :class:`aiohttp.ClientSession`.

    Returns:
        Path to the dataset directory containing saved files.
    """
    ds_mod = _datasets()
    ds = await asyncio.to_thread(ds_mod.load_dataset, name, split=split)

    text_cols: List[str] = []
    image_cols: List[str] = []
    audio_cols: List[str] = []
    sensor_cols: List[str] = []

    for col, feat in ds.features.items():
        cls = feat.__class__.__name__
        if cls == "Image":
            image_cols.append(col)
        elif cls == "Audio":
            audio_cols.append(col)
        elif cls == "Value" and getattr(feat, "dtype", "") == "string":
            text_cols.append(col)
        else:
            sensor_cols.append(col)

    dataset_dir = Path(output_dir) / name.replace("/", "_")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    text_path = dataset_dir / f"{split}_text.jsonl" if text_cols else None
    sensor_path = dataset_dir / f"{split}_sensor.jsonl" if sensor_cols else None

    async with aiohttp.ClientSession() if session is None else session as sess:
        tasks = []
        if text_path:
            txt_f = open(text_path, "w", encoding="utf-8")
        else:
            txt_f = None
        if sensor_path:
            sensor_f = open(sensor_path, "w", encoding="utf-8")
        else:
            sensor_f = None

        seen: set[str] = set()
        for idx, example in enumerate(ds):
            quality, _ = score_record(
                {c: example[c] for c in text_cols + sensor_cols}, seen
            )
            if txt_f:
                rec = {c: example[c] for c in text_cols}
                rec["quality"] = quality
                txt_f.write(json.dumps(rec) + "\n")
            if sensor_f:
                rec = {c: example[c] for c in sensor_cols}
                rec["quality"] = quality
                sensor_f.write(json.dumps(rec) + "\n")
            for col in image_cols:
                img = example[col]
                dest = dataset_dir / "image" / f"{col}_{idx}.jpg"
                dest.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(img, dict) and "path" in img:
                    path = img["path"]
                    if path.startswith("http"):
                        tasks.append(_download_file(sess, path, dest))
                    else:
                        shutil.copy(path, dest)
                elif isinstance(img, str) and img.startswith("http"):
                    tasks.append(_download_file(sess, img, dest))
            for col in audio_cols:
                audio = example[col]
                dest = dataset_dir / "audio" / f"{col}_{idx}.wav"
                dest.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(audio, dict) and "path" in audio:
                    path = audio["path"]
                    if path.startswith("http"):
                        tasks.append(_download_file(sess, path, dest))
                    else:
                        shutil.copy(path, dest)
                elif isinstance(audio, str) and audio.startswith("http"):
                    tasks.append(_download_file(sess, audio, dest))
        if txt_f:
            txt_f.close()
        if sensor_f:
            sensor_f.close()
        if tasks:
            await asyncio.gather(*tasks)

    return dataset_dir


async def _run_once(args: argparse.Namespace) -> None:
    names = search_hf_datasets(args.query, args.max_datasets)
    logger.info("Found %d dataset(s) for '%s'", len(names), args.query)
    for name in names:
        logger.info("Downloading %s...", name)
        path = await download_dataset(name, args.split, args.output_dir)
        logger.info("Saved to %s", path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download datasets from HF Hub")
    parser.add_argument("--query", required=True, help="Search term")
    parser.add_argument("--max_datasets", type=int, default=3)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="datasets")
    parser.add_argument("--schedule", type=int, help="Minutes between retrieval cycles")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    async def runner() -> None:
        if args.schedule:
            while True:
                await _run_once(args)
                logger.info("Waiting %d minute(s) before next run...", args.schedule)
                await asyncio.sleep(args.schedule * 60)
        else:
            await _run_once(args)

    asyncio.run(runner())


if __name__ == "__main__":
    main()
