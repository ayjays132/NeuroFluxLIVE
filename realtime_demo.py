"""Bootstrap script for the NeuroFluxLIVE realtime demo."""

from __future__ import annotations

import argparse
import json
import logging
import threading
from queue import Queue
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
import soundfile as sf
from PIL import Image

from correlation_rag_module import CorrelationRAGMemory
from predictive_coding_temporal_model import PredictiveCodingTemporalModel
from RealTimeDataAbsorber import RealTimeDataAbsorber
from multimodal_dataset_manager import DatasetIndex
from interface.ws_server import run_ws_server
from file_watcher import DataRootWatcher


def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def feed_dataset(absorber: RealTimeDataAbsorber, root: Path) -> None:
    """Feed existing samples from ``root`` into the absorber."""
    if not root.exists():
        logging.warning("Data root %s does not exist", root)
        return
    index = DatasetIndex(str(root), update_cache=False)
    for rec in index.records:
        path = root / rec.rel_path
        try:
            if rec.modality == "text":
                data = path.read_text(encoding="utf-8")
            elif rec.modality == "image":
                data = np.array(Image.open(path).convert("RGB"))
            elif rec.modality == "audio":
                data, _ = sf.read(path)
            elif rec.modality == "sensor":
                data = json.loads(path.read_text(encoding="utf-8"))
            else:
                continue
            absorber.absorb_data(data, rec.modality, source="dataset")
        except Exception as e:
            logging.error("Failed to feed %s: %s", rec.rel_path, e)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run NeuroFluxLIVE bootstrap")
    parser.add_argument("--config", default="config.yaml", help="Path to config")
    parser.add_argument(
        "--no-feed", action="store_true", help="Skip feeding data_root on start"
    )
    parser.add_argument(
        "--watch-interval",
        type=float,
        default=5.0,
        help="Seconds between scans for new data files",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    data_root = Path(paths.get("data_root", "./data"))
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    rag_cfg = cfg.get("rag", {})
    pc_cfg = cfg.get("pc", {})
    embed_dim = int(model_cfg.get("embed_dim", 256))
    context_window = int(pc_cfg.get("context_window", 32))

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    metrics_q: Queue = Queue()
    ws_thread = threading.Thread(
        target=run_ws_server, args=(metrics_q,), daemon=True
    )
    ws_thread.start()

    rag_settings = {**rag_cfg, "embed_dim": embed_dim}
    rag = CorrelationRAGMemory(emb_dim=embed_dim, settings=rag_settings)
    pc = PredictiveCodingTemporalModel(embed_dim=embed_dim, context_window=context_window)
    absorber_settings = {**training_cfg, "db_path": paths.get("db_path", "realtime_learning.db")}
    absorber = RealTimeDataAbsorber(
        model_config=model_cfg,
        settings=absorber_settings,
        metrics_queue=metrics_q,
    )
    absorber.attach_rag(rag)
    absorber.attach_pc(pc)

    absorber.start_absorption()
    watcher = DataRootWatcher(
        absorber=absorber, data_root=data_root, interval=args.watch_interval
    )
    watcher.start()

    try:
        if not args.no_feed:
            feed_dataset(absorber, data_root)
        while True:
            # Keep main thread alive while workers run
            threading.Event().wait(1.0)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        watcher.stop()
        absorber.stop_absorption()


if __name__ == "__main__":
    main()
