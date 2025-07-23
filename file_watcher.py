"""Simple periodic file watcher for the data root.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image
from typing import TYPE_CHECKING

from multimodal_dataset_manager import DatasetIndex

if TYPE_CHECKING:
    from RealTimeDataAbsorber import RealTimeDataAbsorber


class DataRootWatcher:
    """Watches ``data_root`` for new files and feeds them to ``absorber``."""

    def __init__(self, absorber: "RealTimeDataAbsorber", data_root: Path,
                 interval: float = 5.0) -> None:
        self.absorber = absorber
        self.data_root = Path(data_root)
        self.interval = interval
        self._seen: set[str] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    # ------------------------------------------------------------------ #
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                index = DatasetIndex(str(self.data_root))
                for rec in index.records:
                    if rec.checksum in self._seen:
                        continue
                    self._seen.add(rec.checksum)
                    path = self.data_root / rec.rel_path
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
                        self.absorber.absorb_data(
                            data, rec.modality, source="filewatcher"
                        )
                    except Exception as e:
                        logging.error("Failed to load %s: %s", rec.rel_path, e)
            except Exception as e:
                logging.error("File watcher error: %s", e)
            self._stop.wait(self.interval)
