"""
multimodal_dataset_manager.py
─────────────────────────────
A unified dataset‑organiser that

  • Scans folders (or JSONL) for TEXT / IMAGE / AUDIO / SENSOR files
  • Builds a metadata index with SHA‑1 checksums & version stamps
  • Provides GPU‑friendly PyTorch Dataset + DataLoader helpers
  • Supports on‑the‑fly transforms, balanced class / modality sampling,
    pin_memory, async prefetch, automatic device placement
  • Caches per‑item tensors on first load (LMDB OR plain *.pt)
  • Exposes train/val/test splits & easy re‑shuffling

Default directory layout
├── data_root
│   ├── text/
│   │   └── *.txt           (UTF‑8)
│   ├── image/
│   │   └── *.jpg|png|webp
│   ├── audio/
│   │   └── *.wav|flac
│   └── sensor/
│       └── *.json          (dict{sensor:value})
"""

from __future__ import annotations
import os, json, hashlib, random, time, pickle, pathlib, warnings
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import numpy as np
import soundfile as sf
import lmdb                     # 🔒 fast caching; pip install lmdb
import msgpack                  # 🔒 compact binary serialisation
from tqdm.auto import tqdm

# ------------------------------------------------------------------ #
def sha1(path: str | bytes) -> str:
    h = hashlib.sha1()
    if isinstance(path, bytes):
        h.update(path)
    else:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    return h.hexdigest()

# ------------------------------------------------------------------ #
@dataclass
class Record:
    id        : str
    modality  : str
    rel_path  : str
    label     : Optional[str]
    length    : int
    checksum  : str
    extra     : Dict[str, Any]

# ------------------------------------------------------------------ #
class DatasetIndex:
    """Scans filesystem or JSONL and builds a list[Record]."""
    def __init__(self, root: str, recurse: bool = True,
                 jsonl_path: Optional[str] = None,
                 update_cache: bool = True):
        self.root      = pathlib.Path(root)
        self.records   : List[Record] = []
        self.jsonl     = pathlib.Path(jsonl_path) if jsonl_path else self.root/"index.jsonl"
        if self.jsonl.exists() and not update_cache:
            self._load_jsonl()
        else:
            self._scan(recurse)
            self._dump_jsonl()

    # -------------- public helpers -------------------------------- #
    def split(self, ratios=(0.8, 0.1, 0.1), seed=13) -> Tuple[List[Record], ...]:
        """Return train/val/test splits."""
        assert abs(sum(ratios) - 1.0) < 1e-5
        random.Random(seed).shuffle(self.records)
        n = len(self.records)
        a, b = int(ratios[0]*n), int((ratios[0]+ratios[1])*n)
        return self.records[:a], self.records[a:b], self.records[b:]

    def to_json(self) -> List[Dict]:
        return [asdict(r) for r in self.records]

    # -------------- internal -------------------------------------- #
    SUFFIX2MOD = {
        **{s: "image" for s in (".jpg", ".jpeg", ".png", ".webp", ".bmp")},
        **{s: "audio" for s in (".wav", ".flac", ".ogg")},
        **{s: "text"  for s in (".txt",)},
        ".json": "sensor"
    }

    def _scan(self, recurse):
        print("🔍 Scanning dataset...")
        files = self.root.rglob("*") if recurse else self.root.glob("*")
        for fp in tqdm(list(files)):
            if not fp.is_file(): continue
            mod = self.SUFFIX2MOD.get(fp.suffix.lower())
            if not mod: continue
            csum = sha1(fp)
            rel  = fp.relative_to(self.root).as_posix()
            size = fp.stat().st_size
            self.records.append(Record(
                id        = csum[:16],
                modality  = mod,
                rel_path  = rel,
                label     = None,
                length    = size,
                checksum  = csum,
                extra     = {}
            ))

    def _dump_jsonl(self):
        self.jsonl.write_text("\n".join(json.dumps(asdict(r)) for r in self.records))
        print(f"📄 Index written to {self.jsonl}")

    def _load_jsonl(self):
        for line in self.jsonl.read_text().splitlines():
            self.records.append(Record(**json.loads(line)))
        print(f"📄 Loaded index with {len(self.records)} records from {self.jsonl}")

# ------------------------------------------------------------------ #
class LMDBCache
