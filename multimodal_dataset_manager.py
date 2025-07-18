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
class LMDBCache:
    """Simple LMDB-backed cache for tensors."""

    def __init__(self, path: str, readonly: bool = False,
                 map_size: int = 1 << 40) -> None:
        """Create a new cache or open an existing one.

        Args:
            path: Directory where the LMDB environment lives.
            readonly: Open in read-only mode.
            map_size: Maximum size of the database in bytes.
        """
        self.path = pathlib.Path(path)
        self.readonly = readonly
        if not readonly:
            self.path.mkdir(parents=True, exist_ok=True)
        self.env = lmdb.open(
            str(self.path),
            subdir=True,
            readonly=readonly,
            map_size=map_size,
            create=not readonly,
            lock=not readonly,
            readahead=False,
            meminit=False,
        )

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve a tensor by key or ``None`` if missing."""
        with self.env.begin(write=False, buffers=True) as txn:
            buf = txn.get(key.encode("utf-8"))
        if buf is None:
            return None
        obj = msgpack.unpackb(bytes(buf), raw=False)
        dtype = getattr(torch, obj["dtype"].split(".")[-1])
        return torch.tensor(obj["data"], dtype=dtype)

    def put(self, key: str, tensor: torch.Tensor) -> None:
        """Store ``tensor`` under ``key``."""
        if self.readonly:
            raise RuntimeError("Cannot write to read-only LMDB cache")
        arr = tensor.detach().cpu()
        obj = {
            "dtype": str(arr.dtype),
            "data": arr.tolist(),
        }
        data = msgpack.packb(obj, use_bin_type=True)
        with self.env.begin(write=True) as txn:
            txn.put(key.encode("utf-8"), data)

    def close(self) -> None:
        """Close the underlying LMDB environment."""
        if self.env is not None:
            self.env.close()

    def __enter__(self) -> "LMDBCache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()




class BalancedModalitySampler(Sampler[int]):
    """Yield dataset indices round-robin across modalities."""

    def __init__(self, records: List[Record], shuffle: bool = True, seed: int = 42) -> None:
        self.records = list(records)
        self.shuffle = shuffle
        self.seed = seed
        self._groups: Dict[str, List[int]] = {}
        self._order: List[str] = []
        for idx, rec in enumerate(self.records):
            if rec.modality not in self._groups:
                self._groups[rec.modality] = []
                self._order.append(rec.modality)
            self._groups[rec.modality].append(idx)

    def __iter__(self):
        groups = {m: list(idxs) for m, idxs in self._groups.items()}
        if self.shuffle:
            rng = random.Random(self.seed)
            for idxs in groups.values():
                rng.shuffle(idxs)
        remaining = True
        while remaining:
            remaining = False
            for mod in self._order:
                bucket = groups.get(mod)
                if bucket:
                    yield bucket.pop(0)
                    remaining = True

    def __len__(self) -> int:
        return len(self.records)


class MultimodalDataset(Dataset):
    """Dataset that loads items from multiple modalities with caching."""

    def __init__(self, records: List[Record], root: str, cache: Optional[LMDBCache] = None) -> None:
        self.records = list(records)
        self.root = pathlib.Path(root)
        self.cache = cache

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Record]:
        rec = self.records[index]
        tensor = self._load_tensor(rec)
        return tensor, rec

    # -------------------------------------------------------------- #
    def _load_tensor(self, rec: Record) -> torch.Tensor:
        if self.cache:
            cached = self.cache.get(rec.id)
            if cached is not None:
                return cached
        path = self.root / rec.rel_path
        if rec.modality == "text":
            data = path.read_text(encoding="utf-8")
            arr = np.frombuffer(data.encode("utf-8"), dtype=np.uint8)
            tensor = torch.tensor(arr, dtype=torch.uint8)
        elif rec.modality == "image":
            with Image.open(path) as img:
                arr = np.array(img.convert("RGB"))
            tensor = torch.tensor(arr).permute(2, 0, 1)
        elif rec.modality == "audio":
            audio, _ = sf.read(path.as_posix())
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            tensor = torch.tensor(audio)
        elif rec.modality == "sensor":
            obj = json.loads(path.read_text())
            values = list(obj.values()) if isinstance(obj, dict) else obj
            tensor = torch.tensor(values, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown modality: {rec.modality}")
        if self.cache:
            self.cache.put(rec.id, tensor)
        return tensor


def create_balanced_dataloader(
    dataset: MultimodalDataset,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: int = 42,
    **kwargs: Any,
) -> DataLoader:
    """Return a ``DataLoader`` using ``BalancedModalitySampler``."""

    sampler = BalancedModalitySampler(dataset.records, shuffle=shuffle, seed=seed)

    def collate_fn(batch):
        tensors, records = zip(*batch)
        return list(tensors), list(records)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs,
    )
