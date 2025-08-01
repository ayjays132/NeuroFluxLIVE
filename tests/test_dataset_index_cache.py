import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import soundfile as sf

from multimodal_dataset_manager import DatasetIndex, LMDBCache


def test_dataset_index_modalities(tmp_path):
    root = Path(tmp_path)
    (root / "text").mkdir()
    (root / "image").mkdir()
    (root / "audio").mkdir()
    (root / "sensor").mkdir()

    (root / "text" / "sample.txt").write_text("hello")
    img = Image.new("RGB", (2, 2), color="blue")
    img.save(root / "image" / "sample.png")
    sf.write(root / "audio" / "sample.wav", np.zeros(4), 16000)
    with open(root / "sensor" / "sample.json", "w") as f:
        json.dump({"v": 1}, f)

    index = DatasetIndex(str(root))
    modalities = [rec.modality for rec in index.records]
    assert set(modalities) == {"text", "image", "audio", "sensor"}
    assert len(index.records) == 4


def test_lmdb_cache_put_get(tmp_path):
    cache_dir = tmp_path / "db"
    cache = LMDBCache(cache_dir)
    t1 = torch.arange(3)
    t2 = torch.ones(2)
    cache.put("t1", t1)
    cache.put("t2", t2)
    assert torch.equal(cache.get("t1"), t1)
    assert torch.equal(cache.get("t2"), t2)
    assert cache.get("missing") is None
    cache.close()

    # reopen in readonly mode
    ro_cache = LMDBCache(cache_dir, readonly=True)
    assert torch.equal(ro_cache.get("t1"), t1)
    ro_cache.close()
