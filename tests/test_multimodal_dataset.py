import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import soundfile as sf

from multimodal_dataset_manager import (
    Record,
    LMDBCache,
    MultimodalDataset,
    BalancedModalitySampler,
    create_balanced_dataloader,
)


def make_record(modality: str, rel: str, idx: str) -> Record:
    return Record(id=idx, modality=modality, rel_path=rel, label=None, length=0, checksum=idx, extra={})


def test_balanced_modality_sampler_round_robin():
    records = [
        make_record("text", "t0.txt", "t0"),
        make_record("image", "i0.png", "i0"),
        make_record("text", "t1.txt", "t1"),
        make_record("audio", "a0.wav", "a0"),
        make_record("text", "t2.txt", "t2"),
    ]
    sampler = BalancedModalitySampler(records, shuffle=False)
    assert list(iter(sampler)) == [0, 1, 3, 2, 4]


def test_multimodal_dataset_getitem(tmp_path):
    root = Path(tmp_path)
    (root / "text").mkdir()
    (root / "image").mkdir()
    (root / "audio").mkdir()
    (root / "sensor").mkdir()

    (root / "text" / "t.txt").write_text("hi")
    img = Image.new("RGB", (2, 2), color="red")
    img.save(root / "image" / "i.png")
    sf.write(root / "audio" / "a.wav", np.zeros(4), 16000)
    with open(root / "sensor" / "s.json", "w") as f:
        json.dump({"x": 1.0}, f)

    records = [
        make_record("text", "text/t.txt", "t"),
        make_record("image", "image/i.png", "i"),
        make_record("audio", "audio/a.wav", "a"),
        make_record("sensor", "sensor/s.json", "s"),
    ]
    cache = LMDBCache(root / "cache")
    ds = MultimodalDataset(records, root, cache)

    tensor, rec = ds[0]
    assert isinstance(tensor, torch.Tensor)
    assert rec.modality == "text"
    assert cache.get("t") is not None

    loader = create_balanced_dataloader(ds, batch_size=2, shuffle=False)
    ids = []
    for tensors, recs in loader:
        ids.extend([r.id for r in recs])
    assert ids == ["t", "i", "a", "s"]

