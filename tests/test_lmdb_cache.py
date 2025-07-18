import torch
from multimodal_dataset_manager import LMDBCache


def test_lmdb_cache_roundtrip(tmp_path):
    cache = LMDBCache(tmp_path / "db")
    tensor = torch.arange(5)
    cache.put("a", tensor)
    loaded = cache.get("a")
    assert loaded is not None
    assert torch.equal(loaded, tensor)
    cache.close()

