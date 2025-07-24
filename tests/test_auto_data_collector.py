import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock
import json

from data.auto_data_collector import search_hf_datasets, download_dataset


class DummyDataset:
    def __init__(self, items, features):
        self.items = items
        self.features = features

    def __iter__(self):
        return iter(self.items)


def test_search_hf_datasets():
    dummy_mod = SimpleNamespace(list_datasets=lambda: ['ag_news', 'wikitext', 'openwebtext'])
    with patch('data.auto_data_collector._datasets', return_value=dummy_mod):
        result = search_hf_datasets('wiki', limit=2)
    assert result == ['wikitext']


def test_download_dataset(tmp_path: Path):
    class DummyImage:
        pass

    class DummyAudio:
        pass

    class DummyValue:
        def __init__(self, dtype):
            self.dtype = dtype

    DummyImage.__name__ = "Image"
    DummyAudio.__name__ = "Audio"
    DummyValue.__name__ = "Value"

    features = {
        "text": DummyValue("string"),
        "image": DummyImage(),
        "audio": DummyAudio(),
        "sensor": DummyValue("float32"),
    }
    items = [
        {
            "text": "hello",
            "image": {"path": "http://img"},
            "audio": {"path": "http://aud"},
            "sensor": 1.0,
        }
    ]
    dummy_ds = DummyDataset(items, features)
    dummy_mod = SimpleNamespace(load_dataset=lambda name, split=None: dummy_ds)

    async_mock = AsyncMock()

    with patch("data.auto_data_collector._datasets", return_value=dummy_mod), \
         patch("data.auto_data_collector._download_file", async_mock), \
         patch("data.auto_data_collector.score_record", return_value=(1.0, [])):
        out = asyncio.run(download_dataset("dummy", "train", tmp_path))

    assert (out / "train_text.jsonl").exists()
    assert (out / "train_sensor.jsonl").exists()
    assert async_mock.await_count == 2
    with open(out / "train_text.jsonl") as f:
        rec = json.loads(f.readline())
    assert "quality" in rec
