from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from data.auto_data_collector import search_hf_datasets, download_dataset


class DummyDataset:
    def __init__(self, items):
        self.items = items

    def to_json(self, path: str) -> None:
        from json import dumps

        with open(path, "w") as f:
            for item in self.items:
                f.write(dumps(item) + "\n")


def test_search_hf_datasets():
    dummy_mod = SimpleNamespace(list_datasets=lambda: ['ag_news', 'wikitext', 'openwebtext'])
    with patch('data.auto_data_collector._datasets', return_value=dummy_mod):
        result = search_hf_datasets('wiki', limit=2)
    assert result == ['wikitext']


def test_download_dataset(tmp_path: Path):
    dummy_ds = DummyDataset([{'text': 'hello'}])
    dummy_mod = SimpleNamespace(load_dataset=lambda name, split=None: dummy_ds)
    with patch('data.auto_data_collector._datasets', return_value=dummy_mod):
        path = download_dataset('dummy', 'train', tmp_path)
    assert path.exists()
    assert path.read_text().strip() == '{"text": "hello"}'
