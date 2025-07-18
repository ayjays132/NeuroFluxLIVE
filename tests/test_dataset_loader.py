from unittest.mock import patch
from datasets import Dataset

from data.dataset_loader import load_and_tokenize, load_local_json_dataset


class DummyTokenizer:
    def __call__(self, text, truncation=True, padding="max_length", **kwargs):
        if isinstance(text, list):
            return {"input_ids": [[1, 2]] * len(text), "attention_mask": [[1, 1]] * len(text)}
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}


def test_load_and_tokenize_patch():
    dummy_ds = Dataset.from_dict({"text": ["hello", "world"]})
    with patch("data.dataset_loader.load_dataset", return_value=dummy_ds) as load_mock, \
         patch("data.dataset_loader.AutoTokenizer.from_pretrained", return_value=DummyTokenizer()), \
         patch.object(dummy_ds, "shuffle", return_value=dummy_ds) as shuffle_mock:
        tokenized = load_and_tokenize("dummy", "train", "dummy", shuffle=True, seed=123)
        shuffle_mock.assert_called_once_with(seed=123)
        load_mock.assert_called_once()
    assert len(tokenized) == 2
    assert "input_ids" in tokenized.column_names


def test_load_and_tokenize_cache(tmp_path):
    dummy_ds = Dataset.from_dict({"text": ["hello"]})
    with patch("data.dataset_loader.load_dataset", return_value=dummy_ds), \
         patch("data.dataset_loader.AutoTokenizer.from_pretrained", return_value=DummyTokenizer()):
        tokenized = load_and_tokenize("dummy", "train", "dummy", cache_dir=str(tmp_path))
    assert len(tokenized) == 1
    # Call again to load from disk
    with patch("data.dataset_loader.load_from_disk", return_value=tokenized) as load_disk_mock, \
         patch("data.dataset_loader.load_dataset") as load_ds_mock:
        tokenized2 = load_and_tokenize("dummy", "train", "dummy", cache_dir=str(tmp_path))
        load_disk_mock.assert_called_once()
        load_ds_mock.assert_not_called()


def test_load_local_json_dataset(tmp_path):
    json_file = tmp_path / "data.jsonl"
    with open(json_file, "w") as f:
        f.write('{"text": "hello"}\n')
        f.write('{"text": "world"}\n')
    with patch("data.dataset_loader.AutoTokenizer.from_pretrained", return_value=DummyTokenizer()):
        ds = load_local_json_dataset(str(json_file), "dummy")
    assert len(ds) == 2
    assert "input_ids" in ds.column_names


def test_load_and_tokenize_drop_duplicates():
    ds = Dataset.from_dict({"text": ["a", "b", "a"]})
    with patch("data.dataset_loader.load_dataset", return_value=ds), \
         patch("data.dataset_loader.AutoTokenizer.from_pretrained", return_value=DummyTokenizer()):
        tokenized = load_and_tokenize("dummy", "train", "dummy", drop_duplicates=True)
    assert len(tokenized) == 2

