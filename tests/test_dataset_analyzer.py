from unittest.mock import patch
from datasets import Dataset
import numpy as np
import pytest

import torch
from analysis.dataset_analyzer import (
    analyze_tokenized_dataset,
    analyze_dataset,
    cluster_dataset_embeddings,
    compute_tsne_embeddings,
)


def test_analyze_tokenized_dataset():
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3], [2, 3, 4, 5]]})
    stats = analyze_tokenized_dataset(ds, top_k=2, include_trigrams=True)
    assert stats["samples"] == 2
    assert stats["avg_length"] == 3.5
    assert stats["vocab_size"] == 5
    assert stats["length_distribution"] == {3: 1, 4: 1}
    assert stats["top_tokens"] == [2, 3]
    assert stats["top_bigrams"] == [[2, 3], [1, 2]]
    assert stats["lexical_diversity"] == pytest.approx(5 / 7)
    assert stats["token_entropy"] == pytest.approx(2.2359, rel=1e-3)
    assert stats["top_trigrams"] == [[1, 2, 3], [2, 3, 4]]


def test_analyze_dataset_patch():
    ds = Dataset.from_dict({"input_ids": [[1, 2], [3, 4, 5]]})
    with patch("analysis.dataset_analyzer.load_and_tokenize", return_value=ds):
        stats = analyze_dataset("dummy", "train", "dummy", top_k=1)
    assert stats["samples"] == 2
    assert stats["vocab_size"] == 5
    assert stats["top_tokens"] == [1]
    assert stats["top_bigrams"] == [[1, 2]]
    assert stats["lexical_diversity"] == pytest.approx(1.0)


def test_cluster_dataset_embeddings():
    ds = Dataset.from_dict({
        "input_ids": [[1, 2], [3, 4]],
        "attention_mask": [[1, 1], [1, 1]],
    })

    class DummyModel:
        def to(self, device):
            return self

        def __call__(self, input_ids, attention_mask):
            batch_size, seq_len = input_ids.shape
            hidden = torch.ones(batch_size, seq_len, 2)
            return type("Output", (), {"last_hidden_state": hidden})

    with patch("analysis.dataset_analyzer.AutoTokenizer.from_pretrained"), \
         patch("analysis.dataset_analyzer.AutoModel.from_pretrained", return_value=DummyModel()):
        km, labels = cluster_dataset_embeddings(ds, "dummy", num_clusters=2, device="cpu", batch_size=1)

    assert len(labels) == 2
    assert hasattr(km, "cluster_centers_")


def test_compute_tsne_embeddings():
    ds = Dataset.from_dict({
        "input_ids": [[1, 2], [3, 4]],
        "attention_mask": [[1, 1], [1, 1]],
    })

    class DummyModel:
        def to(self, device):
            return self

        def __call__(self, input_ids, attention_mask):
            batch_size, seq_len = input_ids.shape
            hidden = torch.ones(batch_size, seq_len, 2)
            return type("Output", (), {"last_hidden_state": hidden})

    with patch("analysis.dataset_analyzer.AutoTokenizer.from_pretrained"), \
         patch("analysis.dataset_analyzer.AutoModel.from_pretrained", return_value=DummyModel()), \
         patch("analysis.dataset_analyzer.TSNE") as tsne_mock:
        tsne_instance = tsne_mock.return_value
        tsne_instance.fit_transform.return_value = np.array([[0.0, 0.0], [1.0, 1.0]])
        coords = compute_tsne_embeddings(ds, "dummy", device="cpu", batch_size=1, perplexity=0.5)

    assert coords.shape == (2, 2)

