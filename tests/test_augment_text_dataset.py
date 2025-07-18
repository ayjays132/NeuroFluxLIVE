from unittest.mock import patch
from datasets import Dataset

from data.dataset_loader import augment_text_dataset


class DummyAugmenter:
    def augment_prompt(self, prompt, n_variations=1):
        return [f"{prompt} v{i}" for i in range(n_variations)]


def test_augment_text_dataset():
    ds = Dataset.from_dict({"text": ["a", "b"]})
    with patch("data.dataset_loader.PromptAugmenter", return_value=DummyAugmenter()):
        aug = augment_text_dataset(ds, "dummy", n_variations=2)
    assert len(aug) == 6  # original + 2 variations each
    texts = set(aug["text"])
    assert "a v0" in texts and "b v1" in texts
