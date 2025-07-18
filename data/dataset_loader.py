
"""Dataset loading and tokenization utilities."""

from __future__ import annotations

from typing import Any, Optional, Tuple, List
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer
from analysis.prompt_augmenter import PromptAugmenter


def load_and_tokenize(
    dataset_name: str,
    split: str,
    tokenizer_name: str,
    text_column: str = "text",
    batch_size: int = 1000,
    cache_dir: Optional[str] = None,
    num_proc: Optional[int] = None,
    max_length: Optional[int] = None,
    streaming: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    drop_duplicates: bool = False,
) -> Dataset:
    """Load a dataset, optionally shuffle, and tokenize it with optional caching.

    Args:
        dataset_name: Name or path of the dataset to load (e.g. ``"ag_news"``).
        split: Which split to load (e.g. ``"train[:1000]"``).
        tokenizer_name: Name of the tokenizer/model to tokenize with.
        text_column: Name of the text column to tokenize.
        batch_size: Batch size for tokenization.
        cache_dir: Optional path for caching the tokenized dataset.
        num_proc: Optional number of processes for parallel tokenization.
        max_length: Optional maximum token length for truncation.
        streaming: If ``True``, stream the dataset instead of loading it fully.
        shuffle: Whether to shuffle the dataset prior to tokenization (ignored when streaming).
        seed: RNG seed used for shuffling.
        drop_duplicates: If ``True``, remove duplicate samples based on ``text_column``.

    Returns:
        The tokenized dataset formatted with PyTorch tensors.
    """
    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path and (cache_path / "dataset_info.json").exists():
        return load_from_disk(str(cache_path))

    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    if shuffle and not streaming:
        dataset = dataset.shuffle(seed=seed)
    if drop_duplicates and not streaming:
        seen = set()
        def _unique(example):
            text = example[text_column]
            if text in seen:
                return False
            seen.add(text)
            return True
        dataset = dataset.filter(_unique)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if getattr(tokenizer, "pad_token", None) is None:
        eos = getattr(tokenizer, "eos_token", "<PAD>")
        tokenizer.pad_token = eos

    def tokenize(batch: Any) -> Any:
        return tokenizer(
            batch[text_column],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
    )
    tokenized = tokenized.with_format("torch", columns=["input_ids", "attention_mask"])
    if cache_path and not streaming:
        cache_path.mkdir(parents=True, exist_ok=True)
        tokenized.save_to_disk(str(cache_path))
    return tokenized


def load_dataset_splits(
    dataset_name: str,
    tokenizer_name: str,
    train_split: str = "train",
    eval_split: Optional[str] = None,
    text_column: str = "text",
    batch_size: int = 1000,
    train_cache_dir: Optional[str] = None,
    eval_cache_dir: Optional[str] = None,
    num_proc: Optional[int] = None,
    max_length: Optional[int] = None,
    streaming: bool = False,
    shuffle_train: bool = False,
    shuffle_eval: bool = False,
    drop_duplicates_train: bool = False,
    drop_duplicates_eval: bool = False,
    seed: int = 42,
) -> Tuple[Dataset, Optional[Dataset]]:
    """Load and tokenize train/eval dataset splits.

    Args:
        dataset_name: Dataset name or local path.
        tokenizer_name: Pretrained tokenizer name.
        train_split: Split for training data.
        eval_split: Optional evaluation split.
        text_column: Name of the text column to tokenize.
        batch_size: Batch size for tokenization.
        train_cache_dir: Optional path to cache the tokenized train split.
        eval_cache_dir: Optional path to cache the tokenized eval split.
        num_proc: Optional number of processes for parallel tokenization.
        max_length: Optional maximum token length for truncation.
        streaming: If ``True``, stream splits instead of loading them fully.
        shuffle_train: Whether to shuffle the training split prior to tokenization.
        shuffle_eval: Whether to shuffle the evaluation split prior to tokenization.
        drop_duplicates_train: Remove duplicates from the training split.
        drop_duplicates_eval: Remove duplicates from the evaluation split.
        seed: RNG seed used for shuffling.

    Returns:
        A tuple of ``(train_dataset, eval_dataset)`` where ``eval_dataset`` may
        be ``None`` if ``eval_split`` is not provided.
    """
    train_ds = load_and_tokenize(
        dataset_name,
        train_split,
        tokenizer_name,
        text_column=text_column,
        batch_size=batch_size,
        cache_dir=train_cache_dir,
        num_proc=num_proc,
        max_length=max_length,
        streaming=streaming,
        shuffle=shuffle_train,
        seed=seed,
        drop_duplicates=drop_duplicates_train,
    )
    eval_ds = None
    if eval_split:
        eval_ds = load_and_tokenize(
            dataset_name,
            eval_split,
            tokenizer_name,
            text_column=text_column,
            batch_size=batch_size,
            cache_dir=eval_cache_dir,
            num_proc=num_proc,
            max_length=max_length,
            streaming=streaming,
            shuffle=shuffle_eval,
            seed=seed,
            drop_duplicates=drop_duplicates_eval,
        )
    return train_ds, eval_ds


def load_local_json_dataset(
    json_path: str,
    tokenizer_name: str,
    text_key: str = "text",
    batch_size: int = 1000,
    num_proc: Optional[int] = None,
    max_length: Optional[int] = None,
) -> Dataset:
    """Load and tokenize a local JSON or JSONL dataset file.

    Args:
        json_path: Path to the JSON/JSONL file.
        tokenizer_name: Pretrained tokenizer name.
        text_key: Key in each JSON object containing text.
        batch_size: Batch size for tokenization.
        num_proc: Optional number of processes for parallel tokenization.
        max_length: Optional maximum token length for truncation.

    Returns:
        A tokenized ``Dataset``.
    """
    dataset = load_dataset("json", data_files=json_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(batch: Any) -> Any:
        return tokenizer(
            batch[text_key],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized


def augment_text_dataset(
    dataset: Dataset,
    model_name: str,
    text_column: str = "text",
    n_variations: int = 1,
) -> Dataset:
    """Expand a text dataset using prompt augmentation."""
    augmenter = PromptAugmenter(model_name)
    records: List[dict] = []
    for sample in dataset:
        text = sample[text_column]
        records.append({text_column: text})
        for new_text in augmenter.augment_prompt(text, n_variations=n_variations):
            records.append({text_column: new_text})
    return Dataset.from_list(records)


if __name__ == "__main__":
    ds, _ = load_dataset_splits("ag_news", "distilgpt2", train_split="train[:10]")
    print(ds[0])
