from __future__ import annotations

"""Utilities for analyzing tokenized datasets."""

from typing import Optional, Dict, Any, Tuple
import argparse
import json
from collections import Counter
import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer
import torch
from utils.tensor_ops import tensor_to_ndarray

from datasets import Dataset

from data.dataset_loader import load_and_tokenize


def analyze_tokenized_dataset(
    dataset: Dataset,
    max_samples: Optional[int] = None,
    top_k: int = 5,
    include_trigrams: bool = False,
) -> Dict[str, Any]:
    """Compute statistics for a tokenized dataset including length distribution and top tokens.

    Parameters
    ----------
    dataset:
        HuggingFace ``Dataset`` object with an ``input_ids`` column.
    max_samples:
        Optional limit on the number of samples to analyze.

    Returns
    -------
    dict
        Dictionary containing sample count, average token length, vocabulary size,
        token length distribution and the most common tokens.
    """
    total_len = 0
    vocab = set()
    length_counter: Counter[int] = Counter()
    token_freq: Counter[int] = Counter()
    bigram_freq: Counter[Tuple[int, int]] = Counter()
    trigram_freq: Counter[Tuple[int, int, int]] = Counter()
    samples = 0

    for sample in dataset:
        ids = sample["input_ids"]
        total_len += len(ids)
        vocab.update(ids)
        length_counter[len(ids)] += 1
        token_freq.update(ids)
        # Update bigram frequency with consecutive token pairs
        bigram_freq.update(zip(ids[:-1], ids[1:]))
        # Optionally update trigram frequency
        if include_trigrams:
            trigram_freq.update(zip(ids[:-2], ids[1:-1], ids[2:]))
        samples += 1
        if max_samples is not None and samples >= max_samples:
            break

    avg_len = float(total_len / samples) if samples else 0.0
    top_tokens = [tok for tok, _ in token_freq.most_common(top_k)]
    top_bigrams = [list(bg) for bg, _ in bigram_freq.most_common(top_k)]
    top_trigrams = [list(tg) for tg, _ in trigram_freq.most_common(top_k)] if include_trigrams else []
    lexical_diversity = float(len(vocab) / total_len) if total_len else 0.0
    # Shannon entropy of the token distribution
    token_entropy = 0.0
    if total_len:
        for count in token_freq.values():
            p = count / total_len
            token_entropy -= p * math.log2(p)

    result = {
        "samples": samples,
        "avg_length": avg_len,
        "vocab_size": len(vocab),
        "length_distribution": dict(length_counter),
        "top_tokens": top_tokens,
        "top_bigrams": top_bigrams,
        "lexical_diversity": lexical_diversity,
        "token_entropy": token_entropy,
    }
    if include_trigrams:
        result["top_trigrams"] = top_trigrams
    return result


def cluster_dataset_embeddings(
    dataset: Dataset,
    model_name: str,
    num_clusters: int = 5,
    *,
    device: Optional[str] = None,
    batch_size: int = 8,
) -> Tuple[KMeans, list[int]]:
    """Cluster dataset samples using sentence embeddings.

    The dataset must contain ``input_ids`` and ``attention_mask`` columns.

    Parameters
    ----------
    dataset:
        Tokenized ``Dataset`` with ``input_ids`` and ``attention_mask``.
    model_name:
        Name of a transformer model providing hidden states for embeddings.
    num_clusters:
        Number of clusters to form.
    device:
        Optional device specifier (defaults to CUDA if available).
    batch_size:
        Batch size used when computing embeddings.

    Returns
    -------
    tuple
        ``(kmeans, labels)`` where ``labels`` is a list of cluster assignments
        for each dataset sample.
    """

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device_t)

    embeddings = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        if "input_ids" not in batch or "attention_mask" not in batch:
            raise ValueError("Dataset must contain 'input_ids' and 'attention_mask'.")
        input_ids = torch.tensor(batch["input_ids"], device=device_t)
        attention_mask = torch.tensor(batch["attention_mask"], device=device_t)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        embeddings.append(tensor_to_ndarray(pooled))
    embeddings_arr = np.concatenate(embeddings, axis=0)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings_arr)
    return kmeans, labels.tolist()


def compute_tsne_embeddings(
    dataset: Dataset,
    model_name: str,
    *,
    device: Optional[str] = None,
    batch_size: int = 8,
    perplexity: float = 30.0,
) -> np.ndarray:
    """Compute 2D t-SNE embeddings for dataset samples.

    The dataset must contain ``input_ids`` and ``attention_mask`` columns.

    Parameters
    ----------
    dataset:
        Tokenized ``Dataset`` with ``input_ids`` and ``attention_mask``.
    model_name:
        Name of a transformer model providing hidden states for embeddings.
    device:
        Optional device specifier (defaults to CUDA if available).
    batch_size:
        Batch size used when computing embeddings.
    perplexity:
        t-SNE perplexity parameter controlling neighborhood size.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(dataset), 2)`` containing t-SNE coordinates.
    """

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device_t)

    embeddings = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        if "input_ids" not in batch or "attention_mask" not in batch:
            raise ValueError("Dataset must contain 'input_ids' and 'attention_mask'.")
        input_ids = torch.tensor(batch["input_ids"], device=device_t)
        attention_mask = torch.tensor(batch["attention_mask"], device=device_t)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        embeddings.append(tensor_to_ndarray(pooled))
    embeddings_arr = np.concatenate(embeddings, axis=0)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(embeddings_arr)
    return coords


def analyze_dataset(
    dataset_name: str,
    split: str,
    tokenizer_name: str,
    *,
    text_column: str = "text",
    batch_size: int = 1000,
    num_proc: Optional[int] = None,
    max_length: Optional[int] = None,
    streaming: bool = False,
    max_samples: Optional[int] = None,
    top_k: int = 5,
    include_trigrams: bool = False,
) -> Dict[str, Any]:
    """Load and analyze a dataset using ``load_and_tokenize``."""
    ds = load_and_tokenize(
        dataset_name,
        split,
        tokenizer_name,
        text_column=text_column,
        batch_size=batch_size,
        num_proc=num_proc,
        max_length=max_length,
        streaming=streaming,
    )
    return analyze_tokenized_dataset(
        ds,
        max_samples=max_samples,
        top_k=top_k,
        include_trigrams=include_trigrams,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze tokenized dataset statistics")
    parser.add_argument("dataset_name")
    parser.add_argument("split")
    parser.add_argument("tokenizer_name")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--top-k", type=int, default=5, help="Number of top tokens to report")
    parser.add_argument("--trigrams", action="store_true", help="Include trigram analysis")
    args = parser.parse_args()
    stats = analyze_dataset(
        args.dataset_name,
        args.split,
        args.tokenizer_name,
        text_column=args.text_column,
        max_samples=args.max_samples,
        top_k=args.top_k,
        include_trigrams=args.trigrams,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
