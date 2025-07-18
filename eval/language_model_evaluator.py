"""Utilities for evaluating language models on text datasets."""

from __future__ import annotations

import argparse
import math
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_perplexity(
    model_name: str,
    dataset_name: str,
    split: str = "test",
    text_column: str = "text",
    device: Optional[str] = None,
) -> float:
    """Compute perplexity of a model on a given dataset split."""
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device_t)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset(dataset_name, split=split)

    def tokenize(batch):
        return tokenizer(batch[text_column], truncation=True, padding="max_length")

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    nlls = []
    for sample in dataset:
        input_ids = sample["input_ids"].unsqueeze(0).to(device_t)
        with torch.no_grad():
            output = model(input_ids=input_ids, labels=input_ids)
        nlls.append(output.loss.item())
    avg_nll = sum(nlls) / len(nlls)
    return float(math.exp(avg_nll))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model perplexity")
    parser.add_argument("model_name", help="Model identifier")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--text-column", default="text", help="Text column name")
    args = parser.parse_args()
    ppl = evaluate_perplexity(
        model_name=args.model_name,
        dataset_name=args.dataset,
        split=args.split,
        text_column=args.text_column,
    )
    print(f"Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
