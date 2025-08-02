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
    dataset_config: Optional[str] = None,
    split: str = "test",
    text_column: str = "text",
    device: Optional[str] = None,
    max_length: Optional[int] = None,
) -> float:
    """Compute perplexity of a model on a given dataset split."""

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device_t)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    dataset = dataset.filter(lambda e: e[text_column] and not e[text_column].isspace())

    def tokenize(batch):
        enc = tokenizer(
            batch[text_column],
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        labels = [ids.copy() for ids in enc["input_ids"]]
        for label, mask in zip(labels, enc["attention_mask"]):
            for i, m in enumerate(mask):
                if m == 0:
                    label[i] = -100
        enc["labels"] = labels
        return enc

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    nlls = []
    for sample in dataset:
        input_ids = sample["input_ids"].unsqueeze(0).to(device_t)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device_t)
        labels = sample["labels"].unsqueeze(0).to(device_t)
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        nlls.append(output.loss.item())
    avg_nll = sum(nlls) / len(nlls)
    return float(math.exp(avg_nll))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model perplexity")
    parser.add_argument("model_name", help="Model identifier")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("--dataset-config", help="Optional dataset config")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--text-column", default="text", help="Text column name")
    parser.add_argument("--max-length", type=int, help="Truncate sequences to this length")
    args = parser.parse_args()
    ppl = evaluate_perplexity(
        model_name=args.model_name,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column=args.text_column,
        max_length=args.max_length,
    )
    print(f"Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
