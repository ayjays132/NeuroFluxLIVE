from __future__ import annotations
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from language_model_evaluator import evaluate_perplexity


def run_benchmark(
    base_model: str,
    tuned_model: str,
    dataset: str,
    *,
    dataset_config: str | None = None,
    split: str = "test[:50]",
    text_column: str = "text",
    max_length: int | None = 128,
    prompt: str = "Hello",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_ppl = evaluate_perplexity(
        base_model,
        dataset,
        dataset_config=dataset_config,
        split=split,
        text_column=text_column,
        max_length=max_length,
        device=str(device),
    )
    tuned_ppl = evaluate_perplexity(
        tuned_model,
        dataset,
        dataset_config=dataset_config,
        split=split,
        text_column=text_column,
        max_length=max_length,
        device=str(device),
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model).to(device)
    tuned = AutoModelForCausalLM.from_pretrained(tuned_model).to(device)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    base_gen = tokenizer.decode(base.generate(input_ids, max_length=50)[0], skip_special_tokens=True)
    tuned_gen = tokenizer.decode(tuned.generate(input_ids, max_length=50)[0], skip_special_tokens=True)
    print(f"Baseline PPL: {base_ppl:.2f}")
    print(f"Fine-tuned PPL: {tuned_ppl:.2f}")
    print("\nBaseline generation:\n" + base_gen)
    print("\nFine-tuned generation:\n" + tuned_gen)


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-2 benchmarking helper")
    parser.add_argument("base_model")
    parser.add_argument("tuned_model")
    parser.add_argument("dataset")
    parser.add_argument("--dataset-config")
    parser.add_argument("--split", default="test[:50]")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--prompt", default="Hello world")
    args = parser.parse_args()
    run_benchmark(
        args.base_model,
        args.tuned_model,
        args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column=args.text_column,
        max_length=args.max_length,
        prompt=args.prompt,
    )


if __name__ == "__main__":
    main()
