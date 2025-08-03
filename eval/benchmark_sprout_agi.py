from __future__ import annotations

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from language_model_evaluator import evaluate_perplexity


DATASET_ID = "ayjays132/Sprout-AGI"
BASELINE_MODEL = "gpt2"


def run_benchmark(
    tuned_model: str,
    *,
    split: str = "test[:50]",
    text_column: str = "text",
    max_length: int | None = 128,
    prompt: str = "Hello",
) -> None:
    """Run a simple benchmark comparing GPT-2 and a fine-tuned model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_ppl = evaluate_perplexity(
        BASELINE_MODEL,
        DATASET_ID,
        split=split,
        text_column=text_column,
        max_length=max_length,
        device=str(device),
    )
    tuned_ppl = evaluate_perplexity(
        tuned_model,
        DATASET_ID,
        split=split,
        text_column=text_column,
        max_length=max_length,
        device=str(device),
    )
    tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASELINE_MODEL).to(device)
    tuned = AutoModelForCausalLM.from_pretrained(tuned_model).to(device)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    def generate_text(model: AutoModelForCausalLM) -> str:
        output = model.generate(input_ids, max_length=50)
        sequences = output.sequences if hasattr(output, "sequences") else output
        return tokenizer.decode(sequences[0], skip_special_tokens=True)

    base_gen = generate_text(base)
    tuned_gen = generate_text(tuned)
    print(f"Baseline PPL: {base_ppl:.2f}")
    print(f"Fine-tuned PPL: {tuned_ppl:.2f}")
    print("\nBaseline generation:\n" + base_gen)
    print("\nFine-tuned generation:\n" + tuned_gen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprout-AGI benchmarking helper")
    parser.add_argument("tuned_model")
    parser.add_argument("--split", default="test[:50]")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--prompt", default="Hello world")
    args = parser.parse_args()
    run_benchmark(
        args.tuned_model,
        split=args.split,
        text_column=args.text_column,
        max_length=args.max_length,
        prompt=args.prompt,
    )


if __name__ == "__main__":
    main()
