from __future__ import annotations

"""Demonstration of datasetless self-learning using the SynergizedPromptOptimizer."""

import torch

from analysis.synergized_prompt_optimizer import SynergizedPromptOptimizer


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    optimizer = SynergizedPromptOptimizer("distilgpt2", device=device)
    prompt = "Describe a scientific breakthrough:"
    for _ in range(2):
        prompt = optimizer.optimize_prompt(prompt)
        print(f"Refined prompt: {prompt}")
        text = optimizer.generate_variations(prompt, n_variations=1)[0]
        print(f"Generated text: {text[:80]}")


if __name__ == "__main__":
    main()
