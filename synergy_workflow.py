from __future__ import annotations

"""Demonstration of datasetless learning with a universal prompt optimizer."""

import torch
from typing import TYPE_CHECKING

from analysis.hyper_prompt_optimizer import HyperPromptOptimizer

if TYPE_CHECKING:
    from RealTimeDataAbsorber import RealTimeDataAbsorber


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    from RealTimeDataAbsorber import RealTimeDataAbsorber

    absorber = RealTimeDataAbsorber(model_config={}, settings={"db_path": ":memory:"})
    absorber.absorb_data("Initial data point", "text")

    optimizer = HyperPromptOptimizer("distilgpt2", absorber=absorber, device=device)
    prompt = "Describe a scientific breakthrough:"
    for _ in range(2):
        prompt = optimizer.optimize_with_absorber(prompt)
        print(f"Refined prompt: {prompt}")
        text = optimizer.generate_variations(prompt, n_variations=1)[0]
        print(f"Generated text: {text[:80]}")


if __name__ == "__main__":
    main()
