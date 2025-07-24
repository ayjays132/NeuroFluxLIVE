from __future__ import annotations

"""Demonstration of datasetless learning with a universal prompt optimizer."""

import torch
from typing import TYPE_CHECKING

from analysis.hyper_prompt_optimizer import HyperPromptOptimizer

if TYPE_CHECKING:
    from RealTimeDataAbsorber import RealTimeDataAbsorber
    from correlation_rag_module import CorrelationRAGMemory
    from models.vae_compressor import VAECompressor


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    from RealTimeDataAbsorber import RealTimeDataAbsorber
    from models.vae_compressor import VAECompressor
    from correlation_rag_module import CorrelationRAGMemory

    compressor = VAECompressor(latent_dim=16)
    rag = CorrelationRAGMemory(emb_dim=384, save_path="synergy.mem", compressor=compressor)
    if hasattr(rag, "labels"):
        print(f"Loaded {len(rag.labels)} memory items")

    absorber = RealTimeDataAbsorber(model_config={}, settings={"db_path": ":memory:"})
    if hasattr(absorber, "attach_rag"):
        absorber.attach_rag(rag)
    absorber.absorb_data("Initial data point", "text")

    optimizer = HyperPromptOptimizer("distilgpt2", absorber=absorber, device=device)
    prompt = "Describe a scientific breakthrough:"
    for _ in range(2):
        prompt = optimizer.optimize_with_absorber(prompt)
        print(f"Refined prompt: {prompt}")
        text = optimizer.generate_variations(prompt, n_variations=1)[0]
        print(f"Generated text: {text[:80]}")

    if hasattr(absorber, "log_performance_metrics"):
        absorber.log_performance_metrics()


if __name__ == "__main__":
    main()
