from __future__ import annotations
"""Synergized prompt optimizer combining omni strategies with iterative self-learning."""

from typing import Optional

from .prompt_optimizer import PromptOptimizer
from .omni_prompt_optimizer import OmniPromptOptimizer


class SynergizedPromptOptimizer(PromptOptimizer):
    """Iteratively refine prompts using :class:`OmniPromptOptimizer` and generated variations."""

    def __init__(self, model_name: str, *, iterations: int = 2, device: Optional[str] = None) -> None:
        super().__init__(model_name, device=device)
        self.omni = OmniPromptOptimizer(model_name, device=self.device.type)
        self.iterations = iterations

    def optimize_prompt(self, base_prompt: str, n_variations: int = 5) -> str:
        """Return the best prompt after multiple optimization cycles."""
        prompt = base_prompt
        best_prompt = base_prompt
        best_score = float("inf")
        for _ in range(self.iterations):
            prompt = self.omni.optimize_prompt(prompt, n_variations=n_variations)
            variation = self.generate_variations(prompt, n_variations=1)[0]
            score = self.score_prompt(variation)
            if score < best_score:
                best_score = score
                best_prompt = variation
            prompt = variation
        return best_prompt
