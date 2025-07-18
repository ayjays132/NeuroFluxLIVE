"""Prompt optimization using simulated annealing."""

from __future__ import annotations

import math
import random
from typing import Callable, Optional

from .prompt_optimizer import PromptOptimizer


class PromptAnnealingOptimizer(PromptOptimizer):
    """Optimize prompts via simulated annealing."""

    def __init__(
        self,
        model_name: str,
        reward_fn: Optional[Callable[[str], float]] = None,
        *,
        temperature: float = 1.0,
        cooling: float = 0.95,
        steps: int = 50,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(model_name, device=device)
        self.reward_fn = reward_fn
        self.temperature = temperature
        self.cooling = cooling
        self.steps = steps

    def _score(self, prompt: str) -> float:
        if self.reward_fn is not None:
            return self.reward_fn(prompt)
        return super().score_prompt(prompt)

    def optimize_prompt(self, base_prompt: str, n_variations: int = 5) -> str:
        current = base_prompt
        best = current
        current_score = self._score(current)
        best_score = current_score
        temp = self.temperature
        for _ in range(self.steps):
            candidate = self.generate_variations(current, n_variations=1)[0]
            cand_score = self._score(candidate)
            if cand_score < best_score:
                best, best_score = candidate, cand_score
            if cand_score < current_score:
                accept_prob = 1.0
            else:
                accept_prob = math.exp((current_score - cand_score) / max(temp, 1e-8))
            if random.random() < accept_prob:
                current, current_score = candidate, cand_score
            temp *= self.cooling
        return best
