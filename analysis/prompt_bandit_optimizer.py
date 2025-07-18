"""Prompt optimization using a multi-armed bandit approach."""

from __future__ import annotations

import random
from typing import Callable, Optional

from .prompt_optimizer import PromptOptimizer


class PromptBanditOptimizer(PromptOptimizer):
    """Optimize prompts via epsilon-greedy multi-armed bandit."""

    def __init__(
        self,
        model_name: str,
        reward_fn: Callable[[str], float],
        *,
        epsilon: float = 0.1,
        iterations: int = 10,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(model_name, device=device)
        self.reward_fn = reward_fn
        self.epsilon = epsilon
        self.iterations = iterations

    def optimize_prompt(self, base_prompt: str, n_variations: int = 5) -> str:
        """Return the prompt with the highest estimated reward."""
        candidates = self.generate_variations(base_prompt, n_variations=n_variations)
        counts = [0 for _ in candidates]
        values = [0.0 for _ in candidates]

        # Initialize estimates with one sample of each arm
        for i, cand in enumerate(candidates):
            reward = self.reward_fn(cand)
            counts[i] = 1
            values[i] = reward

        for _ in range(self.iterations):
            if random.random() < self.epsilon:
                idx = random.randrange(len(candidates))
            else:
                idx = max(range(len(candidates)), key=lambda i: values[i])
            reward = self.reward_fn(candidates[idx])
            counts[idx] += 1
            # Incremental mean update
            values[idx] += (reward - values[idx]) / counts[idx]

        best_idx = max(range(len(candidates)), key=lambda i: values[i])
        return candidates[best_idx]
