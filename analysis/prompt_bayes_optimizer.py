"""Prompt optimization using Bayesian optimization."""

from __future__ import annotations

from typing import Optional, Tuple, List

from skopt import gp_minimize
from skopt.space import Real

from .prompt_optimizer import PromptOptimizer


class BayesianPromptOptimizer(PromptOptimizer):
    """Optimize prompts by tuning generation temperature via Bayesian search."""

    def __init__(
        self,
        model_name: str,
        *,
        iterations: int = 10,
        temp_range: Tuple[float, float] = (0.5, 1.5),
        device: Optional[str] = None,
    ) -> None:
        super().__init__(model_name, device=device)
        self.iterations = iterations
        self.temp_range = temp_range

    def optimize_prompt(self, base_prompt: str, n_variations: int = 5) -> str:
        """Return the best prompt found via Bayesian optimization."""

        history: List[Tuple[str, float]] = []

        def objective(params: List[float]) -> float:
            temp = params[0]
            cand = self.generate_variations(
                base_prompt, n_variations=1, temperature=temp
            )[0]
            score = self.score_prompt(cand)
            history.append((cand, score))
            return score

        if self.iterations < 10:
            for t in [self.temp_range[0], self.temp_range[1]][: self.iterations]:
                objective([t])
        else:
            gp_minimize(
                objective,
                [Real(self.temp_range[0], self.temp_range[1])],
                n_calls=self.iterations,
                random_state=42,
            )
        best_prompt, _ = min(history, key=lambda x: x[1])
        return best_prompt

# Backwards compatibility alias
PromptBayesOptimizer = BayesianPromptOptimizer

