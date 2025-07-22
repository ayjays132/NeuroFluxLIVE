from __future__ import annotations

"""Unified prompt optimization combining several strategies."""

from typing import Optional, List

from .prompt_optimizer import PromptOptimizer
from .advanced_prompt_optimizer import AdvancedPromptOptimizer
from .prompt_bandit_optimizer import PromptBanditOptimizer
from .prompt_annealing_optimizer import PromptAnnealingOptimizer
from .prompt_rl_optimizer import PromptRLOptimizer
from .prompt_evolver import PromptEvolver


class UnifiedPromptOptimizer(PromptOptimizer):
    """Fuse multiple optimizers and select the best prompt."""

    def __init__(
        self,
        model_name: str,
        *,
        embedding_model: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(model_name, device=device)
        self.advanced = AdvancedPromptOptimizer(
            model_name,
            embedding_model=embedding_model,
            device=self.device.type,
        )
        self.bandit = PromptBanditOptimizer(
            model_name,
            reward_fn=lambda p: -self.score_prompt(p),
            device=self.device.type,
            iterations=3,
            epsilon=0.2,
        )
        self.annealer = PromptAnnealingOptimizer(
            model_name,
            device=self.device.type,
            temperature=1.0,
            cooling=0.8,
            steps=3,
        )
        self.rl = PromptRLOptimizer(
            model_name,
            reward_fn=lambda p: -self.score_prompt(p),
            device=self.device.type,
            episodes=3,
            epsilon=0.2,
            lr=0.3,
        )
        self.evolver = PromptEvolver(model_name, device=self.device.type)

    def optimize_prompt(self, base_prompt: str, n_variations: int = 5) -> str:
        candidates: List[str] = [base_prompt]
        try:
            base_prompt = self.advanced.optimize_prompt(base_prompt, n_variations=n_variations)
            candidates.append(base_prompt)
        except Exception:
            pass
        try:
            base_prompt = self.bandit.optimize_prompt(base_prompt, n_variations=n_variations)
            candidates.append(base_prompt)
        except Exception:
            pass
        try:
            base_prompt = self.annealer.optimize_prompt(base_prompt, n_variations=n_variations)
            candidates.append(base_prompt)
        except Exception:
            pass
        try:
            base_prompt = self.rl.optimize_prompt(base_prompt, n_variations=n_variations)
            candidates.append(base_prompt)
        except Exception:
            pass
        try:
            evolved = self.evolver.evolve_prompt(
                base_prompt, generations=2, population_size=n_variations
            )
            candidates.append(evolved)
        except Exception:
            pass

        return min(candidates, key=self.score_prompt)
