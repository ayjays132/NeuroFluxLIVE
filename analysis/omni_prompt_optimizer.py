from __future__ import annotations

"""Omni prompt optimizer combining every strategy."""

from typing import Optional, List

from .prompt_optimizer import PromptOptimizer
from .advanced_prompt_optimizer import AdvancedPromptOptimizer
from .prompt_bandit_optimizer import PromptBanditOptimizer
from .prompt_annealing_optimizer import PromptAnnealingOptimizer
from .prompt_rl_optimizer import PromptRLOptimizer
from .prompt_bayes_optimizer import BayesianPromptOptimizer
from .prompt_evolver import PromptEvolver
from .prompt_embedding_tuner import PromptEmbeddingTuner


class OmniPromptOptimizer(PromptOptimizer):
    """Fuse all prompt optimizers and select the best result."""

    def __init__(self, model_name: str, *, device: Optional[str] = None) -> None:
        super().__init__(model_name, device=device)
        self.advanced = AdvancedPromptOptimizer(model_name, device=self.device.type)
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
            steps=3,
            temperature=1.0,
            cooling=0.8,
        )
        self.rl = PromptRLOptimizer(
            model_name,
            reward_fn=lambda p: -self.score_prompt(p),
            device=self.device.type,
            episodes=3,
            epsilon=0.2,
            lr=0.3,
        )
        self.bayes = BayesianPromptOptimizer(
            model_name,
            device=self.device.type,
            iterations=3,
        )
        self.evolver = PromptEvolver(model_name, device=self.device.type)
        self.tuner = PromptEmbeddingTuner(model_name, prompt_length=5, device=self.device.type)

    def optimize_prompt(self, base_prompt: str, n_variations: int = 5) -> str:
        candidates: List[str] = [base_prompt]
        for opt in (
            self.advanced,
            self.bandit,
            self.annealer,
            self.rl,
            self.bayes,
        ):
            try:
                base_prompt = opt.optimize_prompt(base_prompt, n_variations=n_variations)
                candidates.append(base_prompt)
            except Exception:
                pass

        try:
            evolved = self.evolver.evolve_prompt(base_prompt, generations=2, population_size=n_variations)
            candidates.append(evolved)
        except Exception:
            pass

        try:
            self.tuner.tune(base_prompt, steps=1)
            tokens = " ".join(self.tuner.get_prompt_tokens())
            candidates.append(f"{tokens} {base_prompt}")
        except Exception:
            pass

        return min(candidates, key=self.score_prompt)
