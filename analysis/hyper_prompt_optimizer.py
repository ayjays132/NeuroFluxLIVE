from __future__ import annotations
"""Universal prompt optimizer leveraging real-time data."""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from RealTimeDataAbsorber import RealTimeDataAbsorber

from .synergized_prompt_optimizer import SynergizedPromptOptimizer


class HyperPromptOptimizer(SynergizedPromptOptimizer):
    """Combine :class:`SynergizedPromptOptimizer` with real-time context."""

    def __init__(
        self,
        model_name: str,
        *,
        absorber: Optional["RealTimeDataAbsorber"] = None,
        iterations: int = 2,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(model_name, iterations=iterations, device=device)
        self.absorber = absorber

    def optimize_with_absorber(self, base_prompt: str, n_variations: int = 5) -> str:
        """Optimize ``base_prompt`` and prepend the latest absorbed text if available."""
        prompt = self.optimize_prompt(base_prompt, n_variations=n_variations)
        if self.absorber and getattr(self.absorber, "data_buffer", None):
            latest = self.absorber.data_buffer[-1]
            if getattr(latest, "modality", "") == "text":
                prompt = f"{latest.data}\n\n{prompt}"
        return prompt
