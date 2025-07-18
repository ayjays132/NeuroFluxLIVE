"""Callbacks for updating prompts during training."""

from __future__ import annotations

from typing import Optional

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from analysis.prompt_optimizer import PromptOptimizer


class PromptUpdateCallback(TrainerCallback):
    """Update training prompts at a fixed epoch interval."""

    def __init__(self, optimizer: PromptOptimizer, *, interval: int = 1, base_prompt: str = "") -> None:
        self.optimizer = optimizer
        self.interval = interval
        self.current_prompt = base_prompt

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if state.epoch is not None and (int(state.epoch) + 1) % self.interval == 0:
            self.current_prompt = self.optimizer.optimize_prompt(self.current_prompt)
            print(f"[PromptUpdateCallback] Updated prompt: {self.current_prompt}")
