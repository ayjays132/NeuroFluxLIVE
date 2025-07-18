# SelfResearch/train/prompt_update_callback.py

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from analysis.prompt_optimizer import PromptOptimizer

class PromptUpdateCallback(TrainerCallback):
    """
    Update training prompts at a fixed epoch interval, using a PromptOptimizer.
    """

    def __init__(
        self,
        optimizer: PromptOptimizer,
        *,
        interval: int = 1,
        base_prompt: str = "",
    ) -> None:
        super().__init__()
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
        # Only update when an integer epoch has completed
        if state.epoch is not None and (int(state.epoch) + 1) % self.interval == 0:
            new_prompt = self.optimizer.optimize_prompt(self.current_prompt)
            # Log to console (or hook into your logging system)
            print(f"[PromptUpdateCallback] Updated prompt: {new_prompt}")
            self.current_prompt = new_prompt
