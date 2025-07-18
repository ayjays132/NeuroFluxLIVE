from analysis.prompt_optimizer import PromptOptimizer
from train.prompt_scheduler import PromptUpdateCallback
from transformers import TrainerState, TrainerControl
from unittest.mock import MagicMock


def test_prompt_update_callback():
    opt = MagicMock(spec=PromptOptimizer)
    opt.optimize_prompt.return_value = "new"
    callback = PromptUpdateCallback(opt, interval=1, base_prompt="base")
    state = TrainerState()
    state.epoch = 0
    control = TrainerControl()
    callback.on_epoch_end(None, state, control)
    opt.optimize_prompt.assert_called_with("base")
    assert callback.current_prompt == "new"
