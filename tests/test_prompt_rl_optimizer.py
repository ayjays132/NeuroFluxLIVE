from analysis.prompt_rl_optimizer import PromptRLOptimizer
from analysis.prompt_optimizer import PromptOptimizer
from unittest.mock import patch


def test_rl_optimize_prompt():
    # reward uses length of prompt
    rl = PromptRLOptimizer("distilgpt2", reward_fn=len, episodes=2, epsilon=0.0, lr=1.0)
    with patch.object(rl, "generate_variations", side_effect=[["a", "bb"], ["bb"]]), \
         patch.object(PromptOptimizer, "score_prompt", return_value=0.0):
        best = rl.optimize_prompt("base", n_variations=2)
    assert best == "bb"
