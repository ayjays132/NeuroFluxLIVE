from analysis.prompt_bandit_optimizer import PromptBanditOptimizer
from analysis.prompt_optimizer import PromptOptimizer
from unittest.mock import patch


def test_bandit_optimize_prompt():
    # Reward is simply the prompt length
    bandit = PromptBanditOptimizer("distilgpt2", reward_fn=len, epsilon=0.0, iterations=3)
    with patch.object(bandit, "generate_variations", return_value=["a", "abcd"]), \
         patch.object(PromptOptimizer, "score_prompt", return_value=0.0):
        best = bandit.optimize_prompt("base", n_variations=2)
    assert best == "abcd"
