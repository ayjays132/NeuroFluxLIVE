from analysis.meta_prompt_optimizer import MetaPromptOptimizer
from analysis.prompt_optimizer import PromptOptimizer
from unittest.mock import patch


def test_meta_optimize_prompt():
    meta = MetaPromptOptimizer("distilgpt2")
    with patch.object(meta.bandit, "optimize_prompt", return_value="p1"), \
         patch.object(meta.annealer, "optimize_prompt", return_value="p2"), \
         patch.object(meta.rl, "optimize_prompt", return_value="p3"), \
         patch.object(PromptOptimizer, "score_prompt", side_effect=[3.0, 2.0, 4.0, 1.0]):
        best = meta.optimize_prompt("base")
    assert best == "p3"
