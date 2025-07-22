from analysis.unified_prompt_optimizer import UnifiedPromptOptimizer
from analysis.prompt_optimizer import PromptOptimizer
from unittest.mock import patch


def test_unified_optimize_prompt():
    uni = UnifiedPromptOptimizer("distilgpt2")
    with patch.object(uni.advanced, "optimize_prompt", return_value="p1"), \
         patch.object(uni.bandit, "optimize_prompt", return_value="p2"), \
         patch.object(uni.annealer, "optimize_prompt", return_value="p3"), \
         patch.object(uni.rl, "optimize_prompt", return_value="p4"), \
         patch.object(uni.evolver, "evolve_prompt", return_value="p5"), \
         patch.object(PromptOptimizer, "score_prompt", side_effect=[5.0, 4.0, 3.0, 2.0, 1.0, 0.5]):
        best = uni.optimize_prompt("base")
    assert best == "p5"
