from analysis.prompt_annealing_optimizer import PromptAnnealingOptimizer
from analysis.prompt_optimizer import PromptOptimizer
from unittest.mock import patch


def test_annealing_optimize_prompt():
    annealer = PromptAnnealingOptimizer("distilgpt2", temperature=1.0, cooling=0.5, steps=3)
    with patch.object(annealer, "generate_variations", side_effect=[["a"], ["b"], ["c"]]), \
         patch.object(PromptAnnealingOptimizer, "_score", side_effect=[3.0, 2.0, 1.5, 1.0]), \
         patch("analysis.prompt_annealing_optimizer.random.random", return_value=0.0):
        best = annealer.optimize_prompt("base", n_variations=1)
    assert best == "c"
