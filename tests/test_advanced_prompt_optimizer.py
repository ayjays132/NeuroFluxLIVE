import torch
from analysis.advanced_prompt_optimizer import AdvancedPromptOptimizer
from analysis.prompt_optimizer import PromptOptimizer
from unittest.mock import patch


def test_advanced_optimize_prompt():
    adv = AdvancedPromptOptimizer("distilgpt2", similarity_weight=1.0)
    with patch.object(adv, "generate_variations", return_value=["a", "b"]), \
         patch.object(PromptOptimizer, "score_prompt", side_effect=[2.0, 1.0]), \
         patch.object(adv, "_embedding", side_effect=[torch.tensor([0.0]), torch.tensor([0.2]), torch.tensor([0.4])]):
        best = adv.optimize_prompt("base")
    assert best == "b"
