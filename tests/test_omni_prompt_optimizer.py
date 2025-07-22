from analysis.omni_prompt_optimizer import OmniPromptOptimizer
from analysis.prompt_optimizer import PromptOptimizer
from unittest.mock import patch


def test_omni_optimize_prompt():
    omni = OmniPromptOptimizer("distilgpt2")
    with patch.object(omni.advanced, "optimize_prompt", return_value="p1"), \
         patch.object(omni.bandit, "optimize_prompt", return_value="p2"), \
         patch.object(omni.annealer, "optimize_prompt", return_value="p3"), \
         patch.object(omni.rl, "optimize_prompt", return_value="p4"), \
         patch.object(omni.bayes, "optimize_prompt", return_value="p5"), \
         patch.object(omni.evolver, "evolve_prompt", return_value="p6"), \
         patch.object(omni.tuner, "tune"), \
         patch.object(omni.tuner, "get_prompt_tokens", return_value=["tok"]), \
         patch.object(PromptOptimizer, "score_prompt", side_effect=[6, 5, 4, 3, 2, 1, 0.5, 0]):
        best = omni.optimize_prompt("base")
    assert best == "tok base"
