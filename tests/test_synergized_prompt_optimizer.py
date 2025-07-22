from analysis.synergized_prompt_optimizer import SynergizedPromptOptimizer
from unittest.mock import patch
from analysis.prompt_optimizer import PromptOptimizer


def test_synergized_optimize_prompt():
    syn = SynergizedPromptOptimizer("distilgpt2", iterations=2)
    with patch.object(syn.omni, "optimize_prompt", side_effect=["p1", "p2"]), \
         patch.object(syn, "generate_variations", return_value=["v1"]), \
         patch.object(PromptOptimizer, "score_prompt", side_effect=[1.0, 0.5]):
        best = syn.optimize_prompt("base")
    assert best == "v1"
