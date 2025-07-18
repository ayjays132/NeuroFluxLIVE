import torch
from analysis.contextual_prompt_optimizer import ContextualPromptOptimizer
from analysis.prompt_optimizer import PromptOptimizer
from unittest.mock import patch


def test_contextual_score_prompt():
    with patch.object(PromptOptimizer, "__init__", return_value=None):
        inst = object.__new__(ContextualPromptOptimizer)
        inst.device = torch.device("cpu")
        inst.tokenizer = None
        inst.embed_model = None
        inst.dataset_embedding = torch.zeros(1)
        inst.similarity_weight = 0.5
        with patch.object(PromptOptimizer, "score_prompt", return_value=1.0), \
             patch.object(ContextualPromptOptimizer, "_embedding", return_value=torch.zeros(1)):
            score = ContextualPromptOptimizer.score_prompt(inst, "test")
    assert isinstance(score, float)
