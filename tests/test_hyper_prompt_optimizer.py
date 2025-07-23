from types import SimpleNamespace
from unittest.mock import patch

from analysis.hyper_prompt_optimizer import HyperPromptOptimizer


def test_optimize_with_absorber():
    absorber = SimpleNamespace(data_buffer=[SimpleNamespace(modality="text", data="ctx")])
    opt = HyperPromptOptimizer("distilgpt2", absorber=absorber, iterations=1)
    with patch.object(opt, "optimize_prompt", return_value="base"):
        result = opt.optimize_with_absorber("prompt")
    assert result.startswith("ctx")
