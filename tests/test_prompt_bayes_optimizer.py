from analysis.prompt_bayes_optimizer import BayesianPromptOptimizer
from analysis.prompt_optimizer import PromptOptimizer
from unittest.mock import patch


def test_bayesian_optimize_prompt():
    opt = BayesianPromptOptimizer("distilgpt2", iterations=2)
    with patch.object(opt, "generate_variations", side_effect=[["a"], ["b"]]), \
         patch.object(PromptOptimizer, "score_prompt", side_effect=[2.0, 1.0]), \
         patch("analysis.prompt_bayes_optimizer.gp_minimize") as gp_mock:
        def runner(func, space, n_calls, random_state=None):
            for _ in range(n_calls):
                func([1.0])
        gp_mock.side_effect = runner
        best = opt.optimize_prompt("base", n_variations=1)
    assert best == "b"
