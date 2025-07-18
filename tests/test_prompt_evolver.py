from analysis.prompt_evolver import PromptEvolver
from unittest.mock import patch


def test_evolve_prompt():
    evolver = PromptEvolver("distilgpt2")
    with patch.object(evolver, "_mutate", side_effect=["a", "b", "c", "d"] * 3), \
         patch.object(evolver, "_score", side_effect=[1.0, 0.5, 0.2, 0.3] * 3):
        best = evolver.evolve_prompt("base", generations=2, population_size=4)
    assert isinstance(best, str)
