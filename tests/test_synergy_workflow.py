import synergy_workflow
from unittest.mock import patch


def test_main_runs(monkeypatch):
    monkeypatch.setattr(
        synergy_workflow.SynergizedPromptOptimizer,
        "optimize_prompt",
        lambda self, p: "prompt",
    )
    monkeypatch.setattr(
        synergy_workflow.SynergizedPromptOptimizer,
        "generate_variations",
        lambda self, p, n_variations=1: ["text"],
    )
    synergy_workflow.main()
