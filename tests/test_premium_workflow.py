import types
from unittest.mock import MagicMock

import torch

import premium_workflow


def test_main_runs(monkeypatch):
    monkeypatch.setattr(premium_workflow, "load_and_tokenize", lambda *a, **k: [])
    monkeypatch.setattr(
        premium_workflow, "analyze_tokenized_dataset", lambda *a, **k: {"samples": 0}
    )
    monkeypatch.setattr(premium_workflow, "train_model", lambda cfg: None)
    monkeypatch.setattr(premium_workflow, "evaluate_perplexity", lambda *a, **k: 0.0)
    monkeypatch.setattr(
        premium_workflow.TopicSelector, "suggest_topic", lambda self, x: "Topic"
    )
    monkeypatch.setattr(
        premium_workflow.TopicSelector, "validate_question", lambda self, q: True
    )
    monkeypatch.setattr(
        premium_workflow.SourceEvaluator,
        "evaluate_source",
        lambda self, url: {"credibility": "high"},
    )
    monkeypatch.setattr(
        premium_workflow.ExperimentSimulator,
        "run_physics_simulation",
        lambda self, *a, **k: torch.tensor([0.0]),
    )
    monkeypatch.setattr(
        premium_workflow.RubricGrader,
        "grade_submission",
        lambda self, t, r: {"Quality": {"score": 5, "max_score": 5}},
    )
    monkeypatch.setattr(
        premium_workflow.AuthAndEthics, "register_user", lambda self, *a: True
    )
    monkeypatch.setattr(
        premium_workflow.AuthAndEthics, "authenticate_user", lambda self, *a: True
    )
    monkeypatch.setattr(
        premium_workflow.AuthAndEthics,
        "flag_ethical_concern",
        lambda self, *a, **k: None,
    )
    monkeypatch.setattr(
        premium_workflow.AuthAndEthics, "get_ethical_flags", lambda self: []
    )

    premium_workflow.main()
