import types
from unittest.mock import MagicMock
import torch
import ultimate_workflow


def test_main_runs(monkeypatch):
    monkeypatch.setattr(ultimate_workflow, "load_and_tokenize", lambda *a, **k: [])
    monkeypatch.setattr(
        ultimate_workflow, "analyze_tokenized_dataset", lambda *a, **k: {"samples": 0}
    )
    monkeypatch.setattr(
        ultimate_workflow,
        "cluster_dataset_embeddings",
        lambda *a, **k: (MagicMock(), [0]),
    )
    monkeypatch.setattr(
        ultimate_workflow, "compute_tsne_embeddings", lambda *a, **k: [[0.0, 0.0]]
    )
    monkeypatch.setattr(ultimate_workflow, "train_model", lambda cfg: None)
    monkeypatch.setattr(ultimate_workflow, "evaluate_perplexity", lambda *a, **k: 0.0)
    monkeypatch.setattr(
        ultimate_workflow.TopicSelector, "suggest_topic", lambda self, x: "Topic"
    )
    monkeypatch.setattr(
        ultimate_workflow.TopicSelector, "validate_question", lambda self, q: True
    )
    monkeypatch.setattr(
        ultimate_workflow.SourceEvaluator,
        "evaluate_source",
        lambda self, url: {"credibility": "high"},
    )
    monkeypatch.setattr(
        ultimate_workflow.ExperimentSimulator,
        "run_physics_simulation",
        lambda self, *a, **k: torch.tensor([0.0]),
    )
    monkeypatch.setattr(
        ultimate_workflow.RubricGrader,
        "grade_submission",
        lambda self, t, r: {"Quality": {"score": 5, "max_score": 5}},
    )
    monkeypatch.setattr(
        ultimate_workflow.AuthAndEthics, "register_user", lambda self, *a: True
    )
    monkeypatch.setattr(
        ultimate_workflow.AuthAndEthics, "authenticate_user", lambda self, *a: True
    )
    monkeypatch.setattr(
        ultimate_workflow.AuthAndEthics,
        "flag_ethical_concern",
        lambda self, *a, **k: None,
    )
    monkeypatch.setattr(
        ultimate_workflow.AuthAndEthics, "get_ethical_flags", lambda self: []
    )
    monkeypatch.setattr(
        ultimate_workflow.requests,
        "get",
        lambda *a, **k: types.SimpleNamespace(ok=True, json=lambda: {"notes": "x"}),
    )

    ultimate_workflow.main()
