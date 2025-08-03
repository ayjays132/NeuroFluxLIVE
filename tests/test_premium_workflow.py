"""Tests for the premium workflow pipeline."""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np

import premium_workflow


class DummyAbsorber:
    """Lightweight stand-in for :class:`RealTimeDataAbsorber`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.started = False

    def start_absorption(self) -> None:  # pragma: no cover - trivial
        self.started = True

    def stop_absorption(self) -> None:  # pragma: no cover - trivial
        self.started = False


class DummyTokenizer:
    class _Encoding(dict):
        def to(self, device: str) -> "DummyTokenizer._Encoding":  # pragma: no cover
            return self

    def __call__(self, prompt: str, return_tensors: str = "pt") -> "DummyTokenizer._Encoding":
        return self._Encoding({"input_ids": np.array([[1, 2]])})

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        return "hello"


class DummyModel:
    def to(self, device: str) -> "DummyModel":  # pragma: no cover
        return self

    def generate(self, **kwargs: Any) -> Any:  # pragma: no cover - simple
        return np.array([[1, 2, 3]])


class DummyGrader:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def grade_submission(self, text: str, rubric: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        return {"response": {"score": 0.5, "max_score": 1.0}}


def test_pipeline_initializes_and_logs(monkeypatch, caplog) -> None:
    """Pipeline should initialize all components and log metrics."""

    monkeypatch.setitem(sys.modules, "RealTimeDataAbsorber", SimpleNamespace(RealTimeDataAbsorber=DummyAbsorber))
    monkeypatch.setitem(
        sys.modules,
        "assessment.rubric_grader",
        SimpleNamespace(RubricGrader=DummyGrader),
    )
    monkeypatch.setattr(
        premium_workflow.AutoTokenizer, "from_pretrained", lambda *a, **k: DummyTokenizer()
    )
    monkeypatch.setattr(
        premium_workflow.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: DummyModel()
    )

    config = {
        "workflow": {
            "enable_absorber": True,
            "enable_rl": True,
            "enable_evaluator": True,
        }
    }

    caplog.set_level(logging.INFO)
    metrics = premium_workflow.run_pipeline(config, use_gym=True, max_steps=1)

    assert metrics["total_reward"] >= 0
    assert "RealTimeDataAbsorber started" in caplog.text
    assert "Episode 0 reward" in caplog.text

