from __future__ import annotations

"""Experiment tracking utilities."""

from dataclasses import dataclass, field
from typing import Any, Dict, List
import json


@dataclass
class ExperimentTracker:
    """Log training metrics and save them to disk."""

    history: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, **metrics: Any) -> None:
        self.history.append(metrics)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


class TrackerCallback:
    """HuggingFace Trainer callback that records logs to an ExperimentTracker."""

    def __init__(self, tracker: ExperimentTracker) -> None:
        self.tracker = tracker

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if logs:
            entry = {"step": state.global_step}
            entry.update(logs)
            self.tracker.log(**entry)
