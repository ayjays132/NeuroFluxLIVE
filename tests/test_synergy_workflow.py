import sys
import types
import synergy_workflow
from types import SimpleNamespace
from unittest.mock import patch


def test_main_runs(monkeypatch):
    monkeypatch.setattr(
        synergy_workflow.HyperPromptOptimizer,
        "optimize_with_absorber",
        lambda self, p: "prompt",
    )
    monkeypatch.setattr(
        synergy_workflow.HyperPromptOptimizer,
        "generate_variations",
        lambda self, p, n_variations=1: ["text"],
    )

    class DummyAbsorber:
        def __init__(self, *a, **k):
            self.data_buffer = []

        def absorb_data(self, *a, **k):
            self.data_buffer.append(SimpleNamespace(modality="text", data="ctx"))

    dummy_mod = types.ModuleType("RealTimeDataAbsorber")
    dummy_mod.RealTimeDataAbsorber = DummyAbsorber
    monkeypatch.setitem(sys.modules, "RealTimeDataAbsorber", dummy_mod)

    synergy_workflow.main()
