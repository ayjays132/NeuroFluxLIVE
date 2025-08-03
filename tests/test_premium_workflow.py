from unittest.mock import MagicMock
import sys
import types

import premium_workflow


def test_pipeline_initializes_and_logs(monkeypatch):
    fake_absorber = MagicMock()
    dummy_module = types.SimpleNamespace(RealTimeDataAbsorber=lambda *a, **k: fake_absorber)
    monkeypatch.setitem(sys.modules, "RealTimeDataAbsorber", dummy_module)

    def fake_trainer(cfg, absorber=None, evaluator=None):
        assert absorber is fake_absorber
        assert evaluator is not None
        evaluator("test")
        return [1.0], ["resp"]

    monkeypatch.setattr(
        premium_workflow, "run_gym_autonomous_trainer", fake_trainer
    )

    monkeypatch.setattr(premium_workflow, "query_model", lambda *a, **k: "ok")
    monkeypatch.setattr(premium_workflow, "AutoTokenizer", MagicMock())
    monkeypatch.setattr(premium_workflow, "AutoModelForCausalLM", MagicMock())
    monkeypatch.setattr(premium_workflow, "RubricGrader", MagicMock())
    monkeypatch.setattr(
        premium_workflow,
        "_load_config",
        lambda path="config.yaml": {
            "paths": {"db_path": ":memory:"},
            "workflow": {
                "enable_absorber": True,
                "enable_rl_trainer": True,
                "enable_evaluator": True,
            },
        },
    )

    premium_workflow.main(["--gym"])
    assert fake_absorber.log_performance_metrics.called

