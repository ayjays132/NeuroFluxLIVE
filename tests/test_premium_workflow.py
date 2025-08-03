import importlib
import sys
import types
from unittest.mock import MagicMock
import torch


def _load_workflow(monkeypatch):
    for name in [
        "cv2",
        "aiohttp",
        "psutil",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    ws_mod = types.ModuleType("interface.ws_server")
    ws_mod.run_ws_server = MagicMock()
    monkeypatch.setitem(sys.modules, "interface.ws_server", ws_mod)
    faiss_mod = types.ModuleType("faiss")
    faiss_mod.__spec__ = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "faiss", faiss_mod)
    tv_mod = types.ModuleType("torchvision")
    tv_mod.__spec__ = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torchvision", tv_mod)
    tv_t_mod = types.ModuleType("torchvision.transforms")
    tv_t_mod.__spec__ = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torchvision.transforms", tv_t_mod)
    sf_mod = types.ModuleType("soundfile")
    sf_mod.__spec__ = types.SimpleNamespace()
    sf_mod.__libsndfile_version__ = "1.0.31"
    monkeypatch.setitem(sys.modules, "soundfile", sf_mod)
    orig_find = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, package=None: types.SimpleNamespace()
        if name == "soundfile"
        else orig_find(name, package),
    )
    fake_cluster = types.ModuleType("sklearn.cluster")
    fake_cluster.DBSCAN = MagicMock()
    fake_cluster.KMeans = MagicMock()
    monkeypatch.setitem(sys.modules, "sklearn.cluster", fake_cluster)
    fake_preproc = types.ModuleType("sklearn.preprocessing")
    fake_preproc.StandardScaler = MagicMock()
    fake_preproc.KernelCenterer = MagicMock()
    monkeypatch.setitem(sys.modules, "sklearn.preprocessing", fake_preproc)
    fake_dq = types.ModuleType("analysis.dataset_quality")
    fake_dq.score_record = MagicMock(return_value=(1.0, []))
    monkeypatch.setitem(sys.modules, "analysis.dataset_quality", fake_dq)
    return importlib.import_module("premium_workflow")


def test_main_runs(monkeypatch):
    premium_workflow = _load_workflow(monkeypatch)
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

    premium_workflow.main([])


def test_autonomous_pipeline_runs(monkeypatch, capsys):
    premium_workflow = _load_workflow(monkeypatch)

    # Patch heavy functions as in main run
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

    class DummyAbsorber:
        started = False

        def __init__(self, *a, **k):
            pass

        def start_absorption(self):
            DummyAbsorber.started = True

        def stop_absorption(self):
            pass

        def absorb_data(self, *a, **k):
            pass

    class DummyTok:
        eos_token_id = 0

        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def __call__(self, text, return_tensors=None):
            class Batch(dict):
                def to(self, device):
                    return self

            return Batch({"input_ids": torch.tensor([[0]])})

        def decode(self, ids, skip_special_tokens=True):
            return "resp"

        def to(self, device):
            return self

    class DummyModel:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def to(self, device):
            return self

        def generate(self, **kwargs):
            return torch.tensor([[0]])

    class DummyGrader:
        def __init__(self, *a, **k):
            pass

        def grade_submission(self, text, rubric):
            return {"Quality": {"score": 1, "max_score": 5}}

    monkeypatch.setattr(premium_workflow, "RealTimeDataAbsorber", DummyAbsorber)
    monkeypatch.setattr(premium_workflow, "AutoModelForCausalLM", DummyModel)
    monkeypatch.setattr(premium_workflow, "AutoTokenizer", DummyTok)
    monkeypatch.setattr(
        premium_workflow,
        "run_gym_autonomous_trainer",
        lambda cfg: ([1.0], ["hi"]),
    )
    monkeypatch.setattr(premium_workflow, "RubricGrader", DummyGrader)

    premium_workflow.main(["--gym", "--benchmark", "sprout-agi"])
    captured = capsys.readouterr()
    assert "Episode rewards" in captured.out
    assert DummyAbsorber.started
