import sys
import types
import importlib
from unittest.mock import MagicMock
import numpy as np
import torch


def _load_absorber(monkeypatch):
    # Provide dummy external modules required for import
    for name in [
        "cv2",
        "soundfile",
        "faiss",
        "aiohttp",
        "requests",
        "torchvision",
        "torchvision.transforms",
        "sklearn.cluster",
        "sklearn.preprocessing",
        "interface.ws_server",
        "psutil",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = MagicMock()
    fake_tf.AutoModel = MagicMock()
    fake_tf.AutoProcessor = MagicMock()
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    fake_corr = types.ModuleType("correlation_rag_module")
    class DummyRag:
        def add(self, *a, **k):
            pass
    fake_corr.CorrelationRAGMemory = DummyRag
    monkeypatch.setitem(sys.modules, "correlation_rag_module", fake_corr)

    fake_pc = types.ModuleType("predictive_coding_temporal_model")
    fake_pc.PredictiveCodingTemporalModel = MagicMock()
    monkeypatch.setitem(sys.modules, "predictive_coding_temporal_model", fake_pc)

    fake_ws = types.ModuleType("interface.ws_server")
    fake_ws.run_ws_server = MagicMock()
    monkeypatch.setitem(sys.modules, "interface.ws_server", fake_ws)

    fake_dynamo = types.ModuleType("torch._dynamo")
    fake_dynamo.config = types.SimpleNamespace(skip_fsdp_hooks=True)
    fake_dynamo.disable = lambda fn, recursive=True: fn
    monkeypatch.setitem(sys.modules, "torch._dynamo", fake_dynamo)

    fake_sklearn_cluster = types.ModuleType("sklearn.cluster")
    fake_sklearn_cluster.DBSCAN = MagicMock()
    monkeypatch.setitem(sys.modules, "sklearn.cluster", fake_sklearn_cluster)

    fake_sklearn_preproc = types.ModuleType("sklearn.preprocessing")
    fake_sklearn_preproc.StandardScaler = MagicMock()
    monkeypatch.setitem(sys.modules, "sklearn.preprocessing", fake_sklearn_preproc)

    fake_dq = types.ModuleType("analysis.dataset_quality")
    fake_dq.score_record = MagicMock(return_value=(1.0, []))
    monkeypatch.setitem(sys.modules, "analysis.dataset_quality", fake_dq)
    return importlib.import_module("RealTimeDataAbsorber")


def test_absorb_data_adds_datapoints(monkeypatch):
    rtda = _load_absorber(monkeypatch)

    class DummyProcessor(rtda.ModalityProcessor):
        def __init__(self, modality):
            super().__init__(modality)
            self.embedding_dim = 3

        def process(self, data):
            return torch.tensor([[0.1, 0.2, 0.3]]), {}

        def extract_embeddings(self, data):
            return np.array([0.1, 0.2, 0.3])

    class DummyDetector:
        def detect_patterns(self, dp):
            return []

    monkeypatch.setattr(rtda, "TextProcessor", lambda *a, **k: DummyProcessor("text"))
    monkeypatch.setattr(rtda, "ImageProcessor", lambda *a, **k: DummyProcessor("image"))
    monkeypatch.setattr(rtda, "AudioProcessor", lambda *a, **k: DummyProcessor("audio"))
    monkeypatch.setattr(rtda, "PatternDetector", lambda *a, **k: DummyDetector())

    class DummyConn:
        def execute(self, *a, **k):
            pass
        def commit(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(rtda.sqlite3, "connect", lambda *a, **k: DummyConn())

    absorber = rtda.RealTimeDataAbsorber(model_config={}, settings={"db_path": ":memory:"})
    rag = MagicMock()
    absorber.attach_rag(rag)

    absorber.absorb_data("hello", "text")
    absorber.absorb_data(np.zeros((2, 2, 3)), "image")
    absorber.absorb_data(np.zeros(16000), "audio")

    assert len(absorber.data_buffer) == 3
    for dp in absorber.data_buffer:
        assert isinstance(dp, rtda.DataPoint)
    assert rag.add.call_count == 3


def test_predictive_coding_observe_step():
    from predictive_coding_temporal_model import PredictiveCodingTemporalModel

    pc = PredictiveCodingTemporalModel(embed_dim=4, context_window=2, device="cpu")
    for _ in range(3):
        pred, err = pc.observe_step(torch.randn(4))

    assert isinstance(pred, torch.Tensor)
    assert pred.shape == (1, 4)
    assert isinstance(err, float)

