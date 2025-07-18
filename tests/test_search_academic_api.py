import types
import torch
from digital_literacy import source_evaluator


def test_search_academic_api_uses_https(monkeypatch):
    called_urls = []

    def dummy_init(self, device="cpu"):
        self.device = torch.device(device)

    def fake_get(url, timeout=15):
        called_urls.append(url)
        return types.SimpleNamespace(text="<feed></feed>", raise_for_status=lambda: None)

    class DummySoup:
        def find_all(self, *a, **k):
            return []

    monkeypatch.setattr(source_evaluator.SourceEvaluator, "__init__", dummy_init)
    monkeypatch.setattr(source_evaluator.requests, "get", fake_get)
    monkeypatch.setattr(source_evaluator, "BeautifulSoup", lambda *a, **k: DummySoup())

    se = source_evaluator.SourceEvaluator(device="cpu")
    se.search_academic_api("test", api_type="arxiv")

    assert called_urls
    assert called_urls[0].startswith("https://export.arxiv.org")
