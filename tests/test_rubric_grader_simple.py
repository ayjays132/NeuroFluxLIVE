from unittest.mock import patch, MagicMock
import torch
from assessment import rubric_grader


class DummyModel:
    config = type("C", (), {"hidden_size": 2})

    def to(self, device):
        return self

    def __call__(self, **kwargs):
        batch = kwargs["input_ids"].shape[0]
        seq_len = kwargs["input_ids"].shape[1]
        return (torch.ones(batch, seq_len, 2),)


class DummyTokenizer:
    model_max_length = 10

    def __call__(self, text, padding=True, truncation=True, return_tensors="pt", max_length=None):
        return {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}


def dummy_pipeline(*args, **kwargs):
    def _gen(*a, **k):
        return [{"generated_text": "feedback"}]

    return _gen


@patch("assessment.rubric_grader.AutoTokenizer.from_pretrained", return_value=DummyTokenizer())
@patch("assessment.rubric_grader.AutoModel.from_pretrained", return_value=DummyModel())
@patch("assessment.rubric_grader.pipeline", dummy_pipeline)
def test_grade_submission_basic(_, __):
    grader = rubric_grader.RubricGrader(device="cpu")
    rubric = {"Quality": {"expected_content": "good", "max_score": 1}}
    result = grader.grade_submission("good", rubric)
    assert "Quality" in result
    assert "overall_summary" in result
