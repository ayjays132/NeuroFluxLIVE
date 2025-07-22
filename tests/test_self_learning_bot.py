from types import SimpleNamespace
from unittest.mock import MagicMock

from self_learning_bot import process_latest


def test_process_latest_text():
    dp = SimpleNamespace(modality="text", data="hello")
    absorber = SimpleNamespace(data_buffer=[dp])
    optimizer = MagicMock(optimize_prompt=MagicMock(return_value="best"))

    best = process_latest(absorber, optimizer)

    assert best == "best"
    optimizer.optimize_prompt.assert_called_with("hello")
