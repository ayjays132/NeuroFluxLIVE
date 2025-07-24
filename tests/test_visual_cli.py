import time
from queue import Queue
from pathlib import Path

from interface.visual_cli import VisualCLI


def test_visual_cli_runs(tmp_path: Path) -> None:
    q: Queue = Queue()
    viewer = VisualCLI(q, svg_path=tmp_path / "m.svg")
    viewer.start()
    q.put({"metric": 1})
    time.sleep(0.2)
    viewer.stop()
    assert (tmp_path / "m.svg").exists()

