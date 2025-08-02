import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
import soundfile as sf

from file_watcher import DataRootWatcher


def test_data_root_watcher(tmp_path: Path) -> None:
    # Create modality subdirectories
    for sub in ("text", "image", "audio", "sensor"):
        (tmp_path / sub).mkdir()

    # Text file
    (tmp_path / "text" / "a.txt").write_text("hello", encoding="utf-8")

    # Image file
    img = Image.new("RGB", (4, 4), color=(255, 0, 0))
    img.save(tmp_path / "image" / "b.png")

    # Audio file
    sf.write(tmp_path / "audio" / "c.wav", np.zeros(16000), 16000)

    # Sensor file
    (tmp_path / "sensor" / "d.json").write_text(json.dumps({"v": 1}), encoding="utf-8")

    # Unknown modality file should be ignored
    (tmp_path / "unknown.bin").write_bytes(b"\x00\x01")

    absorber = MagicMock()
    watcher = DataRootWatcher(absorber=absorber, data_root=tmp_path, interval=0.1)
    watcher.start()
    time.sleep(0.3)
    watcher.stop()

    assert not watcher._thread.is_alive()
    assert absorber.absorb_data.call_count == 4
    modalities = {call.args[1] for call in absorber.absorb_data.call_args_list}
    assert modalities == {"text", "image", "audio", "sensor"}

