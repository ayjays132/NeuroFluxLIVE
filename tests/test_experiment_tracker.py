from train.experiment_tracker import ExperimentTracker


def test_experiment_tracker(tmp_path):
    tracker = ExperimentTracker()
    tracker.log(step=1, loss=0.5)
    tracker.log(step=2, loss=0.4)
    out_file = tmp_path / "log.json"
    tracker.save(out_file)
    data = out_file.read_text()
    assert "loss" in data
    assert "step" in data
