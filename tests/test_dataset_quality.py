from datasets import Dataset

from analysis.dataset_quality import evaluate_dataset, score_record


def test_evaluate_dataset():
    ds = Dataset.from_dict({
        "text": ["a", "b", "", "a"],
        "value": [1, 5, None, 10],
    })
    report = evaluate_dataset(ds, key_columns=["text"], numeric_ranges={"value": (0, 5)})
    assert report["duplicates"] == 1
    assert report["empty_fields"]["text"] == 1
    assert report["empty_fields"]["value"] == 1
    assert report["out_of_range"]["value"] == 1


def test_score_record_duplicate():
    seen = set()
    rec = {"text": "hi"}
    score1, issues1 = score_record(rec, seen)
    score2, issues2 = score_record(rec, seen)
    assert "duplicate" in issues2
    assert score2 < score1

