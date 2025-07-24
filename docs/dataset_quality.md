# Dataset Quality Evaluation

The `analysis.dataset_quality` module scores individual records and
provides reports on entire datasets. It can be used stand‑alone or
combined with `AutoDataCollector` to filter low quality samples before
training.

## Scoring a record
```python
from analysis.dataset_quality import score_record

record = {"text": "hello world"}
score, issues = score_record(record)
```

## Evaluating a dataset file
```bash
python -m analysis.dataset_quality path/to/data.jsonl
```
The script loads a local JSONL file or a dataset from the Hugging Face
Hub and prints a summary of duplicates, empty fields and out‑of‑range
values.
