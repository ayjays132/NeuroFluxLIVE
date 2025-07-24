# Automatic Data Retrieval

The `AutoDataCollector` utility automates the process of searching and
retrieving high quality datasets from the Hugging Face Hub. It is designed to
work with the existing dataset loaders so that newly fetched data can be quickly
used for training and analysis.

## Features
- **Query based search** – datasets are discovered with a search term using the
  `datasets` library.
- **Batch downloading** – multiple datasets can be fetched at once and saved in
  JSONL format.
- **Integration ready** – saved files can be loaded with
  `data.dataset_loader.load_local_json_dataset` or streamed directly into
  `RealTimeDataAbsorber`.

## Usage
```bash
python -m data.auto_data_collector --query "news" --max_datasets 3 --output_dir ./datasets
```
This will search for datasets containing "news" in their name, download up to
three of them and store each split as a JSONL file in `./datasets`.

## Data Quality Tips
1. **Inspect metadata** – review the dataset card on Hugging Face for licensing
   and source information.
2. **Deduplicate** – use `drop_duplicates` in `dataset_loader.load_and_tokenize`
   to remove repeated samples.
3. **Track provenance** – keep a log of dataset names and versions for
   reproducibility.
4. **Validate content** – run `digital_literacy/source_evaluator.py` on samples
   to check for credibility and bias.

For large scale ingestion, combine the collector with the `RealTimeDataAbsorber`
so that models continuously adapt to new, validated data sources.
