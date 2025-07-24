# Automatic Data Retrieval

The `AutoDataCollector` utility automates the process of searching and
retrieving high quality datasets from the Hugging Face Hub. It is designed to
work with the existing dataset loaders so that newly fetched data can be quickly
used for training and analysis.

## Features
- **Query based search** – datasets are discovered with a search term using the
  `datasets` library.
- **Batch downloading** – multiple datasets can be fetched at once and saved in
  modality‑specific folders.
- **Multimodal support** – text, image, audio and numeric sensor fields are
  handled automatically.
- **Asynchronous retrieval** – downloads are performed concurrently with
  `aiohttp`.
- **Integration ready** – saved files can be loaded with
  `data.dataset_loader.load_local_json_dataset` or streamed directly into
  `RealTimeDataAbsorber`.

## Usage
```bash
python -m data.auto_data_collector --query "news" --max_datasets 3 --output_dir ./datasets
```
This command searches for datasets containing "news" in their name and stores
the selected split in `./datasets`.

To run retrieval every hour and fetch images and audio where available:

```bash
python -m data.auto_data_collector --query "environment" --max_datasets 2 \
    --split train --output_dir ./env_data --schedule 60
```

## Data Quality Tips
1. **Inspect metadata** – review the dataset card on Hugging Face for licensing
   and source information.
2. **Deduplicate** – use `drop_duplicates` in `dataset_loader.load_and_tokenize`
   to remove repeated samples.
3. **Track provenance** – keep a log of dataset names and versions for
   reproducibility.
4. **Validate content** – run `digital_literacy/source_evaluator.py` on samples
   to check for credibility and bias.
5. **Check media quality** – confirm image resolution and audio sample rates are
   consistent before training.

For large scale ingestion, combine the collector with the `RealTimeDataAbsorber`
so that models continuously adapt to new, validated data sources.
