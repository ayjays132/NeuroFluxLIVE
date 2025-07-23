# NeuroFluxLIVE

NeuroFluxLIVE is a research playground for rapid experimentation with transformer models. All modules work together so the system can learn from new information in real time without depending on large static datasets. The project demonstrates how prompt optimisation, on‑the‑fly data ingestion and evaluation utilities combine into a single streamlined workflow.

## Features
- **Real‑time learning** – `RealTimeDataAbsorber` continuously trains a model as new text, image or audio streams arrive.
- **Universal prompt optimisers** – the `analysis` package provides evolutionary, bandit and reinforcement learning strategies to improve prompts automatically.
- **Modular workflows** – example scripts (`main.py`, `premium_workflow.py`, `full_demo.py`, `ultimate_workflow.py`) show how to compose modules for different research goals.
- **Datasetless operation** – `synergy_workflow.py` and `self_learning_bot.py` demonstrate how the system grows its knowledge solely from live data.
- **Evaluation tools** – use `eval/language_model_evaluator.py` for perplexity checks and `analysis/dataset_analyzer.py` for dataset statistics and embedding clustering.
- **Collaboration server** – `peer_collab/` enables shared notes and feedback during experiments.

## Installation
1. Create a Python&nbsp;3.10+ environment.
2. Install the core and development dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
PyTorch automatically selects CUDA if available.

## Quick Start
Run the synergy workflow to see datasetless learning in action:
```bash
python3 synergy_workflow.py
```
For a minimal continuously learning bot:
```bash
python3 self_learning_bot.py
```
To explore the entire platform with clustering, perplexity evaluation and collaboration features:
```bash
python3 ultimate_workflow.py
```

## Metrics and Benchmarks
- **Perplexity** – `eval/language_model_evaluator.py` measures language model perplexity on any HuggingFace dataset.
- **Dataset statistics** – `analysis/dataset_analyzer.py` reports lexical diversity, token entropy and clustering quality.
- **Real‑time metrics** – `RealTimeDataAbsorber` streams performance indicators over WebSockets for monitoring.

The project ships with unit tests to ensure each optimizer and workflow behaves as expected. Run them after installing the requirements:
```bash
pytest
```

## Extending the System
The codebase is organised so new modules can plug into the existing pipeline:
```
research_workflow/    topic selection utilities
digital_literacy/     source evaluation and academic search
simulation_lab/       physics and biology simulations
assessment/           rubric-based grading
peer_collab/          collaboration server
analysis/             prompt optimisers and dataset analytics
train/                simple training loops
models/               wrappers around transformer models
```
Edit these components or create new ones to suit your research needs. `config.yaml` provides default settings that any workflow can override with `--config`.

## License
This repository is provided for research and experimentation purposes only.
