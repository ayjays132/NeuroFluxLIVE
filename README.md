# NeuroFluxLIVE

**NeuroFluxLIVE** provides a unified pipeline for building self-learning agents without relying on static datasets. Real-time data ingestion is fused with several prompt optimisation strategies so models can adapt as new information arrives. Version 1.1.1 marks the latest production-ready release on PyPI.

## Key Features
- **Datasetless learning** – `RealTimeDataAbsorber` streams text, audio and images directly into optimisation routines.
- **Automated dataset retrieval** – `AutoDataCollector` searches the Hugging Face Hub and downloads datasets without manual steps.
- **Unified prompt optimisation** – `UnifiedPromptOptimizer` combines evolutionary, bandit, annealing and RL strategies under one interface.
- **Research utilities** – tools for simulation, source evaluation, ethics management and collaboration are included.
- **Modular design** – packages under `analysis`, `train`, `models`, `digital_literacy` and more can be used independently or together.

## Installation
1. Ensure Python 3.10+ is available.
2. Install from PyPI:
   ```bash
   pip install SelfResearch
   ```
   Or install the package from source:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
The requirements files pin `numpy>=1.23,<2.0` to maintain compatibility with PyTorch.
Development extras and tests can be installed with:
```bash
pip install -r requirements-dev.txt
```

## Usage
Run the core datasetless demo (which demonstrates persistent memory across
sessions):
```bash
python synergy_workflow.py
```
Launch the self-learning bot with live metric streaming:
```bash
python self_learning_bot.py
```
Enable the interactive dashboard with SVG metrics:
```bash
python self_learning_bot.py --visual
```
For an end-to-end demonstration that ties everything together, see the
[Ultimate Workflow](#ultimate-workflow) section below.
Configuration defaults live in `config.yaml` and may be overridden with `--config`.

## Ultimate Workflow
`ultimate_workflow.py` orchestrates all major modules in a single run. It
streams data through `RealTimeDataAbsorber`, analyzes the resulting dataset,
trains and evaluates a language model, runs physics simulations, and interacts
with the collaboration server so results can be shared across sessions.

```bash
python ultimate_workflow.py
```

Because the script downloads datasets and model weights, performs clustering
and t-SNE, executes short training loops, and calls the collaboration server,
it benefits from a CUDA‑enabled GPU and at least **8 GB** of system RAM. On a
CPU-only machine the demo may take several minutes to complete.

### Embedding Compression
`VAECompressor` can be attached to `RealTimeDataAbsorber` to reduce the size of stored embeddings. Pass an instance when constructing the absorber. The compressor trains a small variational autoencoder so embeddings are encoded to a latent vector and reconstructed only when accessed:

```python
from models.vae_compressor import VAECompressor
from RealTimeDataAbsorber import RealTimeDataAbsorber

compressor = VAECompressor(latent_dim=32)
absorber = RealTimeDataAbsorber(model_config={}, compressor=compressor)
```
Embeddings will be compressed before being cached and transparently decompressed when accessed.

When `CorrelationRAGMemory` is constructed with a `VAECompressor` and a
`save_path`, compressed memories are persisted on disk. Calling
`RealTimeDataAbsorber.log_performance_metrics()` will periodically save this
state so that subsequent runs can reload it automatically.

## GPT-2 Benchmark 🚀
Using the `eval/gpt2_benchmark.py` helper we fine-tuned GPT-2 on a 100-sample slice of the AG News dataset and measured perplexity on a held-out subset:

| Model | Perplexity (AG News test[:50]) |
|-------|-------------------------------|
| GPT-2 baseline | 67.96 |
| GPT-2 fine-tuned ⚙️ | 51.94 |

Prompt `Breaking news:` showcased the improvement:

> **Baseline:** Breaking news: The FBI has released a list of the people who have been arrested in connection with the shooting death of a black man in Ferguson, Missouri.

> **Fine-tuned:** Breaking news: The U.S. Supreme Court has ruled that the government can't force a company to pay for a product that it says is defective. The case is the latest in a series of cases that have been brought by companies that have sued

`RealTimeDataAbsorber` exposes live metrics over WebSockets during extended sessions.

## Project Layout
```
analysis/         prompt optimisers and dataset analytics
train/            minimal training loops
models/           transformer wrappers
digital_literacy/ source credibility checks
simulation_lab/   physics and biology simulations
assessment/       rubric graders
peer_collab/      collaboration server
```
Create new modules under these directories and adjust settings in `config.yaml`.

## Testing
Run the unit tests before committing changes:
```bash
pytest
```
Any failures will be reported during collection or execution.

## Repository
The project is maintained at [https://github.com/ayjays132/NeuroFluxLIVE](https://github.com/ayjays132/NeuroFluxLIVE).

## License
This repository is provided for research and experimentation purposes only.
