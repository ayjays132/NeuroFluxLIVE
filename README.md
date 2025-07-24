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
The full showcase of every module is available via:
```bash
python ultimate_workflow.py
```
Configuration defaults live in `config.yaml` and may be overridden with `--config`.

### Embedding Compression
`VAECompressor` can be attached to `RealTimeDataAbsorber` to reduce the size of stored embeddings. Pass an instance when constructing the absorber:

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

## Metrics
A quick benchmark with `distilgpt2` on a tiny AG News subset results in:
```
Perplexity: 6785.07
```
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
