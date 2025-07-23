# NeuroFluxLIVE

**NeuroFluxLIVE** provides a unified pipeline for building self-learning agents without relying on static datasets. Real-time data ingestion is fused with several prompt optimisation strategies so models can adapt as new information arrives. Version 1.0.0 marks our first production-ready release on PyPI.

## Key Features
- **Datasetless learning** – `RealTimeDataAbsorber` streams text, audio and images directly into optimisation routines.
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
Run the core datasetless demo:
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

## License
This repository is provided for research and experimentation purposes only.
