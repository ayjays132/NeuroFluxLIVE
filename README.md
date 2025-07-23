# NeuroFluxLIVE

**NeuroFluxLIVE** is a prototype framework for building self‑learning applications. It fuses prompt optimisation, real‑time data ingestion and evaluation utilities into a single pipeline so a model can grow without relying on static datasets.

## Synergised Architecture
- **Unified prompt optimisation** – the `analysis` package contains multiple optimisers that are wrapped by `UnifiedPromptOptimizer` for evolutionary, bandit and RL strategies in one class.
- **Real‑time absorption** – `RealTimeDataAbsorber` streams text, image and audio inputs and updates models on the fly.
- **Datasetless workflows** – scripts such as `synergy_workflow.py` and `self_learning_bot.py` show how the system learns purely from live data.
- **Research utilities** – additional modules handle simulation, source evaluation, security checks and collaborative note sharing.

## Installation
1. Create a Python 3.10+ environment.
2. Install the package and dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e .            # install the `SelfResearch` package
   ```
   Optional extras for development can be installed with `pip install -r requirements-dev.txt`.

## Quick Start
Run the datasetless learning demo:
```bash
python3 synergy_workflow.py
```
Start the self‑learning bot with continuous optimisation:
```bash
python3 self_learning_bot.py
```
To explore every module including clustering and collaboration features:
```bash
python3 ultimate_workflow.py
```

## Metrics and Benchmarks
Example perplexity using `distilgpt2` on a tiny AG News subset:
```
Perplexity: 6785.07
```
`RealTimeDataAbsorber` also streams real‑time metrics over WebSockets for monitoring during long running sessions.

## Extending
The project is organised into clearly separated packages:
```
analysis/         prompt optimisers and dataset analytics
train/            minimal training loops
models/           transformer wrappers
research_workflow/  topic selection helpers
simulation_lab/   physics and biology simulations
digital_literacy/ source credibility checks
assessment/       rubric graders
peer_collab/      simple collaboration server
```
Create new modules in these folders or reuse the provided components. Settings are loaded from `config.yaml` and can be overridden with `--config` when running any workflow script.

## License
This repository is provided for research and experimentation purposes only.
