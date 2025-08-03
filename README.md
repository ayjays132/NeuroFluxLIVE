# NeuroFluxLIVE

**NeuroFluxLIVE** is a unified research pipeline for building self‑learning agents that adapt from live streams instead of fixed datasets.

## Overview
NeuroFluxLIVE combines real‑time data ingestion, online evaluation, and reinforcement learning so models can evolve continuously. Components under `analysis/`, `models/`, `train/`, and `simulation_lab/` interoperate or can be used independently.

## Quickstart 🚀
Run the bot and watch it learn while answering your questions—no static datasets required!
```bash
python self_learning_bot.py  # ask it anything 🧠
```

## Installation
1. Ensure Python 3.10+ is available.
2. Install from PyPI:
   ```bash
   pip install SelfResearch
   ```
   Or install from source:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```
   Development extras:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Unified Pipeline (RealTimeDataAbsorber + Gym RL training)
[RealTimeDataAbsorber.py](RealTimeDataAbsorber.py) streams multimodal data and exposes live metrics. The Gym autonomous trainer [simulation_lab/gym_autonomous_trainer.py](simulation_lab/gym_autonomous_trainer.py) shows how GPT‑2 can learn control policies while generating text responses.
```bash
python -c "from simulation_lab.gym_autonomous_trainer import run_gym_autonomous_trainer; run_gym_autonomous_trainer()"
```

## Sprout‑AGI Benchmark
Evaluate tuned models against the Sprout‑AGI dataset with [eval/benchmark_sprout_agi.py](eval/benchmark_sprout_agi.py) (see [docs/benchmarks.md](docs/benchmarks.md) for details).
```bash
python eval/benchmark_sprout_agi.py ayjays132/CustomGPT2Conversational --split "test[:50]" --prompt "Hello AGI"
```

| Experiment | Metric | Baseline GPT‑2 | Tuned GPT‑2 |
|------------|--------|----------------|-------------|
| Sprout‑AGI Benchmark | Perplexity ↓ | 68.92 | 40.32 |
| Gym Autonomous Trainer | Avg Return ↑ | 13.7 | 11.0 |

## Usage Examples
- Core datasetless demo:
  ```bash
  python synergy_workflow.py
  ```
- Visual self‑learning bot:
  ```bash
  python self_learning_bot.py --visual
  ```
- End‑to‑end orchestration via [ultimate_workflow.py](ultimate_workflow.py):
  ```bash
  python ultimate_workflow.py
  ```

## Testing
Run unit tests before committing changes:
```bash
pytest
```

## License
This repository is provided for research and experimentation purposes only.
