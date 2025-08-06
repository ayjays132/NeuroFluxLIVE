# 🚀 NeuroFluxLIVE – Premium Continuous Learning Pipeline

NeuroFluxLIVE fuses supervised fine‑tuning, reinforcement learning, and evolutionary self‑optimization into a single **always‑on** workflow. The centerpiece is [`premium_workflow.py`](premium_workflow.py), exposed as the `premium_workflow` command after installation.

## ✨ Features
- **Unified pipeline** – dataset analysis, Sprout‑AGI fine‑tuning, Gym RL training, and evolutionary learning all run from one script.
- **Model agnostic** – drop in any Hugging Face causal model (GPT‑2, LLaMA, Qwen, etc.).
- **Continuous adaptation** – `evolutionary_learner.py` tracks performance and mutates hyper‑parameters while the model serves requests.
- **Crash‑safe checkpoints** – tuned weights land in `sprout_benchmark/best_model` only when perplexity improves.

## 🔧 Installation
```bash
pip install -r requirements.txt
pip install -e .
```

## 🚀 Quick Start
Run the full showcase pipeline:
```bash
premium_workflow
```

### 🔬 Sprout‑AGI Benchmark
Evaluate **ayjays132/NeuroReasoner-1-NR-1** on the [Sprout‑AGI](https://huggingface.co/datasets/ayjays132/Sprout-AGI) dataset. The script compares baseline vs. fine‑tuned perplexity and preserves the better model.
```bash
premium_workflow --sprout-benchmark --model ayjays132/NeuroReasoner-1-NR-1 --prompt "Hello world"
```
Observed on CPU:

| Model | Baseline PPL | Tuned PPL |
|-------|--------------|-----------|
| ayjays132/NeuroReasoner-1-NR-1 | 66.44 | 8.94 |

Prompt **"Hello world"** produced:

| Stage | Continuation |
|-------|--------------|
| Baseline | Hello world] [DIFFICULTY:advanced][3.5/4]. rank; |
| Tuned | Hello world where space exploration is a global project, with diverse stakeholders and technologies to solve complex problems in time.' |

### 🕹️ Autonomous Gym Training
```bash
premium_workflow --gym --benchmark CartPole-v1
```
Runs a lightweight policy gradient agent inside Gym while language model components continue to respond.

### 🧬 Evolutionary Learning
To enable background self‑optimization, attach the evolutionary system to your agent:
```python
from evolutionary_learner import integrate_evolutionary_learning
learning_system = integrate_evolutionary_learning(agent)
```
The system periodically mutates hyper‑parameters, evaluates fitness, and persists the best checkpoints.

## 🛠 Development
Run unit tests (requires torch ≥2.6 for full pass):
```bash
pytest
```

## 📄 License
Research use only.
