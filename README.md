# 🌟 NeuroFluxLIVE Premium Pipeline

**NeuroFluxLIVE** merges streaming data absorption, language model fine‑tuning and reinforcement learning into one seamless workflow. The heart of the system is [`premium_workflow.py`](premium_workflow.py), exposed as a console script called `premium_workflow` after installation.

## 📦 Installation
```bash
pip install -r requirements.txt
pip install -e .
```

## 🏁 Quick Start
Run the default demonstration:
```bash
premium_workflow
```

### Sprout‑AGI Benchmark
Fine‑tune any GPT‑style model on the [Sprout‑AGI](https://huggingface.co/datasets/ayjays132/Sprout-AGI) reasoning set and view perplexity shifts and prompt quality.
```bash
premium_workflow --sprout-benchmark --model ayjays132/NeuroReasoner-1-NR-1 --prompt "The future of AI is"
```
Results with `ayjays132/NeuroReasoner-1-NR-1` on a small sample:

| Model | Baseline PPL | Tuned PPL |
|-------|--------------|-----------|
| ayjays132/NeuroReasoner-1-NR-1 | 24.79 | 13.94 |

Prompt **"The future of AI is"**:

| Stage | Continuation |
|-------|-------------|
| Baseline | *The future of AI is uncertain, with growing concerns about bias and privacy. What strategies can be implemented to mitigate these risks?* |
| Tuned | *The future of AI is a complex, interdependent process that requires thoughtful planning and continuous iteration.* |

### Autonomous Gym Training
```bash
premium_workflow --gym --benchmark CartPole-v1
```
Trains a lightweight policy head inside a Gym environment while querying the language model and streaming metrics.

## 🧪 Component Overview
- **RealTimeDataAbsorber** for live ingestion.
- **train.trainer** for supervised fine‑tuning.
- **eval.language_model_evaluator** for perplexity metrics.
- **simulation_lab.gym_autonomous_trainer** for RL demos.

## 🛠️ Development & Testing
```bash
pytest
```

## 📘 License
Research use only.
