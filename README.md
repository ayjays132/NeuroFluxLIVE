# 🌟 NeuroFluxLIVE Premium Pipeline

**NeuroFluxLIVE** combines streaming data absorption, on-the-fly fine-tuning, and reinforcement learning so a model can keep learning while it serves responses. The main entrypoint is [`premium_workflow.py`](premium_workflow.py), installed as the `premium_workflow` console script.

## 📦 Installation
```bash
pip install -r requirements.txt
pip install -e .
```

## 🚀 Using the Premium Workflow
Run the integrated demonstration:
```bash
premium_workflow
```

### 🔬 Sprout-AGI Benchmark
Fine-tune any GPT-style model against the [Sprout-AGI](https://huggingface.co/datasets/ayjays132/Sprout-AGI) reasoning set:
```bash
premium_workflow --sprout-benchmark --model ayjays132/NeuroReasoner-1-NR-1 --prompt "The future of AI is"
```
The command reports baseline and tuned perplexity and, if a prompt is supplied, compares generations. Using `ayjays132/NeuroReasoner-1-NR-1` on a four-example subset yields:

| Model | Baseline PPL | Tuned PPL |
|-------|--------------|-----------|
| ayjays132/NeuroReasoner-1-NR-1 | 221.51 | 15.43 |

Prompt **"The future of AI is"**:

| Stage | Continuation |
|-------|-------------|
| Baseline | The future of AI is in its ability to learn from human experience and adapt. This reflection illustrates that by integrating... |
| Tuned | The future of AI is uncertain, but it’s essential for modern life. 🤖💧 |

### 🕹️ Autonomous Gym Training
```bash
premium_workflow --gym --benchmark CartPole-v1
```
Trains a lightweight policy head in a Gym environment while simultaneously querying the language model.

## 🧩 Key Modules
- **RealTimeDataAbsorber** – streams incoming data to the model.
- **train.trainer** – supervised fine-tuning utilities.
- **eval.language_model_evaluator** – quick perplexity evaluation.
- **simulation_lab.gym_autonomous_trainer** – RL experiments.

## 🛠️ Development & Testing
```bash
pytest
```

## 📘 License
Research use only.
