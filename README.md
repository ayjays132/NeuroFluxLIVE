# 🌐 NeuroFluxLIVE Premium Workflow

**NeuroFluxLIVE** fuses real‑time data absorption, language model fine‑tuning and reinforcement learning into a single, plug‑and‑play research sandbox.  The `premium_workflow.py` entrypoint runs everything: dataset ingestion, model training, prompt evaluation and optional Gym simulations.

## ⚙️ Installation
```bash
pip install -r requirements.txt
pip install -e .
```
The package exposes a console script named `premium_workflow` for immediate use after installation.

## 🚀 Quick Start
Run the full demonstration pipeline:
```bash
premium_workflow
```
Add flags to enable extra benchmarks:
- `--sprout-benchmark` – fine‑tune GPT‑style models on the [Sprout‑AGI](https://huggingface.co/datasets/ayjays132/Sprout-AGI) reasoning dataset.
- `--gym` `--benchmark CartPole-v1` – launch the autonomous RL trainer alongside language prompting.

## 🌱 Sprout‑AGI Benchmark
The benchmark measures perplexity before and after a short fine‑tuning run.  Example with `gpt2`:

| Model | Baseline PPL | Tuned PPL |
|------|--------------|-----------|
| gpt2 | 133.08 | 18.44 |

Prompt quality also improves.  Given the prompt **“The future of AI is”**:

| Stage | Generated Continuation |
|-------|-----------------------|
| Baseline | *The future of AI is uncertain. The future of AI is uncertain.* |
| Fine‑tuned | *The future of AI is a complex and complex subject matter. We will continue to explore the potential of AI to improve human understanding.* |

## 🕹️ Gym RL Demo
```bash
premium_workflow --gym --benchmark CartPole-v1
```
The workflow trains an agent in the environment while periodically querying the language model.  Episode returns are printed together with rubric‑based feedback scores.

## 🧩 Module Showcase
`premium_workflow.py` orchestrates every module:
- **Data ingestion** via `RealTimeDataAbsorber` (optional).
- **Language model training** with `train.trainer`.
- **Evaluation** using `eval.language_model_evaluator` and custom rubric grading.
- **Simulation & RL** through `simulation_lab.gym_autonomous_trainer`.

## ✅ Testing
```bash
pytest
```

## 📄 License
Research use only.
