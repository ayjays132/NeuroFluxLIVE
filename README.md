# 🚀 NeuroFluxLIVE – Premium Continuous Learning Pipeline

NeuroFluxLIVE is an experimental playground where language models stream new data, fine‑tune themselves and act inside reinforcement learning environments without pausing for separate training runs.  The entire system is driven by [`premium_workflow.py`](premium_workflow.py) which installs as the `premium_workflow` command.

## ✨ Features
- **Unified workflow** – dataset analysis, supervised fine‑tuning, real‑time absorption and Gym RL loop in one script.
- **Model agnostic** – works with any causal Hugging Face model (GPT‑2, LLaMA, Qwen, etc.).
- **Always learning** – the language model keeps answering prompts while weights update.
- **Crash‑safe checkpoints** – the best performing weights are promoted to `sprout_benchmark/best_model` only when perplexity improves.

## 🔧 Installation
```bash
pip install -r requirements.txt
pip install -e .
```

## 🚀 Quick Start
Run the showcase workflow:
```bash
premium_workflow
```

### 🔬 Sprout‑AGI Benchmark
Fine‑tune a model on the [Sprout‑AGI](https://huggingface.co/datasets/ayjays132/Sprout-AGI) reasoning dataset and compare generations:
```bash
premium_workflow --sprout-benchmark --model ayjays132/NeuroReasoner-1-NR-1 --prompt "The future of AI is"
```
Example result on a 32/16 split:

| Model | Baseline PPL | Tuned PPL |
|-------|--------------|-----------|
| ayjays132/NeuroReasoner-1-NR-1 | 66.44 | 8.94 |

Prompt **"The future of AI is"**:

| Stage | Continuation |
|-------|--------------|
| Baseline | The future of AI is bright, with promise for humanity’s future. 🤖❄️ |
| Tuned | The future of AI is uncertain. What should the next step be? \|endthought\| A critical question arises in |

Best weights are stored in `sprout_benchmark/best_model/` so progress is preserved between runs.

### 🕹️ Autonomous Gym Training
```bash
premium_workflow --gym --benchmark CartPole-v1
```
Runs a lightweight RL policy in parallel with language model prompting to demonstrate continuous learning without halting service.

## 🛠 Development
Run unit tests (requires torch ≥2.6 for full pass):
```bash
pytest
```

## 📄 License
Research use only.
