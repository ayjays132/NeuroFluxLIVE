# 🌟 NeuroFluxLIVE Premium Workflow

NeuroFluxLIVE fuses data ingestion, reinforcement learning, and evolutionary search into a single always-on research pipeline.  The `premium_workflow` CLI works with any Hugging Face causal language model so the system can answer prompts while continuing to learn in the background.

## ✨ Features
- **Model agnostic** – supply any Hugging Face identifier with `--model`.
- **Sprout‑AGI benchmark** – quick perplexity check and one‑epoch fine‑tuning.
- **Autonomous Gym training** – policy‑gradient agent learns while chatting.
- **Evolutionary learner** – background evolutionary system continually adapts weights.

## 🚀 Getting Started
```bash
pip install -r requirements.txt
pip install -e .
```
Run the full demo:
```bash
premium_workflow
```

### Sprout‑AGI Benchmark
Evaluate and fine‑tune any model on the `ayjays132/Sprout-AGI` dataset:
```bash
premium_workflow --sprout-benchmark --model ayjays132/NeuroReasoner-1-NR-1 --prompt "The future of AI"
```
| Model | Baseline PPL | Tuned PPL |
|-------|--------------|-----------|
| ayjays132/NeuroReasoner-1-NR-1 | 66.44 | 8.94 |

Baseline generation:
> The future of AI in human-like behavior? A decade from now, what is the societal impact on

Fine‑tuned generation:
> The future of AI ethics is uncertain but transformative. What policies can be implemented to ensure ethical behavior in critical systems?

### CartPole Reinforcement Learning
```bash
python - <<'PY'
from premium_workflow import run_autonomous_pipeline
import yaml
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)['workflow']
    cfg['absorber'] = False
run_autonomous_pipeline('CartPole-v1', cfg)
PY
```
Average reward ≈20.6 over three episodes.

### Evolutionary Tuning
```bash
python - <<'PY'
from premium_workflow import run_evolutionary_learner
run_evolutionary_learner('Explain evolution.', model_name='ayjays132/NeuroReasoner-1-NR-1')
PY
```
The function evolves the model's bias vector and then launches a background evolutionary system for continuous improvement.

### Full Premium Pipeline
Run research tools, RL loop, and evolutionary search together:
```bash
python - <<'PY'
from premium_workflow import run_premium_workflow
run_premium_workflow(prompt="The sky is blue because")
PY
```

## 🧪 Testing
```bash
pytest -q
```
(Tests currently fail because the installed PyTorch build lacks the `uint64` dtype.)

## 📄 License
Research use only.
