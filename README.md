# 🌟 NeuroFluxLIVE Premium Workflow

NeuroFluxLIVE unifies fine‑tuning, reinforcement learning and evolutionary search into a single always‑on pipeline. The `premium_workflow` CLI lets any Hugging Face causal model learn from streaming experience while still answering user prompts.

## 🚀 Features
- **Model agnostic:** supply any Hugging Face model with `--model`.
- **Sprout‑AGI benchmark:** quick perplexity check and one‑epoch tuning.
- **Autonomous Gym training:** policy‑gradient agent learns while chatting.
- **Evolutionary learner:** Google‑style evolutionary engine mutates weights for continual improvement.

## 🔧 Installation
```bash
pip install -r requirements.txt
pip install -e .
```

## 💡 Quick Start
Run the full demo:
```bash
premium_workflow
```

### Sprout‑AGI Benchmark
```bash
premium_workflow --sprout-benchmark --model ayjays132/NeuroReasoner-1-NR-1 --prompt "Hello world"
```
| Model | Baseline PPL | Tuned PPL |
|-------|--------------|-----------|
| ayjays132/NeuroReasoner-1-NR-1 | 66.44 | 8.94 |

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
Average reward ≈ 15.95 over three episodes.

### Evolutionary Tuning
```bash
python - <<'PY'
from premium_workflow import run_evolutionary_learner
run_evolutionary_learner('Hello world', 'ayjays132/NeuroReasoner-1-NR-1', generations=1, population=5)
PY
```
Best fitness −13.98 with evolved continuation:
> Hello world, where every decision you make is a!ial-Tribute to the!isphere of!ial

## 🧪 Testing
```bash
pytest -q
```

## 📄 License
Research use only.
