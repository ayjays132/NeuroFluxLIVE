# 🚀 NeuroFluxLIVE Premium Workflow

NeuroFluxLIVE unifies research tools, reinforcement learning, and evolutionary search so a language model can keep learning while answering questions in real time.  The `premium_workflow` CLI works with any Hugging Face causal model and provides a single command entry point for experimentation.

## 🌐 Installation
```bash
pip install -r requirements.txt
pip install -e .
```

## ⚙️ Quick Start
Run the full pipeline with your preferred model and prompt:
```bash
premium_workflow --model ayjays132/NeuroReasoner-1-NR-1 --prompt "The future of AI"
```
The command performs a research demo, trains an RL policy in Gym, and launches the evolutionary learner.

## 📊 Sprout‑AGI Benchmark
Measure baseline perplexity, fine‑tune for one epoch on `ayjays132/Sprout-AGI`, and compare generations:
```bash
premium_workflow --sprout-benchmark --model ayjays132/NeuroReasoner-1-NR-1 --prompt "The future of AI"
```
| Model | Baseline PPL | Tuned PPL |
|-------|--------------|-----------|
| ayjays132/NeuroReasoner-1-NR-1 | 66.44 | 8.94 |

Baseline generation:
> The future of AI is a blend between wonder and caution. |endthought| What are the key challenges

Fine‑tuned generation:
> The future of AI ethics is uncertain but transformative. What policies can be implemented to ensure ethical behavior in critical systems? |

## 🕹️ CartPole Reinforcement Learning
```bash
python - <<'PY'
from premium_workflow import run_autonomous_pipeline
import yaml
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)['workflow']
    cfg['absorber'] = False
    cfg['evaluator'] = False
run_autonomous_pipeline('CartPole-v1', cfg)
PY
```
Example reward trace: `[13.0, 16.0, 20.0]`.

## 🧬 Evolutionary Learner
```bash
python - <<'PY'
from premium_workflow import run_evolutionary_learner
run_evolutionary_learner("Explain evolution.", model_name="ayjays132/NeuroReasoner-1-NR-1")
PY
```
The learner mutates the model's bias vector and spins up a background evolutionary system for continuous self‑improvement.

## 🔗 Full Premium Pipeline
```bash
python - <<'PY'
from premium_workflow import run_premium_workflow
run_premium_workflow(prompt="The sky is blue because")
PY
```
This chains the research demo, Gym training, and evolutionary tuning into one seamless experiment.

## 🧪 Testing
```bash
pytest -q
```
(Currently fails because the installed PyTorch build lacks a `uint64` dtype used by `safetensors`.)

## 📄 License
Research use only.
