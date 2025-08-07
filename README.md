# 🚀 NeuroFluxLIVE – Premium Self-Learning Workflow

NeuroFluxLIVE stitches together dataset analysis, supervised fine‑tuning, reinforcement learning and evolutionary search into a single always‑on pipeline.  The centrepiece is `premium_workflow.py`, a PyPI‑exposed CLI that lets a language model keep learning while still serving user queries.

## ✨ Highlights
- **Model agnostic:** works with any Hugging Face causal language model.  Defaults to [`ayjays132/NeuroReasoner-1-NR-1`](https://huggingface.co/ayjays132/NeuroReasoner-1-NR-1).
- **Sprout‑AGI benchmark:** one‑epoch fine‑tune on the [`ayjays132/Sprout-AGI`](https://huggingface.co/datasets/ayjays132/Sprout-AGI) dataset with automatic perplexity tracking.
- **Autonomous Gym training:** RL agent learns in environments such as CartPole while the model answers prompts.
- **Evolutionary learner:** optional weight‑mutation search for continual improvement.
- **Fully packaged:** `pip install -e .` exposes a `premium_workflow` console script.

## 📦 Installation
```bash
pip install -r requirements.txt
pip install -e .
```

## 🏁 Quick Start
```bash
premium_workflow --sprout-benchmark --prompt "What is the capital of France?"
```

## 📊 Sprout‑AGI Benchmark
The command above fine‑tunes NeuroReasoner for one epoch.  On our run the model improved dramatically:

| Model | Baseline Perplexity | Tuned Perplexity |
|-------|--------------------:|-----------------:|
| ayjays132/NeuroReasoner-1-NR-1 | 66.44 | 8.94 |

Baseline and tuned generations for the prompt `"What is the capital of France?"`:
```
Baseline: What is the capital of France? |startthought| A city's legal center, a place where laws and culture co...
Tuned:    What is the capital of France? |startthought| <context>The French language, known as 'C' for its many...
```

## 🕹️ Autonomous CartPole Training
```bash
python - <<'PY'
from premium_workflow import run_autonomous_pipeline
import yaml
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)['workflow']
    cfg['absorber'] = False  # disable realtime absorption for the demo
run_autonomous_pipeline('CartPole-v1', cfg)
PY
```
The agent interacts with the Gym environment while periodically prompting the language model.  Rubric feedback is folded into the episode returns to encourage coherent answers.

## 🧬 Evolutionary Weight Search
```bash
python - <<'PY'
from premium_workflow import run_evolutionary_learner
run_evolutionary_learner('Explain evolution.')
PY
```
Weights of the language model's LM head are mutated and evaluated to minimise loss on the provided prompt.

## 🧪 Testing
Run the lightweight test suite:
```bash
pytest tests/test_premium_workflow.py -q
```

## 📄 License
Research use only.
