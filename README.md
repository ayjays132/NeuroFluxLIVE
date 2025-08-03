# NeuroFluxLIVE Premium Workflow 🚀

NeuroFluxLIVE provides a modular research sandbox for continual-learning language models.  
The `premium_workflow.py` script weaves data ingestion, training, evaluation and reinforcement learning into a single pipeline.

## Installation
```bash
pip install -r requirements.txt
pip install -e .
```

## Premium Workflow Usage
Run the end-to-end workflow:
```bash
premium_workflow
```

### Sprout‑AGI Benchmark 🌱
Fine‑tune any causal language model on the [Sprout‑AGI](https://huggingface.co/datasets/ayjays132/Sprout-AGI) reasoning dataset and compare perplexity before and after training.
```bash
premium_workflow --sprout-benchmark
```
A quick baseline with `distilgpt2` on a tiny subset reports a perplexity around **194.15** before training.  
Fine‑tuning steps executed by the command will lower this value.

### Gym RL Demo 🕹️
Evaluate autonomous learning in a Gym environment while the model continues to answer prompts:
```bash
premium_workflow --gym --benchmark CartPole-v1
```
Episode returns and textual feedback are printed after each run.

## Metrics
| Experiment | Metric | Example Result |
|------------|--------|----------------|
| Sprout‑AGI baseline | Perplexity ↓ | 194.15 |
| CartPole demo | Avg. return ↑ | printed at runtime |

## Testing
Run the test suite before committing changes:
```bash
pytest
```

## License
Research use only.
