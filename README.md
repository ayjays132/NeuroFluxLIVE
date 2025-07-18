# SelfResearch Platform

SelfResearch provides a modular environment for experimenting with HuggingFace transformer models using PyTorch. The project bundles several tools for virtual research including topic suggestion, source evaluation, simulation, grading and security management.

## Installation
1. Create a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # needed for running tests
   ```
CUDA is automatically detected. If a GPU is available, PyTorch will use it.

The requirements file includes additional libraries used across the project:

* `faiss-cpu` – vector search and clustering
* `opencv-python` – image processing utilities
* `soundfile` – reading and writing audio files
* `aiohttp` – asynchronous HTTP requests
* `psutil` – system resource monitoring
* `pillow` – image loading with `PIL`
* `pyyaml` – YAML configuration handling
* `tqdm` – progress bars for dataset processing
* `torchvision` – vision transforms

## Running the Example Workflow
Execute `main.py` to launch a demonstration of all modules:
```bash
python3 main.py
```
The script showcases topic selection, source evaluation, running physics and biology simulations, grading a submission and basic authentication.

For a more in-depth example that additionally analyzes datasets, evaluates perplexity and optimizes prompts, run `premium_workflow.py`:
```bash
python3 premium_workflow.py
```

For a demonstration that touches **all** modules, including advanced prompt
optimizers and dataset analysis utilities, run `full_demo.py`:
```bash
python3 full_demo.py
```
For the most comprehensive demonstration, which additionally clusters embeddings and interacts with the collaboration server, run `ultimate_workflow.py`:
```bash
python3 ultimate_workflow.py
```

To start the collaboration server used for peer collaboration, run the following in a separate process:
```bash
python3 peer_collab/collaboration_server.py
```

The realtime metrics server starts automatically when running
`RealTimeDataAbsorber.py`. Any WebSocket client can connect to
`ws://<host>:8765` (replace `<host>` if running remotely) to receive
JSON updates describing current processing statistics.

## Configuration
Default settings such as the data directory, model embedding dimension and
memory budget live in `config.yaml`. `main.py` loads this file automatically.
Provide a different file with the `--config` option:

```bash
python3 main.py --config my_config.yaml
```

## Module Walkthrough
The example workflow in `main.py` demonstrates how each component fits together.
The script performs the following steps:

1. **Device detection** using `torch.cuda.is_available()`.
2. **Topic selection** with `TopicSelector`.
3. **Source evaluation** using `SourceEvaluator` to analyze URLs and query
   academic APIs.
4. **Dataset loading** via `load_and_tokenize` followed by model training with
   `TrainingConfig` and `train_model`.
5. **Simulation lab** activities through `ExperimentSimulator` for physics and
   biological experiments and synthetic data generation.
6. **Rubric grading** of sample submissions using `RubricGrader`.
7. **Authentication and ethics** checks with `AuthAndEthics` for user
   management and ethical flagging.
8. The **Collaboration server** in `peer_collab/` can be started separately to
   enable shared notes and feedback.

Refer to `main.py` for concrete code that ties these modules into a single
workflow.

## Loading Different Models
The modules rely on the HuggingFace `transformers` library. You can edit each component to load any model from the Hub by changing the model names in their initialisation calls. Most small models work well on CPU, while larger models benefit from CUDA acceleration.

## Extending the Pipeline
The project follows a simple structure so new functionality can be added easily:
* `research_workflow/` – topic selection utilities
* `digital_literacy/` – source evaluation and academic search
* `simulation_lab/` – physics/biology simulations and data generation
* `assessment/` – rubric-based grading tools
* `peer_collab/` – collaboration server for shared notes and feedback
* `security/` – user authentication and ethical flagging
* `data/` – helpers for loading and tokenizing datasets
* `analysis/` – dataset statistics utilities and prompt engineering tools
* `models/` – wrappers around HuggingFace models
* `train/` – simple training loops
* `eval/` – evaluation utilities such as perplexity measurement

The `train` module contains utilities for fine-tuning language models with
CUDA support. Training now accepts separate train and evaluation splits and
reports perplexity after each epoch. Run `--help` to see all options:

```bash
python3 -m train.trainer --help
```

The `eval` module includes a script to compute perplexity for a model on a
dataset:

```bash
python3 -m eval.language_model_evaluator gpt2 ag_news --split test
```

## Dataset Caching
`load_and_tokenize` and `load_dataset_splits` accept a `cache_dir` argument. If
provided, tokenized datasets will be saved to disk and loaded on subsequent
calls without reprocessing.

```python
from data.dataset_loader import load_and_tokenize
ds = load_and_tokenize("ag_news", "train[:100]", "distilgpt2", cache_dir="./cache/ag_news_train")
```

`load_and_tokenize` also supports `drop_duplicates=True` to remove repeated
text samples before tokenization.
If the chosen tokenizer lacks a pad token (e.g. GPT‑2 models),
`load_and_tokenize` automatically reuses the EOS token so batching works
out of the box.

New training loops, datasets or evaluation scripts can be added under these
modules, keeping the code organized as described in `AGENTS.md`.

## Data Directory Layout
Some utilities expect a folder named `data_root/` containing separate
subdirectories for each modality:

```
data_root/
├── text/    # UTF-8 .txt files
├── image/   # .jpg, .png, .webp images
├── audio/   # .wav or .flac audio
└── sensor/  # JSON sensor readings
```
You can populate these directories with your own sample data to experiment
with the `MultimodalDatasetManager`.

## Dataset Analysis
The `analysis` module provides utilities for computing statistics on tokenized
datasets. `analyze_tokenized_dataset` now reports additional metrics such as
token entropy and optional trigram frequencies alongside average length,
vocabulary size and lexical diversity.
The module also includes `cluster_dataset_embeddings` for grouping dataset
samples by semantic similarity using transformer embeddings.
`compute_tsne_embeddings` generates 2D t-SNE coordinates for visualizing
dataset structure.
Run it from the command line:
```bash
python3 -m analysis.dataset_analyzer ag_news train distilgpt2 --top-k 3 --trigrams
```

## Prompt Optimization
The `PromptOptimizer` class generates prompt variations and scores them using perplexity. It returns the best-scoring prompt so you can iteratively refine instructions or dataset prompts.

Example usage:
```python
from analysis.prompt_optimizer import PromptOptimizer
opt = PromptOptimizer("distilgpt2")
print(opt.optimize_prompt("Summarize the research article:"))
```

### Advanced Prompt Optimization
For more precise control, `AdvancedPromptOptimizer` combines perplexity with
semantic similarity of sentence embeddings. This helps retain the intent of the
original prompt while improving clarity.

```python
from analysis.advanced_prompt_optimizer import AdvancedPromptOptimizer
adv = AdvancedPromptOptimizer(
    "distilgpt2", embedding_model="sentence-transformers/paraphrase-MiniLM-L6-v2"
)
print(adv.optimize_prompt("Summarize the research article:"))
```

### Prompt Augmentation

`PromptAugmenter` generates additional prompt variations to expand training datasets.

```python
from analysis.prompt_augmenter import PromptAugmenter
aug = PromptAugmenter("distilgpt2")
prompts = aug.augment_dataset(["Write an abstract about AI"], n_variations=2)
```

### Evolutionary Prompt Engineering

`PromptEvolver` applies a simple genetic algorithm to refine prompts over several generations.

```python
from analysis.prompt_evolver import PromptEvolver
pe = PromptEvolver("distilgpt2")
print(pe.evolve_prompt("Describe the benefits of renewable energy"))
```

### Bandit-Based Prompt Optimization

`PromptBanditOptimizer` employs an epsilon-greedy multi-armed bandit strategy
to iteratively explore and exploit prompt variations based on a custom reward
function.

```python
from analysis.prompt_bandit_optimizer import PromptBanditOptimizer

# Reward function prefers longer prompts
bandit = PromptBanditOptimizer("distilgpt2", reward_fn=len, epsilon=0.1)
best = bandit.optimize_prompt("Write a summary")
print(best)
```

### Simulated Annealing Prompt Optimization

`PromptAnnealingOptimizer` explores prompt variations using a simulated annealing strategy. It
accepts or rejects new prompts based on a temperature schedule, allowing occasional
worse prompts early on to escape local minima.

```python
from analysis.prompt_annealing_optimizer import PromptAnnealingOptimizer

annealer = PromptAnnealingOptimizer("distilgpt2", temperature=1.0, cooling=0.8, steps=5)
best = annealer.optimize_prompt("Write a summary")
print(best)
```

### Reinforcement Learning Prompt Optimization

`PromptRLOptimizer` applies a simple Q-learning strategy to refine prompts
based on a custom reward function. It explores variations of the base prompt
over multiple episodes and learns which prompts yield the highest reward.

```python
from analysis.prompt_rl_optimizer import PromptRLOptimizer

# Reward prefers longer prompts
rl = PromptRLOptimizer("distilgpt2", reward_fn=len, episodes=5, epsilon=0.1)
best = rl.optimize_prompt("Write a summary")
print(best)
```

### Meta Prompt Optimization

`MetaPromptOptimizer` chains multiple optimizers (bandit, annealing, and
reinforcement learning) to search a broader space of prompt variations and
select the overall best prompt.

```python
from analysis.meta_prompt_optimizer import MetaPromptOptimizer

meta = MetaPromptOptimizer("distilgpt2")
best = meta.optimize_prompt("Write a summary")
print(best)
```


## Running Tests
Run the unit tests with `pytest` from the repository root after installing the
standard and development requirements. Both files must be installed or
`pytest` will raise import errors during collection:

```bash
source .venv/bin/activate  # if not already active
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest
```

The `requirements.txt` file pins a CPU-only build of PyTorch so tests run even
without a GPU.


## License
This repository is provided for research and experimentation purposes only.

## Frontend Assets and Ambient Audio
The `webapp/` directory contains optional web assets used in browser-based demos. `manifest.webmanifest` references a custom SVG icon, and `animated-favicon.svg` provides a simple animated favicon. The example hook in `useAmbientAudio.ts` plays looping background audio while gracefully handling decoding errors that can occur with unsupported files.

