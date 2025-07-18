# AGENTS Guidelines for SelfResearch

This repository contains a prototype pipeline for training and experimenting with HuggingFace transformer models using PyTorch. CUDA acceleration is supported when available. GPT-2 medium acts as the default example but any model from the HuggingFace Hub should run with minimal configuration.

## Objectives
- Build a clean, modular training pipeline leveraging the `transformers` and `datasets` libraries.
- Support GPU acceleration with CUDA via PyTorch while allowing CPU execution for environments without a GPU.
- Encourage exploration and fine-tuning of different models. The system should enable agents to try new prompts, training schedules and architectures in search of novel insights.
- Provide utilities for experiment tracking, evaluation and dataset management so work is reproducible.
- Integrate virtual research tools for topic selection, source evaluation, simulation, grading and ethics management to support online research workflows.

## Coding Conventions
- Use Python 3.10+ with type hints.
- Follow PEP 8 style guidelines and add explanatory comments for complex logic.
- Organize code into clear modules: `data/` for dataset loading and tokenization, `models/` for model wrappers, `train/` for training loops and `eval/` for evaluation scripts.
- When possible, provide small runnable examples or unit tests using `pytest`.

## Workflow Guidelines
1. **Environment Setup**
   - Depend on `torch`, `transformers`, and `datasets`; keep `requirements.txt` updated.
   - Detect CUDA with `torch.cuda.is_available()` and select the appropriate device.
2. **Dataset Preparation**
   - Tokenize datasets with the chosen model tokenizer (e.g., GPT‑2) using efficient batching and padding.
   - Allow loading local files or datasets from the HuggingFace Hub.
3. **Training Loop**
   - Implement a standard loop with gradient accumulation, periodic validation and checkpoint saving.
   - Support learning rate schedules and optional mixed precision training.
4. **Evaluation and Research**
   - Provide metrics such as perplexity and enable easy extension to custom metrics.
   - Encourage experimentation with prompts and datasets so the agent can generate new hypotheses and discoveries.
   - Use the simulation and digital literacy modules to run virtual experiments and validate information sources.
5. **Documentation**
   - Maintain clear README files explaining how to run training, load different models and extend the pipeline.

## Platform Agents
- **Topic Selector (`research_workflow/topic_selector.py`)** – suggests and validates research topics with a language model.
- **Source Evaluation (`digital_literacy/source_evaluator.py`)** – assesses credibility, bias and relevance of references and queries academic APIs.
- **Experiment Simulation (`simulation_lab/experiment_simulator.py`)** – provides physics and biological simulations and generates synthetic data using PyTorch.
- **Rubric Grading (`assessment/rubric_grader.py`)** – grades research submissions based on predefined rubrics and gives constructive feedback.
- **Authentication and Ethics (`security/auth_and_ethics.py`)** – manages user authentication, role permissions and ethical flag reviews.

These components combine to create a virtual research environment where models can be trained, evaluated and applied to novel research tasks.

## Notes for Codex Agents
- Respect the modular project structure when adding or modifying files.
- Keep code and documentation concise but thorough.
- Update `requirements.txt` if new dependencies are introduced.
- If tests are provided, run `pytest` before committing and note failures in the PR message.
