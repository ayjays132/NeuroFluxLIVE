from __future__ import annotations

"""Comprehensive workflow demonstrating all modules."""

import argparse
import math
import shutil
from pathlib import Path
from queue import Queue
from typing import Tuple

import numpy as np
import yaml
import torch

# ---------------------------------------------------------------------------
# ``safetensors`` expects a ``torch.uint64`` dtype which is not defined in some
# PyTorch builds (e.g. 2.2).  To keep the workflow compatible across versions
# and allow installation from PyPI without pinning a specific torch release we
# provide a lightweight shim.  Mapping ``uint64`` to ``int64`` is sufficient for
# the transformer weights used in this project and avoids an ``AttributeError``
# during import.
# ---------------------------------------------------------------------------
if not hasattr(torch, "uint16"):  # pragma: no cover - defensive patch
    torch.uint16 = torch.int16  # type: ignore[attr-defined]
if not hasattr(torch, "uint32"):
    torch.uint32 = torch.int32  # type: ignore[attr-defined]
if not hasattr(torch, "uint64"):
    torch.uint64 = torch.int64  # type: ignore[attr-defined]

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

try:  # optional at import time for environments lacking cv2
    from RealTimeDataAbsorber import RealTimeDataAbsorber  # type: ignore
except Exception:  # pragma: no cover
    RealTimeDataAbsorber = None  # type: ignore

from simulation_lab.gym_autonomous_trainer import (
    TrainerConfig,
    run_gym_autonomous_trainer,
)
from research_workflow.topic_selector import TopicSelector
from digital_literacy.source_evaluator import SourceEvaluator
from simulation_lab.experiment_simulator import ExperimentSimulator
from assessment.rubric_grader import RubricGrader
from security.auth_and_ethics import AuthAndEthics
from data.dataset_loader import load_and_tokenize
from train.trainer import TrainingConfig, train_model
from eval.language_model_evaluator import evaluate_perplexity
from analysis.dataset_analyzer import analyze_tokenized_dataset
from analysis.prompt_optimizer import PromptOptimizer
from evolutionary_learner import GoogleEvolutionaryEngine, integrate_evolutionary_learning
import time


def run_research_workflow() -> None:
    """Original demonstration workflow showing core components."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize modules
    topic_selector = TopicSelector(device=device)
    source_evaluator = SourceEvaluator(device=device)
    experiment_simulator = ExperimentSimulator(device=device)
    rubric_grader = RubricGrader(device=device)
    auth_ethics = AuthAndEthics(device=device)

    # Dataset loading and analysis
    print("\n=== Dataset Loading & Analysis ===")
    tokenized_ds = load_and_tokenize("ag_news", "train[:50]", "distilgpt2")
    stats = analyze_tokenized_dataset(tokenized_ds, max_samples=50)
    print(f"Dataset stats: {stats}")

    # Quick training demo
    print("\n=== Training Demo ===")
    cfg = TrainingConfig(
        model_name="distilgpt2",
        dataset_name="ag_news",
        train_split="train[:10]",
        eval_split="test[:10]",
        epochs=1,
        batch_size=2,
        output_dir="./demo_model",
    )
    train_model(cfg)

    # Evaluate perplexity
    ppl = evaluate_perplexity("distilgpt2", "ag_news", split="test[:10]")
    print(f"Perplexity on small set: {ppl:.2f}")

    # Prompt optimization example
    optimizer = PromptOptimizer("distilgpt2")
    best_prompt = optimizer.optimize_prompt("Summarize the research article:")
    print(f"Optimized prompt: {best_prompt}")

    # Topic suggestion and question validation
    print("\n=== Research Workflow ===")
    topic = topic_selector.suggest_topic("machine learning for health")
    print(f"Suggested topic: {topic}")
    question = "How can ML improve early disease detection?"
    print(f"Valid question: {topic_selector.validate_question(question)}")

    # Source evaluation
    print("\n=== Source Evaluation ===")
    result = source_evaluator.evaluate_source("https://example.com")
    print(result)

    # Simulation lab
    print("\n=== Simulation Lab ===")
    positions = experiment_simulator.run_physics_simulation(0.0, 5.0, 5, 0.1)
    print(f"Positions: {positions.tolist()}")

    # Rubric grading
    print("\n=== Rubric Grading ===")
    rubric = {"Quality": {"expected_content": "Detailed study", "max_score": 5}}
    grades = rubric_grader.grade_submission("A short study.", rubric)
    print(grades)

    # Authentication and ethics
    print("\n=== Security Module ===")
    auth_ethics.register_user("demo", "pass", "researcher")
    print(auth_ethics.authenticate_user("demo", "pass"))
    auth_ethics.flag_ethical_concern("Test flag")
    print(auth_ethics.get_ethical_flags())


def run_evolutionary_learner(
    prompt: str,
    model_name: str = "gpt2",
    generations: int = 2,
    population: int = 10,
) -> None:
    """Evolve the language model's bias vector to improve prompt loss.

    The function demonstrates how the :class:`GoogleEvolutionaryEngine` can
    adapt model parameters without traditional gradient descent.  The
    evolutionary search operates on the LM head bias which keeps the genome
    small enough for quick experimentation while still affecting generation
    quality.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    target = model.lm_head.bias
    if target is None:
        target = model.lm_head.weight[0]

    genome_size = target.numel()
    engine = GoogleEvolutionaryEngine(population_size=population)
    engine.initialize_population(genome_size)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    def evaluator(genome: np.ndarray) -> float:
        delta = torch.tensor(genome, device=device)
        with torch.no_grad():
            original = target.clone()
            target.copy_(original + delta)
            loss = model(**inputs, labels=inputs["input_ids"]).loss.item()
            target.copy_(original)
        # Lower loss == higher fitness
        return -loss

    for _ in range(generations):
        metrics = engine.evolve_generation(evaluator)
        print(
            f"Generation {metrics.generation} | best fitness {metrics.best_fitness:.4f}"
        )

    best_delta = torch.tensor(engine.best_individual["genome"], device=device)
    with torch.no_grad():
        target.add_(best_delta)

    out_ids = model.generate(
        **inputs,
        max_length=inputs["input_ids"].shape[1] + 20,
        pad_token_id=tokenizer.eos_token_id,
    )
    seq = out_ids.sequences[0] if hasattr(out_ids, "sequences") else out_ids[0]
    text = tokenizer.decode(seq, skip_special_tokens=True)
    print("Evolved response:", text)

    # Launch the full evolutionary learning system for continuous background tuning
    agent = type(
        "_TmpAgent",
        (),
        {
            "STATE_DIR": "./agent_state",
            "model": model,
            "last_user_request_time": time.time(),
            "performance_metrics": {},
        },
    )()
    integrate_evolutionary_learning(agent)


def run_autonomous_pipeline(env_name: str, cfg: dict) -> None:
    """Run RL training with feedback and optional components."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_absorber = cfg.get("absorber", True)
    use_trainer = cfg.get("rl_trainer", True)
    use_evaluator = cfg.get("evaluator", True)

    absorber: "RealTimeDataAbsorber" | None = None
    if use_absorber and RealTimeDataAbsorber is not None:
        metrics_q: Queue = Queue()
        absorber = RealTimeDataAbsorber(model_config={}, metrics_queue=metrics_q)
        absorber.start_absorption()
    elif use_absorber:
        print("RealTimeDataAbsorber unavailable: optional dependency missing")
        use_absorber = False

    tokenizer = AutoTokenizer.from_pretrained("ayjays132/NeuroReasoner-1-NR-1")
    model = AutoModelForCausalLM.from_pretrained(
        "ayjays132/NeuroReasoner-1-NR-1"
    ).to(device)

    responses: list[str] = []
    episode_returns: list[float] = []

    if use_trainer:
        gym_cfg = TrainerConfig(env_name=env_name)
        episode_returns, responses = run_gym_autonomous_trainer(gym_cfg)

    # Periodic user prompts
    prompts = ["Describe the research goal."]
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out_ids = model.generate(
            **inputs, max_length=inputs["input_ids"].shape[1] + 20, pad_token_id=tokenizer.eos_token_id
        )
        seq = out_ids.sequences[0] if hasattr(out_ids, "sequences") else out_ids[0]
        text = tokenizer.decode(seq, skip_special_tokens=True)
        responses.append(text)
        if absorber is not None:
            absorber.absorb_data(text, "text", source="prompt", priority=1)

    if use_evaluator and responses:
        grader = RubricGrader(device=device)
        rubric = {"Quality": {"expected_content": "research", "max_score": 5}}
        feedback_scores = []
        for resp in responses:
            grade = grader.grade_submission(resp, rubric)["Quality"]["score"]
            feedback_scores.append(grade)
        if episode_returns:
            episode_returns = [r + sum(feedback_scores) for r in episode_returns]
        print(f"Feedback scores: {feedback_scores}")

    print(f"Episode rewards: {episode_returns}")

    if absorber is not None:
        absorber.stop_absorption()


def run_premium_workflow(
    prompt: str,
    model_name: str = "ayjays132/NeuroReasoner-1-NR-1",
    env_name: str = "CartPole-v1",
) -> None:
    """Execute research, RL and evolutionary steps sequentially.

    This helper shows how all premium components can be chained together for
    a single experiment.  It simply invokes :func:`run_research_workflow`, the
    RL pipeline via :func:`run_autonomous_pipeline` and finally the
    evolutionary learner.  The individual functions remain unchanged so the
    caller can still run them separately if desired.
    """

    run_research_workflow()
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f).get("workflow", {})
    run_autonomous_pipeline(env_name, cfg)
    run_evolutionary_learner(prompt=prompt, model_name=model_name)


def benchmark_sprout_agi(
    model_name: str = "gpt2",
    train_samples: int = 32,
    eval_samples: int = 16,
    prompt: str | None = None,
) -> Tuple[float, float, str, str]:
    """Benchmark GPT-style models on the Sprout-AGI dataset.

    The function measures baseline perplexity of ``model_name`` on a small
    subset of the dataset, fine-tunes for one epoch and reports the new
    perplexity. Optionally, a ``prompt`` can be provided to compare model
    generations before and after fine-tuning.  It returns ``(baseline_ppl,
    tuned_ppl, baseline_text, tuned_text)``.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset("ayjays132/Sprout-AGI", split="train")

    def _format(example: dict) -> dict:
        answer = example["solution"].get("final_answer") or ""
        return {"text": f"{example['context']} {answer}"}

    dataset = dataset.map(_format)
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col != "text"]
    ).shuffle(seed=42)

    train_ds = dataset.select(range(train_samples))
    eval_ds = dataset.select(range(train_samples, train_samples + eval_samples))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _tok(batch: dict) -> dict:
        enc = tokenizer(
            batch["text"], padding="max_length", truncation=True, max_length=64
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    train_tok = train_ds.map(_tok, batched=True)
    eval_tok = eval_ds.map(_tok, batched=True)
    columns = ["input_ids", "attention_mask", "labels"]
    train_tok.set_format(type="torch", columns=columns)
    eval_tok.set_format(type="torch", columns=columns)

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.do_sample = True
        gen_cfg.early_stopping = False
        gen_cfg.length_penalty = 1.0

    baseline_text = ""
    inputs = None
    if prompt is not None:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out_ids = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + 20,
            pad_token_id=tokenizer.eos_token_id,
        )
        seq = out_ids.sequences[0] if hasattr(out_ids, "sequences") else out_ids[0]
        baseline_text = tokenizer.decode(seq, skip_special_tokens=True)

    losses = []
    loader = DataLoader(eval_tok, batch_size=2)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        losses.append(outputs.loss.item())
    baseline_ppl = math.exp(sum(losses) / len(losses))

    args = TrainingArguments(
        output_dir="./sprout_benchmark",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=10,
        report_to="none",
        save_strategy="no",  # avoid writing large checkpoint files during quick benchmarks
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_tok, eval_dataset=eval_tok
    )
    trainer.train()
    metrics = trainer.evaluate()
    tuned_ppl = math.exp(metrics["eval_loss"])
    # Persist the best performing weights.  A temporary directory is used so a
    # partially written checkpoint will not overwrite a previously good model if
    # the process terminates unexpectedly.
    tmp_dir = Path("./sprout_benchmark/tmp")
    best_dir = Path("./sprout_benchmark/best_model")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    if tuned_ppl < baseline_ppl:
        if best_dir.exists():
            shutil.rmtree(best_dir)
        tmp_dir.rename(best_dir)
    else:
        shutil.rmtree(tmp_dir)

    tuned_text = ""
    if inputs is not None:
        out_ids = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + 20,
            pad_token_id=tokenizer.eos_token_id,
        )
        seq = out_ids.sequences[0] if hasattr(out_ids, "sequences") else out_ids[0]
        tuned_text = tokenizer.decode(seq, skip_special_tokens=True)

    print(
        f"Sprout-AGI perplexity | baseline: {baseline_ppl:.2f} | tuned: {tuned_ppl:.2f}"
    )
    return baseline_ppl, tuned_ppl, baseline_text, tuned_text


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for running premium workflow pipelines."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--gym", action="store_true", help="run gym RL trainer")
    parser.add_argument("--benchmark", default="CartPole-v1", help="gym benchmark")
    parser.add_argument(
        "--sprout-benchmark",
        action="store_true",
        help="run Sprout-AGI fine-tuning benchmark",
    )
    parser.add_argument(
        "--model",
        default="gpt2",
        help="model name for the Sprout-AGI benchmark",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="optional prompt to compare generations",
    )
    parser.add_argument(
        "--evolve",
        action="store_true",
        help="run evolutionary learner demo",
    )
    args = parser.parse_args(argv)

    run_research_workflow()

    if args.evolve:
        run_evolutionary_learner(
            prompt=args.prompt or "Explain evolution.", model_name=args.model
        )

    if args.sprout_benchmark:
        base_ppl, tuned_ppl, base_txt, tuned_txt = benchmark_sprout_agi(
            model_name=args.model, prompt=args.prompt
        )
        if args.prompt is not None:
            print("Baseline generation:", base_txt)
            print("Fine-tuned generation:", tuned_txt)

    if args.gym:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f).get("workflow", {})
        run_autonomous_pipeline(args.benchmark, config)


__all__ = [
    "run_research_workflow",
    "run_autonomous_pipeline",
    "run_evolutionary_learner",
    "run_premium_workflow",
    "benchmark_sprout_agi",
    "main",
]


if __name__ == "__main__":
    main()
