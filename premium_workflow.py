from __future__ import annotations

"""Comprehensive workflow demonstrating all modules."""

import argparse
import math
from queue import Queue
from typing import Tuple

import yaml
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from RealTimeDataAbsorber import RealTimeDataAbsorber
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


def run_autonomous_pipeline(env_name: str, cfg: dict) -> None:
    """Run RL training with feedback and optional components."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_absorber = cfg.get("absorber", True)
    use_trainer = cfg.get("rl_trainer", True)
    use_evaluator = cfg.get("evaluator", True)

    absorber: RealTimeDataAbsorber | None = None
    if use_absorber:
        metrics_q: Queue = Queue()
        absorber = RealTimeDataAbsorber(model_config={}, metrics_queue=metrics_q)
        absorber.start_absorption()

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
        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
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


def benchmark_sprout_agi(
    model_name: str = "gpt2", train_samples: int = 32, eval_samples: int = 16
) -> Tuple[float, float]:
    """Benchmark GPT-style models on the Sprout-AGI dataset.

    The function measures baseline perplexity of ``model_name`` on a small
    subset of the dataset, fine-tunes for one epoch and reports the new
    perplexity.  It returns ``(baseline_ppl, tuned_ppl)``.
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
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_tok, eval_dataset=eval_tok
    )
    trainer.train()
    metrics = trainer.evaluate()
    tuned_ppl = math.exp(metrics["eval_loss"])

    print(
        f"Sprout-AGI perplexity | baseline: {baseline_ppl:.2f} | tuned: {tuned_ppl:.2f}"
    )
    return baseline_ppl, tuned_ppl


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
    args = parser.parse_args(argv)

    run_research_workflow()

    if args.sprout_benchmark:
        benchmark_sprout_agi()

    if args.gym:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f).get("workflow", {})
        run_autonomous_pipeline(args.benchmark, config)


__all__ = [
    "run_research_workflow",
    "run_autonomous_pipeline",
    "benchmark_sprout_agi",
    "main",
]


if __name__ == "__main__":
    main()
