from __future__ import annotations

"""Ultimate workflow demonstrating all SelfResearch modules.

This script integrates dataset analysis, training, evaluation,
prompt optimization, simulations, security checks, and
collaboration server interaction in one place.
"""

from typing import Any

import torch
import requests

from research_workflow.topic_selector import TopicSelector
from digital_literacy.source_evaluator import SourceEvaluator
from simulation_lab.experiment_simulator import ExperimentSimulator
from assessment.rubric_grader import RubricGrader
from security.auth_and_ethics import AuthAndEthics
from peer_collab.collaboration_server import CollaborationServer
from data.dataset_loader import load_and_tokenize
from train.trainer import TrainingConfig, train_model
from eval.language_model_evaluator import evaluate_perplexity
from analysis.dataset_analyzer import (
    analyze_tokenized_dataset,
    cluster_dataset_embeddings,
    compute_tsne_embeddings,
)
from analysis.prompt_optimizer import PromptOptimizer
from analysis.advanced_prompt_optimizer import AdvancedPromptOptimizer
from analysis.prompt_bandit_optimizer import PromptBanditOptimizer
from analysis.prompt_bayes_optimizer import PromptBayesOptimizer


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize modules
    topic_selector = TopicSelector(device=device)
    source_evaluator = SourceEvaluator(device=device)
    experiment_simulator = ExperimentSimulator(device=device)
    rubric_grader = RubricGrader(device=device)
    auth_ethics = AuthAndEthics(device=device)
    collab_server = CollaborationServer(device=device)

    # Dataset loading and analysis
    dataset = load_and_tokenize("ag_news", "train[:20]", "distilgpt2")
    stats = analyze_tokenized_dataset(dataset, max_samples=20, include_trigrams=True)
    print("Dataset stats:", stats)

    # Embedding clustering and t-SNE (small subset for speed)
    km, labels = cluster_dataset_embeddings(
        dataset[:10], "distilgpt2", num_clusters=2, device=device
    )
    print("Cluster labels for first 10 samples:", labels)
    coords = compute_tsne_embeddings(
        dataset[:10], "distilgpt2", device=device, perplexity=5.0
    )
    if hasattr(coords, "tolist"):
        coords_list = coords.tolist()
    else:
        coords_list = list(coords)
    print("t-SNE coordinates for first 2 samples:", coords_list[:2])

    # Quick training demo
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
    ppl = evaluate_perplexity("distilgpt2", "ag_news", split="test[:10]")
    print(f"Perplexity: {ppl:.2f}")

    # Prompt optimizers
    base_prompt = "Summarize the research article:"
    optim = PromptOptimizer("distilgpt2")
    adv_optim = AdvancedPromptOptimizer("distilgpt2")
    bandit = PromptBanditOptimizer("distilgpt2", reward_fn=len)
    bayes = PromptBayesOptimizer("distilgpt2", iterations=3)
    print("PromptOptimizer:", optim.optimize_prompt(base_prompt))
    print("AdvancedPromptOptimizer:", adv_optim.optimize_prompt(base_prompt))
    print("BanditOptimizer:", bandit.optimize_prompt(base_prompt))
    print("BayesOptimizer:", bayes.optimize_prompt(base_prompt))

    # Research workflow modules
    topic = topic_selector.suggest_topic("AI for healthcare")
    print("Suggested topic:", topic)
    print("Question valid:", topic_selector.validate_question("How can AI help?"))

    evaluation = source_evaluator.evaluate_source("https://example.com")
    print("Source evaluation:", evaluation)

    positions = experiment_simulator.run_physics_simulation(0.0, 5.0, 5, 0.1)
    print("Physics simulation positions:", positions.tolist())

    rubric = {"Quality": {"expected_content": "Good", "max_score": 5}}
    grades = rubric_grader.grade_submission("A short study.", rubric)
    print("Grades:", grades)

    auth_ethics.register_user("demo", "pass", "researcher")
    print("Authenticated:", auth_ethics.authenticate_user("demo", "pass"))
    auth_ethics.flag_ethical_concern("Test flag")
    print("Ethical flags:", auth_ethics.get_ethical_flags())

    print(
        "\nStart the collaboration server separately with 'python3 peer_collab/collaboration_server.py'"
    )
    try:
        resp = requests.get("http://127.0.0.1:5000/notes/demo", timeout=1)
        if resp.ok:
            print("Collaboration server notes:", resp.json())
    except requests.exceptions.RequestException as exc:  # pragma: no cover - server typically not running
        print("Could not reach collaboration server:", exc)


if __name__ == "__main__":
    main()
