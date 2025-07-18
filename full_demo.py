"""Full demonstration script that exercises all SelfResearch modules."""
from __future__ import annotations

import torch

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
from analysis.advanced_prompt_optimizer import AdvancedPromptOptimizer
from analysis.contextual_prompt_optimizer import ContextualPromptOptimizer
from analysis.prompt_annealing_optimizer import PromptAnnealingOptimizer
from analysis.prompt_bandit_optimizer import PromptBanditOptimizer
from analysis.prompt_bayes_optimizer import PromptBayesOptimizer
from analysis.prompt_embedding_tuner import PromptEmbeddingTuner
from analysis.prompt_evolver import PromptEvolver
from analysis.prompt_rl_optimizer import PromptRLOptimizer
from analysis.prompt_augmenter import PromptAugmenter
from analysis.meta_prompt_optimizer import MetaPromptOptimizer


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize modules
    topic_selector = TopicSelector(device=device)
    source_evaluator = SourceEvaluator(device=device)
    experiment_simulator = ExperimentSimulator(device=device)
    rubric_grader = RubricGrader(device=device)
    auth_ethics = AuthAndEthics(device=device)

    # Dataset loading and analysis
    dataset = load_and_tokenize("ag_news", "train[:20]", "distilgpt2")
    stats = analyze_tokenized_dataset(dataset, max_samples=20)
    print("Dataset stats:", stats)

    # Basic training and evaluation
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

    # Prompt optimization examples
    base_prompt = "Summarize the research article:"
    opt = PromptOptimizer("distilgpt2")
    print("PromptOptimizer:", opt.optimize_prompt(base_prompt))

    adv = AdvancedPromptOptimizer("distilgpt2")
    print("AdvancedPromptOptimizer:", adv.optimize_prompt(base_prompt))

    ctx = ContextualPromptOptimizer("distilgpt2", "ag_news")
    print("ContextualPromptOptimizer:", ctx.optimize_prompt(base_prompt))

    bandit = PromptBanditOptimizer("distilgpt2", reward_fn=len, epsilon=0.2)
    print("BanditOptimizer:", bandit.optimize_prompt(base_prompt))

    anneal = PromptAnnealingOptimizer("distilgpt2", temperature=1.0, steps=5)
    print("AnnealingOptimizer:", anneal.optimize_prompt(base_prompt))

    rl = PromptRLOptimizer("distilgpt2", reward_fn=len, episodes=2)
    print("RLOptimizer:", rl.optimize_prompt(base_prompt))

    bayes = PromptBayesOptimizer("distilgpt2", n_calls=5)
    print("BayesOptimizer:", bayes.optimize_prompt(base_prompt))

    evolver = PromptEvolver("distilgpt2")
    print("PromptEvolver:", evolver.evolve_prompt(base_prompt, generations=1))

    augmenter = PromptAugmenter("distilgpt2")
    print("Augmented prompts:", augmenter.augment_dataset([base_prompt], n_variations=2))

    tuner = PromptEmbeddingTuner("distilgpt2", prompt_length=5)
    tuner.tune("Example text", steps=1)
    print("Tuned tokens:", tuner.get_prompt_tokens())

    meta = MetaPromptOptimizer("distilgpt2")
    print("MetaPromptOptimizer:", meta.optimize_prompt(base_prompt))

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

    print("Start the collaboration server with 'python3 peer_collab/collaboration_server.py'")


if __name__ == "__main__":
    main()
