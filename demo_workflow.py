
"""Entry point demonstrating the various SelfResearch modules."""

import argparse
import yaml
import torch


def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

from research_workflow.topic_selector import TopicSelector
from digital_literacy.source_evaluator import SourceEvaluator
from simulation_lab.experiment_simulator import ExperimentSimulator
from peer_collab.collaboration_server import CollaborationServer
from assessment.rubric_grader import RubricGrader
from security.auth_and_ethics import AuthAndEthics
from data.dataset_loader import load_and_tokenize
from train.trainer import TrainingConfig, train_model

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the SelfResearch demo")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to YAML configuration"
    )
    args = parser.parse_args(argv)

    settings = load_config(args.config)
    print(f"Loaded config from {args.config}: {settings}")

    # Determine device (CUDA or CPU)
    if torch.cuda.is_available():
        device = 'cuda'
        print("CUDA is available! Using GPU.")
    else:
        device = 'cpu'
        print("CUDA not available. Using CPU.")

    print("\n--- Initializing Platform Modules ---")
    topic_selector = TopicSelector(device=device)
    source_evaluator = SourceEvaluator(device=device)
    experiment_simulator = ExperimentSimulator(device=device)
    rubric_grader = RubricGrader(device=device)
    auth_ethics = AuthAndEthics(device=device)

    # --- Demonstrate Dataset Loading ---
    print("\n--- Demonstrating Dataset Loading ---")
    tokenized_ds = load_and_tokenize("ag_news", "train[:100]", "distilgpt2")
    print(f"Loaded {len(tokenized_ds)} tokenized samples")

    # --- Demonstrate Training Pipeline ---
    print("\n--- Demonstrating Training Pipeline ---")
    train_cfg = TrainingConfig(
        model_name="distilgpt2",
        dataset_name="ag_news",
        train_split="train[:20]",
        eval_split="test[:20]",
        epochs=1,
        batch_size=2,
        output_dir="./demo_model",
    )
    train_model(train_cfg)

    # --- Demonstrate Research Workflow --- 
    print("\n--- Demonstrating Research Workflow (Topic Selection) ---")
    research_area = "sustainable agriculture"
    suggested_topic = topic_selector.suggest_topic(research_area)
    print(f"Suggested Topic for '{research_area}': {suggested_topic}")
    question = "How can precision farming techniques optimize resource utilization and minimize environmental impact?"
    print(f"Is '{question}' a valid question? {topic_selector.validate_question(question)}")

    # --- Demonstrate Digital Literacy --- 
    print("\n--- Demonstrating Digital Literacy (Source Evaluation) ---")
    url_to_evaluate = "https://www.nature.com/articles/s41586-023-06012-1"
    evaluation_results = source_evaluator.evaluate_source(url_to_evaluate)
    print(f"Evaluation for {url_to_evaluate}: {evaluation_results}")
    search_query = "precision agriculture AI"
    arxiv_results = source_evaluator.search_academic_api(search_query, api_type="arxiv")
    print(f"ArXiv search results for '{search_query}': {arxiv_results}")
    semantic_scholar_results = source_evaluator.search_academic_api(search_query, api_type="semantic_scholar")
    print(f"Semantic Scholar search results for '{search_query}': {semantic_scholar_results}")

    # --- Demonstrate Simulation Lab --- 
    print("\n--- Demonstrating Simulation Lab (Physics & Biological Simulation, Data Generation) ---")
    # Physics Simulation
    positions = experiment_simulator.run_physics_simulation(initial_position=0.0, initial_velocity=20.0, time_steps=150, dt=0.1)
    print(f"First 5 simulated physics positions: {positions[:5].tolist()}")
    experiment_simulator.visualize_data(positions, "Simulated Projectile Motion Enhanced")

    # Biological Simulation (Lotka-Volterra)
    initial_pop_a = 100.0
    initial_pop_b = 10.0
    growth_a = 0.1
    growth_b = 0.05
    interaction = 0.001
    bio_time_steps = 200
    bio_dt = 0.5
    populations = experiment_simulator.run_biological_simulation(initial_pop_a, initial_pop_b, growth_a, growth_b, interaction, bio_time_steps, bio_dt)
    print(f"First 5 simulated biological populations: {populations[:5].tolist()}")
    experiment_simulator.visualize_data(populations, "Lotka-Volterra Simulation Enhanced", labels=["Prey Population", "Predator Population"])

    # Synthetic Data Generation
    synthetic_data = experiment_simulator.generate_synthetic_data(num_samples=500, num_features=4)
    print(f"Generated synthetic data shape: {synthetic_data.shape}")

    # --- Demonstrate Assessment --- 
    print("\n--- Demonstrating Assessment (Rubric Grader) ---")
    rubric_example = {
        "Introduction Clarity": {"expected_content": "The introduction should clearly state the research question, its significance, and the paper's structure, providing a strong hook for the reader.", "max_score": 10},
        "Methodology Detail": {"expected_content": "The methodology section must provide sufficient detail for replication, including data sources, experimental design, analytical techniques, and ethical considerations.", "max_score": 15},
        "Conclusion Strength": {"expected_content": "The conclusion should summarize key findings, discuss implications, and suggest future research directions, leaving a lasting impression.", "max_score": 8}
    }
    submission_text_good = "This paper investigates the impact of precision farming on crop yield and environmental sustainability. We utilized satellite imagery and soil sensor data, employing advanced machine learning models for analysis. Our methods are detailed to ensure reproducibility. The conclusion highlights significant improvements in resource efficiency and proposes further studies on long-term ecological effects."
    grades_good = rubric_grader.grade_submission(submission_text_good, rubric_example)
    print("\n--- Grading Good Submission ---")
    for criterion, result in grades_good.items():
        print(f"Criterion: {criterion}, Score: {result['score']:.2f}/{result['max_score']}, Feedback: {result['feedback']}")

    submission_text_poor = "My project is about plants. I used some data. It was hard to write the end."
    grades_poor = rubric_grader.grade_submission(submission_text_poor, rubric_example)
    print("\n--- Grading Poor Submission ---")
    for criterion, result in grades_poor.items():
        print(f"Criterion: {criterion}, Score: {result['score']:.2f}/{result['max_score']}, Feedback: {result['feedback']}")

    # --- Demonstrate Security --- 
    print("\n--- Demonstrating Security (Authentication & Ethics) ---")
    auth_ethics.register_user("new_researcher", "secure_pass_123", "researcher")
    print(f"New researcher authenticated: {auth_ethics.authenticate_user('new_researcher', 'secure_pass_123')}")
    print(f"New researcher can access data: {auth_ethics.has_permission('new_researcher', 'access_data')}")
    auth_ethics.flag_ethical_concern("Use of public social media data without explicit consent.", "social_data_proj_005", "new_researcher")
    print("Ethical flags (pending):")
    for flag in auth_ethics.get_ethical_flags(status="pending"):
        print(flag)
    auth_ethics.review_ethical_concern(0, "under_review", "admin_user")
    print("Ethical flags (after review):")
    for flag in auth_ethics.get_ethical_flags():
        print(flag)

    # --- Collaboration Server (Note: This will block if run directly, typically run in a separate process) ---
    print("\n--- Collaboration Server (To run, execute peer_collab/collaboration_server.py separately) ---")
    print("The collaboration server is designed to run as a separate Flask application.")
    print("You can start it by running: python3 peer_collab/collaboration_server.py")
    print("Then interact with it using tools like `curl` or a web browser.")

if __name__ == "__main__":
    main()


