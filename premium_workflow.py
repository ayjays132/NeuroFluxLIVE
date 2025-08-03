from __future__ import annotations

"""End-to-end workflow tying together real-time absorption, RL training and
language model querying.

This module launches :class:`RealTimeDataAbsorber`, loads a GPT-2 style model
from the HuggingFace Hub and optionally kicks off a tiny Gym training loop.  A
``RubricGrader`` can be used to score model responses which are then fed back as
additional rewards for reinforcement learning.  The behaviour of the workflow is
controlled via ``config.yaml`` allowing users to enable or disable the absorber,
trainer or evaluator.

The script exposes a small CLI entry point:

``python premium_workflow.py --gym --benchmark sprout-agi``

Running with ``--gym`` executes the Gym trainer while ``--benchmark`` simply
prints the benchmark name, acting as a placeholder for future integrations.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import argparse
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from assessment.rubric_grader import RubricGrader
from simulation_lab.gym_autonomous_trainer import (
    TrainerConfig,
    run_gym_autonomous_trainer,
)


MODEL_NAME = "ayjays132/NeuroReasoner-1-NR-1"

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from RealTimeDataAbsorber import RealTimeDataAbsorber


def _load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration."""

    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def query_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
) -> str:
    """Generate a response for ``prompt`` using ``model``."""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=inputs["input_ids"].shape[1] + 20,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Premium NeuroFlux workflow")
    parser.add_argument("--gym", action="store_true", help="run Gym RL trainer")
    parser.add_argument(
        "--benchmark", type=str, default="", help="optional benchmark tag"
    )
    args = parser.parse_args(argv)

    cfg = _load_config()
    wf_cfg = cfg.get("workflow", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    absorber: Optional["RealTimeDataAbsorber"] = None
    if wf_cfg.get("enable_absorber", True):
        from RealTimeDataAbsorber import RealTimeDataAbsorber

        absorber = RealTimeDataAbsorber(
            model_config={}, settings={"db_path": cfg["paths"]["db_path"]}
        )

    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForCausalLM] = None
    if wf_cfg.get("enable_rl_trainer", True) or wf_cfg.get("enable_evaluator", True):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    evaluator_fn = None
    if wf_cfg.get("enable_evaluator", True):
        grader = RubricGrader(device=device)
        rubric = {"quality": {"expected_content": "helpful", "max_score": 1}}

        def evaluator_fn(text: str) -> float:
            scores = grader.grade_submission(text, rubric)
            return float(scores["quality"]["score"])

    if args.gym and wf_cfg.get("enable_rl_trainer", True):
        trainer_cfg = TrainerConfig(model_name=MODEL_NAME)
        run_gym_autonomous_trainer(
            trainer_cfg, absorber=absorber, evaluator=evaluator_fn
        )

    if model is not None and tokenizer is not None:
        response = query_model(model, tokenizer, "Hello, researcher!", device)
        print(f"Model response: {response}")
        if absorber is not None:
            absorber.absorb_data(response, "text", source="query")

    if absorber is not None:
        absorber.log_performance_metrics()

    if args.benchmark:
        print(f"Benchmark requested: {args.benchmark}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

