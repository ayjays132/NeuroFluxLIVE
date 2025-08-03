
"""Premium workflow tying together real-time data absorption, GPT-2 inference,
gym-based reinforcement learning and rubric-based feedback.

The workflow is intentionally lightweight so it can be exercised in unit tests
without heavy computation. Each component can be toggled via the ``workflow``
section of ``config.yaml``.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict

import gym
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from RealTimeDataAbsorber import RealTimeDataAbsorber
    from assessment.rubric_grader import RubricGrader


log = logging.getLogger(__name__)


def run_pipeline(
    config: Dict[str, Any],
    *,
    use_gym: bool = False,
    gym_env: str = "CartPole-v1",
    max_steps: int = 100,
) -> Dict[str, float]:
    """Run the premium workflow.

    Parameters
    ----------
    config:
        Parsed configuration dictionary.
    use_gym:
        Whether to launch the Gym RL loop.
    gym_env:
        Environment name to use when ``use_gym`` is ``True``.
    max_steps:
        Maximum number of environment steps to run.

    Returns
    -------
    Dict[str, float]
        Basic metrics collected during the run (e.g., total reward).
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)

    absorber = None
    if config.get("workflow", {}).get("enable_absorber", False):
        from RealTimeDataAbsorber import RealTimeDataAbsorber  # type: ignore

        absorber = RealTimeDataAbsorber(model_config={}, metrics_queue=None)
        absorber.start_absorption()
        log.info("RealTimeDataAbsorber started")

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)

    grader = None
    if config.get("workflow", {}).get("enable_evaluator", False):
        from assessment.rubric_grader import RubricGrader  # type: ignore

        grader = RubricGrader(device=device)

    metrics: Dict[str, float] = {}

    if use_gym and config.get("workflow", {}).get("enable_rl", False):
        env = gym.make(gym_env)
        obs, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        total_reward = 0.0

        rubric = {"response": {"expected_content": "hello", "max_score": 1}}

        for step in range(max_steps):
            action = env.action_space.sample()
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, _ = step_result
            else:  # gym <0.26 style
                next_obs, reward, terminated, _ = step_result
                truncated = False

            prompt = f"Step {step}: say hello"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output_ids = model.generate(**inputs, max_new_tokens=8)
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            reward_total = reward
            if grader is not None:
                grade = grader.grade_submission(response, rubric)
                reward_total += sum(v["score"] for v in grade.values())

            total_reward += reward_total
            log.info(
                "step=%s env_reward=%.2f total_reward=%.2f",
                step,
                reward,
                total_reward,
            )

            obs = next_obs
            if terminated or truncated:
                break

        metrics["total_reward"] = float(total_reward)
        log.info("Episode 0 reward: %.2f", total_reward)

    if absorber is not None:
        absorber.stop_absorption()
        log.info("RealTimeDataAbsorber stopped")

    return metrics


def main() -> None:
    """Entry point for the command line interface."""

    parser = argparse.ArgumentParser(description="Run the premium workflow")
    parser.add_argument("--gym", action="store_true", help="enable Gym RL training")
    parser.add_argument(
        "--benchmark",
        default="CartPole-v1",
        help="Gym environment to use when --gym is supplied",
    )
    args = parser.parse_args()

    with open("config.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    run_pipeline(config, use_gym=args.gym, gym_env=args.benchmark)


if __name__ == "__main__":
    main()

