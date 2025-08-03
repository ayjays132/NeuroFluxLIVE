from __future__ import annotations

"""Minimal autonomous trainer using a Gym environment and GPT-2.

This module demonstrates how a transformer model can drive reinforcement
learning without relying on a fixed dataset.  At each step the model both
selects an action for the environment **and** answers a user prompt.  The
response alongside the reward signal is streamed to ``RealTimeDataAbsorber``
so that metrics can be consumed by other components in real time.
"""

from dataclasses import dataclass
from queue import Queue
from typing import Callable, List, Optional, Tuple

import gym
import logging
import torch
from torch import nn
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer


log = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for :func:`run_gym_autonomous_trainer`"""

    env_name: str = "CartPole-v1"
    episodes: int = 3
    gamma: float = 0.99
    model_name: str = "gpt2"
    prompt: str = "Provide a motivational message for the next move."


def run_gym_autonomous_trainer(
    cfg: TrainerConfig | None = None,
    *,
    absorber: Optional["RealTimeDataAbsorber"] = None,
    evaluator: Optional[Callable[[str], float]] = None,
) -> Tuple[List[float], List[str]]:
    """Train a small policy/value head on top of GPT-2 in a Gym environment.

    Parameters
    ----------
    cfg:
        Optional :class:`TrainerConfig`.  Uses sensible defaults if omitted.
    absorber:
        Optional :class:`RealTimeDataAbsorber` instance used to stream metrics.
        If ``None`` a new absorber is created internally.
    evaluator:
        Optional callable that receives a generated response text and returns
        an additional reward signal which is added to the environment reward.

    Returns
    -------
    Tuple[List[float], List[str]]
        Episode rewards and a sample of textual responses generated during
        interaction.
    """

    cfg = cfg or TrainerConfig()
    if not log.handlers:
        logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(cfg.env_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)

    policy_head = nn.Linear(model.config.n_embd, env.action_space.n).to(device)
    value_head = nn.Linear(model.config.n_embd, 1).to(device)
    optimiser = torch.optim.Adam(
        list(policy_head.parameters()) + list(value_head.parameters()), lr=1e-3
    )

    if absorber is None:
        metrics_queue: Queue = Queue()
        from RealTimeDataAbsorber import RealTimeDataAbsorber  # Lazy import

        absorber = RealTimeDataAbsorber(model_config={}, metrics_queue=metrics_queue)

    episode_returns: List[float] = []
    responses: List[str] = []

    for ep in range(cfg.episodes):
        obs, _ = env.reset()
        done = False
        transitions = []
        step = 0
        ep_return = 0.0

        while not done:
            obs_text = " ".join(f"{o:.2f}" for o in obs)
            tokens = tokenizer(obs_text, return_tensors="pt").to(device)
            with torch.no_grad():
                hidden = model.transformer(**tokens).last_hidden_state[:, -1, :]
            logits = policy_head(hidden)
            value = value_head(hidden).squeeze(-1)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            prompt_ids = tokenizer(cfg.prompt, return_tensors="pt").to(device)
            response_ids = model.generate(
                **prompt_ids,
                max_length=prompt_ids["input_ids"].shape[1] + 20,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
            responses.append(response_text)

            if evaluator is not None:
                try:
                    reward += float(evaluator(response_text))
                except Exception as exc:  # pragma: no cover - safety net
                    log.error("Evaluator failure: %s", exc)

            log.info(
                "Episode %d Step %d | Reward: %.2f | Response: %s",
                ep,
                step,
                reward,
                response_text,
            )

            if absorber is not None:
                absorber.performance_metrics.update(
                    {"latest_reward": float(reward), "episode": ep, "step": step}
                )
                absorber.absorb_data(
                    response_text, "text", source="gym_autonomous_trainer", priority=1
                )
                if hasattr(absorber, "_emit_metrics"):
                    absorber._emit_metrics()

            transitions.append((log_prob, value, reward))
            obs = next_obs
            step += 1
            ep_return += float(reward)

        episode_returns.append(ep_return)

        returns = []
        G = 0.0
        for _, _, r in reversed(transitions):
            G = r + cfg.gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, device=device)
        log_probs = torch.stack([t[0] for t in transitions])
        values = torch.stack([t[1] for t in transitions])
        advantage = returns_t - values.detach()

        policy_loss = -(log_probs * advantage).mean()
        value_loss = (returns_t - values).pow(2).mean()
        loss = policy_loss + value_loss

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    env.close()
    return episode_returns, responses[:5]


__all__ = ["TrainerConfig", "run_gym_autonomous_trainer"]
