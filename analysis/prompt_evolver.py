"""Evolutionary prompt optimization utilities."""

from __future__ import annotations

import math
import random
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class PromptEvolver:
    """Evolve prompts using a simple genetic algorithm."""

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device.type == "cuda" else -1,
        )

    def _mutate(self, prompt: str) -> str:
        """Return a mutated prompt using the language model."""
        out = self.generator(prompt, num_return_sequences=1, max_new_tokens=10)[0]
        gen = out["generated_text"]
        if gen.startswith(prompt):
            gen = gen[len(prompt):].strip()
        return f"{prompt} {gen}".strip()

    def _score(self, prompt: str) -> float:
        """Compute perplexity of a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            loss = self.model(**inputs, labels=inputs["input_ids"]).loss
        return float(math.exp(loss.item()))

    def evolve_prompt(
        self,
        base_prompt: str,
        *,
        generations: int = 3,
        population_size: int = 6,
        retain: float = 0.5,
        mutation_rate: float = 0.3,
    ) -> str:
        """Iteratively evolve a prompt and return the best result."""
        population = [base_prompt] + [self._mutate(base_prompt) for _ in range(population_size - 1)]
        for _ in range(generations):
            scored = [(p, self._score(p)) for p in population]
            scored.sort(key=lambda x: x[1])  # lower perplexity is better
            retain_len = max(1, int(len(scored) * retain))
            parents = [p for p, _ in scored[:retain_len]]
            # Crossover
            children: List[str] = []
            while len(children) + len(parents) < population_size:
                a, b = random.sample(parents, 2)
                split_a = a.split()
                split_b = b.split()
                if not split_a or not split_b:
                    child = a if self._score(a) <= self._score(b) else b
                else:
                    cut_a = random.randint(1, len(split_a))
                    cut_b = random.randint(1, len(split_b))
                    child = " ".join(split_a[:cut_a] + split_b[cut_b:])
                children.append(child)
            population = parents + children
            # Mutation
            for i in range(len(population)):
                if random.random() < mutation_rate:
                    population[i] = self._mutate(population[i])
        best_prompt = min(population, key=self._score)
        return best_prompt
