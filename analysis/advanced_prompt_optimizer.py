"""Advanced prompt optimization with semantic similarity."""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

from .prompt_optimizer import PromptOptimizer


class AdvancedPromptOptimizer(PromptOptimizer):
    """Optimize prompts using perplexity and semantic similarity."""

    def __init__(
        self,
        model_name: str,
        *,
        embedding_model: Optional[str] = None,
        device: Optional[str] = None,
        similarity_weight: float = 0.5,
    ) -> None:
        super().__init__(model_name, device=device)
        emb_model_name = embedding_model or model_name
        self.embed_tokenizer = AutoTokenizer.from_pretrained(emb_model_name)
        self.embed_model = AutoModel.from_pretrained(emb_model_name).to(self.device)
        self.similarity_weight = similarity_weight

    def _embedding(self, text: str) -> torch.Tensor:
        tokens = self.embed_tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.embed_model(**tokens)
            hidden = outputs.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return pooled.squeeze(0)

    def score_prompt(self, prompt: str, *, base_embedding: torch.Tensor) -> float:
        perplexity = super().score_prompt(prompt)
        emb = self._embedding(prompt)
        similarity = F.cosine_similarity(emb, base_embedding, dim=0).item()
        return perplexity - self.similarity_weight * similarity

    def optimize_prompt(self, base_prompt: str, n_variations: int = 5) -> str:
        candidates = self.generate_variations(base_prompt, n_variations=n_variations)
        base_emb = self._embedding(base_prompt)
        best_prompt: str = candidates[0]
        best_score = float("inf")
        for cand in candidates:
            score = self.score_prompt(cand, base_embedding=base_emb)
            if score < best_score:
                best_score = score
                best_prompt = cand
        return best_prompt
