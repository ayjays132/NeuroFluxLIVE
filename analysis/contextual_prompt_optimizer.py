"""Context-aware prompt optimization using dataset embeddings."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel

from .prompt_optimizer import PromptOptimizer


class ContextualPromptOptimizer(PromptOptimizer):
    """Optimize prompts by maximizing similarity to a dataset embedding."""

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        *,
        split: str = "train[:100]",
        similarity_weight: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(model_name, device=device)
        self.embed_model = AutoModel.from_pretrained(model_name).to(self.device)
        self.dataset_embedding = self._compute_dataset_embedding(dataset_name, split)
        self.similarity_weight = similarity_weight

    def _compute_dataset_embedding(self, dataset_name: str, split: str) -> torch.Tensor:
        dataset = load_dataset(dataset_name, split=split)
        texts = [r["text"] for r in dataset]
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            hidden = self.embed_model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return pooled.mean(dim=0)

    def _embedding(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            hidden = self.embed_model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return pooled.squeeze(0)

    def score_prompt(self, prompt: str) -> float:  # type: ignore[override]
        perplexity = super().score_prompt(prompt)
        emb = self._embedding(prompt)
        sim = F.cosine_similarity(emb, self.dataset_embedding, dim=0).item()
        return perplexity - self.similarity_weight * sim
