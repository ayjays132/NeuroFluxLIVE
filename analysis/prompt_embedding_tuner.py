"""Trainable soft prompt embeddings for prompt engineering."""

from __future__ import annotations

from typing import Optional, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class PromptEmbeddingTuner:
    """Tune soft prompt embeddings to minimize language model loss."""

    def __init__(self, model_name: str, prompt_length: int, *, device: Optional[str] = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prompt_length = prompt_length
        emb_dim = self.model.get_input_embeddings().weight.shape[1]
        self.prompt_embeddings = torch.nn.Parameter(
            torch.randn(prompt_length, emb_dim, device=self.device)
        )
        self.optimizer = torch.optim.Adam([self.prompt_embeddings], lr=1e-2)

    def tune(self, target_text: str, *, steps: int = 20) -> None:
        """Optimize the prompt embeddings for the given target text."""
        inputs = self.tokenizer(target_text, return_tensors="pt").to(self.device)
        input_embeds = self.model.get_input_embeddings()(inputs["input_ids"])
        labels_prefix = torch.full(
            (inputs["input_ids"].size(0), self.prompt_length),
            -100,
            dtype=torch.long,
            device=self.device,
        )
        labels = torch.cat([labels_prefix, inputs["input_ids"]], dim=1)
        prompt_batch = self.prompt_embeddings.unsqueeze(0).expand(
            inputs["input_ids"].size(0), -1, -1
        )
        for _ in range(steps):
            self.optimizer.zero_grad()
            embeds = torch.cat([prompt_batch, input_embeds], dim=1)
            outputs = self.model(inputs_embeds=embeds, labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

    def get_prompt_tokens(self) -> List[str]:
        """Return tokens approximating the soft prompt embeddings."""
        embedding_matrix = self.model.get_input_embeddings().weight
        sims = torch.matmul(self.prompt_embeddings, embedding_matrix.t())
        ids = torch.argmax(sims, dim=1)
        return [self.tokenizer.decode([i]) for i in ids]
