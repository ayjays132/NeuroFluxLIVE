import torch
from analysis.prompt_embedding_tuner import PromptEmbeddingTuner
from unittest.mock import patch


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(10, 4)
        self.lm_head = torch.nn.Linear(4, 10)

    def forward(self, input_ids=None, inputs_embeds=None, labels=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        logits = self.lm_head(inputs_embeds)
        loss = logits.mean()
        return type('Output', (), {'loss': loss, 'logits': logits})

    def get_input_embeddings(self):
        return self.embed

    def to(self, device):
        return self


class DummyBatch(dict):
    def to(self, device):
        return self


class DummyTokenizer:
    def __call__(self, text, return_tensors=None):
        ids = torch.tensor([[1, 2]])
        if return_tensors == 'pt':
            return DummyBatch({'input_ids': ids, 'attention_mask': torch.ones_like(ids)})
        return ids.tolist()

    def decode(self, ids):
        return ' '.join(str(i) for i in ids)


def test_prompt_embedding_tuner():
    with patch('analysis.prompt_embedding_tuner.AutoModelForCausalLM.from_pretrained', return_value=DummyModel()), \
         patch('analysis.prompt_embedding_tuner.AutoTokenizer.from_pretrained', return_value=DummyTokenizer()):
        tuner = PromptEmbeddingTuner('dummy', prompt_length=2, device='cpu')
        initial = tuner.prompt_embeddings.clone()
        tuner.tune('hello', steps=1)
        assert not torch.allclose(initial, tuner.prompt_embeddings)
        tokens = tuner.get_prompt_tokens()
        assert len(tokens) == 2
