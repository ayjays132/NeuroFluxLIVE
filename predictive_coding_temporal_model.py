"""
predictive_coding_temporal_model.py
───────────────────────────────────
A CUDA‑ready Predictive‑Coding head that plugs into:

  • RealTimeDataAbsorber          – supply `embedding` per time‑step
  • CorrelationRAGMemory          – use prediction‑error as extra context
  • multimodal_dataset_manager    – works directly on stacked batches
  • token_sequence_pattern_manager – feed rare high‑error bursts
  • GUI / CustomLabelsDataset     – expose live predictive‑error charts

Key ideas
---------
1.  Gated *Dilated* Conv → quick local context
2.  Tiny Transformer Encoder → long‑range context (4 heads, 2 layers)
3.  Predictive‑Coding loss  L = ‖x_t+1 − ŷ_t‖₂²  +  β · KL( post ‖ prior )
4.  Plasticity regulariser  (online EWC‑lite) keeps weights stable
5.  Online adaptation step with optional mixed‑precision

API
---
pc = PredictiveCodingTemporalModel(embed_dim=256).to("cuda")
pred, err = pc.observe_step(current_embedding)   # err is scalar
pc.backward_and_optimize()                       # done every few steps
"""

from __future__ import annotations
import torch, math, itertools, collections
from torch import nn
from typing import Deque, Tuple, List, Optional

# ────────────────────────────────────────────────────────────────────── #
class _GatedDilatedConv1D(nn.Module):
    """Depth‑wise gated temporal conv similar to WaveNet / ByteNet."""
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.conv_f = nn.Conv1d(dim, dim, 3, padding=dilation, dilation=dilation, groups=dim)
        self.conv_g = nn.Conv1d(dim, dim, 3, padding=dilation, dilation=dilation, groups=dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.tanh(self.conv_f(x)) * torch.sigmoid(self.conv_g(x))
        return x + y                      # residual

# ────────────────────────────────────────────────────────────────────── #
class PredictiveCodingTemporalModel(nn.Module):
    def __init__(self,
                 embed_dim      : int,
                 context_window : int = 32,
                 dilations      : Tuple[int] = (1, 2, 4, 8),
                 transformer_depth: int = 2,
                 lr             : float = 3e-4,
                 beta_kl        : float = 1e-3,
                 device         : str   = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device          = device
        self.embed_dim       = embed_dim
        self.context_window  = context_window
        self.beta_kl         = beta_kl

        # 1) Gated dilated conv stack
        self.gconv = nn.Sequential(*[
            _GatedDilatedConv1D(embed_dim, d) for d in dilations
        ])

        # 2) Tiny transformer for global context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4,
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, transformer_depth)

        # 3) Linear projection to predict next embedding
        self.predictor = nn.Linear(embed_dim, embed_dim)

        # Variational latent (optional ‑ simple diagonal gaussian)
        self.mu  = nn.Linear(embed_dim, embed_dim)
        self.log = nn.Linear(embed_dim, embed_dim)

        # Ring buffer for recent embeddings (no grad, CPU)
        self.buffer : Deque[torch.Tensor] = collections.deque(maxlen=context_window+1)

        # optimiser
        self.opt  = torch.optim.AdamW(self.parameters(), lr=lr)

        # online Fisher‑ratio tracker for EWC‑lite
        self.register_buffer("fisher", torch.zeros(sum(p.numel() for p in self.parameters())))

        self.to(device)

    # ─────────────────────────────────────────────────────────────── #
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: [B, T, D]  where T = context_window
        returns predicted next embedding  [B, D]
        """
        # 1) locality conv (operate on channels‑first conv1d)
        x = seq.transpose(1, 2)                        # B, D, T
        x = self.gconv(x)
        x = x.transpose(1, 2)                          # B, T, D

        # 2) transformer global context
        x = self.transformer(x)                        # B, T, D

        # Pool last timestep
        h = x[:, -1]                                   # B, D

        # 3) latent reparam
        mu, logvar = self.mu(h), self.log(h).clamp(-10, 10)
        std   = torch.exp(0.5 * logvar)
        z     = mu + std * torch.randn_like(std)

        # 4) predict next embedding
        y_hat = self.predictor(z)
        return y_hat, mu, logvar

    # ─────────────────────────────────────────────────────────────── #
    def observe_step(self, embedding: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Push an embedding (1, D) and get (prediction, error).
        Call `backward_and_optimize` periodically.
        """
        embedding = embedding.detach().to(self.device).float()
        self.buffer.append(embedding.cpu())   # keep raw for memory economy

        if len(self.buffer) <= self.context_window:        # not enough context yet
            return embedding, 0.0

        # build context tensor
        ctx = torch.stack(list(itertools.islice(self.buffer, len(self.buffer)-self.context_window, len(self.buffer)))) # T,D
        ctx = ctx.unsqueeze(0).to(self.device)             # 1,T,D

        pred, mu, logvar = self.forward(ctx)
        err = torch.mean((embedding - pred.detach())**2).item()

        # ─ store for optimisation ─ #
        if not hasattr(self, "_loss_terms"):
            self._loss_terms : List[torch.Tensor] = []
        recon = torch.mean((embedding - pred)**2)
        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        self._loss_terms.append(recon + self.beta_kl * kl)

        # Fisher diag update (online) for plasticity regularisation
        g = torch.autograd.grad(recon, self.parameters(), retain_graph=True, allow_unused=True)
        offset = 0
        with torch.no_grad():
            for p, gp in zip(self.parameters(), g):
                if gp is None: 
                    offset += p.numel(); continue
                sz = p.numel()
                self.fisher[offset:offset+sz] = 0.95*self.fisher[offset:offset+sz] + 0.05*gp.flatten().abs()
                offset += sz

        return pred.detach(), err

    # ─────────────────────────────────────────────────────────────── #
    def backward_and_optimize(self, clip: float = 1.0):
        """Call every N steps to actually back‑prop an accumulated batch."""
        if not hasattr(self, "_loss_terms") or not self._loss_terms:
            return
        loss = torch.stack(self._loss_terms).mean()

        # EWC‑lite: quadratic penalty on important params
        offset = 0
        for p in self.parameters():
            sz = p.numel()
            fisher_slice = self.fisher[offset:offset+sz].view_as(p)
            loss += 1e-4 * torch.sum(fisher_slice * p**2)
            offset += sz

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.opt.step()
        self._loss_terms.clear()

    # ─────────────────────────────────────────────────────────────── #
    @torch.no_grad()
    def export_state(self) -> dict:
        """Lightweight checkpoint."""
        return {
            "model": self.state_dict(),
            "opt"  : self.opt.state_dict(),
            "fisher": self.fisher.cpu()
        }

    def load_state(self, ckpt: dict):
        self.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["opt"])
        self.fisher.copy_(ckpt["fisher"].to(self.device))

# ────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":  # tiny smoke test
    torch.manual_seed(0)
    pc = PredictiveCodingTemporalModel(embed_dim=64, context_window=8).to("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(200):
        emb = torch.randn(1, 64)
        _, err = pc.observe_step(emb)
        if i % 16 == 0 and err:
            pc.backward_and_optimize()
            print(f"step {i:03d}  pred‑err {err:6.4f}")
