import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Optional


class _VAE(nn.Module):
    """Simple VAE for 1D tensors."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc21 = nn.Linear(128, latent_dim)
        self.fc22 = nn.Linear(128, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)


class VAECompressor(nn.Module):
    """VAE-based compressor for arbitrary tensors."""

    def __init__(self, latent_dim: int = 32, device: Optional[str] = None) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.vaes: Dict[int, _VAE] = {}
        self.optims: Dict[int, torch.optim.Optimizer] = {}

    def _get_vae(self, input_dim: int) -> _VAE:
        if input_dim not in self.vaes:
            vae = _VAE(input_dim, self.latent_dim).to(self.device)
            self.vaes[input_dim] = vae
            self.optims[input_dim] = torch.optim.Adam(vae.parameters(), lr=5e-2)
        return self.vaes[input_dim]

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.view(1, -1).to(self.device)
        vae = self._get_vae(x.size(1))
        with torch.no_grad():
            mu, logvar = vae.encode(x)
            z = vae.reparameterize(mu, logvar)
        return z.squeeze(0).detach().cpu()

    def decode(self, latent: torch.Tensor, output_dim: int) -> torch.Tensor:
        z = latent.view(1, -1).to(self.device)
        vae = self._get_vae(output_dim)
        with torch.no_grad():
            recon = vae.decode(z)
        return recon.view(-1)[:output_dim].cpu()

    def train_step(self, batch: torch.Tensor, inner_steps: int = 2) -> float:
        """Perform one or more optimisation steps and return the last loss."""
        batch = batch.view(batch.size(0), -1).to(self.device)
        vae = self._get_vae(batch.size(1))
        optim = self.optims[batch.size(1)]
        loss_val = 0.0
        for _ in range(inner_steps):
            mu, logvar = vae.encode(batch)
            z = vae.reparameterize(mu, logvar)
            recon = vae.decode(z)
            recon_loss = F.mse_loss(recon, batch)
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.0001 * kld
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_val = float(loss.item())
        return loss_val
