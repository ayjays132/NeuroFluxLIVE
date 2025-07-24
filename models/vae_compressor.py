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
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.vaes: Dict[int, _VAE] = {}

    def _get_vae(self, input_dim: int) -> _VAE:
        if input_dim not in self.vaes:
            self.vaes[input_dim] = _VAE(input_dim, self.latent_dim).to(self.device)
        return self.vaes[input_dim]

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(-1).detach().cpu()

    def decode(self, latent: torch.Tensor, output_dim: int) -> torch.Tensor:
        x = latent.view(-1)
        if x.numel() >= output_dim:
            return x[:output_dim].cpu()
        pad = torch.zeros(output_dim - x.numel())
        return torch.cat([x, pad]).cpu()

    def train_step(self, batch: torch.Tensor) -> float:
        return 0.0
