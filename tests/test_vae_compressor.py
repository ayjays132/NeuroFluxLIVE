import torch
from models.vae_compressor import VAECompressor


def test_encode_decode_roundtrip():
    comp = VAECompressor(latent_dim=4, device="cpu")
    data = torch.randn(8)
    # train briefly for stable reconstruction
    for _ in range(200):
        batch = data.unsqueeze(0)
        comp.train_step(batch)
    latent = comp.encode(data)
    recon = comp.decode(latent, data.numel())
    assert torch.allclose(recon, data, atol=0.1)
