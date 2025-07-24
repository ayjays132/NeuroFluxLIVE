# Persistent Correlation Memory

`CorrelationRAGMemory` can optionally compress stored embeddings with a `VAECompressor`.
Compressed representations and the FAISS index are saved to disk so that memories
persist across sessions. Use `save()` and `load()` to manage state manually or
allow `RealTimeDataAbsorber.log_performance_metrics()` to trigger automatic
saves.

Example:
```python
from correlation_rag_module import CorrelationRAGMemory
from models.vae_compressor import VAECompressor

compressor = VAECompressor(latent_dim=16)
memory = CorrelationRAGMemory(emb_dim=384, save_path="synergy.mem",
                              compressor=compressor)
```
When run again with the same `save_path`, the memory is reloaded
automatically.
