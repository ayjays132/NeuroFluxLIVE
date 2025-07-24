# Persistent Correlation Memory

`CorrelationRAGMemory` can compress stored embeddings with a `VAECompressor` and
persist them to disk. Both the compressed representations and FAISS index are
stored so that memories survive application restarts. Call `save()` to write the
current state or rely on `RealTimeDataAbsorber.log_performance_metrics()` which
invokes `save()` automatically once per hour. Use `load()` to restore a memory
file.

Example:
```python
from correlation_rag_module import CorrelationRAGMemory
from models.vae_compressor import VAECompressor

compressor = VAECompressor(latent_dim=16)
memory = CorrelationRAGMemory(emb_dim=384, save_path="synergy.mem",
                              compressor=compressor)
```
Running again with the same `save_path` will automatically reload the previous
memory, enabling continuous learning across sessions.
