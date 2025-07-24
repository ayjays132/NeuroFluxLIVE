"""
correlation_rag_memory.py
─────────────────────────
Retrieval-Augmented Correlation Memory for real-time robots / agents.
"""

from __future__ import annotations
import faiss, numpy as np, torch, pickle, math, time, pathlib, os
from typing import List, Tuple, Dict, Optional, Any
from models.vae_compressor import VAECompressor

# ----------------------------------------------------------------------
def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

# ----------------------------------------------------------------------
class CorrelationRAGMemory:
    """Lightweight retrieval-augmented memory.

    When a ``VAECompressor`` is provided, added embeddings are compressed and
    saved alongside the FAISS index. ``save()`` writes both structures to disk
    and ``load()`` restores them so that memories persist across sessions.

    Public API:
        • add(emb, label, meta)           → store embedding & metadata
        • retrieve(query, k)              → top-k (emb, label, meta, score)
        • classify(query)                 → returns logits + context
        • update_head(batch_emb, batch_y) → online ridge‑regression update
    """

    def __init__(self,
                 emb_dim: Optional[int] = None,
                 memory_budget: int = 50_000,
                 device: Optional[str] = None,
                 pq_m: int = 16,
                 hnsw_m: int = 32,
                 l2_norm: bool = True,
                 save_path: str = "corr_rag.mem",
                 compressor: Optional[VAECompressor] = None,
                 settings: Optional[Dict[str, Any]] = None):

        if settings:
            emb_dim = settings.get("embed_dim", emb_dim)
            memory_budget = settings.get("memory_budget", memory_budget)
            device = settings.get("device", device)
            pq_m = settings.get("pq_m", pq_m)
            hnsw_m = settings.get("hnsw_m", hnsw_m)
            l2_norm = settings.get("l2_norm", l2_norm)
            save_path = settings.get("save_path", save_path)

        if emb_dim is None:
            raise ValueError("embed_dim must be provided via argument or settings")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.emb_dim        = emb_dim
        self.memory_budget  = memory_budget
        self.device         = device
        self.l2_norm        = l2_norm
        self.save_path      = pathlib.Path(save_path)
        self.compressor     = compressor
        self.compressed: Dict[int, np.ndarray] = {}

        # -------- FAISS index (HNSW + PQ for compression) -------------- #
        quantizer = faiss.IndexHNSWFlat(emb_dim, hnsw_m)
        pq_index  = faiss.IndexPQ(emb_dim, pq_m, 8)        # 8-bit per sub-vec
        self.index = faiss.IndexPreTransform(faiss.LinearTransform(emb_dim, emb_dim), pq_index)
        self.index = faiss.IndexIDMap2(self.index)         # enable add_with_ids
        self.index_is_trained = False

        # -------- lightweight correlation head ------------------------ #
        self.W   = torch.zeros((0, emb_dim), device=device)   # weights per class
        self.bias= torch.zeros(0, device=device)
        self.num_classes = 0
        self.fisher_diag = torch.zeros(0, device=device)      # for EWC

        # -------- memory stores -------------------------------------- #
        self.labels   : Dict[int, int] = {}    # id → class_id
        self.meta     : Dict[int, Any] = {}
        self.next_id  = 0

        if self.save_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def add(self, emb: np.ndarray, label: int, meta: Optional[Dict] = None):
        """Add one embedding."""
        emb = emb.astype("float32").reshape(1, -1)
        if self.l2_norm:
            emb = _l2norm(emb)

        # Train PQ the first time when enough vectors
        if not self.index_is_trained and self.next_id + 1 >= 256:
            self.index.train(emb)
            self.index_is_trained = True

        self.index.add_with_ids(emb, np.array([self.next_id], dtype="int64"))
        self.labels[self.next_id] = label
        self.meta[self.next_id] = meta or {}
        if self.compressor is not None:
            latent = self.compressor.encode(torch.from_numpy(emb.squeeze(0)))
            self.compressed[self.next_id] = latent.numpy()
        self.next_id += 1

        # prune if budget exceeded
        if self.next_id > self.memory_budget:
            self._prune_least_important()

    def retrieve(self, query: np.ndarray, k: int = 8) -> List[Tuple[np.ndarray, int, Dict, float]]:
        query = query.astype("float32").reshape(1, -1)
        if self.l2_norm:
            query = _l2norm(query)
        D, I = self.index.search(query, k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1: break
            results.append((self.index.reconstruct(idx), self.labels[idx], self.meta[idx], dist))
        return results

    def classify(self, query: np.ndarray, temperature: float = 1.0) -> Tuple[np.ndarray, List]:
        ctx = self.retrieve(query, k=16)
        if not ctx or self.num_classes == 0:
            return np.zeros(self.num_classes), ctx

        # compute correlation logits = q · W.T + b
        q  = torch.from_numpy(query).float().to(self.device)
        if self.l2_norm:
            q = torch.nn.functional.normalize(q, dim=1)
        logits = torch.matmul(q, self.W.t()) + self.bias
        return (logits / temperature).cpu().numpy()[0], ctx

    def update_head(self, batch_emb: np.ndarray, batch_labels: np.ndarray, lr: float = 0.5):
        """Online ridge-regression update + Fisher tracking."""
        batch_emb = torch.from_numpy(batch_emb).float().to(self.device)
        if self.l2_norm:
            batch_emb = torch.nn.functional.normalize(batch_emb, dim=1)

        # One-hot targets
        classes = np.unique(batch_labels)
        self._ensure_classes(int(classes.max()) + 1)

        targets = torch.zeros(batch_emb.shape[0], self.num_classes, device=self.device)
        targets[torch.arange(batch_emb.shape[0]), batch_labels] = 1.0

        # closed-form ridge step (W ← W - lr·grad)
        preds   = torch.matmul(batch_emb, self.W.t()) + self.bias
        grad_W  = torch.matmul((preds - targets).t(), batch_emb) / batch_emb.shape[0]
        grad_b  = (preds - targets).mean(0)

        self.W   -= lr * grad_W
        self.bias-= lr * grad_b

        # update Fisher diag for EWC
        with torch.no_grad():
            prob = torch.sigmoid(preds)
            fisher = torch.sum(prob * (1 - prob), dim=0) / batch_emb.shape[0]
            if len(self.fisher_diag) == 0:
                self.fisher_diag = fisher.clone()
            else:
                self.fisher_diag = 0.9 * self.fisher_diag + 0.1 * fisher

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _ensure_classes(self, n: int):
        if n <= self.num_classes: return
        extra = n - self.num_classes
        self.W    = torch.cat([self.W, 0.01*torch.randn(extra, self.emb_dim, device=self.device)])
        self.bias = torch.cat([self.bias, torch.zeros(extra, device=self.device)])
        self.fisher_diag = torch.cat([self.fisher_diag,
                                      torch.zeros(extra, device=self.device)])
        self.num_classes = n

    def _prune_least_important(self):
        """Importance ∝ Fisher_diag for the class of each vector."""
        all_ids = list(self.labels)
        importance = np.array([self.fisher_diag[self.labels[i]].item() for i in all_ids])
        keep_mask  = importance.argsort()[-self.memory_budget:]
        keep_ids   = set(np.array(all_ids)[keep_mask])

        remove_ids = [i for i in all_ids if i not in keep_ids]
        for rid in remove_ids:
            self.index.remove_ids(np.array([rid], dtype="int64"))
            self.labels.pop(rid, None)
            self.meta.pop(rid, None)
            self.compressed.pop(rid, None)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self):
        state = {
            "labels": self.labels,
            "meta": self.meta,
            "W": self.W.cpu(),
            "bias": self.bias.cpu(),
            "fisher": self.fisher_diag.cpu(),
            "next": self.next_id,
            "l2": self.l2_norm,
            "compressed": self.compressed if self.compressed else None,
        }
        self.save_path.with_suffix(".pkl").write_bytes(pickle.dumps(state))
        faiss.write_index(self.index, str(self.save_path))

    def load(self) -> None:
        """Public method to load state from disk if available."""
        self._load()

    def _load(self):
        try:
            state = pickle.loads(self.save_path.with_suffix(".pkl").read_bytes())
            self.labels = state["labels"]
            self.meta   = state["meta"]
            self.W      = state["W"].to(self.device)
            self.bias   = state["bias"].to(self.device)
            self.fisher_diag = state["fisher"].to(self.device)
            self.next_id = state["next"]
            self.l2_norm = state["l2"]
            self.compressed = state.get("compressed", {}) or {}
            self.num_classes = self.W.shape[0]
            if pathlib.Path(str(self.save_path)).exists():
                self.index = faiss.read_index(str(self.save_path))
                self.index_is_trained = True
            elif self.compressed and self.compressor is not None:
                self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(self.emb_dim))
                for idx, latent in self.compressed.items():
                    vec = self.compressor.decode(torch.tensor(latent), self.emb_dim).numpy().reshape(1, -1)
                    if self.l2_norm:
                        vec = _l2norm(vec)
                    self.index.add_with_ids(vec, np.array([int(idx)], dtype="int64"))
            print(f"✅ Loaded CorrelationRAGMemory with {len(self.labels)} items.")
        except Exception as e:
            print("⚠️  Could not load memory:", e)

# ----------------------------------------------------------------------
if __name__ == "__main__":  # quick smoke test
    mem = CorrelationRAGMemory(emb_dim=128, memory_budget=1_000)

    rng = np.random.default_rng(0)
    for i in range(1500):   # add some fake embeddings
        emb = rng.standard_normal(128).astype("float32")
        label = rng.integers(0, 5)
        mem.add(emb, label, meta={"t": i})

        if i % 128 == 0:
            q = rng.standard_normal(128).astype("float32")
            logits, ctx = mem.classify(q)
            print(f"step {i:4d} -> logits {logits[:3]} (ctx {len(ctx)})")

    mem.save()
