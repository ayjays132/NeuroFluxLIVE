"""
token_sequence_pattern_manager.py
─────────────────────────────────
Light-weight pattern recogniser & dataset manager for token sequences.
Designed to pair with RealTimeDataAbsorber or run standalone.

Core features
-------------
• Tokenise text (HF tokenizer optional, regex fallback)
• Maintain rolling frequency stats for n-grams (1–5)
• Detect bursts (rapid ↑ in freq) & rare sequences
• Store sequences in JSONL dataset with dedup + fast lookup
• Sample batches & export vocab for downstream training
"""

from __future__ import annotations
import re, json, time, random, math, pathlib
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional

# Optional: if you want the power of HF tokenisation, uncomment:
# from transformers import AutoTokenizer
# TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
TOKENIZER = None  # fallback to regex

RE_TOKEN = re.compile(r"\w+|\S")  # crude but works

def simple_tokenize(txt: str) -> List[str]:
    return RE_TOKEN.findall(txt.lower())

def tokenize(text: str) -> List[str]:
    if TOKENIZER:
        return TOKENIZER.tokenize(text)
    return simple_tokenize(text)

# --------------------------------------------------------------------------- #
class TokenSequenceDatasetManager:
    """JSONL-backed sequence store with dedup + quick sampling."""
    def __init__(self, path: str = "token_sequences.jsonl"):
        self.path = pathlib.Path(path)
        self._ids  : set[str] = set()
        self._load_existing()

    def _load_existing(self):
        if not self.path.exists(): return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    self._ids.add(obj["id"])
                except Exception:
                    continue

    def _write(self, obj: Dict):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False)+"\n")

    def add_sequence(self, text: str, meta: Optional[Dict]=None) -> bool:
        seq_id = str(hash(text))
        if seq_id in self._ids:
            return False
        obj = {"id": seq_id, "text": text, "meta": meta or {}, "time": time.time()}
        self._write(obj)
        self._ids.add(seq_id)
        return True

    def sample(self, k: int=32) -> List[Dict]:
        """Reservoir-sample k examples without loading all lines."""
        reservoir, n = [], 0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                n += 1
                if len(reservoir) < k:
                    reservoir.append(json.loads(line))
                else:
                    j = random.randrange(n)
                    if j < k:
                        reservoir[j] = json.loads(line)
        return reservoir

    def export_vocab(self, min_freq: int = 2) -> Dict[str, int]:
        vocab = Counter()
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["text"]
                vocab.update(tokenize(text))
        return {tok: c for tok, c in vocab.items() if c >= min_freq}

# --------------------------------------------------------------------------- #
class PatternRecognizer:
    """Tracks n-gram frequencies and flags bursts / rarities."""
    def __init__(self, n_max: int = 5, window: int = 10_000):
        self.n_max     = n_max
        self.window    = window
        self.freqs_now = [Counter() for _ in range(n_max)]
        self.freqs_old = [Counter() for _ in range(n_max)]
        self.seen      = 0

    def _ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def observe(self, text: str) -> Dict[str, List[Tuple[str, ...]]]:
        """Return detected patterns: {'burst': [...], 'rare': [...]}"""
        tokens = tokenize(text)
        self.seen += 1
        bursts, rares = [], []

        for n in range(1, self.n_max+1):
            ngr = self._ngrams(tokens, n)
            now, old = self.freqs_now[n-1], self.freqs_old[n-1]

            for tup in ngr:
                now[tup] += 1

                # Burst detection: freq jumped >2× compared to old window
                if old[tup] and now[tup] / (old[tup] + 1e-9) > 2.0 and now[tup] > 5:
                    bursts.append(tup)

                # Rare pattern: still very low overall
                if now[tup] == 1 and old[tup] == 0:
                    rares.append(tup)

        # Slide window
        if self.seen % self.window == 0:
            self.freqs_old = self.freqs_now
            self.freqs_now = [Counter() for _ in range(self.n_max)]

        return {"burst": bursts, "rare": rares}

# --------------------------------------------------------------------------- #
# Mini-demo (can be removed when importing)
if __name__ == "__main__":
    ds  = TokenSequenceDatasetManager("toy_ds.jsonl")
    pr  = PatternRecognizer()

    texts = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the very lazy dog",
        "deep learning enables quick adaptation",
        "the quick brown fox is quick and clever",
        "deep learning enables deep insights",
    ]
