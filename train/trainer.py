from __future__ import annotations

"""
🚀  Premium Training Utilities v2.0  🚀
A minimal-yet-powerful wrapper around 🤗 Transformers’ `Trainer`
that focuses purely on **training** (no eval) while staying GPU-savvy,
mixed-precision-ready, and logging-capable.
"""

from dataclasses import dataclass
from typing import Optional, List
import argparse
import json
import math
import os
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

# Local utilities
from .experiment_tracker import ExperimentTracker, TrackerCallback
from data.dataset_loader import load_dataset_splits


# ✨--------------------------------------------------------------------✨
#                           CONFIGURATION
# ✨--------------------------------------------------------------------✨
@dataclass
class TrainingConfig:
    """
    🎛️ Hyper-parameters & housekeeping

    Attributes
    ----------
    model_name:       🤗 Hub identifier (or local path)
    dataset_name:     Name / path understood by `load_dataset_splits`
    train_split:      Split string (e.g. 'train[:90%]')
    text_column:      Field containing raw text
    output_dir:       Where checkpoints land
    epochs:           Total epochs
    batch_size:       Per-device batch size
    lr:               Learning rate
    grad_accum:       Gradient-accumulation steps
    warmup_steps:     Scheduler warm-up
    lr_scheduler_type:One of {linear, cosine, …}
    max_grad_norm:    Gradient clipping
    fp16:             Enable float16
    bf16:             Enable bfloat16
    gradient_ckpt:    Activate gradient checkpointing
    device:           Override CUDA/CPU auto-detect
    log_file:         Optional JSON metrics dump
    """

    # core
    model_name: str
    dataset_name: str
    train_split: str
    eval_split: Optional[str] = None

    # data / text
    text_column: str = "text"

    # training
    output_dir: str = "./model_output"
    epochs: int = 1
    batch_size: int = 2
    lr: float = 5e-5
    grad_accum: int = 2
    warmup_steps: int = 0
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0

    # goodies
    fp16: bool = False
    bf16: bool = False
    gradient_ckpt: bool = False

    # misc
    device: Optional[str] = None
    log_file: Optional[str] = None


# ✨--------------------------------------------------------------------✨
#                           TRAINING LOGIC
# ✨--------------------------------------------------------------------✨
def train_model(
    cfg: TrainingConfig, *, extra_callbacks: Optional[List[TrainerCallback]] = None
) -> None:
    """Launch a *train-only* run with 🤗 Trainer."""

    # 🌐 Detect device --------------------------------------------------
    device = torch.device(
        cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()  # keep VRAM tidy
        print(f"🟢 Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("🟡 CUDA not found—falling back to CPU.")

    # 📦 Dataset --------------------------------------------------------
    train_ds, eval_ds = load_dataset_splits(
        cfg.dataset_name,
        cfg.model_name,
        train_split=cfg.train_split,
        eval_split=cfg.eval_split,
        text_column=cfg.text_column,
    )

    # 🧠 Model & Tokenizer ---------------------------------------------
    print("🔄 Loading model & tokenizer…")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    if cfg.gradient_ckpt:
        model.gradient_checkpointing_enable()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ⚙️ TrainingArguments ---------------------------------------------
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        gradient_accumulation_steps=cfg.grad_accum,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_grad_norm=cfg.max_grad_norm,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_steps=10,  # frequent logs 📊
        save_steps=500,
        save_total_limit=3,  # keep disk usage sane
        report_to="none",
    )

    # 🧑‍💻 Callbacks -----------------------------------------------------
    callbacks: List[TrainerCallback] = extra_callbacks or []
    tracker = ExperimentTracker() if cfg.log_file else None
    if tracker:
        callbacks.append(TrackerCallback(tracker))

    # 🚀 Training -------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    print("✨ Starting training!")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    if eval_ds is not None:
        print("\n🔍 Running evaluation...")
        eval_metrics = trainer.evaluate()
        if "eval_loss" in eval_metrics:
            ppl = math.exp(eval_metrics["eval_loss"])
            print(f"Perplexity: {ppl:.2f}")

    # 📝 Metrics dump ---------------------------------------------------
    if cfg.log_file and tracker:
        tracker.save(cfg.log_file)
        print(f"📑 Metrics written to {cfg.log_file}")

    print("✅ Training complete.")


# ✨--------------------------------------------------------------------✨
#                             CLI ENTRY
# ✨--------------------------------------------------------------------✨
def main() -> None:
    parser = argparse.ArgumentParser(description="🚀 Train a causal LM (train-only)")

    # Positional
    parser.add_argument("model_name", help="🤗 Model name-or-path")
    parser.add_argument("dataset", help="Dataset name/path for `load_dataset_splits`")
    parser.add_argument("train_split", help="Training split (e.g. 'train[:1000]')")

    # Optional
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-scheduler", default="linear")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-ckpt", action="store_true")
    parser.add_argument("--output-dir", default="./model_output")
    parser.add_argument("--log-file", help="Write JSON metrics here")
    parser.add_argument("--device", help="Force 'cuda' or 'cpu'")

    args = parser.parse_args()

    cfg = TrainingConfig(
        model_name=args.model_name,
        dataset_name=args.dataset,
        train_split=args.train_split,
        text_column=args.text_column,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_accum=args.grad_accum,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler,
        max_grad_norm=args.max_grad_norm,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_ckpt=args.gradient_ckpt,
        device=args.device,
        log_file=args.log_file,
    )

    train_model(cfg)


if __name__ == "__main__":
    main()
