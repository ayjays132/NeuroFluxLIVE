from __future__ import annotations

"""Self-learning workflow unifying real-time data absorption with prompt optimization."""

import argparse
import logging
import threading
import time
from queue import Queue
from typing import Optional


from main import load_config, initialize_system
from analysis.omni_prompt_optimizer import OmniPromptOptimizer
from interface.ws_server import run_ws_server
from interface.visual_cli import VisualCLI
from rich.logging import RichHandler


def process_latest(absorber, optimizer: OmniPromptOptimizer) -> Optional[str]:
    """Optimize a prompt for the most recent text datapoint."""
    if not getattr(absorber, "data_buffer", None):
        return None
    data_point = absorber.data_buffer[-1]
    if getattr(data_point, "modality", None) != "text":
        return None
    best = optimizer.optimize_prompt(data_point.data)
    return best


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the self-learning bot")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--watch-interval", type=float, default=5.0, help="Seconds between file scans"
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Display live metrics dashboard in the terminal",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])

    cfg = load_config(args.config)
    metrics_q: Queue = Queue()
    ws_thread = threading.Thread(target=run_ws_server, args=(metrics_q,), daemon=True)
    ws_thread.start()

    absorber, watcher, _ = initialize_system(cfg, metrics_q, args.watch_interval)
    optimizer = OmniPromptOptimizer(cfg.get("model", {}).get("name", "distilgpt2"))

    metrics_viewer = VisualCLI(metrics_q) if args.visual else None
    if metrics_viewer:
        metrics_viewer.start()

    absorber.start_absorption()
    watcher.start()

    try:
        while True:
            best = process_latest(absorber, optimizer)
            if best:
                print("Optimized prompt:", best)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        watcher.stop()
        absorber.stop_absorption()
        if metrics_viewer:
            metrics_viewer.stop()


if __name__ == "__main__":
    main()

