from __future__ import annotations

"""Live metrics display with Rich widgets and SVG export."""

from queue import Queue, Empty
from typing import Dict, Optional
import threading
import time

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
import matplotlib.pyplot as plt


class VisualCLI:
    """Render performance metrics as Rich widgets and save SVG charts."""

    def __init__(self, metrics_queue: Queue, svg_path: str = "metrics.svg") -> None:
        self.metrics_queue = metrics_queue
        self.svg_path = svg_path
        self._running = False
        self.thread: Optional[threading.Thread] = None
        self.latest_metrics: Dict[str, float] = {}

    def _render_table(self) -> Table:
        table = Table(title="Live Metrics", expand=True)
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        for key, value in self.latest_metrics.items():
            table.add_row(key, f"{value}")
        return table

    def _update_svg(self) -> None:
        if not self.latest_metrics:
            return
        plt.clf()
        names = list(self.latest_metrics.keys())
        values = list(self.latest_metrics.values())
        plt.bar(names, values, color="#4b9cd3")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(self.svg_path, format="svg")

    def _loop(self) -> None:
        console = Console()
        progress = Progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"))
        task_id = progress.add_task("Processing", total=None)
        with Live(console=console, refresh_per_second=4) as live:
            live.update(Panel(progress))
            while self._running:
                try:
                    data = self.metrics_queue.get_nowait()
                    if isinstance(data, dict):
                        self.latest_metrics.update(data)
                        table = self._render_table()
                        live.update(Panel(table))
                        self._update_svg()
                except Empty:
                    pass
                time.sleep(0.1)
        progress.stop()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._running = False
        if self.thread:
            self.thread.join()

