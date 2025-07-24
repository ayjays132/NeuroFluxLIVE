from __future__ import annotations

"""Rich CLI showcasing premium widgets and logging."""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.syntax import Syntax


def run_demo(steps: int, svg_path: Optional[str], console: Console) -> None:
    """Run a small progress demo with optional SVG rendering."""
    if svg_path:
        path = Path(svg_path)
        if path.is_file():
            svg_content = path.read_text(encoding="utf-8")
            console.print(Syntax(svg_content, "xml", theme="ansi_dark"))
        else:
            console.print(f"[bold red]SVG {svg_path} not found[/bold red]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing", total=steps)
        for _ in range(steps):
            time.sleep(0.2)
            progress.update(task, advance=1)

    console.print(Panel.fit("Process finished", title="Status"))
    logging.info("Demo complete")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Premium CLI demo with widgets")
    parser.add_argument("--svg", help="Path to SVG to display")
    parser.add_argument("--steps", type=int, default=10, help="Progress steps")
    args = parser.parse_args(argv)

    console = Console()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    run_demo(args.steps, args.svg, console)


if __name__ == "__main__":
    main()
