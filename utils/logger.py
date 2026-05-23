from __future__ import annotations
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text

_console = Console()


def get_logger(name: str = "agent", log_file: str = "memory/agent.log") -> logging.Logger:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger   # already configured — return as-is

    logger.setLevel(logging.DEBUG)

    # Console handler: INFO and above, with Rich colours and formatting
    console_handler = RichHandler(
        console         = _console,
        rich_tracebacks = True,
        markup          = True,
        show_path       = False,
    )
    console_handler.setLevel(logging.INFO)

    # File handler: DEBUG and above, plain timestamped text
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def log_section(title: str, subtitle: str = "") -> None:
    text = Text(title, style="bold cyan")
    if subtitle:
        text.append(f"\n{subtitle}", style="dim white")
    _console.print(Panel(text, border_style="cyan", expand=False))


def log_decision(
    decision_log_path: str,
    agent:             str,
    action:            str,
    rationale:         str,
    params:            dict[str, Any] | None = None,
    confidence:        float | None          = None,
) -> None:
    Path(decision_log_path).parent.mkdir(parents=True, exist_ok=True)

    entry: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "agent":     agent,
        "action":    action,
        "rationale": rationale,
    }
    if params:
        entry["params"] = params
    if confidence is not None:
        entry["confidence"] = round(confidence, 4)

    # Read → append → write
    log_path  = Path(decision_log_path)
    existing: list[dict] = []
    if log_path.exists() and log_path.stat().st_size > 0:
        try:
            with open(log_path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
        except json.JSONDecodeError:
            existing = []   # corrupt file — start fresh

    existing.append(entry)
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2, default=str)
