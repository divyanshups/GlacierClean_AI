'''LOGGER MODULE'''

import logging
import os
from logging.handlers import RotatingFileHandler
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text

console = Console()

def get_logger(module_name: str, log_file: str = 'logs/pipeline.log') -> logging.Logger:
    #CREATE output directory
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    #Logger Module
    logger = logging.getLogger(module_name)
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)

    #Console Handler Module
    rich_handler = RichHandler(
        console = console,
        show_time = True,
        show_path = True,
        markup = True,
        rich_tracebacks = True
        )
    rich_handler.setLevel(logging.INFO)

    #Log File Handler Module
    file_handler = RotatingFileHandler(
        filename = log_file,
        maxBytes = 10*1024*1024,
        backupCount = 3,
        encoding = 'utf-8'
    )
    file_handler.setLevel(logging.DEBUG)

    #Log File Formatter Snippet
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(rich_handler)
    logger.addHandler(file_handler)

    return logger

def print_banner():
    banner_text = Text()
    banner_text.append("Intelligent Data Cleaning Agent\n", style="bold cyan")
    banner_text.append("Prototype v0.1.0", style="dim")
    console.print(Panel(banner_text, expand=False, border_style="cyan"))

def print_section(title:str):
    console.print(f"\n[bold yellow]{'─' * 10} {title} {'─' * 10}[/bold yellow]\n")