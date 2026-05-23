from __future__ import annotations
import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from core.pipeline import CleaningPipeline

console = Console()

def parse_args(argv=None) -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Autonomous Data Cleaning Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input dataset (CSV, TSV, XLSX, JSON, or Parquet)",
    )
    parser.add_argument(
        "--config", "-c",
        default="config/config.yaml",
        help="Path to the configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the pipeline without saving the cleaned dataset to disk",
    )
    return parser.parse_args(argv)


def print_summary(result: dict) -> None:
    """Print a formatted summary of the pipeline run to the console."""
    report = result["report"]
    qs     = report["quality_score"]
    delta  = report["delta"]

    console.rule("[bold cyan]Pipeline Summary[/bold cyan]")

    console.print(
        f"\n  Quality score: [bold]{qs['before']}[/bold] → "
        f"[bold green]{qs['after']}[/bold green]  "
        f"([green]+{qs['improvement']}[/green] pts)\n"
    )

    # Build a table showing the before/after values for each key metric
    table = Table(title="Key Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Metric",  style="cyan")
    table.add_column("Before",  justify="right")
    table.add_column("After",   justify="right")
    table.add_column("Change",  justify="right")

    for key, label in [
        ("missing_rate",   "Missing %"),
        ("duplicate_rate", "Duplicate %"),
        ("outlier_rate",   "Outlier %"),
    ]:
        metric = delta[key]
        change = metric["absolute_change"]
        # Green means the metric improved (went down); red means it got worse
        colour = "green" if change <= 0 else "red"
        table.add_row(
            label,
            f"{metric['before']:.2f}%",
            f"{metric['after']:.2f}%",
            f"[{colour}]{change:+.2f}%[/{colour}]",
        )

    row_retention = delta["row_retention"]
    table.add_row(
        "Row retention",
        "100.00%",
        f"{row_retention['after']:.2f}%",
        f"[yellow]{row_retention['rows_removed']:,} removed[/yellow]",
    )
    console.print(table)

    console.print(
        f"\n  Issues detected:  [bold]{report['issues_detected']}[/bold]\n"
        f"  Actions planned:  [bold]{report['actions_planned']}[/bold]\n"
        f"  Actions applied:  [bold green]{report['actions_applied']}[/bold green]\n"
    )
    console.rule()


def main(argv=None) -> int:
    """Run the cleaning pipeline and return an exit code (0 = success, 1 = error)."""
    args = parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {args.input}")
        return 1

    pipeline = CleaningPipeline(config_path=args.config)

    try:
        result = pipeline.run(
            source      = str(input_path),
            file_name   = input_path.name,
            save_output = not args.no_save,
        )
    except Exception as error:
        console.print(f"[bold red]Pipeline failed:[/bold red] {error}")
        raise

    print_summary(result)
    return 0

if __name__ == "__main__":
    sys.exit(main())
