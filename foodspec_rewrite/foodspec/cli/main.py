"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.
CLI main entry point.
"""

import typer
from pathlib import Path

app = typer.Typer(help="FoodSpec 2.0 clean architecture CLI")


@app.command()
def preprocess(
    input_file: Path = typer.Argument(..., help="Input spectral data"),
    output_dir: Path = typer.Option("./output", help="Output directory"),
    method: str = typer.Option("standard", help="Preprocessing method"),
):
    """Preprocess spectral data."""
    typer.echo(f"Preprocessing {input_file} with method={method}")
    typer.echo(f"Output to {output_dir}")


@app.command()
def analyze(
    input_file: Path = typer.Argument(..., help="Input data"),
    task: str = typer.Option("classification", help="Task type"),
):
    """Run analysis workflow."""
    typer.echo(f"Analyzing {input_file} for {task}")


@app.command()
def serve(
    model_path: Path = typer.Argument(..., help="Trained model"),
    port: int = typer.Option(8000, help="Server port"),
):
    """Start prediction server."""
    typer.echo(f"Starting server on port {port} with model {model_path}")


if __name__ == "__main__":
    app()
