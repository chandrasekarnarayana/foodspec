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

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from foodspec.cli.commands.run import run as run_command
from foodspec.deploy import load_bundle, predict_from_bundle_path

app = typer.Typer(
    help="FoodSpec v2.0 - Spectral analysis framework for food authentication",
    add_completion=False,
)

# Register run command from commands module
app.command()(run_command)


@app.command()
def predict(
    bundle: Path = typer.Option(
        ...,
        "--bundle",
        "-b",
        exists=True,
        help="Path to deployment bundle directory",
    ),
    input_csv: Path = typer.Option(
        ...,
        "--input",
        "-i",
        exists=True,
        help="Path to input CSV file with spectra",
    ),
    outdir: Path = typer.Option(
        "./predictions",
        "--outdir",
        "-o",
        help="Output directory for predictions",
    ),
    sample_id_col: str = typer.Option(
        "sample_id",
        "--sample-id-col",
        help="Column name for sample identifiers",
    ),
    wavenumber_col: str = typer.Option(
        "wavenumber",
        "--wavenumber-col",
        help="Column name for wavenumber values",
    ),
    intensity_col: str = typer.Option(
        "intensity",
        "--intensity-col",
        help="Column name for intensity values",
    ),
    no_probabilities: bool = typer.Option(
        False,
        "--no-probabilities",
        help="Skip saving probability matrix",
    ),
) -> None:
    """Make predictions on new data using a deployment bundle.

    Loads a trained model bundle and applies it to new spectral data in CSV format.
    Automatically applies preprocessing pipeline and generates predictions.

    Example:
        foodspec predict --bundle ./bundle --input new_data.csv --outdir predictions
    """
    typer.echo("=" * 70)
    typer.echo("FoodSpec v2.0 - Prediction")
    typer.echo("=" * 70)
    typer.echo(f"Bundle:   {bundle}")
    typer.echo(f"Input:    {input_csv}")
    typer.echo(f"Output:   {outdir}")
    typer.echo("")

    try:
        # Make predictions
        predictions_df = predict_from_bundle_path(
            bundle_dir=bundle,
            input_csv=input_csv,
            output_dir=outdir,
            sample_id_col=sample_id_col,
            wavenumber_col=wavenumber_col,
            intensity_col=intensity_col,
            save_probabilities=not no_probabilities,
        )

        # Print summary
        typer.echo("✓ Predictions completed successfully")
        typer.echo("")
        typer.echo("Summary:")
        typer.echo(f"  Samples processed: {len(predictions_df)}")
        
        # Class distribution
        typer.echo(f"  Prediction distribution:")
        for label, count in predictions_df["predicted_label"].value_counts().items():
            typer.echo(f"    {label}: {count} samples")
        
        typer.echo("")
        typer.echo(f"Output files:")
        typer.echo(f"  Predictions:   {outdir}/predictions.csv")
        if not no_probabilities:
            typer.echo(f"  Probabilities: {outdir}/probabilities.csv")
        typer.echo("=" * 70)

    except FileNotFoundError as e:
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except ValueError as e:
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command()
def report(
    run_dir: Path = typer.Option(
        ...,
        "--run-dir",
        "-r",
        exists=True,
        help="Path to FoodSpec run directory",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output HTML report path (default: <run_dir>/artifacts/report.html)",
    ),
    title: str = typer.Option(
        "FoodSpec Analysis Report",
        "--title",
        "-t",
        help="Report title",
    ),
) -> None:
    """Generate HTML report from a completed FoodSpec run.

    Creates a comprehensive HTML report with run summary, metrics, plots, and analysis results.

    Example:
        foodspec report --run-dir runs/exp1 --output report.html
    """
    typer.echo("=" * 70)
    typer.echo("FoodSpec v2.0 - Report Generation")
    typer.echo("=" * 70)
    typer.echo(f"Run directory: {run_dir}")
    typer.echo("")

    try:
        # Determine output path
        if output is None:
            output = run_dir / "artifacts" / "report.html"
        
        # Load manifest
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            typer.secho(
                f"✗ Error: Manifest not found at {manifest_path}",
                fg=typer.colors.RED,
                err=True,
            )
            typer.secho(
                "  Ensure the run directory contains a valid manifest.json",
                fg=typer.colors.YELLOW,
                err=True,
            )
            raise typer.Exit(code=1)

        # For now, create a simple HTML report
        import json
        
        manifest_data = json.loads(manifest_path.read_text())
        
        # Create simple HTML report
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        .section {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-left: 4px solid #007bff; }}
        .key {{ font-weight: bold; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="section">
        <h2>Run Summary</h2>
        <p><span class="key">Run ID:</span> {manifest_data.get('run_id', 'N/A')}</p>
        <p><span class="key">Seed:</span> {manifest_data.get('seed', 'N/A')}</p>
        <p><span class="key">Protocol Hash:</span> {manifest_data.get('protocol_hash', 'N/A')}</p>
    </div>
    <div class="section">
        <h2>Artifacts</h2>
        <ul>
            <li>Manifest: {run_dir / 'manifest.json'}</li>
            <li>Logs: {run_dir / 'logs.txt'}</li>
        </ul>
    </div>
</body>
</html>"""
        
        # Ensure output directory exists
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html_content)

        typer.echo("✓ Report generated successfully")
        typer.echo("")
        typer.echo(f"Report saved to: {output}")
        typer.echo(f"  Size: {output.stat().st_size} bytes")
        typer.echo("")
        typer.echo(f"Open in browser: file://{output.absolute()}")
        typer.echo("=" * 70)

    except FileNotFoundError as e:
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Show FoodSpec version information."""
    typer.echo("FoodSpec v2.0.0")
    typer.echo("Spectral analysis framework for food authentication")
    typer.echo("")
    typer.echo("Documentation: https://github.com/chandrasekarnarayana/foodspec")


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    app()
